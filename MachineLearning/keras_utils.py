from __future__ import print_function

import json
import os
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import progressbar
import tensorflow.python.keras.backend as K
from numpy.random import choice
from scipy.sparse import load_npz
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop

from tensorflow.python.keras.callbacks import LambdaCallback, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.regularizers import l2
from Tools.io import load_img
from MachineLearning.architectures import get_model

# -1 if the label was not found, since the eval set may contain species not encountered during training
def get_index(d, x):
    r = d.get(x)
    return r if r is not None else -1


def transfer_weights(class_ids: (list, tuple), previous_class_ids: (list, tuple), previous_weights_path: str):
    """
    Converts weights from a previous model for a set of classes to a new set of classes
    :param class_ids: list of new class IDs
    :param previous_class_ids: list of previous class IDs
    :param previous_weights_path: path to HDF5 file with stored weights of previous models
    :return: predictions_name_new, transferred_weights
    """
    predictions_name_new = "new_predictions"
    transferred_weights = None
    if previous_weights_path is not None and os.path.exists(previous_weights_path):
        f_prev = h5py.File(previous_weights_path)
        group = f_prev

        key_prefix = ""
        group = group if "model_weights" not in group.keys() else group["model_weights"]

        predictions_name_old = "new_predictions2" if "new_predictions2" in group.keys() else "new_predictions"
        predictions_name_new = "new_predictions" if "new_predictions2" in group.keys() else "new_predictions2"

        print(group[predictions_name_old])

        if previous_class_ids is not None:
            kernel = group[predictions_name_old][key_prefix + predictions_name_old]["kernel:0"][()]
            bias = group[predictions_name_old][key_prefix + predictions_name_old]["bias:0"][()]

            transferred_weights = np.zeros((kernel.shape[0], len(class_ids)))
            transferred_bias = np.zeros((len(class_ids),))

            previous_labels_set = set(previous_class_ids)

            for l, label in enumerate(class_ids):
                if label in previous_labels_set:
                    prev_index = previous_class_ids.index(label)
                    transferred_weights[:, l] = kernel[:, prev_index]
                    transferred_bias[l] = bias[prev_index]
                else:
                    prev_indices = np.sort(
                        np.random.choice(range(len(previous_class_ids)), replace=False, size=min(50, kernel.shape[1])))
                    transferred_weights[:, l] = np.mean(kernel[:, prev_indices], axis=1)
                    transferred_bias[l] = np.mean(bias[prev_indices])

            transferred_weights = [transferred_weights, transferred_bias]

    return predictions_name_new, transferred_weights


def train(
        training_folder,
        dataset_path,
        previous_labels_path=None,
        previous_weights_path=None,
        stages=None,
        previous_model_folder=None,
        model_configuration={"architecture": "inceptionv3", "num_fully_connected_nodes": 1024,
                             "squaring_method": "crop"},
        last_layer_l2_penalty=0.,
        caller_filename=None,
        replace_paths=None,
        batch_size=16):
    """
    Trains a model from a dataset

    The training takes place in several stages:
    - stage 1: only fine-tune output layer
    - stage 2: fine-tune upper layers
    - stage 3: fine-tune all layers

    Stage configuration is a dictionary with fields:
    * stage: number of stage
    * reduce_lr_patience: number of epochs without improvement (plateau) before learning rate is reduced
    * reduce_lr_factor: factor by which to reduce learning rate when a plateau is reached during training
    * num_epochs: number of epochs for phase
    * epoch_fraction: fraction of whole training set to use per epoch
    * initial_learning_rate: learning rate at the start of a phase

    example {"stage": 1, "reduce_lr_factor": 0.2, "reduce_lr_patience": 2, "num_epochs": 4, "epoch_fraction": 0.5,
         "initial_learning_rate": 1e-4}

    Model configuration is a dictionary with fields:
    * architecture: see get_model_no_top parameter "architecture_name"
    * num_fully_connected_nodes: see get_model_no_top parameter "num_fully_connected_nodes"
    * squaring_method: method to square the image, in ["crop", "pad"]

    :param training_folder: (temporary) folder for storing training results
    :param dataset_path:
    :param logger:
    :param previous_labels_path: experimental, obsolete
    :param previous_weights_path: experimental, obsolete
    :param stages: training stages, see main documentation text
    :param previous_model_folder: model folder to continue training on, typically a Repository Model folder, the folder should contain the weights file and a labels file
    :param model_configuration: model configuration, see main documentation text
    :param last_layer_l2_penalty: experimental
    :param caller_filename: copies the filename of the calling script to the training folder, useful for keeping history/debugging
    :param count_cap: EXPERIMENTAL, if None or 0 no class balancing is performed, if 1 class balancing is performed
    :param replace_paths: experimental, obsolete
    :return:
    """
    Path(training_folder).mkdir(exist_ok=True, parents=True)

    if "squaring_method" not in model_configuration:
        model_configuration["squaring_method"] = "crop"

    stages_predefined = [
        {"stage": 1, "reduce_lr_factor": 0.2, "reduce_lr_patience": 2, "num_epochs": 4, "epoch_fraction": 0.5,
         "initial_learning_rate": 1e-4},
        {"stage": 2, "reduce_lr_factor": 0.1, "reduce_lr_patience": 2, "num_epochs": 4, "epoch_fraction": 0.5,
         "initial_learning_rate": 1e-4},
        {"stage": 3, "reduce_lr_factor": 0.1, "reduce_lr_patience": 3, "num_epochs": 200,
         "epoch_fraction": 1.0, "initial_learning_rate": 1e-4},
    ]

    if stages is None:
        stages = stages_predefined

    if previous_model_folder is not None:
        previous_labels_path = os.path.join(previous_model_folder, "labels.txt")
        previous_weights_path = os.path.join(previous_model_folder, "weights.h5")

    # if caller_filename is not None:
    #     shutil.copy(caller_filename, training_folder)

    json.dump(model_configuration, open(os.path.join(training_folder, "model.json"), "w"))

    # generators
    train_datagen = image.ImageDataGenerator(
        preprocessing_function=lambda x: (x / 255.0 - 0.5) * 2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=90)

    test_datagen = image.ImageDataGenerator(preprocessing_function=lambda x: (x / 255.0 - 0.5) * 2)

    previous_labels = None
    if previous_labels_path is not None and os.path.exists(previous_labels_path):
        previous_labels = list(map(lambda x: x.split(":")[1].strip(), open(previous_labels_path).readlines()))

    img_df = pd.read_csv(dataset_path)
    num_train_images = len(img_df[img_df['set'] == 'train'])
    labels = np.unique(img_df['taxon_id_at_source'].tolist())

    with open(os.path.join(training_folder, "labels.txt"), "w") as labelfile:
        for i, c in enumerate(labels):
            labelfile.write("{}:{}\n".format(i, c))

    num_classes = len(labels)

    loss = "categorical_crossentropy"

    # --
    predictions_name_new, transferred_params = transfer_weights(labels, previous_labels, previous_weights_path)

    architecture_name = model_configuration["architecture"]
    base_model, x, target_size = get_model_no_top(architecture_name, model_configuration[
        "num_fully_connected_nodes"])
    if transferred_params is None:
        predictions = Dense(num_classes, activation='softmax', name=predictions_name_new,
                            activity_regularizer=l2(last_layer_l2_penalty))(x)
    else:
        predictions = Dense(num_classes, activation='softmax', name=predictions_name_new, weights=transferred_params,
                            activity_regularizer=l2(last_layer_l2_penalty))(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # co_occ_input = keras.Input((None, 64))
    # concatenation = keras.layers.concatenate([x, co_occ_input], axis=1)
    # predictions_ensemble = Dense(num_classes, activation='softmax', name=predictions_name_new)(concatenation)
    #
    # model = Model(inputs=[base_model.input, co_occ_input], outputs=predictions_ensemble)

    model.compile(optimizer=RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0), loss=loss, metrics=['accuracy'])

    # load previous weights
    if previous_weights_path is not None and os.path.exists(previous_weights_path):
        print("Loading existing weights from {}".format(previous_weights_path))
        model.load_weights(previous_weights_path, by_name=True)

    test_iterator = DataFrameIterator(
        img_df,
        test_datagen,
        target_size=target_size,
        color_mode="rgb",
        class_mode="categorical",
        data_format=None,
        batch_size=batch_size,
        shuffle=False,
        seed=None,
        set_name="validation",
        verbose=1,
        save_to_dir=None,
        squaring_method=model_configuration["squaring_method"],
        replace_paths=replace_paths)

    label_validation = img_df[img_df['set'] == 'validation']['taxon_id_at_source'].tolist()

    # num_batches = np.ceil(len(label_validation) / float(batch_size))

    test_iterator.set_progressbar(len(label_validation))

    plot_loss_callback = LambdaCallback(on_batch_end=lambda batch, logs: print(batch, logs))

    sample_log_folder = None

    train_iterator = DataFrameIterator(
        img_df,
        train_datagen,
        target_size=target_size,
        color_mode="rgb",
        class_mode="categorical",
        data_format=None,
        save_to_dir=None,
        batch_size=batch_size,
        shuffle=True,
        seed=None,
        squaring_method=model_configuration["squaring_method"],
        sample_log_folder=sample_log_folder,
        replace_paths=replace_paths
    )

    for stage_dict in stages:
        stage = stage_dict["stage"]

        weights_path = os.path.join(training_folder, "weights_stage{}{}.h5".format(stage, stage_dict[
            "sub"] if "sub" in stage_dict else ""))

        if not os.path.exists(weights_path):
            if stage == 1:
                # first: train only the top layers (which were randomly initialized)
                # i.e. freeze all convolutional InceptionV3 layers
                for layer in base_model.layers:
                    layer.trainable = False
            elif stage == 2:
                if architecture_name != "inceptionv3":
                    raise NotImplementedError("Stage 2 not (yet) defined for models other than inceptionv3")
                # we chose to train the top 2 inception blocks, i.e. we will freeze
                # the first 172 layers and unfreeze the rest:
                for layer in model.layers[:172]:
                    layer.trainable = False
                for layer in model.layers[172:]:
                    layer.trainable = True
            elif stage == 3:
                # make all layers trainable
                for layer in model.layers:
                    layer.trainable = True

            # recompile the model (should be done *after* setting layers to non-trainable)
            model.compile(optimizer=RMSprop(lr=stage_dict["initial_learning_rate"], rho=0.9, epsilon=1e-08, decay=0.0),
                          loss=loss,
                          metrics=['accuracy'])

            # train the model on the new data for a few epochs
            epoch_fraction = stage_dict["epoch_fraction"]
            num_epochs = stage_dict["num_epochs"]
            steps_per_epoch = int(num_train_images / float(batch_size) * epoch_fraction)

            reduce_lr_factor = stage_dict["reduce_lr_factor"]
            reduce_lr_patience = stage_dict["reduce_lr_patience"]

            reduce_learningrate_callback = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor,
                                                             patience=reduce_lr_patience, min_lr=1e-8,
                                                             verbose=1)

            checkpointer = ModelCheckpoint(filepath=weights_path,
                                           verbose=1,
                                           save_best_only=True,
                                           monitor="val_accuracy"
                                           )

            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(training_folder, "logs"),
                                                               write_graph=True,
                                                               write_images=True)

            earlyStopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                   patience=8,
                                                                   verbose=1,
                                                                   # restore_best_weights=True
                                                                   )

            # Setting the weights

            model.fit(
                train_iterator,
                steps_per_epoch=steps_per_epoch,
                epochs=num_epochs,
                callbacks=[plot_loss_callback, reduce_learningrate_callback, checkpointer,
                           tensorboard_callback,
                           earlyStopping_callback,
                           ],
                validation_data=test_iterator,
                validation_steps=np.ceil(len(label_validation) / float(batch_size)),
                workers=4)  # , use_multiprocessing = True)

            # model.save(os.path.join(training_folder, 'model_stage_' + str(stage_dict["stage"])))

        else:
            model.load_weights(weights_path, by_name=True)


def get_model_no_top(architecture_name, num_fully_connected_nodes, dropout=0.0):
    """
    Get Keras model based on configuration without the output layer (hence the "no_top")
    :param architecture_name: in ["inceptionv3", "inception-resnet", "mobilenet", "nasnet-mobile"]
    :param num_fully_connected_nodes: number of nodes in layer before output layer
    :param dropout: dropout probability for single last layer
    :return: base_model, last_full, image_size
    """
    input_tensor = None

    # create the base pre-trained model
    base_model, image_size = get_model(architecture_name, input_tensor)

    last_full = base_model.output
    # add a global spatial average pooling layer
    last_full = GlobalAveragePooling2D()(last_full)
    if dropout > 0:
        last_full = Dropout(dropout)(last_full)

    if num_fully_connected_nodes > 0:
        last_full = Dense(num_fully_connected_nodes, activation='relu')(last_full)

    return base_model, last_full, image_size


class DataFrameIterator(image.Iterator):
    """Iterator capable of reading image references from a csv file on disk, for Keras>2.0.8

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self,
                 dataframe,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='jpeg',
                 set_name="train",
                 interpolation='nearest',
                 verbose=0,
                 squaring_method="crop",
                 sample_log_folder=None,
                 replace_paths=None):

        classes = list(map(lambda x: x.split(":")[1].strip(), open(
            os.path.join(os.environ.get('IMG_ROOT'), 'labels.txt')).readlines()))

        self.sample_log_folder = sample_log_folder
        self.squaring_method = squaring_method
        if data_format is None:
            data_format = K.image_data_format()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes

        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        self.verbose = verbose

        # first, count the number of samples and classes
        self.samples = 0

        self.num_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))

        dataframe = dataframe[dataframe['set'] == set_name]
        self.filenames = dataframe['image_id'].tolist()
        labels = dataframe['taxon_id_at_source'].tolist()
        labels = list(map(lambda x: int(x), labels))

        self.filenames = list(
            map(lambda x: os.path.join(os.environ["PROJECT_ROOT"], 'images', x + '.jpg'), self.filenames))

        # -1 if the label was not found, since the eval set may contain species not encountered during training
        if self.class_mode is not None:

            self.labels = np.array(
                list(map(lambda x: get_index(self.class_indices, str(x)), labels)),
                dtype="int32")


        if replace_paths is not None:
            for from_path_part, to_path_part in replace_paths:
                self.filenames = [path.replace(from_path_part, to_path_part) for path in self.filenames]

        self.samples = len(self.filenames)
        self.__myindex = 0

        if self.verbose:
            self.set_progressbar(self.samples)
        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        super(DataFrameIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):

        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        i = 0
        for j in index_array:
            filename = self.filenames[j]
            img = load_img(filename,
                           grayscale=grayscale,
                           target_size=self.target_size,
                           squaring_method=self.squaring_method)
            try:
                x = image.img_to_array(img, data_format=self.data_format)
            except ValueError:
                print(filename)
                continue

            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            # plt.imshow(image.array_to_img(x, self.data_format, scale=True))
            # plt.show()
            batch_x[i] = x
            # optionally save augmented images to disk for debugging purposes
            # if self.save_to_dir:
            #     img = image.array_to_img(batch_x[i], self.data_format, scale=True)
            #     img.save(os.path.join(self.save_to_dir, os.path.basename(filename)))
            #     print("Saved to", os.path.join(self.save_to_dir, os.path.basename(filename)))

            i += 1

        self.__myindex += len(index_array)
        if self.__myindex > self.samples:
            self.__myindex = 0

        if self.verbose:
            if self.progressbar is not None:
                try:
                    self.progressbar.update(self.__myindex)
                except:
                    print("Exception during updating of progressbar: {}".format(traceback.format_exc()))
            else:
                print("next,", self.batch_index * self.batch_size, len(index_array))

        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            # -1 means the eval set contains a species not encountered during training, then all is zero
            for i, label in enumerate(self.labels[index_array]):
                if label > -1:
                    batch_y[i, label] = 1.
        else:
            return batch_x

        return batch_x, batch_y
        # return [batch_x, batch_x_co_occ], batch_y

    def reset(self):
        self.__myindex = 0
        super(DataFrameIterator, self).reset()

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        return self._get_batches_of_transformed_samples(index_array)

    def set_progressbar(self, max_value):
        self.progressbar = progressbar.ProgressBar(maxval=max_value)
        self.progressbar.start()


class DataFrameEnsembleIterator(image.Iterator):
    """Iterator capable of reading image references from a H5 file on disk, for Keras>2.0.8

    # Arguments
        directory: Path to the directory to read images from.
            Each subdirectory in this directory will be
            considered to contain images from one class,
            or alternatively you could specify class subdirectories
            via the `classes` argument.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of subdirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `"input"`: targets are images identical to input images (mainly
                used to work with autoencoders),
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.
    """

    def __init__(self,
                 dataframe,
                 image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='jpeg',
                 set_name="train",
                 interpolation='nearest',
                 verbose=0,
                 squaring_method="crop",
                 sample_log_folder=None,
                 replace_paths=None,
                 co_occ_matrix=None,
                 co_occ_metadata=None,
                 co_occ_model=None,
                 cells_per_degree=12,
                 penultimate_neurons=64,
                 ):

        # This assumes that there are no classes with images but without (imageless) observations.
        # A safe bet but still a bet
        self.classes = list(map(lambda x: x.split(":")[1].strip(), open(
            os.path.join(os.environ.get('ENS_ROOT'), 'labels.txt')).readlines()))

        self.sample_log_folder = sample_log_folder
        self.squaring_method = squaring_method
        if data_format is None:
            data_format = K.image_data_format()
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        if class_mode not in {'categorical', 'binary', 'sparse',
                              'input', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", "input"'
                             ' or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.interpolation = interpolation
        self.verbose = verbose

        # first, count the number of samples and classes
        self.samples = 0

        dataframe = dataframe[dataframe['set'] == set_name]
        self.filenames = dataframe['image_id'].tolist()

        self.num_class = len(self.classes)
        self.class_indices = dict(zip(self.classes, range(len(self.classes))))

        # -1 if the label was not found, since the eval set may contain species not encountered during training
        if self.class_mode is not None:
            self.labels = np.array(
                list(map(lambda x: get_index(self.class_indices, str(x)), dataframe['taxon_id_at_source'].tolist())),
                dtype="int32")

        dataframe = dataframe[['image_id', 'location_latitude', 'location_longitude']]
        dataframe.set_index(['image_id'], inplace=True)
        dataframe.columns = ['lat', 'lon']

        dataframe['lat'] = np.floor(dataframe['lat'] * cells_per_degree) / cells_per_degree + (
                .5 / cells_per_degree)
        dataframe['lon'] = np.floor(dataframe['lon'] * cells_per_degree) / cells_per_degree + (
                .5 / cells_per_degree)

        self.coordinates = dataframe.to_dict(orient='index')
        self.co_occ_model = load_model(co_occ_model)

        # Get the name of the last dense layer that is not the output layer
        last_dense_name = None
        for l in reversed(self.co_occ_model.layers[:-1]):
            if isinstance(l, keras.layers.Dense):
                last_dense_name = l.name
                break

        self.co_occ_model = Model(inputs=self.co_occ_model.input,
                                  outputs=[self.co_occ_model.output, self.co_occ_model.get_layer(last_dense_name).output])

        self.co_occ_matrix = load_npz(co_occ_matrix)
        cells = list(map(tuple, np.load(co_occ_metadata)['matrix_rows']))
        self.cell_to_index = dict(zip(cells, range(len(cells))))
        del cells
        self.penultimate_co_occ = {}
        self.penultimate_neurons = penultimate_neurons

        self.filenames = list(
            map(lambda x: os.path.join(os.environ["PROJECT_ROOT"], 'images', x + '.jpg'), self.filenames))

        if replace_paths is not None:
            for from_path_part, to_path_part in replace_paths:
                self.filenames = [path.replace(from_path_part, to_path_part) for path in self.filenames]

        self.samples = len(self.filenames)
        self.__myindex = 0

        if self.verbose:
            self.set_progressbar(self.samples)
        print('Found %d images belonging to %d classes.' % (self.samples, self.num_class))

        super(DataFrameEnsembleIterator, self).__init__(self.samples, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        batch_x_co_occ = np.zeros((len(index_array),) + (self.penultimate_neurons,), dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        i = 0
        for j in index_array:
            filename = self.filenames[j]

            coord = self.coordinates[filename.split('/')[-1].split('.')[0]]
            tup = (coord['lat'], coord['lon'])

            penult = self.penultimate_co_occ.get(tup, [])
            if len(penult) == 0:
                cellindex = self.cell_to_index.get(tup, False)
                if cellindex == False:
                    model_in = [
                        [(tup[0] - 45) / 45, tup[1] / 33] + np.zeros(self.co_occ_matrix.shape[1]).tolist()]
                else:
                    model_in = [
                        [(tup[0] - 45) / 45, tup[1] / 33] + self.co_occ_matrix[cellindex, :].todense().tolist()[0]]
                self.penultimate_co_occ[tup] = self.co_occ_model.predict(model_in)[1]
                penult = self.penultimate_co_occ[tup]

            batch_x_co_occ[i] = penult

            img = load_img(filename,
                           grayscale=grayscale,
                           target_size=self.target_size,
                           squaring_method=self.squaring_method)
            try:
                x = image.img_to_array(img, data_format=self.data_format)
            except ValueError:
                print(filename)
                continue

            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            i += 1

        self.__myindex += len(index_array)
        if self.__myindex > self.samples:
            self.__myindex = 0

        if self.verbose:
            if self.progressbar is not None:
                try:
                    self.progressbar.update(self.__myindex)
                except:
                    print("Exception during updating of progressbar: {}".format(traceback.format_exc()))
            else:
                print("next,", self.batch_index * self.batch_size, len(index_array))

        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.labels[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.labels[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
            for i, label in enumerate(self.labels[index_array]):
                # -1 means the eval set contains a species not encountered during training, then all is zero
                if label > -1:
                    batch_y[i, label] = 1.
        else:
            return [batch_x, batch_x_co_occ]

        # return batch_x, batch_y
        return [batch_x, batch_x_co_occ], batch_y

    def reset(self):
        self.__myindex = 0
        super(DataFrameEnsembleIterator, self).reset()

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        return self._get_batches_of_transformed_samples(index_array)

    def set_progressbar(self, max_value):
        self.progressbar = progressbar.ProgressBar(maxval=max_value)
        self.progressbar.start()
