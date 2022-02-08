import os

from MachineLearning.keras_utils import train


def train_(dataset_path: str,
           # job_name: str,
           previous_model_folder=None, squaring_method=None,
           architecture=None, batch_size=12):
    """
    Trains a model from a dataset
    Requires PROJECT_ROOT
    :param dataset_path: path to HDF5 file in dataset
    :param job_name: name of job (warning: will remove previous job folder with the same name)
    :param previous_model_folder: model to continue training from, if the classes have changed will remap the classes to match them
    :param squaring_method: in {"crop", "pad"}
    :param architecture:
    :return:
    """

    if squaring_method is None:
        squaring_method = "crop"
    if architecture is None:
        architecture = "inception-resnet"

    # stage 2 (fine tuning parts of the layers) is rarely used
    stages = [
        {"stage": 1, "reduce_lr_factor": 0.2, "reduce_lr_patience": 2, "num_epochs": 2, "epoch_fraction": 1.0,
         "initial_learning_rate": 1e-3},  # the first stage only tunes the top layer
        {"stage": 3, "reduce_lr_factor": 0.1, "reduce_lr_patience": 3, "num_epochs": 200,
         "epoch_fraction": 1.0, "initial_learning_rate": 1e-4},  # the third stage tunes all layers
    ]

    train_folder = os.path.join(os.environ.get("IMG_ROOT"))

    train(
        train_folder,
        dataset_path,
        stages=stages,
        previous_model_folder=previous_model_folder,
        model_configuration={
            "architecture": architecture,  # deep learning architecture
            "num_fully_connected_nodes": 0,  # number of fully connected nodes before output layer
            "squaring_method": squaring_method  # method to make images square
        },
        caller_filename=__file__,
        # count_cap=0,  # force balancing of classes during training
        batch_size=batch_size
    )


if __name__ == "__main__":
    print("Denaturalisalized, might not work the way you are used to")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("dataset_path", help="path to HDF5 file in dataset")
    # parser.add_argument("job_name", help="name of job")
    # parser.add_argument("--previous_model_folder", dest="previous_model_folder", default=None,
    #                     help="model to continue training from")
    # parser.add_argument("--squaring_method", dest="squaring_method", default=None, choices=["crop", "pad"],
    #                     help="'crop' or 'pad'")
    # parser.add_argument("--architecture", dest="architecture", default=None, help="")
    # args = parser.parse_args()
    # train_(
    #     args.dataset_path,
    #     args.job_name,
    #     args.previous_model_folder,
    #     args.squaring_method,
    #     args.architecture
    # )
