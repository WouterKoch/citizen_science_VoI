from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.nasnet import NASNetMobile


def get_params(architecture_name):
    if architecture_name == "inceptionv3":
        image_size = (299, 299)
    elif architecture_name == "inception-resnet":
        image_size = (299, 299)
    elif architecture_name == "mobilenet":
        image_size = (224, 224)
    elif architecture_name == "nasnet-mobile":
        image_size = (224, 224)
    elif architecture_name == 'EfficientNetB3':
        image_size = (300, 300)
    else:
        raise ValueError("Unknown model_name {}".format(architecture_name))

    return image_size


def get_model(architecture_name, input_tensor=None):
    if architecture_name == "inceptionv3":
        base_model = InceptionV3(weights='imagenet' if input_tensor is None else None, input_tensor=input_tensor,
                                 include_top=False)
    elif architecture_name == "inception-resnet":
        base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    elif architecture_name == "mobilenet":
        base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, alpha=1.0)
    elif architecture_name == "nasnet-mobile":
        base_model = NASNetMobile(input_shape=(224, 224, 3), include_top=False)
    elif architecture_name == 'EfficientNetB3':
        base_model = EfficientNetB3(input_shape=(300, 300, 3), include_top=False, weights='imagenet')
    else:
        raise ValueError("Unknown model_name {}".format(architecture_name))

    return base_model, get_params(architecture_name)
