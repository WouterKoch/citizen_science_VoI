import os
from Tools.prefetch_images import fetch as prefetch_images
from Scripts.Image.train_multiclass import train_ as train_img

def train_models():
    for root, dirs, files in os.walk(os.environ.get("JOBS_DIR")):
        for name in files:
            if name == "all_images.csv":
                os.environ["PROJECT_ROOT"] = "/".join(root.split("/")[:-2])
                os.environ["IMG_ROOT"] = "/".join(root.split("/")[:-1])

                if not os.path.isfile(os.path.join(os.environ.get("IMG_ROOT"), 'trained_model.h5')):
                    print("Preparing to train model", ": ".join(root.split("/")[-3:-1]))
                    prefetch_images(os.path.join(os.environ.get("IMG_ROOT"), "source_data/all_images.csv"))
                    train_img(
                        os.path.join(os.environ.get("IMG_ROOT"), "source_data/all_images.csv"),
                        architecture="inception-resnet",
                        batch_size=os.getenv('DEFAULT_BATCH_SIZE')
                    )
                    os.rename(os.path.join(os.environ.get("IMG_ROOT"), 'weights_stage3.h5'), os.path.join(os.environ.get("IMG_ROOT"), 'trained_model.h5'))


                else:
                    print("Image model", ": ".join(root.split("/")[-3:-1]), "exists, skipping training")



if __name__ == "__main__":

    print("USAGE")
    print("train_models(): based on folders in the directory defined by the JOBS_DIR environment variable")
