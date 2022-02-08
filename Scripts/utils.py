import os
from tqdm import tqdm
from shutil import copyfile

def copy_all(filename, from_folder, to_folder):
    for root, dirs, files in tqdm(os.walk(from_folder)):
        for name in files:
            if name == filename:
                print(f"create {root.replace(from_folder, to_folder)} if it does not exist")
                os.makedirs(os.path.join(root.replace(from_folder, to_folder)), exist_ok=True)
                print(f"copy {os.path.join(root, name)} to {os.path.join(root.replace(from_folder, to_folder), name)}")
                copyfile(os.path.join(root, name), os.path.join(root.replace(from_folder, to_folder), name))

if __name__ == "__main__":
    copy_all("tested.csv", "/media/wouter/hephaestus/home/grunnkart/wouter/datasets/VOI", "/home/wouter/Projects/PhD/Voi_data/Laat maar zien dan")