import os.path
import socket
import urllib.request
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

#
#
# Images will be downloaded later in the workflow as well, but to work with bad internet connections fetching them
# one by one first ensures an unsupervised workflow.
#
#

# timeout in seconds
timeout = 10
socket.setdefaulttimeout(timeout)


def fetch(filename):
    project_root = os.environ.get("PROJECT_ROOT")

    def download_image(url, output_file):
        try:
            urllib.request.urlretrieve(url, output_file)
        except:
            return

    def read_file(input_file):
        return pd.read_csv(input_file, usecols=['record_id', 'image_url'])

    def download_all(df):
        print('Downloading images')
        folder = os.path.join(project_root, 'images')

        if not os.path.exists(folder):
            os.mkdir(folder)

        record_id_to_sequential = defaultdict(lambda: -1)

        def get_sequence_number(record_id):
            record_id_to_sequential[record_id] += 1
            return record_id_to_sequential[record_id]

        for index, image_row in tqdm(df.iterrows(), total=df.shape[0]):
            imgfile = os.path.join(folder, str(image_row['record_id']) + '_' + str(
                get_sequence_number(image_row['record_id'])) + '.jpg')

            if not os.path.isfile(imgfile) or os.path.getsize(imgfile) < 100:
                download_image(image_row['image_url'], imgfile)

    # df = read_file(os.path.join(project_root, "source_data/all_images.csv"))
    df = read_file(filename)
    download_all(df)


if __name__ == "__main__":
    print("Use fetch()")
