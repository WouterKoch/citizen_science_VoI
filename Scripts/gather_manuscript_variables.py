import numpy as np
import pandas as pd
import requests, json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import date

def write_csv(output_folder):
    df = pd.DataFrame()
    url = requests.get("https://api.gbif.org/v1/occurrence/count")
    gbif_all = json.loads(url.text)

    url = requests.get("https://api.gbif.org/v1/occurrence/search?mediaType=StillImage&limit=0")
    gbif_img = json.loads(url.text)['count']

    df = df.append({"Parameter": "Generation_date", "Value": date.today().strftime("%Y-%m-%d")}, ignore_index=True)
    df = df.append({"Parameter": "GBIF_observations_total", "Value": f"{gbif_all:,}"}, ignore_index=True)
    df = df.append({"Parameter": "GBIF_observations_img", "Value": f"{gbif_img:,}"}, ignore_index=True)
    df = df.append({"Parameter": "GBIF_observations_img_percentage", "Value": f"{100 * gbif_img/gbif_all:,.2f} \\%"}, ignore_index=True)
    df.to_csv(os.path.join(output_folder, "Parameters.csv"), index=False)

if __name__ == "__main__":
    print("USAGE:")
    print("write_csv(output_folder)")