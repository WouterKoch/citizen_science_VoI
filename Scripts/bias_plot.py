import logging

import numpy as np
import pandas as pd
import requests, json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def get_gbif_data(params):
    url = "https://www.gbif.org/api/occurrence/breakdown?advanced=false&limit=1"
    for key, value in params.items():
        if isinstance(value, list):
            for v in value:
                url += f"&{key}={v}"
        else:
            url += f"&{key}={value}"

    url = requests.get(url)
    return json.loads(url.text)


def retrieve_stats(csv_dir, dataset, taxa, output_file):
    # Assuming all taxa are of the same rank
    rank = taxa[0]["rank"]

    en_to_no = {
        "class": "Klasse",
        "order": "Orden",
        "species": "Art",
    }

    if not os.path.isfile(output_file):
        counts = {}
        for root, dirs, files in os.walk(csv_dir):
            for name in files:
                dat = pd.read_csv(os.path.join(csv_dir, name), encoding="cp1252", sep=";",
                                  usecols=[en_to_no[rank], en_to_no["species"]]).dropna().drop_duplicates()
                dat = dat.groupby(by=en_to_no[rank]).size().to_dict()
                counts = counts | dat
                del dat

        observations = []
        for taxon in tqdm(taxa):
            # url = requests.get(f"https://api.gbif.org/v1/species/search?q={taxon['name']}&rank={taxon['rank']}&limit=1")
            url = requests.get(f"https://api.gbif.org/v1/species/match?rank={taxon['rank']}&name={taxon['name']}")
            data = json.loads(url.text)
            # taxonKey = data["results"][0]["nubKey"]
            taxonKey = data["usageKey"]

            data = get_gbif_data(
                {"dimension": taxon["rank"] + "Key", "taxon_key": taxonKey, "dataset_key": dataset, "country": "NO"})
            imgdata = get_gbif_data(
                {"dimension": taxon["rank"] + "Key", "taxon_key": taxonKey, "dataset_key": dataset, "country": "NO",
                 "media_type": "StillImage"})
            totaldata = get_gbif_data({"dimension": taxon["rank"] + "Key", "taxon_key": taxonKey, "country": "NO"})

            name = taxon["norwegian"] if "norwegian" in taxon else taxon["name"]
            observations = observations + [
                {"name": name, "cs_observations": data["total"], "cs_img_observations": imgdata["total"],
                 "all_observations": totaldata["total"], "species": counts[name]}]

        df = pd.DataFrame(observations)
        df.to_csv(output_file, index=False)
    else:
        print(f"{output_file} exisits, skipping generation")
        df = pd.read_csv(output_file)

    stats = df[["cs_observations", "cs_img_observations", "all_observations", "species"]].sum().to_dict()
    stats["cs_observations_per_species"] = stats["cs_observations"] / stats["species"]
    stats["cs_img_observations_per_species"] = stats["cs_img_observations"] / stats["species"]
    stats["all_observations_per_species"] = stats["all_observations"] / stats["species"]
    stats["cs_per_total"] = stats["cs_observations"] / stats["all_observations"]
    stats["img_per_cs"] = stats["cs_img_observations"] / stats["cs_observations"]
    stats["img_per_total"] = stats["cs_img_observations"] / stats["all_observations"]

    df["all_bias"] = df.apply(lambda x: get_field_value(x, stats, "all_observations"), axis=1)
    df["cs_obs_bias"] = df.apply(lambda x: get_field_value(x, stats, "cs_observations"), axis=1)
    df["cs_img_bias"] = df.apply(lambda x: get_field_value(x, stats, "cs_img_observations"), axis=1)

    df["cs_per_total_bias"] = df.apply(
        lambda x: (x["cs_observations"] - (stats["cs_per_total"] * x["all_observations"])), axis=1)
    df["cs_img_per_cs_bias"] = df.apply(
        lambda x: (x["cs_img_observations"] - (stats["img_per_cs"] * x["all_observations"])), axis=1)
    df["cs_img_per_total_bias"] = df.apply(
        lambda x: (x["cs_img_observations"] - (stats["img_per_total"] * x["all_observations"])), axis=1)

    print(df[["name", "cs_img_bias"]])

    return df


def get_field_value(x, stats, field):
    return x[field] - (x["species"] * stats[field + "_per_species"])


def get_bias_plot(df, subset, position, title, colorscheme=None):
    if isinstance(subset, list):
        get_side_by_side_bias_plot(df, subset, position, title, colorscheme)
        return

    if colorscheme == "secondary":
        colors = {"negative": "tomato", "positive": "yellowgreen"}
    else:
        colors = {"negative": "red", "positive": "limegreen"}

    ticks = [-500, -200, -50, -10, -1, 0, 1, 10, 50, 200, 500]
    factor = 1
    while df[subset].max() > 500 or df[subset].min() < -500:
        factor *= 10
        df[subset] /= 10

    if (df.columns[-1] == "Value"):
        df.sort_values(by="Value", inplace=True, ascending=True)
    else:
        df.sort_values(by=subset, inplace=True, ascending=False)
    df = df.reset_index()

    ax = sns.barplot(ax=position, x=subset, y="name", data=df,
                     palette=list(
                         map(lambda x: colors["negative"] if x < 0 else colors["positive"], df[subset].tolist())))
    ax.axvline(0, 0, 1, label="pyplot vertical line", color="black", alpha=.7)
    ax.set_xscale("function", functions=(np.arcsinh, np.sinh))
    ax.set_xlabel(f"Observations (×{factor:,})", size=18)
    ax.set_ylabel("")
    ax.set_title(title, size=22)
    ax.set_xticks(ticks)


def get_side_by_side_bias_plot(df, subsets, position, title, colorscheme=None):

    if colorscheme == "secondary":
        colors = {"negative": "tomato", "positive": "yellowgreen"}
    else:
        colors = {"negative": "red", "positive": "limegreen"}

    ticks = [-500, -200, -50, -10, -1, 0, 1, 10, 50, 200, 500]

    print(df.head())
    print(df.columns)

    fig, axes = plt.subplots(1, len(subsets), figsize=(8 * len(subsets), 10))
    df.sort_values(by=subsets[0], inplace=True, ascending=False)
    df = df.reset_index()

    for i, subset in enumerate(subsets):
        factor = 1
        while df[subset].max() > 500 or df[subset].min() < -500:
            factor *= 10
            df[subset] /= 10
        ax = sns.barplot(ax=axes[i], x=subset, y="name", data=df,
                    palette=list(
                        map(lambda x: colors["negative"] if x < 0 else colors["positive"], df[subset].tolist())))
        ax.axvline(0, 0, 1, label="pyplot vertical line", color="black", alpha=.7)
        ax.set_xscale("function", functions=(np.arcsinh, np.sinh))
        ax.set_xlabel(f"Observations (×{factor:,})", size=14)
        ax.set_ylabel("")
        ax.set_title(title, size=22)
        ax.set_xticks(ticks)
        if i > 0:
            ax.set_yticks([])




    

    





def get_bias_plot_with_y_values(df, subset, position, title, colorscheme=None):
    if colorscheme == "secondary":
        colors = {"negative": "tomato", "positive": "yellowgreen"}
    else:
        colors = {"negative": "red", "positive": "limegreen"}

    ticks = [-500, -200, -50, -10, -1, 0, 1, 10, 50, 200, 500]
    factor = 1
    while df[subset].max() > 500 or df[subset].min() < -500:
        factor *= 10
        df[subset] /= 10

    if (df.columns[-1] == "Value"):
        df.sort_values(by="Value", inplace=True, ascending=True)
    else:
        df.sort_values(by=subset, inplace=True, ascending=False)
    df = df.reset_index()

    ax = sns.barplot()

    ax.hlines(df["Value"].to_list(), 0, df[subset], lw=10, color=list(
        map(lambda x: colors["negative"] if x < 0 else colors["positive"], df[subset].tolist())))
    ax.axvline(0, 0, 1, label="pyplot vertical line", color="black", alpha=.7)

    ax.set_xscale("function", functions=(np.arcsinh, np.sinh))
    ax.set_yscale("log")
    ax.set_xlabel(f"Observations (×{factor:,})", size=18)
    ax.set_ylabel("VoI (for F1)", size=18)
    ax.set_title(title, size=22)
    ax.set_xticks(ticks)


def get_bias_plot_with_y_dots(df, subset, position, title, colorscheme=None):
    if colorscheme == "secondary":
        colors = {"negative": "tomato", "positive": "yellowgreen"}
    else:
        colors = {"negative": "red", "positive": "limegreen"}

    ticks = [-500, -200, -50, -10, -1, 0, 1, 10, 50, 200, 500]
    factor = 1
    while df[subset].max() > 500 or df[subset].min() < -500:
        factor *= 10
        df[subset] /= 10

    if (df.columns[-1] == "Value"):
        df.sort_values(by="Value", inplace=True, ascending=True)
    else:
        df.sort_values(by=subset, inplace=True, ascending=False)

    df.sort_values(by=subset, inplace=True, ascending=False)
    df = df.reset_index()

    ax = sns.barplot(ax=position, x=subset, y="name", data=df, alpha=.2,
                     palette=list(
                         map(lambda x: colors["negative"] if x < 0 else colors["positive"], df[subset].tolist())))

    size_factor = 10000 / df["Value"].max()

    for i, row in df.iterrows():
        ax.plot(row[subset], i, marker="o", markersize=(np.sqrt((row["Value"] * size_factor) / np.pi)),
                markeredgecolor="black",
                markerfacecolor=(colors["negative"] if row[subset] < 0 else colors["positive"]))

    # ax.hlines(df["Value"].to_list(), 0, df[subset], lw=10, color=list(
    #                      map(lambda x: colors["negative"] if x < 0 else colors["positive"], df[subset].tolist())))
    ax.axvline(0, 0, 1, label="pyplot vertical line", color="black", alpha=.7)

    ax.set_xscale("function", functions=(np.arcsinh, np.sinh))
    ax.set_xlabel(f"Observations (×{factor:,})", size=18)
    ax.set_ylabel("", size=18)
    ax.set_title(title, size=22)
    ax.set_xticks(ticks)


def save_single_plot(df, output_path, extension, rank, plottype):
    sns.set(rc={"axes.facecolor": "#ececec"})
    sns.set_style({"grid.color": "white"})
    fig, axes = plt.subplots(1, 1, figsize=(8, 9))
    fig.subplots_adjust(left=0.2)

    if df.columns[-1] == "Value":
        get_bias_plot_with_y_dots(df, subset=plottype, position=None, title="")
    else:
        get_bias_plot(df, subset=plottype, position=None, title="", colorscheme=("secondary" if plottype in ["cs_per_total_bias", "cs_img_per_total_bias", "cs_img_per_cs_bias"] else None))

    plt.savefig(os.path.join(output_path,
                             f"Bias statistics ({rank}, {str(plottype)}{', sorted by ' + df.iloc[0]['Metric'] if df.columns[-1] == 'Value' else ''}).{extension}"),
                dpi=300, bbox_inches='tight')
    plt.close()


    # # Uncomment to save as scatterplot too

    # if (df.columns[-1] == "Value"):
    #     fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    #     fig.suptitle("Performance")
    #     plt.subplots_adjust(wspace=.25, hspace=.4)
    #     ax = sns.scatterplot(data=df, x=plottype, y="Value")
    #     ax.set_xlabel(f"Bias in cs img")
    #     ax.set_ylabel(f"VoI per image")
    #     plt.savefig(os.path.join(output_path, f"Scatter VoI of {plottype} VS {df.iloc[0]['Metric']}.{extension}"),
    #                 dpi=300)
    #     plt.close()


def save_plot(df, output_path, extension, rank):
    sns.set(rc={"axes.facecolor": "#ececec"})
    sns.set_style({"grid.color": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(26, 18))
    plt.subplots_adjust(wspace=.4, hspace=.2)

    get_bias_plot(df, subset="all_bias", position=axes[0, 0], title="Total observations")
    get_bias_plot(df, subset="cs_obs_bias", position=axes[0, 1], title="CS observations")
    get_bias_plot(df, subset="cs_img_bias", position=axes[0, 2], title="CS observations with images")
    get_bias_plot(df, subset="cs_per_total_bias", position=axes[1, 0], title="CS obs, corrected for total obs",
                  colorscheme="secondary")
    get_bias_plot(df, subset="cs_img_per_total_bias", position=axes[1, 1],
                  title="CS obs w/ images, corrected for total obs", colorscheme="secondary")
    get_bias_plot(df, subset="cs_img_per_cs_bias", position=axes[1, 2], title="CS obs w/ images, corrected for CS obs",
                  colorscheme="secondary")

    plt.savefig(os.path.join(output_path,
                             f"Bias statistics ({rank}{', sorted by ' + df.iloc[0]['Metric'] if df.columns[-1] == 'Value' else ''}).{extension}"),
                dpi=300)
    plt.close()



def get_rank_stats(rank, csv_output_dir):
    if rank == "order":
        taxa = [
            {"name": "Agaricales", "rank": "order"},
            {"name": "Anseriformes", "rank": "order"},
            {"name": "Asparagales", "rank": "order"},
            {"name": "Charadriiformes", "rank": "order"},
            {"name": "Coleoptera", "rank": "order"},
            {"name": "Diptera", "rank": "order"},
            {"name": "Lamiales", "rank": "order"},
            {"name": "Lecanorales", "rank": "order"},
            {"name": "Lepidoptera", "rank": "order"},
            {"name": "Odonata", "rank": "order"},
            {"name": "Passeriformes", "rank": "order"},
            {"name": "Polyporales", "rank": "order"},
        ]
    else:
        taxa = [
            {"name": "Actinopterygii", "rank": "class"},
            {"name": "Agaricomycetes", "rank": "class"},
            {"name": "Amphibia", "rank": "class"},
            {"name": "Anthozoa", "rank": "class"},
            {"name": "Arachnida", "rank": "class"},
            {"name": "Aves", "rank": "class"},
            {"name": "Bacillariophyceae", "rank": "class"},
            {"name": "Bivalvia", "rank": "class"},
            {"name": "Bryopsida", "rank": "class"},
            {"name": "Florideophyceae", "rank": "class"},
            {"name": "Gastropoda", "rank": "class"},
            {"name": "Globothalamea", "rank": "class"},
            {"name": "Insecta", "rank": "class"},
            {"name": "Jungermanniopsida", "rank": "class"},
            {"name": "Lecanoromycetes", "rank": "class"},
            {"name": "Liliopsida", "rank": "class", "norwegian": "Monocots"},
            {"name": "Magnoliopsida", "rank": "class", "norwegian": "Eudicots"},
            {"name": "Malacostraca", "rank": "class"},
            {"name": "Mammalia", "rank": "class"},
            {"name": "Maxillopoda", "rank": "class"},
            {"name": "Pinopsida", "rank": "class"},
            {"name": "Polychaeta", "rank": "class"},
            {"name": "Polypodiopsida", "rank": "class"},
            {"name": "Reptilia", "rank": "class"},
        ]

    return retrieve_stats(
        csv_dir=os.environ.get("TAXONOMY_DIR"),
        dataset=[
            "b124e1e0-4755-430f-9eab-894f25a9b59c",  # Norwegian Species Observation Service
        ],
        taxa=taxa,
        output_file=os.path.join(csv_output_dir, f"Bias statistics ({rank}).csv")
    )


def create_plots(rank, csv_output_dir, plot_output_dir, extension, plottype=None, sortby=None):
    df = get_rank_stats(rank, csv_output_dir)

    if sortby is not None:
        df_supplement = pd.read_csv(os.path.join(csv_output_dir, f"Stats per taxon.csv"))
        df_supplement = df_supplement[df_supplement["Metric"] == sortby]
        df = df.merge(df_supplement, left_on="name", right_on="Taxon")

    if plottype is None:
        save_plot(df, output_path=plot_output_dir, extension=extension, rank=rank)
    else:
        save_single_plot(df, output_path=plot_output_dir, extension=extension, rank=rank, plottype=plottype)


if __name__ == "__main__":
    print("USAGE:\ncreate_plots(rank, csv_dir, csv_output_dir, plot_output_dir)")
