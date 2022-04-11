import os
import json
import sys
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from MachineLearning.keras_utils import DataFrameIterator
import numpy as np
from Tools.prefetch_images import fetch as prefetch_images
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from tqdm import tqdm
import scipy
from Scripts.bias_plot import create_plots as create_bias_plots
from Scripts.bias_plot import get_rank_stats as get_rank_stats

sns.set_style("whitegrid")


def get_colormap(path):
    taxa = pd.read_csv(os.path.join(path, "Species (total).csv"))["Taxon"].to_list()
    colors = plt.cm.get_cmap('tab20', len(taxa))
    return dict(zip(taxa, colors(range(len(taxa)))))


def non_standardized_vbgf(t, k, asymp, t0):
    return asymp * (1 - np.exp(-k * (t - t0)))


def non_standardized_vbgf_derivative(t, k, asymp, t0):
    b = asymp * np.exp(k * t0)
    return b * k * np.exp(-k * t)


def standardized_vbgf(val, k):
    return 1 - np.exp(-k * (val - 12))


def standardized_vbgf_derivative(k, t):
    return np.exp(12 * k) * k * np.exp(-k * t)


def run_test_set(root):
    if not os.path.isfile(os.path.join(root, "tested.csv")):

        os.environ["PROJECT_ROOT"] = "/".join(root.split("/")[:-1])
        os.environ["IMG_ROOT"] = root

        img_model = load_model(os.path.join(root, "trained_model.h5"))
        img_labels = list(map(lambda x: x.split(":")[1].strip(),
                              open(os.path.join(root, "labels.txt")).readlines()))

        df_images = pd.read_csv(
            os.path.join(os.environ.get("PROJECT_ROOT"), "Testing observations with media.csv"))

        id_to_name = df_images[["taxon_full_name", "taxon_id_at_source"]].drop_duplicates()
        id_to_name = dict(zip(id_to_name["taxon_id_at_source"].tolist(),
                              id_to_name["taxon_full_name"].tolist()))

        df_images = df_images[df_images["taxon_id_at_source"].isin(list(map(lambda x: int(x), img_labels)))]

        df_images["set"] = "validation"
        image_data = df_images[["image_id", "taxon_id_at_source", "set"]]

        prefetch_images(os.path.join(os.environ.get("PROJECT_ROOT"), "Testing observations with media.csv"))

        test_datagen = image.ImageDataGenerator(
            preprocessing_function=lambda x: (x / 255.0 - 0.5) * 2
        )

        with open(os.path.join(root, "model.json")) as f:
            jsondata = json.load(f)

        squaring_method = jsondata["squaring_method"]

        img_iterator = DataFrameIterator(
            image_data,
            test_datagen,
            color_mode="rgb",
            class_mode="categorical",
            target_size=(500, 500),
            data_format=None,
            batch_size=int(os.environ.get("DEFAULT_BATCH_SIZE")),
            shuffle=False,
            seed=None,
            set_name="validation",
            verbose=1,
            save_to_dir=None,
            squaring_method=squaring_method,
            replace_paths=None,
        )

        img = img_model.predict(img_iterator)
        stats = pd.DataFrame()
        stats["img"] = img.tolist()

        true_labels = df_images["taxon_id_at_source"].tolist()
        true_labels = list(map(lambda x: str(x), true_labels))
        taxon_map = dict(
            zip(map(lambda x: str(x), df_images["taxon_id_at_source"].tolist()),
                df_images["taxon_full_name"].tolist()))

        stats["truth"] = true_labels

        stats["img_highscore"] = list(map(lambda x: max(x), img))
        stats["img_highscore_species"] = list(map(lambda x: img_labels[x.tolist().index(max(x))], img))

        stats["img_truescore"] = stats[["img", "truth"]].apply(lambda x: x["img"][img_labels.index(x["truth"])],
                                                               axis=1)

        stats["img_true_ranking"] = stats[["img", "truth"]].apply(
            lambda x: list(reversed(list(list(zip(*sorted(zip(x["img"], img_labels))))[1]))).index(
                x["truth"]) + 1, axis=1)

        stats["truth"] = stats["truth"].apply(lambda x: taxon_map.get(int(x), x))
        stats["truth_name"] = stats["truth"].apply(lambda x: id_to_name.get(int(x)))

        stats["img_highscore_species"] = stats["img_highscore_species"].apply(
            lambda x: taxon_map.get(int(x), x))
        stats["img_highscore_species_name"] = stats["img_highscore_species"].apply(lambda x: id_to_name.get(int(x)))

        stats.drop(columns=["img"], inplace=True)

        stats.to_csv(os.path.join(root, "tested.csv"))
    else:
        print(os.path.join(root, "tested.csv"), "exists. Skipping test.")


def run_all_test_sets(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name == "trained_model.h5":
                run_test_set(root)


def collect_metrics(path, skip_species_level=False):
    df_collected = pd.DataFrame()
    df_species = pd.DataFrame()

    df_taxon_stats = pd.DataFrame()
    df_species_stats = pd.DataFrame()

    df_species_totals = pd.read_csv(os.path.join(path, "Species (total).csv"))

    df_bias_stats = get_rank_stats("order", path)

    for root, dirs, files in tqdm(os.walk(path)):
        for name in files:
            if name == "tested.csv":
                rootpath = root.split("/")
                metadata = rootpath[-1].split(" ")
                df = pd.read_csv(os.path.join(root, "tested.csv"))

                accuracy = len(df[df["img_true_ranking"] == 1]) / len(df)
                avg_recall = 0
                avg_precision = 0

                all_species = df["truth_name"].unique()
                for species in all_species:
                    tp = len(df[(df["truth_name"] == species) & (df["img_highscore_species_name"] == species)])
                    pred = len(df[(df["img_highscore_species_name"] == species)])
                    true = len(df[(df["truth_name"] == species)])
                    precision = tp / pred
                    recall = tp / true
                    avg_recall += recall / len(all_species)
                    avg_precision += precision * (true / len(df))
                    df_species = df_species.append(
                        {
                            "Taxon": rootpath[-2],
                            "Observations (training)": int(metadata[0]),
                            "Species subset": metadata[6][:-1],
                            "Observation subset (training)": metadata[-1][:-1],
                            "Species": species,
                            "Images (testing)": true,
                            "Average score": df[df["truth_name"] == species]["img_truescore"].mean(),
                            "Precision": precision,
                            "Recall": recall,
                            "F1": 2 * ((recall * precision) / (recall + precision)) if (recall + precision) != 0 else 0
                        },
                        ignore_index=True
                    )

                df_collected = df_collected.append(
                    {
                        "Taxon": rootpath[-2],
                        "Observations (training)": int(metadata[0]),
                        "Species subset": metadata[6][:-1],
                        "Observation subset (training)": metadata[-1][:-1],
                        "Images (testing)": len(df),
                        "Accuracy": accuracy,
                        "Average recall": avg_recall,
                        "Average precision": avg_precision,
                        "F1": 2 * ((avg_recall * avg_precision) / (avg_recall + avg_precision))
                    },
                    ignore_index=True
                )

    for taxon in df_collected["Taxon"].unique():
        df_current_taxon = df_collected[df_collected["Taxon"] == taxon]
        df_taxon_stats = df_taxon_stats.append([
            {"Taxon": taxon, "Metric": "Accuracy (max)",
             "Value": df_current_taxon["Accuracy"].max()},

            {"Taxon": taxon, "Metric": "Accuracy (min)",
             "Value": df_current_taxon["Accuracy"].min()},

            {"Taxon": taxon, "Metric": "Accuracy (mean)",
             "Value": df_current_taxon["Accuracy"].mean()},

            {"Taxon": taxon, "Metric": "Average recall (max)",
             "Value": df_current_taxon["Average recall"].max()},

            {"Taxon": taxon, "Metric": "Average recall (min)",
             "Value": df_current_taxon["Average recall"].min()},

            {"Taxon": taxon, "Metric": "Average recall (mean)",
             "Value": df_current_taxon["Average recall"].mean()},

            {"Taxon": taxon, "Metric": "Average precision (max)",
             "Value": df_current_taxon["Average precision"].max()},

            {"Taxon": taxon, "Metric": "Average precision (min)",
             "Value": df_current_taxon["Average precision"].min()},

            {"Taxon": taxon, "Metric": "Average precision (mean)",
             "Value": df_current_taxon["Average precision"].mean()},

            {"Taxon": taxon, "Metric": "F1 (max)",
             "Value": df_current_taxon["F1"].max()},

            {"Taxon": taxon, "Metric": "F1 (min)",
             "Value": df_current_taxon["F1"].min()},

            {"Taxon": taxon, "Metric": "F1 (mean)",
             "Value": df_current_taxon["F1"].mean()},
        ],
            ignore_index=True)

        df_all_species = pd.read_csv(os.path.join(path, taxon, "All species.csv"))

        df_taxon_stats = df_taxon_stats.append([
            {"Taxon": taxon, "Metric": "Observations (total)",
             "Value": df_all_species["count"].sum()},

            {"Taxon": taxon, "Metric": "Observations (mean)",
             "Value": df_all_species["count"].mean()},

            {"Taxon": taxon, "Metric": "Species (dataset)",
             "Value": len(df_all_species)},

            {"Taxon": taxon, "Metric": "Species (total)",
             "Value": df_species_totals[df_species_totals["Taxon"] == taxon].reset_index().at[0, "Number of species"]},

            {"Taxon": taxon, "Metric": "Species representation",
             "Value": len(df_all_species) / df_species_totals[df_species_totals["Taxon"] == taxon].reset_index().at[
                 0, "Number of species"]},

        ],
            ignore_index=True)

    for species in df_species["Species"].unique():
        df_current_species = df_species[df_species["Species"] == species].reset_index()
        taxon = df_current_species.at[0, "Taxon"]

        df_species_stats = df_species_stats.append([
            {"Taxon": taxon, "Species": species, "Metric": "Images (testing)",
             "Value": df_current_species["Images (testing)"].max()},

            {"Taxon": taxon, "Species": species, "Metric": "Average score (max)",
             "Value": df_current_species["Average score"].max()},

            {"Taxon": taxon, "Species": species, "Metric": "Precision (max)",
             "Value": df_current_species["Precision"].max()},

            {"Taxon": taxon, "Species": species, "Metric": "Recall (max)",
             "Value": df_current_species["Recall"].max()},

            {"Taxon": taxon, "Species": species, "Metric": "F1 (max)",
             "Value": df_current_species["F1"].max()},


            {"Taxon": taxon, "Species": species, "Metric": "Average score (mean)",
             "Value": df_current_species[df_current_species["Observations (training)"] == 200]["Average score"].mean()},

            {"Taxon": taxon, "Species": species, "Metric": "Precision (mean)",
             "Value": df_current_species[df_current_species["Observations (training)"] == 200]["Precision"].mean()},

            {"Taxon": taxon, "Species": species, "Metric": "Recall (mean)",
             "Value": df_current_species[df_current_species["Observations (training)"] == 200]["Recall"].mean()},

            {"Taxon": taxon, "Species": species, "Metric": "F1 (mean)",
             "Value": df_current_species[df_current_species["Observations (training)"] == 200]["F1"].mean()},
        ],
            ignore_index=True)

    # -------------------------------------#####

    df_collected, df_species = convert_dummy_obs_subsets(df_collected, df_species)
    df_collected["Run"] = df_collected["Species subset"].astype(str) + "_" + df_collected[
        "Observation subset (training)"].astype(str)

    df_ci_metrics = pd.DataFrame()

    # ----- Functions to fit curves with

    def vbgf(val, k, a):
        return asymptote * (1 - np.exp(-k * (val - a)))

    df_collected.sort_values(by="Observations (training)", inplace=True)
    df_species.sort_values(by="Observations (training)", inplace=True)

    for taxon in tqdm(df_collected["Taxon"].unique()):
        df_subset = df_collected[df_collected["Taxon"] == taxon].reset_index()

        for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):

            x = df_subset["Observations (training)"]
            y = df_subset[metric]

            row = df_bias_stats[df_bias_stats["name"] == taxon].iloc[0]
            avg_img_per_species = row["cs_img_observations"] / row["species"]

            asymptote = max(df_subset.groupby(["Observations (training)"])[metric].mean().to_list())

            popt, pcov = scipy.optimize.curve_fit(vbgf, x, y)

            df_taxon_stats = df_taxon_stats.append([
                {"Taxon": taxon, "Metric": f"Curve k of {metric}", "Value": popt[0]},
                {"Taxon": taxon, "Metric": f"Curve a of {metric}", "Value": popt[1]},
                {"Taxon": taxon, "Metric": f"Curve asymp of {metric}", "Value": asymptote},
            ],
                ignore_index=True)

            df_taxon_stats = df_taxon_stats.append(
                [{"Taxon": taxon, "Metric": f"Curve derivative of {metric} at average img",
                  "Value": non_standardized_vbgf_derivative(avg_img_per_species, popt[0], asymptote, popt[1])}],
                ignore_index=True)

            df_taxon_stats = df_taxon_stats.append(
                [{"Taxon": taxon, "Metric": f"Curve derivative of {metric} at average img per species",
                  "Value": non_standardized_vbgf_derivative(avg_img_per_species, popt[0], asymptote, popt[1]) / row[
                      "species"]}],
                ignore_index=True)

            y_norm = (np.asarray(y) - min(y)) / (max(y) - min(y))
            popt, pcov = scipy.optimize.curve_fit(standardized_vbgf, x, y_norm)
            df_taxon_stats = df_taxon_stats.append(
                [{"Taxon": taxon, "Metric": f"Exponent factor of normalized range of {metric}", "Value": popt[0]}],
                ignore_index=True)

            df_taxon_stats = df_taxon_stats.append(
                [{"Taxon": taxon, "Metric": f"Derivative of {metric} at average img",
                  "Value": standardized_vbgf_derivative(popt[0], avg_img_per_species)}],
                ignore_index=True)

            x = np.log10(x)
            y_norm = (np.asarray(y) - min(y)) / min(y)
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y_norm)
            df_taxon_stats = df_taxon_stats.append(
                [{"Taxon": taxon, "Metric": "Slope of normalized percentage of min " + metric, "Value": slope},
                 {"Taxon": taxon, "Metric": "Intercept of normalized percentage of min " + metric, "Value": intercept}],
                ignore_index=True)

            y_norm = (np.asarray(y) - min(y))
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y_norm)
            df_taxon_stats = df_taxon_stats.append(
                [{"Taxon": taxon, "Metric": "Slope of normalized absolute of min " + metric, "Value": slope},
                 {"Taxon": taxon, "Metric": "Intercept of normalized absolute of min " + metric, "Value": intercept}],
                ignore_index=True)

            y_norm = (np.asarray(y) - min(y)) / (max(y) - min(y))
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y_norm)
            df_taxon_stats = df_taxon_stats.append(
                [{"Taxon": taxon, "Metric": f"Slope of normalized range of {metric}", "Value": slope},
                 {"Taxon": taxon, "Metric": f"Intercept of normalized range of {metric}", "Value": intercept}],
                ignore_index=True)

    if not skip_species_level:
        for i, metric in enumerate(["Average score", "Recall", "Precision", "F1"]):
            for species in df_species["Species"].unique():
                df_fit = df_species[df_species["Species"] == species]
                xfit = df_fit["Observations (training)"].to_list()
                yfit = df_fit[metric].to_list()
                asymptote = np.mean(df_fit[df_fit["Observations (training)"] == max(xfit)][metric])
                popt, pcov = scipy.optimize.curve_fit(vbgf, xfit, yfit, maxfev=5000)
                df_species_stats = df_species_stats.append([{"Taxon": df_fit.iloc[0]["Taxon"], "Species": species,
                                                             "Metric": f"Curve k of {metric}", "Value": popt[0]},
                                                            {"Taxon": df_fit.iloc[0]["Taxon"], "Species": species,
                                                             "Metric": f"Curve a of {metric}", "Value": popt[1]}],
                                                           ignore_index=True)

    df_collected.to_csv(os.path.join(path, "Collected_stats.csv"), index=False)
    df_species.to_csv(os.path.join(path, "Collected_stats_per_species.csv"), index=False)
    df_taxon_stats.to_csv(os.path.join(path, "Stats per taxon.csv"), index=False)
    df_ci_metrics.to_csv(os.path.join(path, "ci_metrics.csv"), index=False)
    df_species_stats.to_csv(os.path.join(path, "Stats per species.csv"), index=False)


def convert_dummy_obs_subsets(df_collected, df_species):
    """
    Data is pre-capped at the number of images per species required for the first, largest model.
    This means it makes no sense to specify the seed for the subsetting of observations there; all will be part of that
    first model. So the subset is set to 0 to ensure that an (identical) model with the same species subset will not
    train again. The seed for subsetting observations is however used for identifying which run it belongs to in
    analysis. This function sets the seed from 0 to what it is for the rest of the set, and duplicates the stats if
    needed (when multiple models have been trained with the same species subset, but different observation subsets).
    """

    max_obs = int(df_collected["Observations (training)"].max())

    df_collected_max = df_collected[
        (df_collected["Observations (training)"] == max_obs)]
    df_species_max = df_species[
        (df_species["Observations (training)"] == max_obs)]
    df_collected = df_collected[
        (df_collected["Observations (training)"] < max_obs)]
    df_species = df_species[
        (df_species["Observations (training)"] < max_obs)]

    df_process = df_collected[["Taxon", "Species subset", "Observation subset (training)"]].drop_duplicates()

    for index, row in df_process.iterrows():
        df_collected_append = df_collected_max[
            (df_collected_max["Species subset"] == row["Species subset"]) & (df_collected_max["Taxon"] == row["Taxon"])]
        df_collected_append.loc[:, "Observation subset (training)"] = row["Observation subset (training)"]
        df_collected = df_collected.append(df_collected_append)

        df_species_append = df_species_max[
            (df_species_max["Species subset"] == row["Species subset"]) & (df_species_max["Taxon"] == row["Taxon"])]
        df_species_append.loc[:, "Observation subset (training)"] = row["Observation subset (training)"]
        df_species = df_species.append(df_species_append)

    return df_collected, df_species


def create_all_graphs(path, output_path, skip_species_plots=False, extension="svg"):
    os.makedirs(os.path.join(output_path), exist_ok=True)

    plot_normalized_performances(path=path, output_path=output_path, extension="svg")
    plot_pictures_vs_slopes(path=path, output_path=output_path, extension="svg")
    plot_pictures_vs_exp(path=path, output_path=output_path, extension="svg")
    plot_bias_vs_slopes(path=path, output_path=output_path, extension="svg")
    plot_taxon_performance_vs_images(path=path, output_path=output_path, extension="svg")
    plot_curve_metrics_vs_images_per_taxon(path=path, output_path=output_path, extension="svg")
    plot_curve_metrics_vs_images(path=path, output_path=output_path, extension="svg")
    plot_performance_vs_images(path=path, output_path=output_path, extension="svg")
    plot_curve_parameters_vs_species_per_taxon(path=path, output_path=output_path, extension="svg")
    plot_ci_metrics_vs_species_per_taxon(path=path, output_path=output_path, extension="svg")
    plot_performance_vs_species(path=path, output_path=output_path, extension="svg")
    if not skip_species_plots:
        plot_performance_per_species(path=path, output_path=output_path, extension="svg", with_fitted_curves=False)
    plot_min_normalized_means(path=path, output_path=output_path, extension="svg")
    plot_normalized_means(path=path, output_path=output_path, extension="svg")
    # plot_normalized_means_with_estimated_asymptote(path=path, output_path=output_path, extension="svg")
    plot_mean_curve_per_taxon(path=path, output_path=output_path, extension="svg")
    plot_performance_per_run_of_taxon(path=path, output_path=output_path, extension="svg")
    plot_taxon_slope_vs_curve_k(path=path, output_path=output_path, extension=extension)
    plot_taxon_slope_vs_min_performance(path=path, output_path=output_path, extension=extension)
    plot_taxon_slope_vs_max_performance(path=path, output_path=output_path, extension=extension)


def create_selected_graphs(path, output_path, extension="svg"):
    os.makedirs(os.path.join(output_path), exist_ok=True)

    create_bias_plots(rank="class", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="all_bias")

    create_bias_plots(rank="class", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="cs_img_bias")

    create_bias_plots(rank="order", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="cs_img_per_total_bias")

    plot_mean_curve_per_taxon_together(path=path, output_path=output_path, extension=extension, metrics=["F1"])




    plot_means_derivatives_together(path=path, output_path=output_path, extension=extension, metrics=["F1"])
    plot_means_derivatives_together(path=path, output_path=output_path, extension=extension, metrics=["F1"],
                                    per_species=True)

    create_bias_plots(rank="order", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="cs_img_bias",
                      sortby="Curve derivative of F1 at average img")

    create_bias_plots(rank="order", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="cs_img_bias",
                      sortby="Curve derivative of F1 at average img per species")

    create_bias_plots(rank="class", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension)

    create_bias_plots(rank="order", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension)


    sys.exit()



    plot_mean_curve_per_taxon_together(path=path, output_path=output_path, extension=extension)
    plot_mean_curve_per_taxon_together(path=path, output_path=output_path, extension=extension, metrics=["F1"])
    plot_normalized_means_together(path=path, output_path=output_path, extension=extension, metrics=["F1"])
    plot_normalized_fitted_means(path=path, output_path=output_path, extension=extension, metrics=["F1"])
    plot_fitted_means_together(path=path, output_path=output_path, extension=extension, metrics=["F1"])
    plot_means_derivatives_together(path=path, output_path=output_path, extension=extension, metrics=["F1"])

    plot_taxon_slope_vs_max_performance(path=path, output_path=output_path, extension=extension)
    plot_mean_curve_per_taxon(path=path, output_path=output_path, extension=extension, metrics=["Accuracy"])

    create_bias_plots(rank="order", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="cs_img_bias",
                      sortby="Exponent factor of normalized range of Accuracy")

    create_bias_plots(rank="order", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="cs_img_bias",
                      sortby="Derivative of Accuracy at average img")

    create_bias_plots(rank="order", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="cs_img_bias",
                      sortby="Derivative of F1 at average img")

    create_bias_plots(rank="order", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="cs_img_bias",
                      sortby="Curve derivative of F1 at average img")

    exit(0)

    create_bias_plots(rank="order", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension)

    create_bias_plots(rank="class", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension)

    create_bias_plots(rank="class", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="all_bias")

    create_bias_plots(rank="class", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="cs_img_bias")

    create_bias_plots(rank="class", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="cs_img_per_total_bias")

    create_bias_plots(rank="order", plot_output_dir=output_path,
                      csv_output_dir=path, extension=extension, plottype="cs_img_per_total_bias")

    plot_normalized_means(path=path, output_path=output_path, extension="svg")
    # plot_normalized_means_with_estimated_asymptote(path=path, output_path=output_path, extension="svg")
    plot_pictures_vs_exp(path=path, output_path=output_path, extension="svg")
    plot_pictures_vs_slopes(path=path, output_path=output_path, extension="svg")
    plot_taxon_exp_vs_max_performance(path=path, output_path=output_path, extension="svg")


def plot_normalized_performances(path, output_path, extension):
    """
        Plots the fitted performance increase over runs relative to the first run for each taxon in a single plot
    """

    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))
    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"), usecols=["Observations (training)"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Normalized performance regressions")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    x = df_collected["Observations (training)"].unique()
    x = np.log10(x)

    for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
        ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)])
        ax.set_xticks(x)
        ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))
        ax.set(ylim=(0, 1), ylabel=metric, xlabel="Observations in train set")

        for taxon in np.sort(df_taxon_stats["Taxon"].unique()):
            intercept = df_taxon_stats[
                (df_taxon_stats["Taxon"] == taxon) & (
                        df_taxon_stats["Metric"] == "Intercept of normalized absolute of min " + metric)].iloc[
                0]["Value"]
            slope = df_taxon_stats[
                (df_taxon_stats["Taxon"] == taxon) & (
                        df_taxon_stats["Metric"] == "Slope of normalized absolute of min " + metric)].iloc[0][
                "Value"]
            ax.plot(x, slope * x + intercept)

    plt.savefig(os.path.join(output_path, f"Normalized performance regressions.{extension}"), dpi=300)
    plt.close()


def plot_pictures_vs_slopes(path, output_path, extension):
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))
    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"),
                               usecols=["Taxon", "Images (testing)"]).drop_duplicates()
    df_collected = pd.DataFrame(df_collected.groupby(by="Taxon").mean()).reset_index()

    df_taxon_stats = df_taxon_stats.merge(df_collected, on="Taxon")
    df_taxon_stats["Images (testing)"] = df_taxon_stats["Images (testing)"].apply(lambda x: np.log10(x))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Pictures vs slopes")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
        df_subset = df_taxon_stats[df_taxon_stats["Metric"] == "Slope of normalized absolute of min " + metric]
        ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], data=df_subset, x="Images (testing)", y="Value",
                             hue="Taxon")
        ax.legend([], [], frameon=False)
        ax.set_ylabel("Slope")
        ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))
        add_linregress(ax, df_subset["Images (testing)"], df_subset["Value"])

    plt.savefig(os.path.join(output_path, f"Pictures vs slopes.{extension}"), dpi=300)
    plt.close()


def plot_pictures_vs_exp(path, output_path, extension):
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))
    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"),
                               usecols=["Taxon", "Images (testing)"]).drop_duplicates()
    df_collected = pd.DataFrame(df_collected.groupby(by="Taxon").mean()).reset_index()

    df_taxon_stats = df_taxon_stats.merge(df_collected, on="Taxon")
    df_taxon_stats["Images (testing)"] = df_taxon_stats["Images (testing)"].apply(lambda x: np.log10(x))
    df_taxon_stats = df_taxon_stats[df_taxon_stats["Taxon"] != "Lepidoptera"]  # Crappy fit :(

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Pictures vs slopes")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
        df_subset = df_taxon_stats[df_taxon_stats["Metric"] == f"Exponent factor of normalized range of {metric}"]
        ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], data=df_subset, x="Images (testing)", y="Value",
                             hue="Taxon")
        ax.legend([], [], frameon=False)
        ax.set_ylabel("Exponent factor")
        ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))

        add_linregress(ax, df_subset["Images (testing)"], df_subset["Value"])

    plt.savefig(os.path.join(output_path, f"Pictures vs exp factor.{extension}"), dpi=300)
    plt.close()


def plot_bias_vs_slopes(path, output_path, extension):
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))
    df_collected = pd.read_csv(os.path.join(path, "Bias_stats_orders.csv"))
    df_collected.columns = ["Taxon", *df_collected.columns[1:]]
    df_taxon_stats = df_taxon_stats.merge(df_collected, on="Taxon")

    biases = df_collected.columns[1:-1]

    for bias in biases:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Bias of " + bias + " vs slopes")
        plt.subplots_adjust(wspace=.25, hspace=.4)

        for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
            df_subset = df_taxon_stats[df_taxon_stats["Metric"] == "Slope of normalized absolute of min " + metric]
            ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], data=df_subset, x=bias, y="Value",
                                 hue="Taxon")
            ax.legend([], [], frameon=False)
            ax.set_ylabel("Slope")

            add_linregress(ax, df_subset[bias], df_subset["Value"])

        plt.savefig(os.path.join(output_path, f"Bias of {bias} vs slopes.{extension}"), dpi=300)
        plt.close()


def plot_taxon_performance_vs_images(path, output_path, extension):
    df_species_stats = pd.read_csv(os.path.join(path, "Stats per species.csv")).sort_values(by="Species")

    for taxon in df_species_stats["Taxon"].unique():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Performance over number of images of " + taxon)
        plt.subplots_adjust(wspace=.25, hspace=.4)

        x = np.log10(
            df_species_stats[(df_species_stats["Metric"] == "Images (testing)") & (df_species_stats["Taxon"] == taxon)][
                "Value"].tolist())

        for i, metric in enumerate(["Average score (max)", "Recall (max)", "Precision (max)", "F1 (max)"]):
            y = \
                df_species_stats[(df_species_stats["Metric"] == metric) & (df_species_stats["Taxon"] == taxon)][
                    "Value"].tolist()

            ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], x=x, y=y)
            ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))

            add_linregress(ax, x, y, position="bottom")

            ax.set(ylim=(0, 1.1))
            ax.set_yticks(np.arange(0, 1.1, .2))
            ax.set_xlabel("Number of images in test set")
            ax.legend([], [], frameon=False)

        plt.savefig(os.path.join(output_path, f"Species vs number of images ({taxon}).{extension}"), dpi=300)
        plt.close()


def plot_curve_metrics_vs_images_per_taxon(path, output_path, extension):
    df_species_stats = pd.read_csv(os.path.join(path, "Stats per species.csv")).sort_values(by="Species")

    for taxon in df_species_stats["Taxon"].unique():
        # x = np.log10(
        #     df_species_stats[(df_species_stats["Taxon"] == taxon) & (df_species_stats["Metric"] == "Images (testing)")][
        #         "Value"])

        x = np.log10(
            df_species_stats[(df_species_stats["Taxon"] == taxon) & (df_species_stats["Metric"] == "Images (testing)")][
                "Value"].tolist())

        for i, curve_metric in enumerate(["a", "k"]):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Performance over number of images")
            plt.subplots_adjust(wspace=.25, hspace=.4)

            for j, metric in enumerate(["Average score", "Recall", "Precision", "F1"]):
                y = df_species_stats[(df_species_stats["Taxon"] == taxon) & (
                        df_species_stats["Metric"] == f"Curve {curve_metric} of {metric}")]["Value"].tolist()

                ax = sns.scatterplot(ax=axes[int(j / 2), int(j % 2)], x=x, y=y)
                add_linregress(ax, x, y)

                ax.set_xlabel("Number of images in test set")
                ax.set_ylabel(metric + " " + curve_metric)
                ax.legend([], [], frameon=False)
                ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))

            plt.savefig(
                os.path.join(output_path, f"{taxon} species curves {curve_metric} vs number of images.{extension}"),
                dpi=300)
            plt.close()


def plot_curve_metrics_vs_images(path, output_path, extension):
    df_species_stats = pd.read_csv(os.path.join(path, "Stats per species.csv")).sort_values(by="Taxon")
    x = np.log10(df_species_stats[df_species_stats["Metric"] == "Images (testing)"]["Value"].tolist())
    taxa = df_species_stats[df_species_stats["Metric"] == "Images (testing)"]["Taxon"].tolist()

    for i, curve_metric in enumerate(["a", "k"]):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Performance over number of images")
        plt.subplots_adjust(wspace=.25, hspace=.4)

        for j, metric in enumerate(["Average score", "Recall", "Precision", "F1"]):
            y = df_species_stats[df_species_stats["Metric"] == f"Curve {curve_metric} of {metric}"]["Value"].tolist()

            ax = sns.scatterplot(ax=axes[int(j / 2), int(j % 2)], x=x, y=y,
                                 hue=taxa)

            add_linregress(ax, x, y)

            ax.set_xlabel("Number of images in test set")
            ax.set_ylabel(metric + " " + curve_metric)
            ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))
            ax.legend([], [], frameon=False)

        plt.savefig(os.path.join(output_path, f"Species curves {curve_metric} vs number of images.{extension}"),
                    dpi=300)
        plt.close()


def plot_performance_vs_images(path, output_path, extension):
    df_species_stats = pd.read_csv(os.path.join(path, "Stats per species.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Performance over number of images")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    df_x = df_species_stats[df_species_stats["Metric"] == "Images (testing)"]
    df_subset = df_x.reset_index()
    df_subset["Value"] = np.log10(df_subset["Value"])

    for i, metric in enumerate(["Average score (max)", "Recall (max)", "Precision (max)", "F1 (max)"]):
        df_subset[metric] = df_species_stats[df_species_stats["Metric"] == metric]["Value"].tolist()
        ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], data=df_subset, x="Value", y=metric, hue="Taxon")
        ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))

        add_linregress(ax, df_subset["Value"], df_subset[metric])

        ax.set_xlabel("Number of images in test set")
        ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(output_path, f"Species vs number of images.{extension}"), dpi=300)
    plt.close()


def plot_curve_parameters_vs_species_per_taxon(path, output_path, extension):
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv")).sort_values(by="Taxon")

    for x, x_metric in enumerate(["Species (dataset)", "Species (total)", "Species representation"]):
        x = df_taxon_stats[df_taxon_stats["Metric"] == x_metric]["Value"].tolist()
        taxa = df_taxon_stats[df_taxon_stats["Metric"] == x_metric]["Taxon"].tolist()

        if x_metric != "Species representation":
            x = np.log10(x)

        for ci_metric in ["k", "a"]:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Curve parameter over number of images")
            plt.subplots_adjust(wspace=.25, hspace=.4)

            for j, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
                y = df_taxon_stats[df_taxon_stats["Metric"] == f"Curve {ci_metric} of {metric}"]["Value"].tolist()

                ax = sns.scatterplot(ax=axes[int(j / 2), int(j % 2)], x=x, y=y,
                                     hue=taxa)
                if x_metric != "Species representation":
                    ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))

                ax.set_ylabel("Images needed for " + metric)
                # if (max(df_subset[ci_metric]) > 100):
                #     ax.set(ylim=(0, np.log10(220)))

                add_linregress(ax, x, y)
                ax.set_xlabel(x_metric)

                if x_metric == "Species representation":
                    ax.set(xlim=(0, np.max(x) * 1.1))
                ax.legend([], [], frameon=False)

            plt.savefig(os.path.join(output_path, f"Taxon vs log {x_metric} ({ci_metric}).{extension}"), dpi=300)
            plt.close()


def plot_ci_metrics_vs_species_per_taxon(path, output_path, extension):
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))
    df_ci_stats = pd.read_csv(os.path.join(path, "ci_metrics.csv"))

    for x, x_metric in enumerate(["Species (dataset)", "Species (total)", "Species representation"]):
        df_x = df_taxon_stats[df_taxon_stats["Metric"] == x_metric]

        for ci_metric in df_ci_stats.columns[2:]:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Curve parameter over number of images")
            plt.subplots_adjust(wspace=.25, hspace=.4)

            for j, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
                df_subset = df_x.reset_index()
                df_subset = df_subset.merge(df_ci_stats[df_ci_stats["Metric"] == metric], on="Taxon")

                transform_x = x_metric != "Species representation"
                transform_y = max(df_subset[ci_metric]) > 20

                if transform_x:
                    df_subset["Value"] = np.log10(df_subset["Value"])

                if transform_y:
                    df_subset[ci_metric] = np.log10(df_subset[ci_metric])

                ax = sns.scatterplot(ax=axes[int(j / 2), int(j % 2)], data=df_subset, x="Value", y=ci_metric,
                                     hue="Taxon")
                if transform_x:
                    ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))
                if transform_y:
                    ax.yaxis.set_major_formatter(lambda x, y: int(10 ** x))

                ax.set_ylabel("Images needed for " + metric)
                # if (max(df_subset[ci_metric]) > 100):
                #     ax.set(ylim=(0, np.log10(220)))

                add_linregress(ax, df_subset["Value"], df_subset[ci_metric])

                ax.set_xlabel(x_metric)

                if x_metric == "Species representation":
                    ax.set(xlim=(0, df_subset["Value"].max() * 1.1))
                ax.legend([], [], frameon=False)

            plt.savefig(os.path.join(output_path, f"Taxon vs log {x_metric} ({ci_metric}).{extension}"), dpi=300)
            plt.close()


def plot_performance_vs_species(path, output_path, extension):
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv")).sort_values(by="Taxon")
    for x, x_metric in enumerate(["Species (dataset)", "Species (total)", "Species representation"]):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Performance over number of species")
        plt.subplots_adjust(wspace=.25, hspace=.4)

        x = df_taxon_stats[df_taxon_stats["Metric"] == x_metric]["Value"]
        taxa = df_taxon_stats[df_taxon_stats["Metric"] == x_metric]["Taxon"]

        transform_x = x_metric != "Species representation"
        if transform_x:
            x = np.log10(x)

        for i, metric in enumerate(
                ["Accuracy (max)", "Average recall (max)", "Average precision (max)", "F1 (max)"]):

            y = df_taxon_stats[df_taxon_stats["Metric"] == metric]["Value"]
            ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], x=x, y=y, hue=taxa)

            if transform_x:
                ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))
            ax.set_xlabel(x_metric)

            add_linregress(ax, x, y)

            if x_metric == "Species representation":
                ax.set(xlim=(0, x.max() * 1.1))
            ax.legend([], [], frameon=False)

        plt.savefig(os.path.join(output_path, f"Taxon vs {x_metric}.{extension}"), dpi=300)
        plt.close()


def plot_performance_per_species(path, output_path, extension, with_fitted_curves=True):
    def vbgf(val, k, a):
        return asymptote * (1 - np.exp(-k * (val - a)))

    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
    df_species = pd.read_csv(os.path.join(path, "Collected_stats_per_species.csv"))
    df_collected, df_species = convert_dummy_obs_subsets(df_collected, df_species)
    df_curves = pd.read_csv(os.path.join(path, "Stats per species.csv"))

    xfit = np.sort(df_species["Observations (training)"].unique())

    for taxon in tqdm(df_collected["Taxon"].unique()):
        df_subset = df_species[df_species["Taxon"] == taxon]
        df_subset.reset_index(inplace=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(taxon)
        plt.subplots_adjust(wspace=.25, hspace=.4)

        for i, metric in enumerate(["Average score", "Recall", "Precision", "F1"]):
            ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)], data=df_subset, x="Observations (training)", y=metric,
                              hue="Species", markersize=6, marker="o")
            ax.set_ylabel = metric

            if with_fitted_curves:
                for species in df_subset["Species"].unique():
                    df_fit = df_subset[df_subset["Species"] == species]
                    asymptote = np.mean(df_fit[df_fit["Observations (training)"] == max(xfit)][metric])

                    a = \
                        df_curves[
                            (df_curves["Species"] == species) & (df_curves["Metrix"] == f"Curve a of {metric}")].iloc[
                            0]["Value"]
                    k = \
                        df_curves[
                            (df_curves["Species"] == species) & (df_curves["Metrix"] == f"Curve k of {metric}")].iloc[
                            0]["Value"]

                    ax.plot(xfit, vbgf(xfit, k, a), "r-")

            ax.set(ylim=(0, 1))
            ax.legend([], [], frameon=False)

        plt.savefig(os.path.join(output_path, f"{taxon} per species.{extension}"), dpi=300)
        plt.close()


def plot_min_normalized_means(path, output_path, extension):
    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))

    ticks = np.log10(df_collected["Observations (training)"].sort_values().unique())

    for taxon in df_collected["Taxon"].unique():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(taxon)
        plt.subplots_adjust(wspace=.25, hspace=.4)

        df_subset = df_collected[df_collected["Taxon"] == taxon].sort_values(
            by=["Observations (training)"]).reset_index()

        for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
            x = df_subset["Observations (training)"].unique()
            y = df_subset.groupby(["Observations (training)"])[metric].mean().to_list()

            x = np.log10(x)
            y = (np.asarray(y) - min(y)) / (max(y) - min(y))

            slope = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Slope of normalized range of {metric}")].iloc[0]["Value"]
            intercept = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Intercept of normalized range of {metric}")].iloc[0]["Value"]

            ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)], x=x, y=y,
                              markersize=6, marker="o", ci="sd")
            ax.xaxis.set_major_formatter(lambda x, y: int(10 ** x))
            ax.set_xticks(ticks)

            ax.set(ylim=(0, 1), ylabel=f"{metric} (relative to lowest)", xlabel="Observations in train set", )
            ax.plot(x, slope * x + intercept)

        plt.savefig(os.path.join(output_path, f"{taxon} (normalized means).{extension}"), dpi=300)
        plt.close()


def plot_normalized_means(path, output_path, extension):
    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))

    ticks = np.log10(df_collected["Observations (training)"].sort_values().unique())

    for taxon in df_collected["Taxon"].unique():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(taxon)
        plt.subplots_adjust(wspace=.25, hspace=.4)

        df_subset = df_collected[df_collected["Taxon"] == taxon].sort_values(
            by=["Observations (training)"]).reset_index()

        for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
            x = df_subset["Observations (training)"].unique()
            y = df_subset.groupby(["Observations (training)"])[metric].mean().to_list()

            y = (np.asarray(y) - min(y)) / (max(y) - min(y))

            exp_factor = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Exponent factor of normalized range of {metric}")].iloc[0]["Value"]

            ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)], x=x, y=y,
                              markersize=6, marker="o", ci="sd")
            ax.set_xticks(ticks)

            ax.set(ylim=(0, 1), ylabel=f"{metric} (standardized from 0 to 1)", xlabel="Observations in train set", )
            ax.plot(x, standardized_vbgf(x, exp_factor))

        plt.savefig(os.path.join(output_path, f"{taxon} (normalized means with exp curve).{extension}"), dpi=300)
        plt.close()


def plot_normalized_fitted_means(path, output_path, extension, metrics=None):
    if metrics is None:
        metrics = ["Accuracy", "Average recall", "Average precision", "F1"]

    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))

    df_collected = df_collected.groupby(["Observations (training)", "Taxon"]).mean()[metrics].reset_index().sort_values(
        by=["Observations (training)"])

    if len(metrics) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(wspace=.25, hspace=.4)

    ticks = np.log10(df_collected["Observations (training)"].sort_values().unique())

    x = np.arange(12, 200)

    for i, metric in enumerate(metrics):
        if len(metrics) == 1:
            ax = sns.lineplot()
        else:
            ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)])

        ax.set(ylim=(0, 1), ylabel=f"{metric} (standardized from 0 to 1)", xlabel="Observations in train set", )

        for taxon in df_collected["Taxon"].unique():
            exp_factor = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Exponent factor of normalized range of {metric}")].iloc[0]["Value"]
            ax.plot(x, standardized_vbgf(x, exp_factor))

    plt.savefig(os.path.join(output_path, f"Normalized means exp curves.{extension}"), dpi=300)
    plt.close()


def plot_normalized_means_together(path, output_path, extension, metrics=None):
    if metrics is None:
        metrics = ["Accuracy", "Average recall", "Average precision", "F1"]

    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))

    df_collected = df_collected.groupby(["Observations (training)", "Taxon"]).mean()[metrics].reset_index().sort_values(
        by=["Taxon"])

    if len(metrics) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(wspace=.25, hspace=.4)

    for i, metric in enumerate(metrics):
        df_collected[metric] = df_collected[["Taxon", metric]].apply(
            lambda y: (y[metric] - np.min(df_collected[df_collected["Taxon"] == y["Taxon"]][metric])) / (
                    np.max(df_collected[df_collected["Taxon"] == y["Taxon"]][metric]) - np.min(df_collected[
                                                                                                   df_collected[
                                                                                                       "Taxon"] == y[
                                                                                                       "Taxon"]][
                                                                                                   metric])), axis=1)

        if len(metrics) == 1:
            ax = sns.lineplot(data=df_collected, x="Observations (training)", y=metric, hue="Taxon",
                              markersize=4, marker="o", ci="sd")
        else:
            ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)], data=df_collected, x="Observations (training)", y=metric,
                              hue="Taxon",
                              markersize=4, marker="o", ci="sd")

        # ax.fill_between(x, lower, upper, color="r", alpha=.1)
        ax.set(ylim=(0, 1), ylabel=f"{metric} (average)", xlabel="Observations in train set")
        ax.legend([], [], frameon=False)

        # ax.plot(x, standardized_vbgf(x, exp_factor))

        plt.savefig(os.path.join(output_path,
                                 f"Normalized performances{(' (' + metric + ')') if len(metrics) == 1 else ''} with exp curves.{extension}"),
                    dpi=300)
        plt.close()


def plot_fitted_means_together(path, output_path, extension, metrics=None):
    if metrics is None:
        metrics = ["Accuracy", "Average recall", "Average precision", "F1"]

    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))

    df_collected = df_collected.groupby(["Observations (training)", "Taxon"]).mean()[
        metrics].reset_index().sort_values(
        by=["Observations (training)"])

    if len(metrics) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(wspace=.25, hspace=.4)

    x = np.arange(12, 200)

    for i, metric in enumerate(metrics):
        if len(metrics) == 1:
            ax = sns.lineplot()
        else:
            ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)])

        ax.set(ylim=(0, 1), ylabel=f"{metric} (standardized from 0 to 1)", xlabel="Observations in train set", )

        for taxon in df_collected["Taxon"].unique():
            k = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Curve k of {metric}")].iloc[0][
                "Value"]
            a = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Curve a of {metric}")].iloc[0][
                "Value"]
            asymp = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Curve asymp of {metric}")].iloc[0][
                "Value"]

            ax.plot(x, non_standardized_vbgf(x, k, asymp, a))

    plt.savefig(os.path.join(output_path, f"Fitted VBGF.{extension}"), dpi=300)
    plt.close()


def plot_means_derivatives_together(path, output_path, extension, metrics=None, per_species=False):
    colormap = get_colormap(path)
    colormap = sns.color_palette("tab20", n_colors=12)


    if metrics is None:
        metrics = ["Accuracy", "Average recall", "Average precision", "F1"]

    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))

    df_bias_stats = pd.read_csv(os.path.join(path, "Bias statistics (order).csv"))

    df_collected = df_collected.groupby(["Observations (training)", "Taxon"]).mean()[
        metrics].reset_index().sort_values(
        by=["Taxon"])

    if len(metrics) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plt.subplots_adjust(wspace=.25, hspace=.4)

    x = np.arange(2, 340, .1)

    for i, metric in enumerate(metrics):
        if len(metrics) == 1:
            ax = sns.lineplot()
        else:
            ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)])

        metricName = metric.replace("F1", "F$_1$")
        ax.set(ylabel=f"VoI ({metricName} increase)", xlabel="Observations with images", xscale="log",
               xticks=[2, 10, 20, 100, 200])
        ax.xaxis.set_major_formatter(ScalarFormatter())

        if per_species:
            ax.set(yscale="log")

        for t, taxon in enumerate(df_collected["Taxon"].unique()):
            k = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Curve k of {metric}")].iloc[0][
                "Value"]
            a = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Curve a of {metric}")].iloc[0][
                "Value"]
            asymp = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Curve asymp of {metric}")].iloc[0][
                "Value"]

            factor = \
                df_taxon_stats[
                    (df_taxon_stats["Taxon"] == taxon) & (df_taxon_stats["Metric"] == "Species (total)")].iloc[
                    0]["Value"] if per_species else 1

            ax.plot(x, non_standardized_vbgf_derivative(x, k, asymp, a) / factor, c=colormap[t])

            row = df_bias_stats[df_bias_stats["name"] == taxon].iloc[0]
            avg_img_per_species = row["cs_img_observations"] / row["species"]

            ax.vlines(x=avg_img_per_species, colors=colormap[t], ls="dotted",
                      ymin=(-0.001 * (.01 if per_species else 1)),
                      ymax=non_standardized_vbgf_derivative(avg_img_per_species, k, asymp, a) / factor, zorder=99)
            ax.scatter(avg_img_per_species, non_standardized_vbgf_derivative(avg_img_per_species, k, asymp, a) / factor, 
                marker='X',
                s=35,
                facecolor=colormap[t], 
                zorder=100,
                edgecolor="white",
                linewidth=1, label=taxon)
        
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


    plt.savefig(
        os.path.join(output_path, f"Fitted VBGF derivative{' (per species)' if per_species else ''}.{extension}"),
        dpi=300, bbox_inches='tight')
    plt.close()


# def plot_normalized_means_with_estimated_asymptote(path, output_path, extension):
#     df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
#     df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))
#
#     ticks = np.log10(df_collected["Observations (training)"].sort_values().unique())
#
#     for taxon in df_collected["Taxon"].unique():
#         fig, axes = plt.subplots(2, 2, figsize=(12, 8))
#         fig.suptitle(taxon)
#         plt.subplots_adjust(wspace=.25, hspace=.4)
#
#         df_subset = df_collected[df_collected["Taxon"] == taxon].sort_values(
#             by=["Observations (training)"]).reset_index()
#
#         for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
#             x = df_subset["Observations (training)"].unique()
#             y = df_subset.groupby(["Observations (training)"])[metric].mean().to_list()
#
#             y = (np.asarray(y) - min(y)) / (max(y) - min(y))
#
#             exp_factor = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
#                     df_taxon_stats["Metric"] == f"Exponent factor of normalized range with estimated asymptote of {metric}")].iloc[0]["Value"]
#             a = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
#                     df_taxon_stats[
#                         "Metric"] == f"Estimated asymptote of normalized range of {metric}")].iloc[0]["Value"]
#
#             ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)], x=x, y=y,
#                               markersize=6, marker="o", ci="sd")
#             ax.set_xticks(ticks)
#
#             ax.set(ylim=(0, 1), ylabel=f"{metric} (standardized from 0 to 1)", xlabel="Observations in train set", )
#             ax.plot(x, exponential(x, exp_factor, a))
#
#         plt.savefig(os.path.join(output_path, f"{taxon} (normalized means with exp curve and estimated asymptote).{extension}"), dpi=300)
#         plt.close()


def plot_mean_curve_per_taxon(path, output_path, extension, metrics=None):
    if metrics is None:
        metrics = ["Accuracy", "Average recall", "Average precision", "F1"]

    def vbgf(val, k, a):
        return asymptote * (1 - np.exp(-k * (val - a)))

    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
    df_ci_metrics = pd.read_csv(os.path.join(path, "ci_metrics.csv"))
    taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))

    for taxon in df_collected["Taxon"].unique():
        df_subset = df_collected[df_collected["Taxon"] == taxon].sort_values(
            by=["Observations (training)"]).reset_index()

        if len(metrics) == 1:
            fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(taxon)
        plt.subplots_adjust(wspace=.25, hspace=.4)

        for i, metric in enumerate(metrics):
            x = df_subset["Observations (training)"].unique()
            y = df_subset.groupby(["Observations (training)"])[metric].mean().to_list()
            sds = df_subset.groupby(["Observations (training)"])[metric].std().to_list()
            lower = np.subtract(y, np.multiply(sds, 1))
            upper = np.add(y, np.multiply(sds, 1))

            asymptote = max(y)

            if len(metrics) == 1:
                ax = sns.lineplot(x=x, y=y,
                                  markersize=6, marker="o", ci="sd")
            else:
                ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)], x=x, y=y,
                                  markersize=6, marker="o", ci="sd")

            colors = ["g", "b", "r"]
            for j, ci_metric in enumerate(df_ci_metrics.columns[2:-1]):
                ax.axvline(
                    x=df_ci_metrics[(df_ci_metrics["Taxon"] == taxon) & (df_ci_metrics["Metric"] == metric)].iloc[0][
                        ci_metric], linestyle=":", color=colors[j],
                    label=ci_metric)

            ax.fill_between(x, lower, upper, color="r", alpha=.1)
            ax.set(ylim=(0, 1), ylabel=metric, xlabel="Observations in train set")
            if i == 0:
                ax.legend(loc="lower right")

            a = \
                taxon_stats[(taxon_stats["Taxon"] == taxon) & (taxon_stats["Metric"] == f"Curve a of {metric}")].iloc[
                    0][
                    "Value"]
            k = \
                taxon_stats[(taxon_stats["Taxon"] == taxon) & (taxon_stats["Metric"] == f"Curve k of {metric}")].iloc[
                    0][
                    "Value"]

            ax.plot(x, vbgf(x, k, a), "r-")

        plt.savefig(os.path.join(output_path, f"{taxon} (means).{extension}"), dpi=300)
        plt.close()


def plot_mean_curve_per_taxon_together(path, output_path, extension, metrics=None):
    colormap = sns.color_palette("tab20", n_colors=12)

    if metrics is None:
        metrics = ["Accuracy", "Average recall", "Average precision", "F1"]

    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
    df_collected = df_collected.sort_values(by=["Taxon"])
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))

    if len(metrics) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(wspace=.25, hspace=.4)

    for i, metric in enumerate(metrics):
        if len(metrics) == 1:
            ax = sns.scatterplot(data=df_collected, x="Observations (training)", y=metric,
                                 hue='Taxon', palette=colormap)
        else:
            ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], data=df_collected, x="Observations (training)",
                                 y=metric, hue='Taxon', palette=colormap)


        metricName = metric.replace('F1', 'F$_{1}$')
        ax.set(ylim=(0, 1), ylabel=f"{metricName}", xlabel="Observations in train set")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        num_taxa = len(df_collected["Taxon"].unique())
        cols = int(np.ceil(np.sqrt(num_taxa)))
        rows = int(np.ceil(num_taxa/cols))

        fig_r, axes_r = plt.subplots(rows, cols, figsize=(24, 16))

        for i, taxon in enumerate(df_collected["Taxon"].unique()):
            k = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Curve k of {metric}")].iloc[0][
                "Value"]
            a = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Curve a of {metric}")].iloc[0][
                "Value"]
            asymp = df_taxon_stats[(df_taxon_stats["Taxon"] == taxon) & (
                    df_taxon_stats["Metric"] == f"Curve asymp of {metric}")].iloc[0][
                "Value"]
            x = np.arange(10, 200)
            ax.plot(x, non_standardized_vbgf(x, k, asymp, a), c=colormap[i])

            df_res = df_collected[(df_collected["Taxon"] == taxon)]
            df_res["residual"] = df_res.apply(lambda row: non_standardized_vbgf(row["Observations (training)"], k, asymp, a) - row[metric], axis=1)
            residual = sns.scatterplot(ax=axes_r[int(i / cols), int(i % cols)], data=df_res, x="Observations (training)", y="residual")

            # if int(i / cols) != rows or int(i % cols) != 0:
            #     residual.set(xlabel="", ylabel="")
            # else:
            #     residual.set(xlabel="Hatsee", ylabel="Hops")

            residual.plot([0, 200], [0, 0])
            residual.set_title(taxon)
            residual.set(ylim=(-.25, .25))

    ## Uncomment to save residuals plot
    # fig_r.savefig(os.path.join(output_path, f"{metric} residuals.{extension}"), dpi=300, bbox_inches='tight')
    
    fig.savefig(os.path.join(output_path,
                             f"Performance{(' (' + metrics[0] + ')') if len(metrics) == 1 else ''} means.{extension}"),
                dpi=300, bbox_inches='tight')
   

def plot_performance_per_run_of_taxon(path, output_path, extension):
    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
    df_species = pd.read_csv(os.path.join(path, "Collected_stats_per_species.csv"))
    df_collected, df_species = convert_dummy_obs_subsets(df_collected, df_species)
    df_collected["Run"] = df_collected["Species subset"].astype(str) + "_" + df_collected[
        "Observation subset (training)"].astype(str)

    for taxon in tqdm(df_collected["Taxon"].unique()):
        df_subset = df_collected[df_collected["Taxon"] == taxon].reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(taxon)
        plt.subplots_adjust(wspace=.25, hspace=.4)

        for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
            x = df_subset["Observations (training)"]
            y = df_subset[metric]
            hue = df_subset["Run"]

            ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)], x=x, y=y,
                              hue=hue,
                              markersize=6, marker="o")

            ax.set(ylim=(0, 1), ylabel=metric, xlabel="Observations in train set")
            ax.legend([], [], frameon=False)

        plt.savefig(os.path.join(output_path, f"{taxon}.{extension}"), dpi=300)
        plt.close()


# Plot normalized slopes against curve k
def plot_taxon_slope_vs_curve_k(path, output_path, extension):
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv")).sort_values(by="Taxon")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Slopes vs curves")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
        x = df_taxon_stats[df_taxon_stats["Metric"] == f"Slope of normalized absolute of min {metric}"][
            "Value"].tolist()
        y = df_taxon_stats[df_taxon_stats["Metric"] == f"Curve k of {metric}"]["Value"].tolist()
        taxa = df_taxon_stats[df_taxon_stats["Metric"] == f"Slope of normalized absolute of min {metric}"][
            "Taxon"].tolist()

        ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], x=x, y=y,
                             hue=taxa)
        ax.legend([], [], frameon=False)
        add_linregress(ax, x, y)
        ax.set_xlabel("Slope")

    plt.savefig(os.path.join(output_path, f"Slope vs k.{extension}"), dpi=300)
    plt.close()


# Plot normalized slopes against max performance
def plot_taxon_slope_vs_max_performance(path, output_path, extension):
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Slopes vs max performance")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
        df_subset = df_taxon_stats[df_taxon_stats["Metric"] == f"Slope of normalized absolute of min {metric}"].rename(
            columns={"Value": "Slope"})
        df_performance = df_taxon_stats[df_taxon_stats["Metric"] == f"{metric} (max)"].rename(
            columns={"Value": metric})
        df_subset = df_subset.merge(df_performance, on="Taxon")

        ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], data=df_subset, x="Slope", y=metric,
                             hue="Taxon")
        ax.legend([], [], frameon=False)
        add_linregress(ax, df_subset["Slope"], df_subset[metric])

    plt.savefig(os.path.join(output_path, f"Slope vs max performance.{extension}"), dpi=300)
    plt.close()


def plot_taxon_exp_vs_max_performance(path, output_path, extension):
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Exp vs max performance")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
        df_subset = df_taxon_stats[
            df_taxon_stats["Metric"] == f"Exponent factor of normalized range of {metric}"].rename(
            columns={"Value": "Exp"})
        df_performance = df_taxon_stats[df_taxon_stats["Metric"] == f"{metric} (max)"].rename(
            columns={"Value": metric})
        df_subset = df_subset.merge(df_performance, on="Taxon")

        ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], data=df_subset, x="Exp", y=metric,
                             hue="Taxon")
        ax.legend([], [], frameon=False)
        add_linregress(ax, df_subset["Exp"], df_subset[metric])

    plt.savefig(os.path.join(output_path, f"Exp vs max performance.{extension}"), dpi=300)
    plt.close()


# Plot normalized slopes against min performance
def plot_taxon_slope_vs_min_performance(path, output_path, extension):
    df_taxon_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Slopes vs min performance")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    for i, metric in enumerate(["Accuracy", "Average recall", "Average precision", "F1"]):
        df_subset = df_taxon_stats[df_taxon_stats["Metric"] == f"Slope of normalized absolute of min {metric}"].rename(
            columns={"Value": "Slope"})
        df_performance = df_taxon_stats[df_taxon_stats["Metric"] == f"{metric} (min)"].rename(
            columns={"Value": metric})
        df_subset = df_subset.merge(df_performance, on="Taxon")

        ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], data=df_subset, x="Slope", y=metric,
                             hue="Taxon")
        ax.legend([], [], frameon=False)
        add_linregress(ax, df_subset["Slope"], df_subset[metric])

    plt.savefig(os.path.join(output_path, f"Slope vs min performance.{extension}"), dpi=300)
    plt.close()


def add_linregress(ax, x, y, position="top"):
    x = np.asarray(x)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

    if p_value < .05:
        ax.plot(x, slope * x + intercept)
        stats_str = "R² = " + "{:.2f}".format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + "{:.2e}".format(p_value)
        else:
            stats_str += "p = " + "{:.5f}".format(np.round(p_value, 5))
        ax.text(0.7, 0.05 if position == "bottom" else 0.95, stats_str, horizontalalignment="left",
                verticalalignment=position,
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))


def evaluate(skip_species_plots=False):
    # run_all_test_sets(os.environ.get("JOBS_DIR"))
    # collect_metrics(os.environ.get("JOBS_DIR"))
    create_all_graphs(path=os.environ.get("JOBS_DIR"), output_path=os.path.join(os.environ.get("JOBS_DIR"), "GRAPHS"),
                      skip_species_plots=False, extension="svg")
    create_selected_graphs(path=os.environ.get("JOBS_DIR"),
                           output_path=os.path.join(os.environ.get("JOBS_DIR"), "GRAPHS_SELECTED"), extension="png")


if __name__ == "__main__":
    print("USAGE")
    print("evaluate(): based on folders in the directory defined by the JOBS_DIR environment variable")
