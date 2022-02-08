import os
import json
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from MachineLearning.keras_utils import DataFrameIterator
import numpy as np
from Tools.prefetch_images import fetch as prefetch_images
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scipy


def create_graphs(path, skip_species_plots=False):
    os.makedirs(os.path.join(path, "GRAPHS"), exist_ok=True)
    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
    df_species = pd.read_csv(os.path.join(path, "Collected_stats_per_species.csv"))
    df_collected, df_species = convert_dummy_obs_subsets(df_collected, df_species)
    df_collected['Run'] = df_collected['Species subset'].astype(str) + '_' + df_collected[
        'Observation subset (training)'].astype(str)

    # ----- Functions to fit curves with

    asymptote = 0

    # def exponential(val, a):
    #     return asymptote - np.exp(-a * val)
    #
    # def exponential_non_origin(val, a, b):
    #     return asymptote - np.exp(-a * (val + b))
    #
    # def logarithmic(val, a, b):
    #     return a * np.log(val) + b
    #
    # def gompertz(val, a, b):
    #     return max(y) * np.exp(-np.exp(a - b * val))

    def vbgf(val, k, a):
        return asymptote * (1 - np.exp(-k * (val - a)))

    sns.set_style("whitegrid")

    df_collected.sort_values(by='Observations (training)', inplace=True)
    df_species.sort_values(by='Observations (training)', inplace=True)

    ci_metrics = {}
    df_curves_taxa = pd.DataFrame()
    df_curves = pd.DataFrame()

    for group in tqdm(df_collected['Taxon'].unique()):
        df_subset = df_collected[df_collected['Taxon'] == group].reset_index()

        ci_metrics[group] = {}

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(group)
        plt.subplots_adjust(wspace=.25, hspace=.4)

        for i, metric in enumerate(['Accuracy', 'Average recall', 'Average precision', 'F1']):
            x = df_subset['Observations (training)']
            y = df_subset[metric]
            hue = df_subset['Run']

            ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)], x=x, y=y,
                              hue=hue,
                              markersize=6, marker='o')

            ax.set(ylim=(0, 1), ylabel=metric, xlabel='Observations in train set')
            ax.legend([], [], frameon=False)

        plt.savefig(os.path.join(path, "GRAPHS", group + '.svg'), dpi=300)
        plt.close()

        # --------------------------------------------------------

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(group)
        plt.subplots_adjust(wspace=.25, hspace=.4)

        for i, metric in enumerate(['Accuracy', 'Average recall', 'Average precision', 'F1']):
            x = df_subset['Observations (training)'].unique()
            y = df_subset.groupby(['Observations (training)'])[metric].mean().to_list()
            sds = df_subset.groupby(['Observations (training)'])[metric].std().to_list()
            lower = np.subtract(y, np.multiply(sds, 1))
            upper = np.add(y, np.multiply(sds, 1))

            asymptote = max(y)
            popt, pcov = scipy.optimize.curve_fit(vbgf, x, y)

            df_curves_taxa = df_curves_taxa.append(
                {'taxon': group, 'metric': metric, 'k': popt[0], 'a': popt[1]},
                ignore_index=True)

            ci_metrics[group][metric] = {
                # find the point where the upper bound is equal to or higher than the highest average
                'ci_includes_best_avg': x[np.min((upper >= np.max(y)).nonzero())],

                # find the point where the avg performance is within the bounds of the best performing model
                'y_within_best_ci': x[np.min((y >= np.max(lower)).nonzero())],

                # find the point where the upper bound is within the bounds of the best performing model
                'ci_overlap': x[np.min((upper >= np.max(lower)).nonzero())],

                'vbgg_k': popt[0]

            }

            ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)], x=x, y=y,
                              markersize=6, marker='o', ci='sd')
            ax.axvline(x=ci_metrics[group][metric]['ci_includes_best_avg'], linestyle=":", color='g',
                       label="CI includes best mean")
            ax.axvline(x=ci_metrics[group][metric]['y_within_best_ci'], linestyle=":", color='b',
                       label="Mean is within best CI")
            ax.axvline(x=ci_metrics[group][metric]['ci_overlap'], linestyle=":", color='r',
                       label="CI overlaps with best model")

            ax.fill_between(x, lower, upper, color='r', alpha=.1)
            ax.set(ylim=(0, 1), ylabel=metric, xlabel='Observations in train set')
            if i == 0:
                ax.legend(loc="lower right")

            ax.plot(x, vbgf(x, *popt), 'r-')

        plt.savefig(os.path.join(path, "GRAPHS", group + ' (means).svg'), dpi=300)
        plt.close()

        # --------------------------------------------------------

        df_subset = df_species[df_species['Taxon'] == group]

        if not skip_species_plots:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(group)
            plt.subplots_adjust(wspace=.25, hspace=.4)
            x = df_subset['Observations (training)'].to_list()
            hue = df_subset['Species'].to_list()

            for i, metric in enumerate(['Average score', 'Recall', 'Precision', 'F1']):

                y = df_subset[metric].to_list()
                ax = sns.lineplot(ax=axes[int(i / 2), int(i % 2)], x=x, y=y, hue=hue, markersize=6, marker='o')

                for species in df_subset['Species'].unique():
                    df_fit = df_subset[df_subset['Species'] == species]
                    xfit = df_fit['Observations (training)'].to_list()
                    yfit = df_fit[metric].to_list()

                    asymptote = np.mean(df_fit[df_fit['Observations (training)'] == max(xfit)][metric])
                    popt, pcov = scipy.optimize.curve_fit(vbgf, xfit, yfit, maxfev=5000)
                    ax.plot(xfit, vbgf(xfit, *popt), 'r-')
                    df_curves = df_curves.append(
                        {'taxon': group, 'metric': metric, 'species': species, 'k': popt[0], 'a': popt[1]},
                        ignore_index=True)

                ax.set(ylim=(0, 1))
                ax.legend([], [], frameon=False)

            plt.savefig(os.path.join(path, "GRAPHS", group + ' per species.svg'), dpi=300)
            plt.close()

    if not skip_species_plots:
        df_curves.to_csv(os.path.join(path, "Curve parameters per species.csv"), index=False)
    else:
        df_curves = pd.read_csv(os.path.join(path, "Curve parameters per species.csv"))

    df_curves_taxa.to_csv(os.path.join(path, "Curve parameters per taxon.csv"), index=False)

    # --------------------------------------------------------

    df_group_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))
    for x, x_metric in enumerate(['Species (dataset)', 'Species (total)', 'Species representation']):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Performance over number of species")
        plt.subplots_adjust(wspace=.25, hspace=.4)

        df_x = df_group_stats[df_group_stats['Statistic'] == x_metric]
        for i, metric in enumerate(
                ['Accuracy (max)', 'Average recall (max)', 'Average precision (max)', 'F1 (max)']):
            df_subset = df_x.reset_index()
            df_subset[metric] = df_group_stats[df_group_stats['Statistic'] == metric]['Value'].tolist()
            ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], data=df_subset, x="Value", y=metric, hue="Taxon")
            ax.set_xlabel(x_metric)
            if x_metric == 'Species representation':
                ax.set(xlim=(0, df_subset['Value'].max() * 1.1))
            ax.legend([], [], frameon=False)

        plt.savefig(os.path.join(path, "GRAPHS", 'Taxon vs ' + x_metric + '.svg'), dpi=300)
        plt.close()

        # ..........

        for i, ci_metric in enumerate(list(next(iter(next(iter(ci_metrics.values())).values())).keys())):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Performance CI metrics over number of species")
            plt.subplots_adjust(wspace=.25, hspace=.4)

            for j, metric in enumerate(['Accuracy', 'Average recall', 'Average precision', 'F1']):
                df_subset = df_x.reset_index()
                df_subset[metric] = df_subset['Taxon'].apply(lambda x: ci_metrics[x][metric][ci_metric]).tolist()

                ax = sns.scatterplot(ax=axes[int(j / 2), int(j % 2)], data=df_subset, x="Value", y=metric, hue="Taxon")

                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                                     df_subset[metric])
                if p_value < .05:
                    ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
                    stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
                    stats_str += "\n"
                    if p_value < 0.0001:
                        stats_str += "p = " + '{:.2e}'.format(p_value)
                    else:
                        stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
                    ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

                ax.set_xlabel(x_metric)
                ax.set_ylabel("Images needed for " + metric)
                if (max(df_subset[metric]) > 100):
                    ax.set(ylim=(0, 220))
                ax.legend([], [], frameon=False)

                if x_metric == 'Species representation':
                    ax.set(xlim=(0, df_subset['Value'].max() * 1.1))

            plt.savefig(os.path.join(path, "GRAPHS", 'Taxon vs ' + x_metric + ' (' + ci_metric + ').svg'), dpi=300)
            plt.close()

            # .....

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Log performance CI metrics over number of species")
            plt.subplots_adjust(wspace=.25, hspace=.4)

            for j, metric in enumerate(['Accuracy', 'Average recall', 'Average precision', 'F1']):
                df_subset = df_x.reset_index()
                df_subset[metric] = df_subset['Taxon'].apply(lambda x: ci_metrics[x][metric][ci_metric]).tolist()

                df_subset[metric] = np.log(df_subset[metric])

                ax = sns.scatterplot(ax=axes[int(j / 2), int(j % 2)], data=df_subset, x="Value", y=metric, hue="Taxon")
                ax.yaxis.set_major_formatter(lambda x, y: int(np.e ** x))
                ax.set_ylabel("Images needed for " + metric)
                if (max(df_subset[metric]) > 100):
                    ax.set(ylim=(0, np.log(220)))

                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                                     df_subset[metric])

                if p_value < .05:
                    ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
                    stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
                    stats_str += "\n"
                    if p_value < 0.0001:
                        stats_str += "p = " + '{:.2e}'.format(p_value)
                    else:
                        stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
                    ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

                ax.set_xlabel(x_metric)

                if x_metric == 'Species representation':
                    ax.set(xlim=(0, df_subset['Value'].max() * 1.1))
                ax.legend([], [], frameon=False)

            plt.savefig(os.path.join(path, "GRAPHS", 'Taxon vs log ' + x_metric + ' (' + ci_metric + ').svg'), dpi=300)
            plt.close()

    # --------------------------------------------------------

    df_curves_taxa.sort_values(by=['taxon'], inplace=True)

    for x, x_metric in enumerate(['Species (dataset)', 'Species (total)', 'Species representation']):
        df_x = df_group_stats[df_group_stats['Statistic'] == x_metric]
        df_x.sort_values(by=['Taxon'], inplace=True)

        for parameter in ['a', 'k']:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Curve " + parameter + " vs " + x_metric)
            plt.subplots_adjust(wspace=.25, hspace=.4)

            for i, metric in enumerate(
                    ['Accuracy', 'Average recall', 'Average precision', 'F1']):
                y = df_curves_taxa[df_curves_taxa['metric'] == metric][parameter].tolist()

                ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], data=df_x, x="Value", y=y, hue="Taxon")

                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_x['Value'], y)

                if p_value < .05:
                    ax.plot(df_x['Value'], slope * df_x['Value'] + intercept)
                    stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
                    stats_str += "\n"
                    if p_value < 0.0001:
                        stats_str += "p = " + '{:.2e}'.format(p_value)
                    else:
                        stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
                    ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

                ax.set_xlabel(x_metric)
                ax.set_ylabel(metric)

                ax.legend([], [], frameon=False)

            plt.savefig(os.path.join(path, "GRAPHS", "Curve " + parameter + " vs " + x_metric + ').svg'), dpi=300)
            plt.close()

    for j, metric in enumerate(['Accuracy', 'Average recall', 'Average precision', 'F1']):
        df_subset = df_x.reset_index()
        df_subset[metric] = df_subset['Taxon'].apply(lambda x: ci_metrics[x][metric][ci_metric]).tolist()

        df_subset[metric] = np.log(df_subset[metric])

        ax = sns.scatterplot(ax=axes[int(j / 2), int(j % 2)], data=df_subset, x="Value", y=metric, hue="Taxon")
        ax.yaxis.set_major_formatter(lambda x, y: int(np.e ** x))
        ax.set_ylabel("Images needed for " + metric)
        if (max(df_subset[metric]) > 100):
            ax.set(ylim=(0, np.log(220)))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                             df_subset[metric])

        if p_value < .05:
            ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
            stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
            stats_str += "\n"
            if p_value < 0.0001:
                stats_str += "p = " + '{:.2e}'.format(p_value)
            else:
                stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
            ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        ax.set_xlabel(x_metric)

        if x_metric == 'Species representation':
            ax.set(xlim=(0, df_subset['Value'].max() * 1.1))
        ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", 'Taxon vs log ' + x_metric + ' (' + ci_metric + ').svg'), dpi=300)
    plt.close()

    # --------------------------------------------------------

    df_species_stats = pd.read_csv(os.path.join(path, "Stats per species.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Performance over number of images")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    df_x = df_species_stats[df_species_stats['Statistic'] == 'Images (testing)']
    for i, metric in enumerate(['Average score (max)', 'Recall (max)', 'Precision (max)', 'F1 (max)']):
        df_subset = df_x.reset_index()
        df_subset['Value'] = np.log(df_subset['Value'])
        df_subset[metric] = df_species_stats[df_species_stats['Statistic'] == metric]['Value'].tolist()
        ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], data=df_subset, x="Value", y=metric, hue="Taxon")
        ax.xaxis.set_major_formatter(lambda x, y: int(np.e ** x))

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'], df_subset[metric])

        if p_value < .05:
            ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
            stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
            stats_str += "\n"
            if p_value < 0.0001:
                stats_str += "p = " + '{:.2e}'.format(p_value)
            else:
                stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
            ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        ax.set_xlabel("Number of images in test set")
        ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", 'Species vs number of images.svg'), dpi=300)
    plt.close()

    # --------------------------------------------------------

    df_x = df_species_stats[df_species_stats['Statistic'] == 'Images (testing)'][
        ['Taxon', 'Species', 'Value']].drop_duplicates(ignore_index=True)

    df_x = df_x.reset_index()
    # df_x['Value'] = np.log(df_x['Value'])

    df_x = df_x.merge(df_curves, left_on="Species", right_on="species")

    for i, curve_metric in enumerate(['a', 'k']):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Performance over number of images")
        plt.subplots_adjust(wspace=.25, hspace=.4)

        for j, metric in enumerate(['Average score', 'Recall', 'Precision', 'F1']):
            df_subset = df_x[df_x['metric'] == metric]

            ax = sns.scatterplot(ax=axes[int(j / 2), int(j % 2)], data=df_subset, x="Value", y=curve_metric,
                                 hue="Taxon")
            # ax.xaxis.set_major_formatter(lambda x, y: int(np.e ** x))

            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                                 df_subset[curve_metric])

            if p_value < .05:
                ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
                stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
                stats_str += "\n"
                if p_value < 0.0001:
                    stats_str += "p = " + '{:.2e}'.format(p_value)
                else:
                    stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
                ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            ax.set_xlabel("Number of images in test set")
            ax.set_ylabel(metric + " " + curve_metric)
            ax.legend([], [], frameon=False)

        plt.savefig(os.path.join(path, "GRAPHS", 'Species curves ' + curve_metric + ' vs number of images.svg'),
                    dpi=300)
        plt.close()

    # --------------------------------------------------------

    for taxon in df_species_stats['Taxon'].unique():
        for i, curve_metric in enumerate(['a', 'k']):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle("Performance over number of images")
            plt.subplots_adjust(wspace=.25, hspace=.4)

            for j, metric in enumerate(['Average score', 'Recall', 'Precision', 'F1']):
                df_subset = df_x[(df_x['metric'] == metric) & (df_x['Taxon'] == taxon)]

                ax = sns.scatterplot(ax=axes[int(j / 2), int(j % 2)], data=df_subset, x="Value", y=curve_metric,
                                     hue="Taxon")
                # ax.xaxis.set_major_formatter(lambda x, y: int(np.e ** x))

                slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                                     df_subset[curve_metric])

                if p_value < .05:
                    ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
                    stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
                    stats_str += "\n"
                    if p_value < 0.0001:
                        stats_str += "p = " + '{:.2e}'.format(p_value)
                    else:
                        stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
                    ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

                ax.set_xlabel("Number of images in test set")
                ax.set_ylabel(metric + " " + curve_metric)
                ax.legend([], [], frameon=False)

            plt.savefig(
                os.path.join(path, "GRAPHS", taxon + ' species curves ' + curve_metric + ' vs number of images.svg'),
                dpi=300)
            plt.close()

    # --------------------------------------------------------
    for taxon in df_species_stats['Taxon'].unique():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Performance over number of images of " + taxon)
        plt.subplots_adjust(wspace=.25, hspace=.4)

        df_x = df_species_stats[
            (df_species_stats['Statistic'] == 'Images (testing)') & (df_species_stats['Taxon'] == taxon)]
        for i, metric in enumerate(['Average score (max)', 'Recall (max)', 'Precision (max)', 'F1 (max)']):
            df_subset = df_x.reset_index()
            df_subset['Value'] = np.log(df_subset['Value'])
            df_subset[metric] = \
                df_species_stats[(df_species_stats['Statistic'] == metric) & (df_species_stats['Taxon'] == taxon)][
                    'Value'].tolist()

            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'], df_subset[metric])
            ax = sns.scatterplot(ax=axes[int(i / 2), int(i % 2)], data=df_subset, x="Value", y=metric, hue="Taxon")
            ax.xaxis.set_major_formatter(lambda x, y: int(np.e ** x))

            if p_value < .05:
                ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
                stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
                stats_str += "\n"
                if p_value < 0.0001:
                    stats_str += "p = " + '{:.2e}'.format(p_value)
                else:
                    stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))

                ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            ax.set(ylim=(0, 1))
            ax.set_xlabel("Number of images in test set")
            ax.legend([], [], frameon=False)

        plt.savefig(os.path.join(path, "GRAPHS", 'Species vs number of images (' + taxon + ').svg'), dpi=300)
        plt.close()


def generate_selected_png(path):
    os.makedirs(os.path.join(path, "GRAPHS", "PNG"), exist_ok=True)
    df_collected = pd.read_csv(os.path.join(path, "Collected_stats.csv"))
    df_species = pd.read_csv(os.path.join(path, "Collected_stats_per_species.csv"))
    df_collected, df_species = convert_dummy_obs_subsets(df_collected, df_species)
    df_collected['Run'] = df_collected['Species subset'].astype(str) + '_' + df_collected[
        'Observation subset (training)'].astype(str)

    sns.set_style("whitegrid")
    df_collected.sort_values(by='Observations (training)', inplace=True)

    ci_metrics = {}

    for group in tqdm(df_collected['Taxon'].unique()):
        df_subset = df_collected[df_collected['Taxon'] == group].reset_index()
        ci_metrics[group] = {}

        for i, metric in enumerate(['Accuracy', 'Average recall', 'Average precision', 'F1']):
            x = df_subset['Observations (training)'].unique()
            y = df_subset.groupby(['Observations (training)'])[metric].mean().to_list()
            sds = df_subset.groupby(['Observations (training)'])[metric].std().to_list()
            lower = np.subtract(y, np.multiply(sds, 1))
            upper = np.add(y, np.multiply(sds, 1))

            ci_metrics[group][metric] = {
                # find the point where the upper bound is equal to or higher than the highest average
                'ci_includes_best_avg': x[np.min((upper >= np.max(y)).nonzero())],

                # find the point where the avg performance is within the bounds of the best performing model
                'y_within_best_ci': x[np.min((y >= np.max(lower)).nonzero())],

                # find the point where the upper bound is within the bounds of the best performing model
                'ci_overlap': x[np.min((upper >= np.max(lower)).nonzero())]
            }

    group = "Polyporales"
    metric = 'Accuracy'

    df_subset = df_collected[df_collected['Taxon'] == group].reset_index()
    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle(group)
    plt.subplots_adjust(wspace=.25, hspace=.4)

    x = df_subset['Observations (training)'].unique()
    y = df_subset.groupby(['Observations (training)'])[metric].mean().to_list()
    sds = df_subset.groupby(['Observations (training)'])[metric].std().to_list()
    lower = np.subtract(y, np.multiply(sds, 1))
    upper = np.add(y, np.multiply(sds, 1))

    ax = sns.lineplot(x=x, y=y,
                      markersize=6, marker='o', ci='sd')
    ax.axvline(x=ci_metrics[group][metric]['ci_includes_best_avg'], linestyle=":", color='g',
               label="CI includes best mean")
    ax.axvline(x=ci_metrics[group][metric]['y_within_best_ci'], linestyle=":", color='b',
               label="Mean is within best CI")
    ax.axvline(x=ci_metrics[group][metric]['ci_overlap'], linestyle=":", color='r',
               label="CI overlaps with best model")

    ax.fill_between(x, lower, upper, color='r', alpha=.1)
    ax.set(ylim=(0, 1), ylabel=metric, xlabel='Observations in train set')
    ax.legend(loc="lower right")
    plt.savefig(os.path.join(path, "GRAPHS", "PNG", group + ' (means).png'), dpi=300)
    plt.close()

    # ..................

    df_species_stats = pd.read_csv(os.path.join(path, "Stats per species.csv"))

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle("Performance over number of images")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    df_x = df_species_stats[df_species_stats['Statistic'] == 'Images (testing)']
    metric = 'Recall (max)'

    df_subset = df_x.reset_index()
    df_subset['Value'] = np.log(df_subset['Value'])
    df_subset[metric] = df_species_stats[df_species_stats['Statistic'] == metric]['Value'].tolist()
    ax = sns.scatterplot(data=df_subset, x="Value", y=metric, hue="Taxon")
    ax.xaxis.set_major_formatter(lambda x, y: int(np.e ** x))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'], df_subset[metric])

    if p_value < .05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
        ax.text(0.05, 0.05, stats_str, horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel("Number of images in test set")
    ax.legend([], [], frameon=False)
    ax.set(ylim=(0, 1.03))

    plt.savefig(os.path.join(path, "GRAPHS", "PNG", 'Species vs number of images.png'), dpi=300)
    plt.close()

    # ..........................................................

    taxon = 'Anseriformes'
    metric = 'Precision (max)'

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle("Performance over number of images of " + taxon)
    plt.subplots_adjust(wspace=.25, hspace=.4)

    df_x = df_species_stats[
        (df_species_stats['Statistic'] == 'Images (testing)') & (df_species_stats['Taxon'] == taxon)]

    df_subset = df_x.reset_index()
    df_subset['Value'] = np.log(df_subset['Value'])
    df_subset[metric] = \
        df_species_stats[(df_species_stats['Statistic'] == metric) & (df_species_stats['Taxon'] == taxon)][
            'Value'].tolist()

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'], df_subset[metric])
    ax = sns.scatterplot(data=df_subset, x="Value", y=metric, hue="Taxon")
    ax.xaxis.set_major_formatter(lambda x, y: int(np.e ** x))

    if p_value < .05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))

        ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set(ylim=(0, 1))
    ax.set_xlabel("Number of images in test set")
    ax.legend([], [], frameon=False)

    plt.savefig(
        os.path.join(path, "GRAPHS", "PNG", 'Species vs number of images (' + taxon + ').png'),
        dpi=300)
    plt.close()

    # ................................................................

    metric = 'Average recall'
    ci_metric = 'ci_overlap'
    x_metric = 'Species (dataset)'

    df_group_stats = pd.read_csv(os.path.join(path, "Stats per taxon.csv"))
    df_x = df_group_stats[df_group_stats['Statistic'] == x_metric]

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle("Log performance CI metrics over number of species")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    df_subset = df_x.reset_index()
    df_subset[metric] = df_subset['Taxon'].apply(lambda x: ci_metrics[x][metric][ci_metric]).tolist()

    # df_subset = df_subset[df_subset["Value"] < 160]

    df_subset[metric] = np.log(df_subset[metric])

    ax = sns.scatterplot(data=df_subset, x="Value", y=metric, hue="Taxon")
    ax.yaxis.set_major_formatter(lambda x, y: int(np.e ** x))
    ax.set_ylabel("Images needed for " + metric)
    ax.set(ylim=(0, np.log(220)))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                         df_subset[metric])

    ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
    stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
    stats_str += "\n"
    if p_value < 0.0001:
        stats_str += "p = " + '{:.2e}'.format(p_value)
    else:
        stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
    ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel(x_metric)
    ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", "PNG",
                             'Taxon vs log ' + x_metric + ' (' + ci_metric + ').png'), dpi=300)
    plt.close()

    # ................................................................

    y_metric = 'Observations (mean)'
    x_metric = 'Species (dataset)'

    df_subset = df_group_stats[df_group_stats['Statistic'] == x_metric]
    df_subset[y_metric] = list(df_group_stats[df_group_stats['Statistic'] == y_metric]['Value'])

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle(x_metric + " over " + y_metric)
    plt.subplots_adjust(wspace=.25, hspace=.4)

    # df_subset[y_metric] = np.log(df_subset[y_metric])

    ax = sns.scatterplot(data=df_subset, x="Value", y=y_metric, hue="Taxon")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                         df_subset[y_metric])
    if p_value < .05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
        ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel(x_metric)
    ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", "PNG",
                             y_metric + ' vs ' + x_metric + '.png'), dpi=300)
    plt.close()

    # ................................................................

    y_metric = 'Observations (total)'
    x_metric = 'Species (dataset)'

    df_subset = df_group_stats[df_group_stats['Statistic'] == x_metric]
    df_subset[y_metric] = list(df_group_stats[df_group_stats['Statistic'] == y_metric]['Value'])

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle(x_metric + " over " + y_metric)
    plt.subplots_adjust(wspace=.25, hspace=.4)

    # df_subset[y_metric] = np.log(df_subset[y_metric])

    ax = sns.scatterplot(data=df_subset, x="Value", y=y_metric, hue="Taxon")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                         df_subset[y_metric])
    if p_value < .05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
        ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel(x_metric)
    ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", "PNG",
                             y_metric + ' vs ' + x_metric + '.png'), dpi=300)
    plt.close()

    # ................................................................

    y_metric = 'Observations (mean)'
    x_metric = 'Species representation'

    df_subset = df_group_stats[df_group_stats['Statistic'] == x_metric]
    df_subset[y_metric] = list(df_group_stats[df_group_stats['Statistic'] == y_metric]['Value'])

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle(x_metric + " over " + y_metric)
    plt.subplots_adjust(wspace=.25, hspace=.4)

    # df_subset[y_metric] = np.log(df_subset[y_metric])

    ax = sns.scatterplot(data=df_subset, x="Value", y=y_metric, hue="Taxon")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                         df_subset[y_metric])
    if p_value < .05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
        ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel(x_metric)
    ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", "PNG",
                             y_metric + ' vs ' + x_metric + '.png'), dpi=300)
    plt.close()

    # ................................................................

    x_metric = 'Observations (mean)'
    y_metric = 'Average recall (max)'

    df_subset = df_group_stats[df_group_stats['Statistic'] == x_metric]
    df_subset[y_metric] = list(df_group_stats[df_group_stats['Statistic'] == y_metric]['Value'])

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle(x_metric + " over " + y_metric)
    plt.subplots_adjust(wspace=.25, hspace=.4)

    # df_subset[y_metric] = np.log(df_subset[y_metric])

    ax = sns.scatterplot(data=df_subset, x="Value", y=y_metric, hue="Taxon")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                         df_subset[y_metric])
    if p_value < .05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
        ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel(x_metric)
    ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", "PNG",
                             y_metric + ' vs ' + x_metric + '.png'), dpi=300)
    plt.close()

    # ................................................................

    x_metric = 'Observations (mean)'
    y_metric = 'Average precision (max)'

    df_subset = df_group_stats[df_group_stats['Statistic'] == x_metric]
    df_subset[y_metric] = list(df_group_stats[df_group_stats['Statistic'] == y_metric]['Value'])

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle(x_metric + " over " + y_metric)
    plt.subplots_adjust(wspace=.25, hspace=.4)

    # df_subset[y_metric] = np.log(df_subset[y_metric])

    ax = sns.scatterplot(data=df_subset, x="Value", y=y_metric, hue="Taxon")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                         df_subset[y_metric])
    if p_value < .05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
        ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel(x_metric)
    ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", "PNG",
                             y_metric + ' vs ' + x_metric + '.png'), dpi=300)
    plt.close()

    # ................................................................

    x_metric = 'Observations (total)'
    y_metric = 'Average recall (max)'

    df_subset = df_group_stats[df_group_stats['Statistic'] == x_metric]
    df_subset[y_metric] = list(df_group_stats[df_group_stats['Statistic'] == y_metric]['Value'])

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle(x_metric + " over " + y_metric)
    plt.subplots_adjust(wspace=.25, hspace=.4)

    # df_subset[y_metric] = np.log(df_subset[y_metric])

    ax = sns.scatterplot(data=df_subset, x="Value", y=y_metric, hue="Taxon")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                         df_subset[y_metric])
    if p_value < .05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
        ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel(x_metric)
    ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", "PNG",
                             y_metric + ' vs ' + x_metric + '.png'), dpi=300)
    plt.close()

    # ................................................................

    x_metric = 'Observations (total)'
    y_metric = 'Average precision (max)'

    df_subset = df_group_stats[df_group_stats['Statistic'] == x_metric]
    df_subset[y_metric] = list(df_group_stats[df_group_stats['Statistic'] == y_metric]['Value'])

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle(x_metric + " over " + y_metric)
    plt.subplots_adjust(wspace=.25, hspace=.4)

    # df_subset[y_metric] = np.log(df_subset[y_metric])

    ax = sns.scatterplot(data=df_subset, x="Value", y=y_metric, hue="Taxon")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                         df_subset[y_metric])
    if p_value < .05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
        ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel(x_metric)
    ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", "PNG",
                             y_metric + ' vs ' + x_metric + '.png'), dpi=300)
    plt.close()

    # ................................................................

    metric = 'Average recall'
    ci_metric = 'ci_overlap'
    x_metric = 'Observations (mean)'

    df_x = df_group_stats[df_group_stats['Statistic'] == x_metric]

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle("Log performance CI metrics over number of species")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    df_subset = df_x.reset_index()
    df_subset[metric] = df_subset['Taxon'].apply(lambda x: ci_metrics[x][metric][ci_metric]).tolist()

    # df_subset = df_subset[df_subset["Value"] < 160]

    df_subset[metric] = np.log(df_subset[metric])

    ax = sns.scatterplot(data=df_subset, x="Value", y=metric, hue="Taxon")
    ax.yaxis.set_major_formatter(lambda x, y: int(np.e ** x))
    ax.set_ylabel("Images needed for " + metric)
    ax.set(ylim=(0, np.log(220)))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                         df_subset[metric])
    if p_value < 0.05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
        ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel(x_metric)
    ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", "PNG",
                             'Taxon vs log ' + x_metric + ' (' + ci_metric + ').png'), dpi=300)
    plt.close()

    # ................................................................

    metric = 'Average recall'
    ci_metric = 'ci_overlap'
    x_metric = 'Observations (total)'

    df_x = df_group_stats[df_group_stats['Statistic'] == x_metric]

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle("Log performance CI metrics over number of species")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    df_subset = df_x.reset_index()
    df_subset[metric] = df_subset['Taxon'].apply(lambda x: ci_metrics[x][metric][ci_metric]).tolist()

    # df_subset = df_subset[df_subset["Value"] < 160]

    df_subset[metric] = np.log(df_subset[metric])

    ax = sns.scatterplot(data=df_subset, x="Value", y=metric, hue="Taxon")
    ax.yaxis.set_major_formatter(lambda x, y: int(np.e ** x))
    ax.set_ylabel("Images needed for " + metric)
    ax.set(ylim=(0, np.log(220)))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                         df_subset[metric])
    if p_value < 0.05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
        ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel(x_metric)
    ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", "PNG",
                             'Taxon vs log ' + x_metric + ' (' + ci_metric + ').png'), dpi=300)
    plt.close()

    # ................................................................

    metric = 'Average precision'
    ci_metric = 'ci_overlap'
    x_metric = 'Observations (mean)'

    df_x = df_group_stats[df_group_stats['Statistic'] == x_metric]

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle("Log performance CI metrics over number of species")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    df_subset = df_x.reset_index()
    df_subset[metric] = df_subset['Taxon'].apply(lambda x: ci_metrics[x][metric][ci_metric]).tolist()

    # df_subset = df_subset[df_subset["Value"] < 160]

    df_subset[metric] = np.log(df_subset[metric])

    ax = sns.scatterplot(data=df_subset, x="Value", y=metric, hue="Taxon")
    ax.yaxis.set_major_formatter(lambda x, y: int(np.e ** x))
    ax.set_ylabel("Images needed for " + metric)
    ax.set(ylim=(0, np.log(220)))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                         df_subset[metric])
    if p_value < 0.05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
        ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel(x_metric)
    ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", "PNG",
                             'Taxon vs log ' + x_metric + ' (' + ci_metric + ').png'), dpi=300)
    plt.close()

    # ................................................................

    metric = 'Average precision'
    ci_metric = 'ci_overlap'
    x_metric = 'Observations (total)'

    df_x = df_group_stats[df_group_stats['Statistic'] == x_metric]

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))
    fig.suptitle("Log performance CI metrics over number of species")
    plt.subplots_adjust(wspace=.25, hspace=.4)

    df_subset = df_x.reset_index()
    df_subset[metric] = df_subset['Taxon'].apply(lambda x: ci_metrics[x][metric][ci_metric]).tolist()

    # df_subset = df_subset[df_subset["Value"] < 160]

    df_subset[metric] = np.log(df_subset[metric])

    ax = sns.scatterplot(data=df_subset, x="Value", y=metric, hue="Taxon")
    ax.yaxis.set_major_formatter(lambda x, y: int(np.e ** x))
    ax.set_ylabel("Images needed for " + metric)
    ax.set(ylim=(0, np.log(220)))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_subset['Value'],
                                                                         df_subset[metric])
    if p_value < 0.05:
        ax.plot(df_subset['Value'], slope * df_subset['Value'] + intercept)
        stats_str = "R² = " + '{:.2f}'.format(np.round(r_value * r_value, 2))
        stats_str += "\n"
        if p_value < 0.0001:
            stats_str += "p = " + '{:.2e}'.format(p_value)
        else:
            stats_str += "p = " + '{:.5f}'.format(np.round(p_value, 5))
        ax.text(0.7, 0.95, stats_str, horizontalalignment='left', verticalalignment='top',
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax.set_xlabel(x_metric)
    ax.legend([], [], frameon=False)

    plt.savefig(os.path.join(path, "GRAPHS", "PNG",
                             'Taxon vs log ' + x_metric + ' (' + ci_metric + ').png'), dpi=300)
    plt.close()


def evaluate(skip_species_plots=False):
    # run_all_test_sets(os.environ.get("JOBS_DIR"))
    # collect_metrics(os.environ.get("JOBS_DIR"))
    create_graphs(os.environ.get("JOBS_DIR"), skip_species_plots=skip_species_plots)


if __name__ == "__main__":
    print("USAGE")
    print("evaluate(): based on folders in the directory defined by the JOBS_DIR environment variable")
