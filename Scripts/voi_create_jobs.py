import os
import zipfile
import pandas as pd
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def add_image_ids(df_media):
    # Adds a media_id column (id's based on the observation id and an enumerator)
    record_id_to_sequential = defaultdict(lambda: -1)

    def get_sequence_number(record_id):
        record_id_to_sequential[record_id] += 1
        return str(record_id) + "_" + str(record_id_to_sequential[record_id])

    df_media['image_id'] = df_media['record_id'].apply(get_sequence_number)
    return df_media


def get_media_df(DwCA_path):
    # Retrieves the multimedia entries from a DwCA file
    df_media = pd.read_csv(zipfile.ZipFile(DwCA_path).open('multimedia.txt'), sep='\t',
                           error_bad_lines=False, usecols=["gbifID", "identifier"])
    df_media.columns = ['record_id', 'image_url']
    df_media = add_image_ids(df_media)
    return df_media


def evaluate_jobs(path):
    # Summarizes all job files in all subdirs of the path
    for root, dirs, files in os.walk(path):
        for name in files:
            if name == "all_images.csv":
                eval_file = os.path.join(root, name)
                df = pd.read_csv(eval_file)
                print("\n", root.split("/")[-2])
                print(len(df), "images")
                print(len(df['record_id'].unique()), "observations")
                print(len(df['taxon_full_name'].unique()), "species")
                print(len(df['record_id'].unique()) / len(df['taxon_full_name'].unique()),
                      "observations per species")

                print("\n")

                df = df.drop_duplicates(subset=['record_id'])

                for s in df['taxon_full_name'].unique().tolist():
                    print(s, "-", len(df[(df['taxon_full_name'] == s) & (df['set'] == "train")]), "training,",
                          len(df[(df['taxon_full_name'] == s) & (df['set'] == "validation")]), "validation")


def make_subset(df_obs, num_obs, validation_proportion, shuffle_seed):
    # Returns media of num_obs observations of each species from df_obs.
    # Splits into val/train according to validation_proportion,
    # or subsets existing spilt with same proportion if available.

    rng = np.random.RandomState(shuffle_seed)
    num_validation = int(np.ceil(num_obs * validation_proportion))
    df = pd.DataFrame()

    for species in df_obs['taxon_full_name'].unique():
        if "set" not in df_obs.columns:
            obs_ids = df_obs[df_obs['taxon_full_name'] == species]['record_id'].unique().tolist()
            rng.shuffle(obs_ids)
            if len(obs_ids) < num_obs:
                print(species, "has too few observations")
                exit(1)

            obs_ids = obs_ids[:num_obs]
            df_selection = df_obs[df_obs['record_id'].isin(obs_ids)]
            df_selection['set'] = df_obs['record_id'].apply(
                lambda x: "validation" if x in obs_ids[:num_validation] else "train")
        else:
            train_ids = df_obs[(df_obs['taxon_full_name'] == species) & (df_obs['set'] == "train")][
                'record_id'].unique().tolist()
            rng.shuffle(train_ids)

            val_ids = df_obs[(df_obs['taxon_full_name'] == species) & (df_obs['set'] == "validation")][
                'record_id'].unique().tolist()
            rng.shuffle(val_ids)

            obs_ids = train_ids[:num_obs - num_validation] + val_ids[:num_validation]
            df_selection = df_obs[df_obs['record_id'].isin(obs_ids)]

        df = df.append(df_selection, ignore_index=True)
    return df


def save_job(df, num_obs, offset, shuffle_seed, path):
    # Saves the csv, creating the directory if needed
    os.makedirs(os.path.join(path, str(num_obs) + " observations with media (species subset " + str(
        offset) + ", observation subset " + str(
        shuffle_seed) + ")", "source_data"), exist_ok=True)
    df.to_csv(os.path.join(path, str(num_obs) + " observations with media (species subset " + str(
        offset) + ", observation subset " + str(
        shuffle_seed) + ")", "source_data", "all_images.csv"),
              index=False)


def create_jobs(num_species, num_obs, reduced_factor, validation_proportion, grouping_csv, species_csv, dwca_file, observations_minimum):

    random.seed()

    jobs_dir = os.environ.get("JOBS_DIR")

    df_grouping = pd.read_csv(grouping_csv)
    df_all_species = pd.read_csv(species_csv)

    # NBIC does not provide id's, so the index is used instead
    if "taxon_id_at_source" not in df_all_species.columns:
        df_all_species['taxon_id_at_source'] = df_all_species.index

    df_all_species.rename(columns={'scientificName': 'taxon_full_name'}, inplace=True)

    columns = ['gbifID', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'scientificName']
    df_verbatim = pd.read_csv(zipfile.ZipFile(dwca_file).open('verbatim.txt'), sep='\t',
                              error_bad_lines=False, usecols=columns)
    df_verbatim.rename(columns={'scientificName': 'taxon_full_name', 'gbifID': 'record_id'}, inplace=True)

    df_media = get_media_df(dwca_file)

    df_verbatim = df_media.merge(df_verbatim, on="record_id", how="left")

    grouping_level = df_grouping.at[0, "level"]

    rng = np.random.RandomState(42)
    for group in tqdm(df_grouping[grouping_level]):
        os.makedirs(os.path.join(jobs_dir, group), exist_ok=True)

        if not os.path.isfile(
                os.path.join(jobs_dir, group, "Training observations with media.csv")) or not os.path.isfile(
            os.path.join(jobs_dir, group, "Testing observations with media.csv")):
            df_group = df_all_species[df_all_species[grouping_level] == group]
            df_group.sort_values(by=["kingdom", "phylum", "class", "order", "family", "genus", "taxon_full_name"],
                                 inplace=True)
            df_group.to_csv(os.path.join(jobs_dir, group, "All species.csv"), index=False)

            df_train_obs = pd.DataFrame()
            df_test_obs = pd.DataFrame()

            for species in df_group['taxon_full_name'].to_list():
                df_obs = df_verbatim[df_verbatim["taxon_full_name"] == species]
                df_obs = df_obs.merge(df_all_species[['taxon_full_name', 'taxon_id_at_source']], on="taxon_full_name",
                                      how="left")

                gbifids = df_obs["record_id"].unique()
                rng.shuffle(gbifids)
                gbifids = gbifids[:num_obs]

                df_train_obs = df_train_obs.append(df_obs[df_obs["record_id"].isin(gbifids)], ignore_index=True)
                df_test_obs = df_test_obs.append(df_obs[~df_obs["record_id"].isin(gbifids)], ignore_index=True)

            df_train_obs.to_csv(os.path.join(jobs_dir, group, "Training observations with media.csv"), index=False)
            df_test_obs.to_csv(os.path.join(jobs_dir, group, "Testing observations with media.csv"), index=False)

        else:
            df_train_obs = pd.read_csv(os.path.join(jobs_dir, group, "Training observations with media.csv"))
            df_test_obs = pd.read_csv(os.path.join(jobs_dir, group, "Testing observations with media.csv"))
            df_group = pd.read_csv(os.path.join(jobs_dir, group, "All species.csv"))

        if len(df_group) == num_species:
            offset = 0
        else:
            offset = random.randrange(len(df_group))

        indices = list(range(num_species))
        indices = list(map(lambda x: (x + offset) % len(df_group), indices))

        df_species = df_group.iloc[indices]
        df_species.to_csv(
            os.path.join(jobs_dir, group, "Species (subset " + str(offset) + ").csv"))

        df_obs = df_train_obs[df_train_obs['taxon_full_name'].isin(df_species['taxon_full_name'])]
        df_obs.to_csv(
            os.path.join(jobs_dir, group, "Training observations (species subset " + str(offset) + ") with media.csv"),
            index=False)

        shuffle_seed = random.randrange(9999)
        selection_size = num_obs
        while selection_size >= observations_minimum:
            df_obs = make_subset(df_obs, num_obs=selection_size, validation_proportion=validation_proportion,
                                 shuffle_seed=shuffle_seed)

            effective_shuffle_seed = shuffle_seed if selection_size < num_obs else 0
            save_job(df_obs, num_obs=selection_size, offset=offset, shuffle_seed=effective_shuffle_seed,
                     path=os.path.join(jobs_dir, group))
            selection_size = int(np.ceil(selection_size * reduced_factor))


if __name__ == "__main__":
    '''
    --------------------------------------------------------------------------------------------------------------
    This script creates training jobs from a DwCA file
    --------------------------------------------------------------------------------------------------------------
    
    Makes a directory for each group in the jobs dir
    Prepares a sorted species list in each group dir
    
    --------------------------------------------------------------------------------------------------------------
    '''

    print("USAGE")
    print("create_jobs(num_species, num_obs, reduced_factor, validation_proportion, grouping_csv, species_csv, dwca_file)")

