import os

import pandas as pd
import zipfile

import matplotlib.pyplot as plt
import seaborn as sns

"""
    Reads verbatim names from a DwCA file with observations
    Filters species based on a numerical threshold
    Gets the number of species within each represented taxonomic level
"""


def get_groups(path, threshold, numberOfSpecies, taxonlevel):
    columns = ['scientificName', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    df_verbatim = pd.read_csv(zipfile.ZipFile(path).open('verbatim.txt'), sep='\t', usecols=columns,
                              error_bad_lines=False)
    df_counts = pd.DataFrame(df_verbatim.value_counts(subset=columns)).reset_index()
    df_counts.columns = columns + ['count']
    df_counts = df_counts[df_counts['count'] >= threshold]

    df_groups = pd.DataFrame()
    levels = columns[1:columns.index(taxonlevel)+1]

    for row in range(len(df_counts)):
        row_object = {'level': taxonlevel,
                      'species': len(df_counts[df_counts[taxonlevel] == df_counts.at[row, taxonlevel]])
                      }
        for level in levels:
            row_object[level] = df_counts.at[row, level]

        df_groups = df_groups.append(row_object, ignore_index=True)

    df_groups.drop_duplicates(inplace=True)
    df_groups.sort_values(by=['species'], inplace=True, ascending=False, ignore_index=True)
    df_groups = df_groups[df_groups['species'] >= numberOfSpecies]
    df_groups["species"] = df_groups["species"].astype(int)
    df_groups.sort_values(by=levels, inplace=True, ignore_index=True)

    output_path = path.split(".")[0] + " - grouped by " + taxonlevel + ".csv"
    df_groups.to_csv(output_path, index=None)
    return output_path


def propse_groups(path, threshold, groups):
    columns = ['scientificName', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']
    df_verbatim = pd.read_csv(zipfile.ZipFile(path).open('verbatim.txt'), sep='\t', usecols=columns,
                              error_bad_lines=False)

    df_counts = pd.DataFrame(df_verbatim.value_counts(subset=columns)).reset_index()
    df_counts.columns = columns + ['count']
    df_counts = df_counts[df_counts['count'] >= threshold]

    print(len(df_counts), "species with at least", threshold, "observations with images\n\n")

    df_groups = pd.DataFrame(columns=['level', 'name', 'count'])

    for level in columns[1:]:
        for name in df_counts[level].unique():
            df_groups = df_groups.append({'level': level, 'name': name,
                                      'count': len(df_counts[df_counts[level] == name])},
                                     ignore_index=True)

    df_groups.drop_duplicates(inplace=True)
    df_groups.sort_values(by=['count'], inplace=True, ascending=False, ignore_index=True)

    for rank in columns[1:]:
        df_rank = df_groups[(df_groups['level'] == rank)]
        df_rank.sort_values(by=['count'], inplace=True, ascending=False, ignore_index=True)
        groupsize = df_rank.at[min(groups - 1, len(df_rank) - 1), 'count']
        print("The rank of", rank, "can" if len(df_rank) >= groups else "cannot", "be divided into", groups, "groups. There are enough species to fill",
              len(df_rank[(df_rank['count'] >= groupsize)]), "out of", len(df_rank), rank + "-based", "groups with at least",
              groupsize, "species")

    df_counts.to_csv(path.split('.zip')[0] + ' - observations per species.csv', index=None)
    df_groups.to_csv(path.split('.zip')[0] + ' - species per taxonomic level.csv', index=None)

    return path.split('.zip')[0] + ' - observations per species.csv'



if __name__ == '__main__':
    print("USAGE")
    print("propse_groups(path, threshold, groups) to propose taxonomic groups based on a DwCA file, a minimum number of observations per species, and a number of groups.")
    print("get_groups(path, threshold, numberOfSpecies, taxonlevel) to retrieve taxonomic groups based on a DwCA file, a minimum number of observations per species, a minimum of number of species, and the taxoomic level to group on.")