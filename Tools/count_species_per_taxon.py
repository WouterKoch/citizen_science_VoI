import os
import pandas as pd


def get_taxonlevel_column_names(level, language):
    languages = {
        'no': {'order': ['Orden', 'Slekt', 'Art'], 'family': ['Familie', 'Slekt', 'Art']}}

    return languages[language][level]


def count_species_per_taxon(source_folder, encoding, sep, output_file, grouping_csv, language):
    if os.path.isfile(os.path.join(os.getenv('JOBS_DIR'), output_file)):
        return

    taxa = pd.read_csv(grouping_csv)
    columns = get_taxonlevel_column_names(taxa.at[0, 'level'], language)
    taxa = taxa[taxa.at[0, 'level']].to_list()

    counts = pd.DataFrame()

    for root, dirs, files in os.walk(os.path.join(os.environ.get("JOBS_DIR"), source_folder)):
        for file in files:
            df = pd.read_csv(os.path.join(root, file), encoding=encoding, sep=sep,
                             usecols=columns).dropna()

            df = df[df[columns[0]].isin(taxa)].drop_duplicates()
            counts = counts.append(df.groupby([columns[0]]).size().reset_index(name='Species'), ignore_index=True)

    counts.columns = ['Taxon', 'Number of species']
    counts.to_csv(os.path.join(os.getenv('JOBS_DIR'), output_file), index=None)


if __name__ == "__main__":
    print("USAGE")
    print("count_species_per_taxon(): based on folders in the directory defined by the JOBS_DIR environment variable")
