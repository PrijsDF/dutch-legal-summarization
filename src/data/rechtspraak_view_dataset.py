import time

import pandas as pd

from src.utils import DATA_DIR, REPORTS_DIR, load_dataset

# Some pandas options that allow to view all collumns and rows at once
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 400)
pd.options.display.width = None


def main():
    """View Open Rechtspraak dataset with pandas."""
    # Load the raw dataset
    #all_cases = load_dataset(DATA_DIR / 'open_data_uitspraken/interim')
    # all_cases = pd.read_parquet(DATA_DIR / 'open_data_uitspraken/processed/test_rechtspraak.parquet')
    # print(len(all_cases))

    # Get a sample of the dataset and save the sample as csv
    # samples_df = create_sample_of_df(all_cases, number_of_items=100, only_complete_items=False,
    #                                  save_sample=False, save_dir=REPORTS_DIR)
    #
    # print(samples_df)

    # # Get a subset of the dataset derived from a list of ecli-identifiers
    # eclis = ['ECLI:NL:RVS:2012:BW3031', 'ECLI:NL:RVS:2012:BW3031']
    # ecli_subset = create_subset_from_ecli(all_cases, eclis=eclis, only_complete_items=False,
    #                                       save_sample=False, save_dir=REPORTS_DIR)

    # # View the sample
    # print(samples_df)
    # print(samples_df.dtypes)

    # View a cluster dataset split (e.g. check whether this is correct
    # {0: 23584, 5: 21875, 2: 19156, 4: 15413, 1: 13504, 3: 6669})
    # cluster_file = '3_test_rechtspraak'
    # cluster_file_path = DATA_DIR / f'open_data_uitspraken/processed/cluster_subsets/{cluster_file}.parquet'
    # df = pd.read_parquet(cluster_file_path)
    # print(df)


def create_sample_of_df(df, number_of_items=20, only_complete_items=True, save_sample=False, save_dir=None):
    """ Returns a subset of the df containing number_of_items cases. By default, only complete cases are included (these
    are cases with both a summary and a description). Furthermore, if needed the sample can be saved."""
    # Subset the df to only include complete cases; this should only be done by the raw dataset
    if only_complete_items:
        mask = (df['missing_parts'] == 'none') \
               & (df['summary'] != '-') \
               & (df['summary'].str.split().str.len() >= 10)

        df = df.loc[mask, ]

    # Pick sample
    samples_df = df.sample(n=number_of_items, random_state=1)

    if save_sample:
        samples_df.to_csv(save_dir / 'sample_cases_content.csv', mode='w', index=False, header=True)

    return samples_df


def create_subset_from_ecli(df, eclis, only_complete_items=False, save_sample=False, save_dir=None):
    """ Returns a subset of the df containing number_of_items cases. By default, only complete cases are included (these
    are cases with both a summary and a description). Furthermore, if needed the sample can be saved."""
    # Subset the df to only include complete cases
    if only_complete_items:
        mask = (df['missing_parts'] == 'none') \
               & (df['summary'] != '-') \
               & (df['summary'].str.split().str.len() >= 10)

        df = df.loc[mask, ]

    # Pick sample
    # samples_df = df.sample(n=number_of_items, random_state=1)
    ecli_df = df[df['identifier'].isin(eclis)]

    if save_sample:
        ecli_df.to_csv(save_dir / 'ecli_subset.csv', mode='w', index=False, header=True)

    return ecli_df


def view_cluster_split(df):
    """This function can be used to load a specific cluster dataset split. It may be useful to sanity-check whether
    the number of cases in each of the cluster_splits corresponds to the following distribution (which was learned and
    used by k-means):
    {0: 23584, 5: 21875, 2: 19156, 4: 15413, 1: 13504, 3: 6669}
     """


if __name__ == '__main__':
    main()
