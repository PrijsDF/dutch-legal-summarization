import time
import re
from pprint import pprint

import pandas as pd

from src.utils import DATA_DIR, REPORTS_DIR, load_dataset


def main():
    """This creates a dataframe containing the average score for each of the features of the dataset. The resulting
    scores can be compared with other datasets from Bommasani and Cardie (2020).

    Importantly, we will filter out all outliers and litter."""
    # Load the features csv; this csv was created in the rechtspraak_compute_features.py file
    features_df = pd.read_csv(DATA_DIR / 'open_data_uitspraken/features/descriptive.csv')

    # Remove cases with a cmp of 999 in either cmp
    features_df = rm_cmp_litter(features_df)

    # Remove cases with a redundancy of 999
    features_df = rm_red_litter(features_df)

    # Create an aggregate view of the dataset to derive the dataset-wide features
    agg_df = features_df.mean(axis=0)
    #print(agg_df)


def rm_cmp_litter(df):
    """Remove cases with a cmp_words or cmp_sents of 999. This value was given when the case had either 0 words
    or zero sentences (see rechtspraak_compute_features.py).

    Todo: really remove the cases; currently it only prints them"""
    df_temp = df.loc[(df['cmp_words'] == 999) | (df['cmp_sents'] == 999), ]

    if len(df_temp) > 0:
        # Show the cases where either of the cmp's was littered
        print(df_temp)

    return df


def rm_red_litter(df):
    """Remove cases with a redundancy of 999. This value was given when the case had zero sentences
    (see rechtspraak_compute_features.py).

    Todo: really remove the cases; currently it only prints them"""
    df_temp = df.loc[(df['redundancy'] == 999), ]

    if len(df_temp) > 0:
        # Show the cases where either of the cmp's was littered
        print(df_temp)

    return df


if __name__ == '__main__':
    main()
