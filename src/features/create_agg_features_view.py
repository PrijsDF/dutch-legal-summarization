import time
import re
from pprint import pprint

import pandas as pd

from src.utils import DATA_DIR, REPORTS_DIR, load_dataset

# Some pandas options that allow to view all collumns and rows at once
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 400)
pd.options.display.width = None
# This will prevent a warning from happening during the interpunction removal in the LDA function
pd.options.mode.chained_assignment = None


def main():
    """This creates a dataframe containing the average score for each of the features of the dataset. The resulting
    scores can be compared with other datasets from Bommasani and Cardie (2020).

    Importantly, we will filter out all outliers and litter."""
    # Load the features csv; this csv was created in the compute_bommasani_features.py file
    #features_df = pd.read_csv(DATA_DIR / 'open_data_uitspraken/features/descriptive_features_full_1024.csv')
    features_df = pd.read_csv(REPORTS_DIR / 'descriptive_features_full_1024.csv')
    print(len(features_df))
    # Remove cases with a score of 999 in any of the features (besides the simple word and sentence counts)
    features_df = remove_garbage(features_df)
    print(features_df)

    # Create an aggregate view of the dataset to derive the dataset-wide features; this view is presented in my thesis
    # in a table to allow for a comparison with Bommasani-reported datasets
    agg_df = features_df.mean(axis=0)
    print(agg_df)


def remove_garbage(df):
    """Remove cases with a score of 999 in any of the non-count features. This value was given when the case caused
    unusual behaviour in the compute_..._features scripts."""
    df_temp = df.loc[
        (df['topic_similarity'] == 999)
        | (df['abstractivity'] == 999)
        | (df['redundancy'] == 999)
        | (df['semantic_coherence'] == 999)
        | (df['cmp_words'] == 999)
        | (df['cmp_sents'] == 999)

    ]
    print(df_temp)

    df_clean = df.loc[
        ~(df['topic_similarity'] == 999)
        & ~(df['abstractivity'] == 999)
        & ~(df['redundancy'] == 999)
        & ~(df['semantic_coherence'] == 999)
        & ~(df['cmp_words'] == 999)
        & ~(df['cmp_sents'] == 999)
    ]
    #print(len(df_clean), '\n')
    # if len(df_temp) > 0:
    #     # Show the cases where either of the cmp's was littered
    #     print(df_temp)

    return df_clean


if __name__ == '__main__':
    main()
