import time
import re
from pprint import pprint
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import DATA_DIR, REPORTS_DIR, MODELS_DIR, load_dataset


def main():
    """In this file a table is printed that shows the number of cases in the raw dataset train, val and test split and
    in each of the cluster's train, val and test splits. This table is reported in the thesis. It is also used when
    computing the weighted loss of each of the cluster model training epochs in order to create an aggregate loss value
    for the complete clustering framework from the individual models (see ft_create_plot_full_models in
    plot_training_graphs.py"""
    dataset_folder = DATA_DIR / 'processed'

    datasets = ['full', '0', '1', '2', '3', '4', '5']

    results = []
    for dataset in datasets:
        if dataset == 'full':
            train = pd.read_parquet(dataset_folder / 'train_rechtspraak.parquet')
            val = pd.read_parquet(dataset_folder / 'val_rechtspraak.parquet')
            test = pd.read_parquet(dataset_folder / 'test_rechtspraak.parquet')
        else:
            train = pd.read_parquet(dataset_folder / f'cluster_subsets/{dataset}_train_rechtspraak.parquet')
            val = pd.read_parquet(dataset_folder / f'cluster_subsets/{dataset}_val_rechtspraak.parquet')
            test = pd.read_parquet(dataset_folder / f'cluster_subsets/{dataset}_test_rechtspraak.parquet')

        results.append([dataset, len(train), len(val), len(test)])

    # Transform the results to a pandas dataframe and print it
    results_df = pd.DataFrame(results, columns=['dataset', 'train_len', 'val_len', 'test_len'])
    print(results_df)

    # Sanity checks
    print(results_df.sum(axis=0))
    print(results_df.sum(axis=1))


if __name__ == '__main__':
    main()
