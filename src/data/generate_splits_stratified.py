import itertools
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.model_selection import train_test_split

from src.utils import DATA_DIR, REPORTS_DIR, MODELS_DIR, load_dataset


def main():
    """This file looks at the k-means-clustered data and then generates a train-test-val split in a stratified
    manner. That is, the train, val and test split each will have the same distribution of classes.

    What happens is; we load the mapping of ECLIs to clusters and we load the dataset. Then we use sklearn to split the
    mapping df in a stratified manner. Finally, we map the obtained stratified mapping splits to the real dataset and
    store the final obtained splits in the processed data dir folder.
    """
    # Load the interim dataset
    all_cases = load_dataset(DATA_DIR / 'interim')

    # Load the cluster file that contains a mapping of each ECLI to a cluster
    class_mapping = pd.read_csv(REPORTS_DIR / 'ecli_cluster_mapping.csv')

    # Get the data splits and save them in save_dir
    save_dir = DATA_DIR / 'processed'
    get_data_splits(class_mapping, all_cases, save_dir, save_full_ds=False, save_cluster_ds=False)


def get_data_splits(class_mapping, all_cases, save_dir, save_full_ds, save_cluster_ds):
    """This function looks at the k-means-clustered data and then generates a train-test-val split in a stratified
    manner. That is, the train, val and test split each will have the same distribution of classes.

    The function expects as input a dataset with a column 'cluster' indicating the cluster each case was assigned to.

    If save_full_ds is set, the complete dataset will be stored in three splits; this ds corresponds to the standard
    stratified dataset.
    If save_cluster_ds is set, each of the cluster datasets will be stored in three splits; these ds correspond to
    the dataset where only cases of one class are included in the splits

    Note: when the cluster dataset files are stored there likely will be discrepancies between the total file size of
    the all_cases dataset splits and the cluster splits. I checked it with an excel sheet and it seems that the total
    file size of the cluster files is approx 95% of the size of the three main dataset splits. I guess this difference
    stems from the nature of parquet (with brotli compression?).

    In the view_dataset.py file a split can be inspected to check whether indeed only contains the cases
    of the right class and the right number of cases. I sample-checked on both these properties and it looks fine.
    """
    # We have to split twice in a row; first to get the train split and second to get the val and test split
    # For split sizes we want 70% train, 20% val, 10% test
    train_mapping, valtest_mapping = train_test_split(class_mapping, test_size=0.3, stratify=class_mapping['class']
                                                      , random_state=42)
    val_mapping, test_mapping = train_test_split(valtest_mapping, test_size=0.33, stratify=valtest_mapping['class']
                                                 , random_state=42)

    # Now that we have the stratified class mapping splits, we want to use the mapping to divide the data from all_cases
    # into three splits
    train_split = pd.merge(all_cases, train_mapping, how='inner', on=['identifier'])
    val_split = pd.merge(all_cases, val_mapping, how='inner', on=['identifier'])
    test_split = pd.merge(all_cases, test_mapping, how='inner', on=['identifier'])

    # Verify whether each has an equal distribution of classes (just a sanity check)
    #print(f'Train has the following class frequencies: {train_split.groupby("class")["class"].count()}')
    #print(f'Val has the following class frequencies: {val_split.groupby("class")["class"].count()}')
    #print(f'Test has the following class frequencies: {test_split.groupby("class")["class"].count()}')

    # Finally, save each of the three splits to the processing data folder
    if save_full_ds:
        train_split.to_parquet(save_dir / f'train_rechtspraak.parquet', compression='brotli')
        val_split.to_parquet(save_dir / f'val_rechtspraak.parquet', compression='brotli')
        test_split.to_parquet(save_dir / f'test_rechtspraak.parquet', compression='brotli')
        print(f'Saved each of the full dataset splits to {save_dir}')

    if save_cluster_ds:
        classes = [0, 1, 2, 3, 4, 5]
        for cluster in classes:
            c_train_split = train_split.loc[train_split['class'] == cluster, ]
            c_val_split = val_split.loc[val_split['class'] == cluster, ]
            c_test_split = test_split.loc[test_split['class'] == cluster, ]

            cluster_save_dir = save_dir / 'cluster_subsets'
            c_train_split.to_parquet(cluster_save_dir / f'{cluster}_train_rechtspraak.parquet', compression='brotli')
            c_val_split.to_parquet(cluster_save_dir / f'{cluster}_val_rechtspraak.parquet', compression='brotli')
            c_test_split.to_parquet(cluster_save_dir / f'{cluster}_test_rechtspraak.parquet', compression='brotli')

        print(f'Saved each of the cluster datasets splits to {save_dir}')


if __name__ == '__main__':
    main()
