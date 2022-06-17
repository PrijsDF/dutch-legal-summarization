import sys
from pathlib import Path
import time

import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt


# To easily refer to common dirs use these in other files
ROOT_DIR = Path(sys.path[1])
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
REPORTS_DIR = ROOT_DIR / 'reports'
LOG_DIR = ROOT_DIR / 'reports/logs'

# These colors will be loaded into files that need to access them
COLORS = {
    'blue': '#1F77B4',
    'red': '#ff0000',
    'pink': '#ff00f0',
    'purple': '#9400d3',
    'orange': '#ff4500',
    'yellow': '#ffa500',
    'green': '#009900',
    'brown': '#800000',
}

# We specify layout rules for the plots here, as utils will be loaded in all graphing files
small_size = 16
medium_size = 20
big_size = 16

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=small_size)
plt.rc('text', usetex=True)
plt.figure(figsize=(14, 7))
plt.rc('axes', titlesize=small_size)
plt.rc('axes', labelsize=medium_size)
plt.rc('xtick', labelsize=small_size)
plt.rc('ytick', labelsize=small_size)
plt.rc('legend', fontsize=14)
plt.rc('figure', titlesize=big_size)


# This name will give a conflict if used in HF datasets scripts; as they also called their function this way
def load_dataset(data_dir, use_dask=False, columns=None):
    """ Read all data and combine these in a single df."""
    start = time.time()

    # Dask is used when computing the LDA model; without it there would be memory errors
    if use_dask:
        cases_content = dd.read_parquet(data_dir / '*cases_chunk_*.parquet', engine='pyarrow')

        # We want to repartion to contain approx. 100mb of data per partition (according to Dask's best practices)
        # our total dataset contains 15390mb
        cases_content = cases_content.repartition(npartitions=154)
    else:
        # Use Pandas
        if columns:
            cases_content = pd.concat(
                pd.read_parquet(parquet_file)[columns] for parquet_file in data_dir.glob('*cases_chunk_*.parquet')
            )
        else:
            # Columns parameter was supplied; thus only load the requested columns
            cases_content = pd.concat(
                pd.read_parquet(parquet_file) for parquet_file in data_dir.glob('*cases_chunk_*.parquet')
            )

    print(f'Time taken to load in dataset: {round(time.time() - start, 2)} seconds')

    return cases_content


if __name__ == '__main__':
    all_cases = load_dataset(DATA_DIR / 'raw')  # 'raw/bu_13-6_results_ds'
    print(len(all_cases))
