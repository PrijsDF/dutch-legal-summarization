import sys
from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(sys.path[1])
DATA_DIR = ROOT_DIR / 'data'
REPORTS_DIR = ROOT_DIR / 'reports'
LOG_DIR = ROOT_DIR / 'reports/logs'


def load_dataset(data_dir):
    """ Read all data and combine these in a single df. Preferably, only the meta information should be fetched;
        otherwise this function might take up to 5 minutes to run."""
    #dataset_dir = data_dir #DATA_DIR / 'open_data_uitspraken/interim'  # 'raw/OpenDataUitspraken'

    # Get all data and load these into a df
    #all_cases = read_dataset(dataset_dir)
    #print(all_cases.head())

    start = time.time()

    # Read multiple parquet files into a df, preferably dont load in the summaries and case descriptions when loading
    # all data; otherwise it might take around 5 min to load the data. Without these, it takes .. min
    # columns = ['identifier', 'missing_parts', 'judgment_date']
    cases_content = pd.concat(
        pd.read_parquet(parquet_file) for parquet_file in data_dir.glob('*cases_chunk_*.parquet')
        # pd.read_parquet(parquet_file)[columns] for parquet_file in dataset_dir.glob('cases_chunk_*.parquet')
    )
    print(f'Time taken to load in dataset: {round(time.time() - start, 2)} seconds')

    return cases_content

