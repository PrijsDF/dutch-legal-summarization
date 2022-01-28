import sys
from pathlib import Path
import time

import pandas as pd

ROOT_DIR = Path(sys.path[1])
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
REPORTS_DIR = ROOT_DIR / 'reports'
LOG_DIR = ROOT_DIR / 'reports/logs'


def load_dataset(data_dir):
    """ Read all data and combine these in a single df. Preferably, only the meta information should be fetched;
        otherwise this function might take up to 5 minutes to run."""
    start = time.time()

    # columns = ['identifier', 'missing_parts', 'judgment_date']
    cases_content = pd.concat(
        pd.read_parquet(parquet_file) for parquet_file in data_dir.glob('*cases_bk2_chunk_*.parquet')  # temp _bk_
        # pd.read_parquet(parquet_file)[columns] for parquet_file in dataset_dir.glob('cases_chunk_*.parquet')
    )
    print(f'Time taken to load in dataset: {round(time.time() - start, 2)} seconds')

    return cases_content


# all_cases = load_dataset(DATA_DIR / 'open_data_uitspraken/interim')
# print(all_cases)
# casee = all_cases.loc[all_cases['identifier'] == 'ECLI:NL:PHR:2006:AY7459', ].values
#
# print(f'Case summary: {casee[0][1]}\n\nCase Text: {casee[0][2]}')
