import sys
from pathlib import Path
import time

import pandas as pd
import dask.dataframe as dd

ROOT_DIR = Path(sys.path[1])
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
REPORTS_DIR = ROOT_DIR / 'reports'
LOG_DIR = ROOT_DIR / 'reports/logs'


# This name will give a conflict if used in HF datasets scripts; as they also called their function this way
def load_dataset(data_dir, use_dask=False, columns=None):
    """ Read all data and combine these in a single df."""
    start = time.time()

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


# test = '. Aanvraag niet gedaan door een belanghebbende. daarom geen aanvraag in de zin van de Awb. Verweerder was ' \
#        'daarom niet gehouden om een besluit te nemen. Niet voldaan aan de voorwaarden voor het instellen van beroep ' \
#        'NT. '

# print(test.strip()[1:].strip())
# print(test)
#
# all_cases = load_dataset(DATA_DIR / 'open_data_uitspraken/interim')
#
# a = all_cases[all_cases['summary'].str.startswith('. ')]['summary']
# print(a)
# b = all_cases.iloc[[43508]]['summary']
# print(b)
#
# print(all_cases.iloc[2903])
#all_cases = load_dataset(DATA_DIR / 'open_data_uitspraken/interim', columns=['identifier', 'summary'])
#print(all_cases)
#print(f'Size of dataset as df: {round(sys.getsizeof(all_cases) / 1024 / 1024, 2)} mb')
#print(all_cases['judgment_date'])

# casee = all_cases.loc[all_cases['identifier'] == 'ECLI:NL:PHR:2006:AY7459', ].values
#
# print(f'Case summary: {casee[0][1]}\n\nCase Text: {casee[0][2]}')
