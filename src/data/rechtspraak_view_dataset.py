import pandas as pd
from src.utils import DATA_DIR, REPORTS_DIR
import time

import matplotlib.pyplot as plt
import numpy as np
import spacy

# Some pandas options that allow to view all collumns and rows at once
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 400)
pd.options.display.width = None


def main():
    """View Open Rechtspraak dataset with pandas."""
    dataset_dir = DATA_DIR / 'interim/OpenDataUitspraken'  # 'raw/OpenDataUitspraken'

    # Get all data and load these into a df
    # all_cases = read_dataset(dataset_dir)
    # print(all_cases)

    # mask = (all_cases['missing_parts'] == 'none') \
    #    & (all_cases['summary'] != '-') \
    #    & (all_cases['summary'].str.split().str.len() > 10)

    # filtered = all_cases.loc[mask, ['identifier', 'summary', 'description']]
    # print(filtered)

    # Create interim dataset (only containing completete, viable cases)
    # create_interim_dataset(all_cases, save_dir=DATA_DIR / 'interim/OpenDataUitspraken')

    # # Get a sample of the dataset and save the sample as csv
    # samples_df = create_sample_of_df(all_cases, number_of_items=100, only_complete_items=True,
    #                                 save_sample=True, save_dir=dataset_dir)

    # # View the sample
    # print(samples_df)
    # print(samples_df.dtypes)


def read_dataset(dataset_dir):
    """Read all data and combine these in a single df. Preferably, only the meta information should be fetched;
    otherwise this function might take up to 5 minutes to run."""
    start = time.time()

    # Read multiple parquet files into a df, preferably dont load in the summaries and case descriptions when loading
    # all data; otherwise it might take around 5 min to load the data. Without these, it takes .. min
    # columns = ['identifier', 'missing_parts', 'judgment_date']
    cases_content = pd.concat(
        pd.read_parquet(parquet_file) for parquet_file in dataset_dir.glob('viable_cases_chunk_*.parquet')
        # pd.read_parquet(parquet_file)[columns] for parquet_file in dataset_dir.glob('cases_chunk_*.parquet')
    )
    print(f'Time taken to load in dataset: {round(time.time() - start, 2)} seconds')

    return cases_content


def create_sample_of_df(df, number_of_items=20, only_complete_items=True, save_sample=False, save_dir=None):
    """ Returns a subset of the df containing number_of_items cases. By default, only complete cases are included (these
    are cases with both a summary and a description). Furthermore, if needed the sample can be saved."""
    # Subset the df to only include complete cases
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


def create_interim_dataset(df, save_dir, chunks=4):
    """In the interim dataset, only those cases are included that contain both a case description and a summary, and
    the summary has to be at least 10 words."""
    mask = (df['missing_parts'] == 'none') \
        & (df['summary'] != '-') \
        & (df['summary'].str.split().str.len() >= 10)

    df = df.loc[mask, ['identifier', 'summary', 'description']]

    print(f'Saving the interim dataset of {len(df)} cases...')

    cases_per_chunk = int(len(df)/chunks)

    # for chunk in chunks:
        # chunk_df = df[chunk*cases_per_chunk-cases_per_chunk:chunk*cases_per_chunk]

    df[:100000].to_parquet(save_dir / 'viable_cases_chunk_1.parquet', compression='brotli')
    print(f'Saved Chunk 1.')

    df[100000:200000].to_parquet(save_dir / 'viable_cases_chunk_2.parquet', compression='brotli')
    print(f'Saved Chunk 2.')

    df[200000:300000].to_parquet(save_dir / 'viable_cases_chunk_3.parquet', compression='brotli')
    print(f'Saved Chunk 3.')

    df[300000:].to_parquet(save_dir / 'viable_cases_chunk_4.parquet', compression='brotli')
    print(f'Saved Chunk 4.')

    del df

    print(f'Saved all cases.')


if __name__ == '__main__':
    main()
