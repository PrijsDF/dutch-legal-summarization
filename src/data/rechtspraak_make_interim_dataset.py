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
    all_cases = load_dataset(DATA_DIR / 'open_data_uitspraken/raw')

    # Create interim dataset (only containing completete, viable cases)
    create_interim_dataset(all_cases, save_dir=DATA_DIR / 'open_data_uitspraken/interim')


def create_interim_dataset(df, save_dir, chunks=4):
    """In the interim dataset, only those cases are included that contain both a case description and a summary, and
    the summary has to be at least 10 words."""
    mask = (df['missing_parts'] == 'none') \
        & (df['summary'] != '-') \
        & (df['summary'].str.split().str.len() >= 10)

    df = df.loc[mask, ['identifier', 'summary', 'description']]

    print(f'Saving the interim dataset of {len(df)} cases...')

    cases_per_chunk = int(len(df)/chunks)

    for chunk in range(chunks):
        # If it is the final chunk, take the remaining cases
        if chunk == chunks - 1:
            chunk_cases = df[chunk*cases_per_chunk:]
        else:
            chunk_cases = df[chunk*cases_per_chunk:(chunk+1)*cases_per_chunk]

        chunk_cases. to_parquet(save_dir / f'viable_cases_chunk_{chunk+1}.parquet', compression='brotli')
        print(f'Saved Chunk {chunk+1}.')

    del df

    print(f'Saved all cases.')


if __name__ == '__main__':
    main()
