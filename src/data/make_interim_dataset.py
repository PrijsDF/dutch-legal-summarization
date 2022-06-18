import pandas as pd
from tqdm import tqdm

from src.utils import DATA_DIR, load_dataset


def main():
    """View Open Rechtspraak dataset with pandas."""
    # Load the raw dataset
    all_cases = load_dataset(DATA_DIR / 'raw'
                             , columns=['identifier', 'missing_parts', 'description', 'summary'])

    # Create interim dataset (only containing completete, viable cases)
    create_interim_dataset(all_cases, save_dir=DATA_DIR / 'interim')


def create_interim_dataset(df, save_dir, chunks=10):
    """In the interim dataset, only those cases are included that contain both a case description and a summary, and
    the summary has to contain at least 10 words. Note; we use 10 chunks, using less might result in memory errors."""
    # Before starting the process the dataset, we want to shuffle the data so training uses randomized data later on
    # See https://stackoverflow.com/a/34879805
    print('Shuffling the dataset...')
    df = df.sample(frac=1).reset_index(drop=True)

    cases_per_chunk = int(len(df)/chunks)

    for chunk in tqdm(range(chunks), desc=f'Saving the interim dataset of {len(df)} cases...'):
        # If it is the final chunk, take the remaining cases
        if chunk == chunks - 1:
            chunk_cases = df[chunk*cases_per_chunk:]
        else:
            chunk_cases = df[chunk*cases_per_chunk:(chunk+1)*cases_per_chunk]

        # We only want a subset of all cases to be included in our interim dataset. These are deemed the 'viable' cases
        mask = (chunk_cases['missing_parts'] == 'none') \
               & (chunk_cases['summary'] != '-') \
               & (chunk_cases['summary'].str.split().str.len() >= 10) \
               & (chunk_cases['description'].str.split().str.len() <= 1024)

        chunk_cases = chunk_cases.loc[mask, ['identifier', 'summary', 'description']]

        chunk_cases.to_parquet(save_dir / f'viable_cases_chunk_{chunk+1}.parquet', compression='brotli')

    print(f'Saved all cases.')


if __name__ == '__main__':
    main()
