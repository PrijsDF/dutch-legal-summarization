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
    the summary has to contain at least 10 words. Note; we use 6 chunks, 4 gives a SIGKILL error with exit code 137."""
    # TODO: here, also the desc length must be specified; it should be more than 5 words or so
    # TODO: Make sure to remove leading-dots (.) from summaries; these are sometimes included (in three cases) and cause
    # errors when computing rouge scores

    # We only want a subset of all cases to be included in our interim dataset. These are deemed the 'viable' cases
    mask = (df['missing_parts'] == 'none') \
        & (df['summary'] != '-') \
        & (df['summary'].str.split().str.len() >= 10)  # \
        # & (df['description'].str.split().str.len() >= 10)

    df = df.loc[mask, ['identifier', 'summary', 'description']]

    # Only now, we use the more expensive mask; I expect that otherwise this function will be too expensive
    # This might actually better be done in a seperate file using the datasets library because the model uses sub-word
    # tokenization and not word-level tokenization. But it is what it is
    #mask = (df['description'].str.split().str.len() <= 1024)
    #print('Removing all descriptions with more than 1024 tokens...')
    #df = df[df['description'].str.split().str.len() <= 1024]

    # Before saving the dataset, we want to shuffle the data so training uses randomized data later on
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

        chunk_cases.to_parquet(save_dir / f'viable_cases_chunk_{chunk+1}.parquet', compression='brotli')

    print(f'Saved all cases.')


if __name__ == '__main__':
    main()
