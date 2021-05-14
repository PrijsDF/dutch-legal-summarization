import pandas as pd
from src.utils import DATA_DIR, REPORTS_DIR
import time

# Some pandas options that allow to view all collumns and rows at once
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 400)
pd.options.display.width = None


def main():
    """View Open Rechtspraak dataset with pandas."""
    dataset_dir = DATA_DIR / 'raw/OpenDataUitspraken'

    # Get all data and load these into a df
    all_cases = read_dataset(dataset_dir)

    mask = (all_cases['missing_parts'] == 'none') \
        & (all_cases['summary'] != '-') \
        & (all_cases['summary'].str.split().str.len() > 10)

    filtered = all_cases.loc[mask, ['identifier', 'summary', 'description']]
    # print(filtered)

    # # Create aggregate stats dataframe
    # decade_stats = create_year_counts_df(dataset_dir)
    # print(decade_stats)

    # # Get a sample of the dataset and save the sample as csv
    samples_df = create_sample_of_df(all_cases, number_of_items=100, only_complete_items=True,
                                     save_sample=True, save_dir=dataset_dir)

    # # View the sample
    print(samples_df)
    # print(samples_df.dtypes)


def read_dataset(dataset_dir):
    """Read all data and combine these in a single df. Preferably, only the meta information should be fetched;
    otherwise this function might take up to 5 minutes to run."""
    start = time.time()

    # Read multiple parquet files into a df, preferably dont load in the summaries and case descriptions when loading
    # all data; otherwise it might take around 5 min to load the data. Without these, it takes .. min
    # columns = ['identifier', 'missing_parts', 'judgment_date']
    cases_content = pd.concat(
        pd.read_parquet(parquet_file) for parquet_file in dataset_dir.glob('cases_chunk_*.parquet')
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
               & (df['summary'].str.split().str.len() > 10)

        df = df.loc[mask, ]
        #df = df.loc[df['missing_parts'] == 'none']

    # Pick sample
    samples_df = df.sample(n=number_of_items, random_state=1)

    if save_sample:
        samples_df.to_csv(save_dir / 'sample_cases_content.csv', mode='w', index=False, header=True)

    return samples_df


def create_year_counts_df(dataset_dir):
    """Read the chunks one at a time and derive aggregate stats of the chunk. Returned is a df containing these stats,
    per decade
    """
    start = time.time()

    chunks = dataset_dir.glob('cases_chunk_*.parquet')

    # Loop over the chunks and process the cases for each year
    all_years = range(1940, 2022)
    decades = list(range(1910, 2021, 10))

    years_list = [[year, 0, 0, 0] for year in all_years]
    decades_list = [[decade, 0, 0, 0] for decade in decades]

    for chunk in chunks:
        chunk_df = pd.read_parquet(chunk)[['identifier', 'missing_parts', 'summary']]

        for current_decade in decades:
            # Subset the df to get all cases of current year
            # cases_of_year = chunk_df[chunk_df['identifier'].apply(lambda x: x.split(':')[3] == str(current_year))]
            cases_of_decade = chunk_df[
                chunk_df['identifier'].apply(lambda x: int(int(x.split(':')[3]) / 10) * 10 == current_decade)]
            number_of_cases = len(cases_of_decade)

            # Get missing counts
            missing_counts = cases_of_decade['missing_parts'].value_counts()
            if 'none' in missing_counts:
                number_of_completes = missing_counts['none']
            else:
                number_of_completes = 0

            # Get number of short summaries
            summaries = cases_of_decade['summary'].values
            summary_lengths = [len(summary.replace('|', ' ').split()) for summary in summaries if summary != 'none']
            short_summaries = len([sl for sl in summary_lengths if 1 < sl < 10])

            # years_list = [
            #     [year[0], year[1]+number_of_cases, year[2]+number_of_completes, year[3]+short_summaries]
            #     if year[0] == current_year
            #     else [year[0], year[1], year[2], year[3]]
            #     for year in years_list
            # ]
            decades_list = [
                [decade[0], decade[1] + number_of_cases, decade[2] + number_of_completes, decade[3] + short_summaries]
                if decade[0] == current_decade
                else [decade[0], decade[1], decade[2], decade[3]]
                for decade in decades_list
            ]

    decade_stats_df = pd.DataFrame(columns=['decade', 'cases', 'complete_cases', 'short_summaries'], data=decades_list)

    # Save the df to a csv
    decade_stats_df.to_csv(REPORTS_DIR / 'decades_stats.csv')

    print(f'Time taken to load in dataset: {round(time.time() - start, 2)} seconds')

    return decade_stats_df


if __name__ == '__main__':
    main()
