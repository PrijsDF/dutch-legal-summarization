import time
import re
from pprint import pprint

import pandas as pd

from src.utils import DATA_DIR, REPORTS_DIR, load_dataset


def main():
    """Create a df with some counts on viable cases in the raw dataset."""
    # Load the features csv; this csv was created in the compute_bommasani_features.py file
    decade_counts_df = create_year_counts_df(DATA_DIR / 'open_data_uitspraken/raw')
    print(decade_counts_df)


def create_year_counts_df(dataset_dir):
    """Read the chunks one at a time and derive aggregate stats of the chunk. Returned is a df containing these stats,
    per decade
    """
    start = time.time()

    chunks = dataset_dir.glob('*cases_chunk_*.parquet')

    # Loop over the chunks and process the cases for each year
    all_years = range(1940, 2022)
    decades = list(range(1910, 2021, 10))

    years_list = [[year, 0, 0, 0, 0, 0, 0] for year in all_years]
    decades_list = [[decade, 0, 0, 0, 0, 0, 0] for decade in decades]

    for chunk in chunks:
        chunk_df = pd.read_parquet(chunk)[['identifier', 'missing_parts', 'summary']]

        for current_decade in decades:
            # Subset the df to get all cases of current decade
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

            # Get missing summary count
            missing_counts = cases_of_decade['missing_parts'].value_counts()
            if 'summary' in missing_counts:
                missing_summaries = missing_counts['summary']
            else:
                missing_summaries = 0

            # Get number of short summaries
            summaries = cases_of_decade['summary'].values
            summary_lengths = [len(summary.replace('|', ' ').split()) for summary in summaries if summary != 'none']
            short_summaries = len([sl for sl in summary_lengths if 1 < sl < 10])
            oneword_summaries = len([sl for sl in summary_lengths if sl == 1])
            viable_cases = number_of_completes - (short_summaries + oneword_summaries)

            # print(cases_of_decade.loc[cases_of_decade['missing_parts'] == 'summary', ])

            # years_list = [
            #     [year[0], year[1]+number_of_cases, year[2]+number_of_completes, year[3]+short_summaries]
            #     if year[0] == current_year
            #     else [year[0], year[1], year[2], year[3]]
            #     for year in years_list
            # ]
            decades_list = [
                [
                    decade[0],
                    decade[1] + number_of_cases,
                    decade[2] + number_of_completes,
                    decade[3] + oneword_summaries,
                    decade[4] + short_summaries,
                    decade[5] + missing_summaries,
                    decade[6] + viable_cases
                ]
                if decade[0] == current_decade
                else [decade[0], decade[1], decade[2], decade[3], decade[4], decade[5], decade[6]]
                for decade in decades_list
            ]

    # Viable: all components and an informative summary, Complete: all components but not an informative summary (e.g
    # too short)
    decade_stats_df = pd.DataFrame(columns=['decade',
                                            'cases',
                                            'complete_cases',
                                            'oneword_summaries',
                                            'short_summaries',
                                            'missing_summaries',
                                            'viable_cases'],
                                   data=decades_list)

    # Print totals
    print(f'Total number of cases: {decade_stats_df["cases"].sum()}')
    print(f'Total number of complete cases: {decade_stats_df["complete_cases"].sum()}')
    print(f'Total number of one-word summaries: {decade_stats_df["oneword_summaries"].sum()}')
    print(f'Total number of short summaries: {decade_stats_df["short_summaries"].sum()}')
    print(f'Total number of missing summaries: {decade_stats_df["missing_summaries"].sum()}')
    print(f'Total number of viable cases: {decade_stats_df["viable_cases"].sum()}')

    # Save the df to a csv
    decade_stats_df.to_csv(REPORTS_DIR / 'decades_stats.csv')

    print(f'Time taken to load in dataset: {round(time.time() - start, 2)} seconds')

    return decade_stats_df


if __name__ == '__main__':
    main()
