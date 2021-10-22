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

    # cases_features = compute_features_of_df(all_cases, save_df=True, save_dir=REPORTS_DIR)
    # print(cases_features)

    # agg_cases_features = create_agg_df(REPORTS_DIR / 'dataset_metrics.csv')
    # print(agg_cases_features)

    x = 'cases_featfures = comput'  # ['dit', 'is', 'een', 'testzin.', 'Of', 'niet.']
    y = 'cases_features = comp'  # ['dit', 'is', 'niet', 'testzin.', 'Of', 'wel.']
    c = temp(x, y)
    print(c)

    create_hist_fig(REPORTS_DIR / 'dataset_metrics.csv')

    # mask = (all_cases['missing_parts'] == 'none') \
    #    & (all_cases['summary'] != '-') \
    #    & (all_cases['summary'].str.split().str.len() > 10)

    # filtered = all_cases.loc[mask, ['identifier', 'summary', 'description']]
    # print(filtered)

    # Create interim dataset (only containing completete, viable cases)
    # create_interim_dataset(all_cases, save_dir=DATA_DIR / 'interim/OpenDataUitspraken')

    # Create aggregate stats dataframe
    # decade_stats = create_year_counts_df(dataset_dir)
    # print(decade_stats)

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


def temp(x, y):
    """Code from https://www.geeksforgeeks.org/longest-common-substring-dp-29/"""
    m = len(x)
    n = len(y)

    # Table storing the lcs for each token pair onward
    lcs_uff = [[0 for k in range(n+1)] for l in range(m+1)]

    result = 0

    # Following steps to build
    # LCSuff[m+1][n+1] in bottom up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                lcs_uff[i][j] = 0
            elif x[i-1] == y[j-1]:
                lcs_uff[i][j] = lcs_uff[i-1][j-1] + 1
            else:
                lcs_uff[i][j] = 0

    for r in lcs_uff:
        print(r)

    return max([max(value) for value in lcs_uff])


def compute_features_of_df(df, save_df, save_dir):
    """ Generate and safe a df with the scores for the metrics as described by Bommasani and Cardie (2020). Currently,
    we don't filter the texts on punctuation and stopwords."""
    # Create the spacy nlp object; we disable those features that are not needed
    nlp = spacy.load("nl_core_news_sm", exclude=[
        'tok2vec',
        'morphologizer',
        'tagger',
        'parser',
        'attribute_ruler',
        'lemmatizer',
        'ner'])
    nlp.add_pipe('sentencizer')  # Because we removed the parser, we need to manualy add sentencizer back
    nlp.max_length = 1500000  # Otherwise the limit will be 1000000 which is too little

    start = time.time()

    print(f'Starting computation of features. Total cases: {len(df)}')

    metrics_df = pd.DataFrame(columns=[
        'identifier',
        'sum_words',
        'sum_sents',
        'desc_words',
        'desc_sents',
        'cmp_words',
        'cmp_sents',
        'rouge_l'
    ])

    for i in range(len(df)):
        if (i + 1) % 500 == 0:
            print(f'{len(df) - (i + 1)} cases left. Time elapsed since start: {round(time.time() - start, 2)} seconds')

        identifier = df['identifier'].values[i]
        summ = df['summary'].values[i]
        dsc = df['description'].values[i]

        # Create the spacy documents
        sum_doc = nlp(summ)
        dsc_doc = nlp(dsc)

        def get_metrics(metricss_df, sum_docc, dsc_docc):
            # Show summary tokens
            # temp = [token for token in sum_docc]
            # print(temp)

            # Get summary length in words and sentences
            sum_words = len([token for token in sum_docc])
            # sum_words_cleaned = len([token for token in sum_docc if not token.is_punct and not token.is_stop])
            sum_sents = len([sent for sent in sum_docc.sents])

            # Get description length in words and sentences
            dsc_words = len([token for token in dsc_docc])
            # sum_words_cleaned = len([token for token in sum_docc if not token.is_punct and not token.is_stop])
            dsc_sents = len([sent for sent in dsc_docc.sents])

            # Compute the compression scores
            if sum_words > 0 and dsc_words > 0:
                cmp_words = 1 - sum_words / dsc_words
            else:
                cmp_words = 999

            if sum_sents > 0 and dsc_sents > 0:
                cmp_sents = 1 - sum_sents / dsc_sents
            else:
                cmp_sents = 999

            # Scores for current case
            case_row = {
                'identifier': identifier,
                'sum_words': sum_words,
                'sum_sents': sum_sents,
                'desc_words': dsc_words,
                'desc_sents': dsc_sents,
                'cmp_words': cmp_words,
                'cmp_sents': cmp_sents,
                'rouge_l': 0  # rouge_l
            }

            # Append row to the dataframe
            metricss_df = metricss_df.append(case_row, ignore_index=True)

            # print(i, 'sum_words:', sum_words, 'sum_sents:', sum_sents)
            return metricss_df

        metrics_df = get_metrics(metrics_df, sum_doc, dsc_doc)

    if save_df:
        metrics_df.to_csv(save_dir / 'dataset_metrics.csv')

    print(f'Total time taken to compute metrics of dataset: {round(time.time() - start, 2)} seconds')
    return metrics_df  # cases_df


def create_agg_df(data_dir):
    df = pd.read_csv(data_dir)

    df = df.mean(axis=0)

    return df


def create_hist_fig(data_dir):
    """Create a figure showing histograms with the word and sentence length distributions."""
    df = pd.read_csv(data_dir)

    # Plot the data
    # We use below layout
    plt.style.use('seaborn-whitegrid')  # ggplot - seaborn - searborn-darkgrid
    plt.figure(figsize=(14, 7))  # , dpi=300)
    plt.suptitle('Probability Density of Case Summary and Case Description Length Measures')
    plt.subplots_adjust(left=0.055, bottom=0.06, right=0.98, top=0.9, wspace=0.1, hspace=None) ###

    # Make a 2x2 fig (4 subplots)
    # Sum words
    iqr = df[df['sum_words'].between(df['sum_words'].quantile(.0), df['sum_words'].quantile(.975), inclusive=True)]
    uniq_count = iqr.nunique()['sum_words']
    plt.subplot(2, 2, 1)
    plt.hist(iqr['sum_words'], density=True, bins=uniq_count)
    plt.title('Summary Length')
    plt.xlabel('Length (words)')
    plt.ylabel('Density')
    x_max = iqr['sum_words'].max()
    x_min = iqr['sum_words'].min()
    plt.xlim(0, x_max)
    plt.xticks(np.arange(0, x_max, 20))

    iqr = df[df['desc_words'].between(df['desc_words'].quantile(.0), df['desc_words'].quantile(.975), inclusive=True)]
    uniq_count = iqr.nunique()['desc_words']
    plt.subplot(2, 2, 2)
    plt.hist(iqr['desc_words'], density=True, bins=uniq_count)
    plt.title('Description Length')
    plt.xlabel('Length (words)')
    x_max = iqr['desc_words'].max()
    x_min = iqr['desc_words'].min()
    plt.xlim(x_min, x_max)
    plt.xticks(np.arange(x_min, x_max, 1000))

    iqr = df[df['sum_sents'].between(df['sum_sents'].quantile(.0), df['sum_sents'].quantile(.975), inclusive=True)]
    uniq_count = iqr.nunique()['sum_sents']
    plt.subplot(2, 2, 3)
    plt.hist(iqr['sum_sents'], density=True, bins=uniq_count)
    plt.ylabel('Density')
    plt.xlabel('Length (sentences)')
    x_max = iqr['sum_sents'].max()
    x_min = iqr['sum_sents'].min()
    plt.xlim(0, x_max)
    plt.xticks(np.arange(0, x_max, 1))

    iqr = df[df['desc_sents'].between(df['desc_sents'].quantile(.0), df['desc_sents'].quantile(.975), inclusive=True)]
    uniq_count = iqr.nunique()['desc_sents']
    plt.subplot(2, 2, 4)
    plt.hist(iqr['desc_sents'], density=True, bins=uniq_count)
    plt.xlabel('Length (sentences)')
    x_max = iqr['desc_sents'].max()
    x_min = iqr['desc_sents'].min()
    plt.xlim(x_min, x_max)
    plt.xticks(np.arange(x_min, x_max, 20))

    plt.show()

    return True


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


def create_year_counts_df(dataset_dir):
    """Read the chunks one at a time and derive aggregate stats of the chunk. Returned is a df containing these stats,
    per decade
    """
    start = time.time()

    chunks = dataset_dir.glob('cases_chunk_*.parquet')

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
