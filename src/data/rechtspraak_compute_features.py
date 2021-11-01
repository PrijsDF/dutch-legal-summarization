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
    all_cases = read_dataset(dataset_dir)

    cases_features = compute_features_of_df(all_cases, save_df=False, save_dir=REPORTS_DIR)
    print(cases_features)

    # agg_cases_features = create_agg_df(REPORTS_DIR / 'dataset_metrics.csv')
    # print(agg_cases_features)

    # compute_fragments(dataset_dir)

    # create_hist_fig(REPORTS_DIR / 'dataset_metrics.csv')

    # Create aggregate stats dataframe
    # decade_stats = create_year_counts_df(dataset_dir)
    # print(decade_stats)


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


def compute_fragments(dir):
    """See the paper by Grusky et al. (2018) for a textual description of the algorithm"""

    sum = "Table 1 shows the aggregate statistics about our corpus. The whole list of texts consists of more than " \
          "27 million words, of which 320302 are distinct."
    text = "Table 1 shows the basic statistics about our corpus. We obtained more than 3 million segments from 30554 " \
           "documents, and we have 362811 segments in the gist sections. Overall, 11.54% of the segments were " \
           "chosen as gist statements. The whole corpus contains more than 27 million words, among which 320302 are " \
           "distinct."

    s_tokens = sum.replace(".", "").replace(",", "").split()
    a_tokens = text.replace(".", "").replace(",", "").split()

    s_len = len(s_tokens)
    a_len = len(a_tokens)

    fragments = []
    i = 0
    j = 0

    # See the paper of Grusky et al. (2018) for a textual description
    while i < s_len:
        fragment = []

        while j < a_len:
            if s_tokens[i] == a_tokens[j]:
                i_spar = i
                j_spar = j
                # print(s_tokens[i], a_tokens[j])

                while s_tokens[i_spar] == a_tokens[j_spar]:
                    i_spar += 1
                    j_spar += 1

                    if j_spar == a_len or i_spar == s_len:
                        break

                if len(fragment) < (i_spar - i):
                    fragment = []
                    for r in range(i, i_spar):
                        fragment.append(s_tokens[r])
                else:
                    print(f"No new fragment token to add in i: {i}")

                j = j_spar
            else:
                j += 1
        i += max(len(fragment), 1)
        j = 0

        if fragment:
            fragments.append(fragment)

    print(fragments)


def compute_features_of_df(df, save_df, save_dir):
    """ Generate and save a df with the scores for the metrics as described by Bommasani and Cardie (2020). Currently,
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
    return metrics_df


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
