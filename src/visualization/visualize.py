import time
import re
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import DATA_DIR, REPORTS_DIR, load_dataset


def main():
    """Create a plot of the cases' text and summary lengths in words and sentences."""
    # Show the lengths of summaries and cases and put findings in a plot
    create_hist_fig(REPORTS_DIR / 'dataset_metrics.csv')


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


if __name__ == '__main__':
    main()
