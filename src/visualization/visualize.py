import time
import re
from pprint import pprint
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import DATA_DIR, REPORTS_DIR, load_dataset


blue = '#1F77B4'  # Use solely for the full model
red = '#ff0000'  # Use solely for the cluster combined model

geel = '#ff00f0'
marine = '#9400d3' #'#35193E'#141E8C'
olive = '#ff4500' #'#AD1759' #'#808000'
purple = '#ffa500' #'#F37651' #'#2A0800'
grass = '#009900' #'#E13342' #'#28b463'
pink = '#800000' #'#701F57'  #'#F6B48F' #'#b428a7'

cluster_alpha = 0.7  # The opacity of cluster model lines; to make hte main model line better visible
linestyle = 'dashed'  # The type of line for the validation curves in the val loss all models graph

# Font sizes for graph texts
small_size = 16
medium_size = 20
big_size = 16

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}, size=small_size)  # controls default text sizes
plt.rc('text', usetex=True)
plt.figure(figsize=(14, 7))#, dpi=160)  #, dpi=300)
plt.rc('axes', titlesize=small_size)     # fontsize of the axes title
plt.rc('axes', labelsize=medium_size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small_size)    # fontsize of the tick labels
plt.rc('legend', fontsize=14)    # legend fontsize
plt.rc('figure', titlesize=big_size)     # fontsize of the figure title


def main():
    """Create a plot of the cases' text and summary lengths in words and sentences and other handy visualizations."""
    # Show the lengths of summaries and cases and put findings in a plot
    # create_hist_fig(REPORTS_DIR / 'descriptive_features_full_1024.csv', save_figure=True)

    # There are amounts of words that are more frequent than others, here we can check some of them
    # check_len = [300, 350]
    # for i in range(check_len[0], check_len[1]):
    #     cases_of_len = fetch_cases_of_n_len(REPORTS_DIR / 'descriptive_features_full_1024.csv', desc_len=i)
    #     #print(f'There are {len(cases_of_len)} cases with a description length of {i} words')
    #
    #     if i == 331:
    #         print(cases_of_len)

    # We want to see what the correlation is between ds size and rouge scores
    # plot_avg_rouge_ds_len(save_figure=False)

    # Inspect the summaries of a specific summary (similar to a function found in rechtspraak_view_dataset.py)
    ecli = 'ECLI:NL:CRVB:2005:AU5952'
    print_ecli_summaries(REPORTS_DIR / 'human_evaluation/evaluation_sample_evaluated.csv'
                         , ecli=ecli)


def print_ecli_summaries(data_dir, ecli):
    """Create a figure showing histograms with the word and sentence length distributions."""
    df = pd.read_csv(data_dir)
    case_df = df.loc[df['identifier'] == ecli, ]
    ref_sum = case_df['summary'].iloc[0]
    full_model_sum = case_df['summary_full_model'].iloc[0]
    cluster_sum = case_df['summary_cluster_model'].iloc[0]
    # full_text = case_df['description'].iloc[0]

    print(f'Reference summary:\n{ref_sum}'
          f'\n\nFull model summary:\n{full_model_sum}'
          f'\n\nCluster model summary:\n{cluster_sum}')

    return True


def plot_avg_rouge_ds_len(save_figure=False):
    """This function is used to show the correlation of the rouge scores that the model achieves and size of the dataset
    the model was trained on.
    Note: This plot is no longer used in the thesis document (as of 31-5-2022) as I found out that the full model
    performs equally well on these cluster subsets."""
    annotations = ["Full", "0", "1", "2", "3", "4", "5"]
    rouge_1_scores = [46.52, 39.74, 46.74, 43.82, 52.57, 46.56, 41.21]
    rouge_2_scores = [33.74, 26.41, 35.42, 31.20, 43.15, 34.35, 28.13]
    rouge_l_scores = [44.88, 37.73, 45.60, 41.98, 52.15, 44.84, 39.26]
    dataset_sizes = [100201, 23484, 13504, 19156, 6669, 15413, 21875]

    # Plot the data
    # We use below layout
    # plt.suptitle('ROUGE-1 scores and dataset size')
    plt.scatter(dataset_sizes, rouge_1_scores, label='ROUGE-1', zorder=100)
    plt.scatter(dataset_sizes, rouge_2_scores, label='ROUGE-2', zorder=100)
    plt.scatter(dataset_sizes, rouge_l_scores, label='ROUGE-L', zorder=100)
    plt.xlabel('Dataset size')
    plt.ylabel('ROUGE score')
    x_max = max(dataset_sizes)
    y_max = max(rouge_1_scores)
    plt.xlim(0, 110000)
    plt.ylim(0, 100)

    # Annotate the plotted values and a vertical guide line
    for i, label in enumerate(annotations):
        plt.axvline(x=dataset_sizes[i], color='black', linewidth=1.5, linestyle=':')
        plt.annotate(label, (dataset_sizes[i] + 400, 1))

        # For the full model, we want to print horizontal guide lines to show which of the cluster models outperform it
        if label == 'Full':
            # This colar map is the default one; it was also used to draw the scatter plots
            cmap = plt.get_cmap("tab10")
            plt.axhline(y=rouge_1_scores[i], color='C0', linewidth=1.5, linestyle=':')
            plt.axhline(y=rouge_2_scores[i], color='C1', linewidth=1.5, linestyle=':')
            plt.axhline(y=rouge_l_scores[i], color='C2', linewidth=1.5, linestyle=':')

    plt.legend(loc=1)

    plt.grid()
    plt.tight_layout()

    if save_figure:
        plt.savefig(REPORTS_DIR / 'correlation_ds_size_rouge.svg', format='svg', dpi=1200)

    plt.show()


def fetch_cases_of_n_len(data_dir, desc_len=1000):
    """There is a relatively large fraction of cases with exactly n words in the description; this function fetches
    those cases."""
    df = pd.read_csv(data_dir)

    df = df.loc[df['desc_words'] == desc_len]

    return df


def create_hist_fig(data_dir, save_figure=False):
    """Create a figure showing histograms with the word and sentence length distributions."""
    df = pd.read_csv(data_dir)

    # Plot the data
    features = ['sum_words', 'desc_words', 'sum_sents', 'desc_sents']

    # The mapping is used to retrieve the plot's title and axis labels
    texts_mapping = {
        'sum_words': ['Summary Length', 'Length (words)', 'Density'],
        'desc_words': ['Description Length', 'Length (words)', 'Density'],
        'sum_sents': ['Summary Length', 'Length (sentences)', 'Density'],
        'desc_sents': ['Description Length', 'Length (sentences)', 'Density']
    }
    # Make a subplot for each of the four columns
    for i in range(len(features)):
        feature = features[i]
        iqr = df[df[feature].between(df[feature].quantile(.0), df[feature].quantile(.99), inclusive='both')]
        uniq_count = iqr.nunique()[feature]

        plt.subplot(2, 2, i+1)
        plt.hist(iqr[feature], density=True, bins=uniq_count, zorder=10)
        # Only the first plot of both gets a title to prevent clutter
        if feature == 'sum_words' or feature == 'desc_words':
            plt.title(texts_mapping[feature][0])

        plt.xlabel(texts_mapping[feature][1])

        # Only the left column plots need to have a y label, again to prevent clutter
        if feature == 'sum_words' or feature == 'sum_sents':
            plt.ylabel(texts_mapping[feature][2])
        plt.grid(zorder=-1)
        x_max = iqr[feature].max()
        x_min = iqr[feature].min()
        plt.xlim(0, x_max)

    plt.tight_layout()

    if save_figure:
        file_format = 'svg'
        plt.savefig(REPORTS_DIR / f'hist_word_sent_len.{file_format}', format=file_format, dpi=1200)

    plt.show()

    return True


if __name__ == '__main__':
    main()
