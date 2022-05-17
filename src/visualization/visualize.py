import time
import re
from pprint import pprint

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
    #create_hist_fig(REPORTS_DIR / 'descriptive_features_full_1024.csv')

    # There are amounts of words that are more frequent than others, important to check the cause of this
    # check_len = [300, 350]
    # for i in range(check_len[0], check_len[1]):
    #     cases_of_len = fetch_cases_of_n_len(REPORTS_DIR / 'descriptive_features_full_1024.csv', desc_len=i)
    #     #print(f'There are {len(cases_of_len)} cases with a description length of {i} words')
    #
    #     if i == 331:
    #         print(cases_of_len)

    # We want to see what the correlation is between ds size and rouge scores
    plot_avg_rouge_ds_len()


def plot_avg_rouge_ds_len():
    """This function is used to show the correlation of the rouge scores that the model achieves and size of the dataset
    the model was trained on."""
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

    # Annotate the plotted values
    for i, label in enumerate(annotations):
        #plt.annotate(label, (dataset_sizes[i] + 5, rouge_1_scores[i] + 1))
        #plt.annotate(label, (dataset_sizes[i] + 5, rouge_2_scores[i] + 1))
        #plt.annotate(label, (dataset_sizes[i] + 5, rouge_l_scores[i] + 1))
        plt.axvline(x=dataset_sizes[i], color='black', linewidth=1.5, linestyle=':')
        plt.annotate(label, (dataset_sizes[i] + 400, 1))

    plt.legend(loc=1)

    plt.grid()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'correlation_ds_size_rouge.svg', format='svg', dpi=1200)
    plt.show()


def fetch_cases_of_n_len(data_dir, desc_len=1000):
    """There is a relatively large fraction of cases with exactly n words in the description; this function fetches
    those cases."""
    df = pd.read_csv(data_dir)

    df = df.loc[df['desc_words'] == desc_len]

    return df


def create_hist_fig(data_dir):
    """Create a figure showing histograms with the word and sentence length distributions."""
    df = pd.read_csv(data_dir)

    # Plot the data
    # We use below layout
    plt.style.use('seaborn-whitegrid')  # ggplot - seaborn - searborn-darkgrid
    plt.figure(figsize=(14, 7))  # , dpi=300)
    plt.suptitle('Probability Density of Case Summary and Case Description Length Measures')
    plt.subplots_adjust(left=0.055, bottom=0.06, right=0.98, top=0.9, wspace=0.1, hspace=None)

    # Make a 2x2 fig (4 subplots)
    # Summary words
    iqr = df[df['sum_words'].between(df['sum_words'].quantile(.0), df['sum_words'].quantile(.99), inclusive=True)]
    #iqr = df[df['sum_words'].between(df['sum_words'].quantile(.0), df['sum_words'].quantile(1), inclusive=True)]
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

    # Description words
    iqr = df[df['desc_words'].between(df['desc_words'].quantile(.0), df['desc_words'].quantile(.99), inclusive=True)]
    #iqr = df[df['desc_words'].between(df['desc_words'].quantile(.0), df['desc_words'].quantile(1), inclusive=True)]
    uniq_count = iqr.nunique()['desc_words']
    plt.subplot(2, 2, 2)
    plt.hist(iqr['desc_words'], density=True, bins=uniq_count)
    plt.title('Description Length')
    plt.xlabel('Length (words)')
    x_max = iqr['desc_words'].max()
    x_min = iqr['desc_words'].min()
    plt.xlim(x_min, x_max)
    plt.xticks(np.arange(0, x_max, 100))

    iqr = df[df['sum_sents'].between(df['sum_sents'].quantile(.0), df['sum_sents'].quantile(.99), inclusive=True)]
    #iqr = df[df['sum_sents'].between(df['sum_sents'].quantile(.0), df['sum_sents'].quantile(1), inclusive=True)]
    uniq_count = iqr.nunique()['sum_sents']
    plt.subplot(2, 2, 3)
    plt.hist(iqr['sum_sents'], density=True, bins=uniq_count)
    plt.ylabel('Density')
    plt.xlabel('Length (sentences)')
    x_max = iqr['sum_sents'].max()
    x_min = iqr['sum_sents'].min()
    plt.xlim(0, x_max)
    plt.xticks(np.arange(0, x_max, 1))

    iqr = df[df['desc_sents'].between(df['desc_sents'].quantile(.0), df['desc_sents'].quantile(.99), inclusive=True)]
    #iqr = df[df['desc_sents'].between(df['desc_sents'].quantile(.0), df['desc_sents'].quantile(1), inclusive=True)]
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
