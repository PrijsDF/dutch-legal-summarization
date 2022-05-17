import itertools
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn import mixture

from src.utils import DATA_DIR, MODELS_DIR, load_dataset

red = '#ff0000' #'#FF0000' #'#FF0000'
marine = '#9400d3' #'#35193E'#141E8C'
olive = '#ff4500' #'#AD1759' #'#808000'
purple = '#ffa500' #'#F37651' #'#2A0800'
grass = '#009900' #'#E13342' #'#28b463'
pink = '#800000' #'#701F57'  #'#F6B48F' #'#b428a7'
blue = '#1F77B4' #'#1F77B4' # '#1f77ba

cluster_alpha = 0.7  # The opacity of cluster model lines; to make hte main model line better visible
linestyle = 'solid'  # The type of line for the validation curves in the val loss all models graph

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
    """ We will learn a Guassian Mixture model that, later, can be used to cluster cases on. We adhered to the
    sci-kit guide found here: https://scikit-learn.org/stable/modules/mixture.html
    """
    # Load the dataset and remove the irrelevant columns
    data = pd.read_csv(DATA_DIR / 'open_data_uitspraken/features/clustering_features_full_1024.csv')
    # data = data.drop(columns=['redundancy', 'semantic_coherence'])

    # Plot two of the dataset's cols
    # plot_two_cols(data, x_col='topic_class', y_col='desc_sents')

    # Find suitable number of components and convariance type first
    learn_mixture_using_bic(data)

    # Learn Gaussian mixture model now we know the number of components
    # learn_mixture(data, n_components=6, cov_type='full')


def plot_two_cols(data, x_col='desc_words', y_col='desc_words'):
    plt.scatter(x=data[x_col], y=data[y_col])
    plt.show()


def learn_mixture(data, n_components, cov_type='full'):
    """First use the learn_mixture_using_bic function to find the appropiate number of components and the best working
    covariance_type; only then, use this function to learn a mixture model and predict for the dataset according to
    that number of components."""
    gmm = mixture.GaussianMixture(
        n_components=n_components, covariance_type=cov_type
    )

    predictions = gmm.fit_predict(data[[
        'topic_class',
        'redundancy',
        'semantic_coherence',
        'desc_words',
        'desc_sents'
    ]])

    print(Counter(predictions))


def learn_mixture_using_bic(data):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 10)
    cv_types = ["tied", "diag", "full"]  # ["spherical", "tied", "diag", "full"] # Spherical was bad in all cases
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type=cv_type
            )
            gmm.fit(data[[
                'topic_class',
                'redundancy',
                'semantic_coherence',
                'desc_words',
                'desc_sents'
            ]])
            bic.append(gmm.bic(data[[
                'topic_class',
                'redundancy',
                'semantic_coherence',
                'desc_words',
                'desc_sents'
            ]]))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 0.2 * (i - 2)
        bars.append(
            plt.bar(
                xpos,
                bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                width=0.2,
                color=color,
            )
        )
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
    plt.title("BIC score per model")
    xpos = (
            np.mod(bic.argmin(), len(n_components_range))
            + 0.65
            + 0.2 * np.floor(bic.argmin() / len(n_components_range))
    )
    plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
    spl.set_xlabel("Number of components")
    spl.legend([b[0] for b in bars], cv_types)

    print(lowest_bic)
    plt.show()


if __name__ == '__main__':
    main()
