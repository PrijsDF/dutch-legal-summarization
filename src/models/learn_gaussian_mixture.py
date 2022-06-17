import itertools
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture

# src.utils also loads layout parameters for pyplot
from src.utils import DATA_DIR, REPORTS_DIR


def main():
    """ We will learn a Guassian Mixture model that, later, can be used to cluster cases on. We adhered to the
    sci-kit guide found here: https://scikit-learn.org/stable/modules/mixture.html
    """
    # Load the dataset and remove the irrelevant columns
    data = pd.read_csv(DATA_DIR / 'features/clustering_features_full_1024.csv')
    # data = data.drop(columns=['redundancy', 'semantic_coherence'])

    # Find suitable number of components and convariance type first
    learn_mixture_using_bic(data, save_figure=False)

    # Learn Gaussian mixture model now we know the number of components
    # learn_mixture(data, n_components=6, cov_type='full')


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


def learn_mixture_using_bic(data, save_figure):
    """This code is largely taken from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html"""
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
    color_iter = itertools.cycle(["#1F77B4", "#834187", "#ffa500", "#009900"])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 0.2 * (i - 2)
        bars.append(
            plt.bar(
                xpos,
                bic[i * len(n_components_range) : (i + 1) * len(n_components_range)],
                width=0.2,
                color=color,
                zorder=20,
            )
        )
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
    xpos = (
            np.mod(bic.argmin(), len(n_components_range))
            + 0.65
            + 0.2 * np.floor(bic.argmin() / len(n_components_range))
    )
    plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
    plt.xlabel("Number of components")
    plt.ylabel("BIC score")
    plt.legend([b[0] for b in bars], cv_types)

    print(lowest_bic)

    plt.grid()
    # This method magically fixes the bugged layout that otherwise will be present. It also fixes margins
    plt.tight_layout()

    if save_figure:
        plt.savefig(REPORTS_DIR / 'gaussian_mixture_plot.svg', format='svg', dpi=1200)

    plt.show()


if __name__ == '__main__':
    main()
