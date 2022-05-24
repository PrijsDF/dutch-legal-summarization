import itertools
from collections import Counter
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import yellowbrick
from yellowbrick.cluster import KElbowVisualizer

from src.utils import DATA_DIR, REPORTS_DIR, MODELS_DIR, load_dataset


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

plt.rc('patch', edgecolor='black')  # otherwise the elbow plot is printed with grey border
plt.rc('patch', linewidth='1')
#yellowbrick.style.rcmod.set_style(style='white')  # otherwise the elbow plot is printed in seaborn


def main():
    """ We will learn a k-means model that can be used to cluster cases on later.
    """
    # Load the dataset and remove the irrelevant columns
    data = pd.read_csv(DATA_DIR / 'open_data_uitspraken/features/clustering_features_full_1024.csv')
    # data = data.drop(columns=['redundancy', 'semantic_coherence'])

    # Plot two of the dataset's cols
    # plot_two_cols(data, x_col='topic_class', y_col='desc_sents')

    # Find suitable number of clusters using elbow method first
    # learn_k_means_elbow(data, save_figure=False)

    # Learn Gaussian mixture model now we know the number of components
    save_mapping_path = REPORTS_DIR / 'ecli_cluster_mapping.csv'
    save_model_path = MODELS_DIR / 'k_means_model.pkl'
    clustered_cases = learn_k_means(data, n_clusters=6, save_mapping=False, save_mapping_path=save_mapping_path
                                    , save_model=False, save_model_path=save_model_path)

    print(clustered_cases)


def plot_two_cols(data, x_col='desc_words', y_col='desc_words'):
    plt.scatter(x=data[x_col], y=data[y_col])
    plt.show()


def learn_k_means(data, n_clusters, save_mapping, save_mapping_path, save_model, save_model_path):
    """First use the learn_mixture_using_bic function to find the appropiate number of components and the best working
    covariance_type; only then, use this function to learn a mixture model and predict for the dataset according to
    that number of components.

    If save_mapping is true, the mapping between clusters and ECLIs will be stored. This mapping is used in
    generate_splits_stratified.py to generate the dataset splits.
    """
    k_means_model = KMeans(n_clusters=n_clusters, random_state=0)

    k_means_model.fit(data[[
        'topic_class',
        'redundancy',
        'semantic_coherence',
        'desc_words',
        'desc_sents'
    ]])

    predictions = k_means_model.predict(data[[
        'topic_class',
        'redundancy',
        'semantic_coherence',
        'desc_words',
        'desc_sents'
    ]])

    print(Counter(predictions))

    # We merge the predictions with the dataframe so we can associate the predictions with the cases
    predictions_df = pd.DataFrame(predictions, columns=['class'])
    data = pd.concat([data, predictions_df], axis=1)

    # We don't need these extra columns anymore
    data = data.drop(columns=['topic_class', 'redundancy', 'semantic_coherence', 'desc_words', 'desc_sents'])

    if save_mapping:
        data.to_csv(save_mapping_path)
        print('Saved ecli-class mapping')

    # Saving is not a native method unfortunately, so we need pickle to save and later load the model
    # From https://stackoverflow.com/a/56107843
    if save_model:
        with open(save_model_path, 'wb') as f:
            pickle.dump(k_means_model, f)

        print('Saved the k-means model')

    return data


def learn_k_means_elbow(data, save_figure):
    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1, 12), timings=False)

    visualizer.fit(data[[
        'topic_class',
        'redundancy',
        'semantic_coherence',
        'desc_words',
        'desc_sents'
    ]])
    # If we print the plot ourselves instead of using this line, we can adjust its style and save it etc.
    #visualizer.show()

    plt.xlabel('k')
    plt.ylabel('Distortion score')
    plt.legend()

    # plt.grid()
    plt.tight_layout()

    if save_figure:
        plt.savefig(REPORTS_DIR / 'k_means_elbow_plot.svg', format='svg', dpi=1200)

    plt.show()


if __name__ == '__main__':
    main()
