from collections import Counter
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# src.utils also loads layout parameters for pyplot
from src.utils import DATA_DIR, REPORTS_DIR


# Due to conflicts with yellowbrick functionality, the plots in this file require extra styling
plt.rc('patch', edgecolor='black')  # otherwise the elbow plot is printed with grey border
plt.rc('patch', linewidth='1')


def main():
    """ We will learn a k-means model that can be used to cluster cases on later.
    """
    # Load the dataset and remove the irrelevant columns
    data = pd.read_csv(DATA_DIR / 'features/clustering_features_full_1024.csv')
    # data = data.drop(columns=['redundancy', 'semantic_coherence'])

    # Find suitable number of clusters using elbow method first
    learn_k_means_elbow(data, save_figure=False)

    # Learn Gaussian mixture model now we know the number of components
    # save_mapping_path = REPORTS_DIR / 'ecli_cluster_mapping.csv'
    # save_model_path = MODELS_DIR / 'k_means_model.pkl'
    # clustered_cases = learn_k_means(data, n_clusters=6, save_mapping=False, save_mapping_path=save_mapping_path
    #                                 , save_model=False, save_model_path=save_model_path)
    #
    # print(clustered_cases)


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

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 1, 1)
    visualizer.fit(data[[
        'topic_class',
        'redundancy',
        'semantic_coherence',
        'desc_words',
        'desc_sents'
    ]])

    # If we print the plot ourselves instead of using this line, we can adjust its style and save it etc.
    # visualizer.show()

    # We need to add some extra styles to make the plot consistent with other plots; yellowbricks doesn't seem to offer
    # these options unfortunately. The linewidth and tick lengths are the defaults used in matplotlib
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set(color='black')
        ax.spines[spine].set(linewidth='0.8')

    ax.tick_params(which='major', length=3.5)

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
