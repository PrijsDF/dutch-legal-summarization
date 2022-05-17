import time
import re
from pprint import pprint
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import DATA_DIR, REPORTS_DIR, MODELS_DIR, load_dataset


def main(graphs_dir=REPORTS_DIR / 'training_graphs'):
    """Verify the k-means class mapping to see whether a case really gets assigned the class that is found in the class
    mapping."""
    # Load the clustering features
    clustering_features = pd.read_csv(DATA_DIR / 'open_data_uitspraken/features/clustering_features_full_1024.csv')

    # Load the cluster file that contains a mapping of each ECLI to a cluster
    class_mapping = pd.read_csv(REPORTS_DIR / 'ecli_cluster_mapping.csv')

    model_load_path = MODELS_DIR / 'k_means_model.pkl'
    ecli = 'ECLI:NL:RVS:2006:AW3972'
    verify_mapping(ecli, class_mapping, clustering_features, model_load_path)


def verify_mapping(ecli, class_mapping, clustering_features, model_load_path):
    """We check the class of the ecli in the mapping and compute the k-means class for the same case using the k-means
    model to see whether they are the same (which should be the case)."""
    class_in_mapping = class_mapping.loc[class_mapping['identifier'] == ecli, ].iloc[0]['class']
    print(f'Testing for ecli: {ecli}, which has class {class_in_mapping}')

    # Load the k-means model using pickle (as saving is not natively supported in sklearn)
    with open(model_load_path, 'rb') as f:
        k_means_model = pickle.load(f)

    case = clustering_features.loc[clustering_features['identifier'] == ecli, ]
    predictions = k_means_model.predict(case[[
        'topic_class',
        'redundancy',
        'semantic_coherence',
        'desc_words',
        'desc_sents'
    ]])

    predicted = predictions[0]
    if predicted == class_in_mapping:
        print(f'The predicted class was {predicted}, thus the class was correctly mapped')
    else:
        print(f'The predicted class was {predicted}, which differs from the mapped clas ({class_in_mapping})')


if __name__ == '__main__':
    main()
