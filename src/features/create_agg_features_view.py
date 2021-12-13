import time
import re
from pprint import pprint

import pandas as pd

from src.utils import DATA_DIR, REPORTS_DIR, load_dataset


def main():
    """This creates a dataframe containing the average score for each of the features of the dataset. The resulting
    scores can be compared with other datasets from Bommasani."""
    # Load the features csv; this csv was created in the rechtspraak_compute_features.py file
    features_df = pd.read_csv(DATA_DIR / 'open_data_uitspraken/interim')

    agg_df = features_df.mean(axis=0)
    print(agg_df)


if __name__ == '__main__':
    main()
