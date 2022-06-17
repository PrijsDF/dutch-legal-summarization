import time
import re
from pprint import pprint

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import DATA_DIR, REPORTS_DIR, load_dataset


def main():
    """In this file, rouge scores will be computed for the generated summaries. We load in a file containing generated
    summaries for either the full dataset or one of the clusters. """
    # Now, we also need the dataset to fetch the corresponding real summary; we can use the test split with all cases
    all_cases = pd.read_parquet(DATA_DIR / 'processed/test_rechtspraak.parquet')

    # We take a sample of 40 cases to evaluate manually; the out-commented line is the preferred way, but unfortunately
    # it does not properly stratifies the sample. Furthermore, it has trouble with classes that are 0 and omits these
    # eval_sample = all_cases.sample(n=40, weights='class')
    eval_sample = all_cases.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=0.004, random_state=10))

    # We need to reset the index for later steps
    eval_sample = eval_sample.reset_index()

    # Add columns for the generated summaries, one for the full model and one for the cluster model. These will be
    # filled later
    full_model_summaries = ['' for case in range(len(eval_sample))]
    cluster_model_summaries = ['' for case in range(len(eval_sample))]

    eval_sample['summary_full_model'] = full_model_summaries
    eval_sample['summary_cluster_model'] = cluster_model_summaries

    #print(eval_sample.columns)
    #print(eval_sample)

    # This mapping will be used to load in the right data file
    file_mapping = {
        'full': REPORTS_DIR / 'generated_summaries/model_full_sums_12-05_22-16-15.csv',
        '0': REPORTS_DIR / 'generated_summaries/model_0_sums_13-05_05-58-44.csv',
        '1': REPORTS_DIR / 'generated_summaries/model_1_sums_11-05_13-59-15.csv',
        '2': REPORTS_DIR / 'generated_summaries/model_2_sums_11-05_20-17-49.csv',
        '3': REPORTS_DIR / 'generated_summaries/model_3_sums_11-05_10-25-57.csv',
        '4': REPORTS_DIR / 'generated_summaries/model_4_sums_13-05_22-28-03.csv',
        '5': REPORTS_DIR / 'generated_summaries/model_5_sums_13-05_14-14-09.csv',
    }

    models = ['full', '0', '1', '2', '3', '4', '5']

    pbar = tqdm(models)
    for model in pbar:
        pbar.set_description(f"Adding summaries of model {model} to the evaluation sample")

        # Load in the specific model's results by using the mapping
        results_df = pd.read_csv(file_mapping[model])

        # Clean the tags that were generated with the summary
        results_df['c_summary'] = results_df['c_summary'].str.slice(7, -4)  # </s><s> en <s>

        # For the full model, we need to extract every candidate summary, for the other models only if the same class
        if model == 'full':
            for i in range(len(eval_sample)):
                ecli = eval_sample.iloc[i]['identifier']

                # Fetch the generated summary
                c_summary = results_df.loc[results_df['identifier'] == ecli, ]['c_summary'].item()

                # Add it to the sample
                eval_sample.loc[i, 'summary_full_model'] = c_summary
        else:
            for i in range(len(eval_sample)):
                # Only do something if the class corresponds to the model
                if str(eval_sample.iloc[i]['class']) == model:
                    ecli = eval_sample.iloc[i]['identifier']

                    # Fetch the generated summary
                    c_summary = results_df.loc[results_df['identifier'] == ecli, ]['c_summary'].item()

                    # Add it to the sample
                    eval_sample.loc[i, 'summary_cluster_model'] = c_summary

    # Check output if necessary
    # for i in range(len(eval_sample)):
    #     a = eval_sample.iloc[i]['summary']
    #     b = eval_sample.iloc[i]['summary_full_model']
    #     c = eval_sample.iloc[i]['summary_cluster_model']
    #     print(f'{a}\n{b}\n{c}\n')

    # Finally, we append columns for each of the evaluation metrics in order to easily be able to work with the file
    # in the evaluation web app. We give the dummy value 0 as the metrics will be valued between 1 and 5
    dummy_values = [0 for i in range(len(eval_sample))]

    # We have to add four columns for each of the summaries
    summary_names = ['summary', 'summary_full_model', 'summary_cluster_model']
    for summary_name in summary_names:
        eval_sample[f'inf_{summary_name}'] = dummy_values
        eval_sample[f'rel_{summary_name}'] = dummy_values
        eval_sample[f'flu_{summary_name}'] = dummy_values
        eval_sample[f'coh_{summary_name}'] = dummy_values

    # Before saving we drop the columns that are litter
    eval_sample = eval_sample.drop(columns=['index', 'Unnamed: 0'])

    # Save the evaluation sample
    eval_sample.to_csv(REPORTS_DIR / f'evaluation_sample.csv')


if __name__ == '__main__':
    main()
