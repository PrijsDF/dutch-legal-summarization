import time
import re
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from rouge import Rouge
from tqdm import tqdm

from src.utils import DATA_DIR, REPORTS_DIR, load_dataset

# Initialize the rouge scorer
rouge_scorer = Rouge()


def main():
    """In this file, rouge scores will be computed for the generated summaries. We load in a file containing generated
    summaries for either the full dataset or one of the clusters. """
    # This mapping is used to load in the appriopiate dataset
    file_mapping = {
        'full': REPORTS_DIR / 'generated_summaries/model_full_sums_12-05_22-16-15.csv',
        '0': REPORTS_DIR / 'generated_summaries/model_0_sums_13-05_05-58-44.csv',
        '1': REPORTS_DIR / 'generated_summaries/model_1_sums_11-05_13-59-15.csv',
        '2': REPORTS_DIR / 'generated_summaries/model_2_sums_11-05_20-17-49.csv',
        '3': REPORTS_DIR / 'generated_summaries/model_3_sums_11-05_10-25-57.csv',
        '4': REPORTS_DIR / 'generated_summaries/model_4_sums_13-05_22-28-03.csv',
        '5': REPORTS_DIR / 'generated_summaries/model_5_sums_13-05_14-14-09.csv',
    }

    # Now, we also need the dataset to fetch the corresponding real summary; we can use the test split with all cases
    all_cases = pd.read_parquet(DATA_DIR / 'open_data_uitspraken/processed/test_rechtspraak.parquet')

    # First we compute the scores for each of the models
    models = ['full', '0', '1', '2', '3', '4', '5']
    rouge_scores = []
    pbar = tqdm(models)
    for model in pbar:
        pbar.set_description(f"Computing scores for model {model}")

        model_scores = compute_rouge_scores(model, file_mapping, all_cases)
        rouge_scores.append([model, model_scores])

    # Finally we update the scores with the weighted score for the cluster framework, using the invidivual scores
    rouge_scores = compute_clusters_combined(rouge_scores)

    # Print the scores
    for model_scores in rouge_scores:
        print(model_scores)


def compute_rouge_scores(model, file_mapping, all_cases):
    results_df = pd.read_csv(file_mapping[model])

    # Clean the tags that were generated with the summary
    results_df['c_summary'] = results_df['c_summary'].str.slice(7, -4)  # </s><s> en <s>

    # Compute and append the scores of each case
    case_scores = []
    for i in range(len(results_df)):
        ecli = results_df.iloc[i]['identifier']
        can_summary = results_df.iloc[i]['c_summary']
        ref_summary = all_cases.loc[all_cases['identifier'] == ecli]['summary'].item()

        # print(ref_summary)
        # print(can_summary)

        scores = rouge_scorer.get_scores(hyps=can_summary, refs=ref_summary)
        rouge_1_f = scores[0]['rouge-1']['f'] * 100
        rouge_2_f = scores[0]['rouge-2']['f'] * 100
        rouge_l_f = scores[0]['rouge-l']['f'] * 100
        # print(f'ROUGE-1-F: {rouge_1_f}, ROUGE-2-F: {rouge_2_f} and ROUGE-l-F: {rouge_l_f}')

        case_scores.append([rouge_1_f, rouge_2_f, rouge_l_f])

    avg_rouge_1 = round(sum([scores[0] for scores in case_scores]) / len(case_scores), 2)
    avg_rouge_2 = round(sum([scores[1] for scores in case_scores]) / len(case_scores), 2)
    avg_rouge_l = round(sum([scores[2] for scores in case_scores]) / len(case_scores), 2)

    # print(f'Model {model} achieves the following rouge scores: '
    #       f'avg. ROUGE-1-F: {avg_rouge_1}, avg. ROUGE-2-F: {avg_rouge_2} and avg. ROUGE-l-F: {avg_rouge_l}')

    return [avg_rouge_1, avg_rouge_2, avg_rouge_l]


def compute_clusters_combined(rouge_scores):
    """Simply outputs the ROUGE scores for the combined cluster model/framework. The dataset sizes are from the file
    generate_eval_sample."""
    ds_size_mapping = {
        '0': 23584,
        '1': 13504,
        '2': 19156,
        '3': 6669,
        '4': 15413,
        '5': 21875
    }

    total_cases = 0
    for key, val in ds_size_mapping.items():
        total_cases += val

    # sanity check
    print(f'Total cases: {total_cases}')

    combined_rouge_1 = 0
    combined_rouge_2 = 0
    combined_rouge_l = 0

    # We only loop over the 1th to nth as 0th is the full dataset
    for i in range(1, len(rouge_scores)):
        # The 0th value holds the dataset name
        ds_size = ds_size_mapping[rouge_scores[i][0]]

        # This is the weight we use to rescale the values
        cluster_weight = ds_size / total_cases

        # Weight each of the dataset scores
        weighted_rouge_1 = rouge_scores[i][1][0] * cluster_weight
        weighted_rouge_2 = rouge_scores[i][1][1] * cluster_weight
        weighted_rouge_l = rouge_scores[i][1][2] * cluster_weight

        # Add the weighted scores to total combined scores
        combined_rouge_1 += weighted_rouge_1
        combined_rouge_2 += weighted_rouge_2
        combined_rouge_l += weighted_rouge_l

    # Once all cluster scores have been weighted and combined, we can append the combined values and return the scores
    rouge_scores.append(['combined_clusters', [round(combined_rouge_1, 2),
                                               round(combined_rouge_2, 2),
                                               round(combined_rouge_l, 2)]])
    return rouge_scores


if __name__ == '__main__':
    main()
