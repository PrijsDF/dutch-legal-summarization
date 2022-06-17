import time
import argparse

import pandas as pd
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizerFast
from datasets import load_dataset

from src.utils import REPORTS_DIR, DATA_DIR, MODELS_DIR


def main():
    """Use this file to generate summaries for the test set, using the fine-tuned models from ft_bart.py. Choose what
    model to infer for by adding the models as an argument when calling the file. You will have to change the
    paths in the model_mapping dictionary to correspond with your own models."""
    parser = argparse.ArgumentParser(description='Script for generating summaries for the test sets')

    # Choose the data to load; is it a cluster dataset or the complete dataset?
    # Choose from {'full', '0', '1', '2', '3', '4', '5'}
    parser.add_argument("--dataset",
                        choices=["full", "0", "1", "2", "3", "4", "5"],
                        required=True, type=str, help="Which dataset and model should be inferred for?")

    args = parser.parse_args()

    dataset = args.dataset
    dataset_name = dataset

    # Load the dataset, depending on whether we want to train on a cluster or the whole dataset
    if dataset_name == 'full':
        raw_dataset = load_dataset(
            path='parquet', data_files={'test': [DATA_DIR / 'test_rechtspraak.parquet']}
        )
    else:  # that is, we have a cluster number (from 0 to 5)
        raw_dataset = load_dataset(
            path='parquet', data_files={'test': [DATA_DIR / f'cluster_subsets/{dataset_name}_test_rechtspraak.parquet']}
        )

    print(raw_dataset)

    # Change these to your own model paths
    model_mapping = {
        'full': 'sum_full_bart_nl_09-05_21-47-58/best_model',
        '0': 'sum_0_bart_nl_10-05_09-26-02/best_model',
        '1': 'sum_1_bart_nl_10-05_12-23-11/best_model',
        '2': 'sum_2_bart_nl_10-05_16-45-27/best_model',
        '3': 'sum_3_bart_nl_09-05_20-52-08/best_model',
        '4': 'sum_4_bart_nl_10-05_19-26-19/best_model',
        '5': 'sum_5_bart_nl_10-05_14-06-55/best_model',
    }
    print(f'{model_mapping[dataset_name]} model will be loaded.')
    model = BartForConditionalGeneration.from_pretrained(MODELS_DIR / f"{model_mapping[dataset_name]}")
    tokenizer = BartTokenizerFast.from_pretrained(MODELS_DIR / f"bart_nl_tiny_tk")

    # print(model.config.min_length)
    # print(model.config.max_length)

    output_summaries = []
    for i in tqdm(range(len(raw_dataset['test'])), desc="Summarizing the test set's cases"):
        model_inputs = tokenizer(
            raw_dataset['test'][i]['description'], max_length=1024, return_tensors='pt', truncation=True
        )
        # print(model_inputs)
        model_outputs = model.generate(
            inputs=model_inputs["input_ids"],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        candidate_summary = tokenizer.decode(model_outputs[0])
        output_summaries.append([raw_dataset['test'][i]['identifier'], candidate_summary])

    df = pd.DataFrame(output_summaries, columns=['identifier', 'c_summary'])

    # Create file name to store
    date_and_time = time.strftime("%d-%m_%H-%M-%S", time.localtime())

    save_dir = REPORTS_DIR / f'inf_results/model_{dataset_name}_sums_{date_and_time}.csv'

    df.to_csv(save_dir)


if __name__ == '__main__':
    main()
