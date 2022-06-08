import time
import argparse

import pandas as pd
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset


parser = argparse.ArgumentParser(description='Script for inferring for the test set of each of the summarization models!=')

# Choose the data to load; is it a cluster or the complete dataset?
# Choose from {'full', '0', '1', '2', '3', '4', '5'}
parser.add_argument("--dataset", 
                    choices=["full", "0", "1", "2", "3", "4", "5"],
                    required=True, type=str, help="Which dataset and model should be inferred for?")

args = parser.parse_args()

dataset = args.dataset
dataset_name = dataset

# Load the dataset, depending on whether we want to train on a cluster or the whole dataset 
if dataset_name == 'full':
    raw_dataset = load_dataset(path='parquet', data_files={'test': ['all_cases_1024/test_rechtspraak.parquet']})
else:  # that is, we have a cluster number (from 0 to 5)
    raw_dataset = load_dataset(path='parquet', data_files={'test': [f'all_cases_1024/cluster_subsets/{dataset_name}_test_rechtspraak.parquet']})

print(raw_dataset)

dataset_mapping = {
    'full': 'sum_full_bart_nl_09-05_21-47-58/best_model',
    '0': 'sum_0_bart_nl_10-05_09-26-02/best_model',
    '1': 'sum_1_bart_nl_10-05_12-23-11/best_model',
    '2': 'sum_2_bart_nl_10-05_16-45-27/best_model',
    '3': 'sum_3_bart_nl_09-05_20-52-08/best_model',
    '4': 'sum_4_bart_nl_10-05_19-26-19/best_model',
    '5': 'sum_5_bart_nl_10-05_14-06-55/best_model',
}
print(f'{dataset_mapping[dataset_name]} model will be loaded.')
model = BartForConditionalGeneration.from_pretrained(dataset_mapping[dataset_name])  # /checkpoints/checkpoint-10000") #8000/9000
tokenizer = BartTokenizerFast.from_pretrained("bart_nl_tiny_tk")

# print(model.config.min_length)
# print(model.config.max_length) # This is 20, which is strange and hard to believe. Also max_length is not printed as param with model.config 

output_summaries = []
for i in tqdm(range(len(raw_dataset['test'])), desc="Summarizing the test set's cases"): 
    model_inputs = tokenizer(raw_dataset['test'][i]['description'], max_length=1024, return_tensors='pt', truncation=True) # padding=max_length
    #print(model_inputs)
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

df = pd.DataFrame(output_summaries ,columns=['identifier', 'c_summary'])

# Create file name to store 
date_and_time = time.strftime("%d-%m_%H-%M-%S", time.localtime())

save_dir = f'inf_results/model_{dataset_name}_sums_{date_and_time}.csv'

df.to_csv(save_dir)