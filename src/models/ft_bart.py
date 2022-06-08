import time 

import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from datasets import load_metric

from transformers import BartModel, BartConfig


def main():
    """ See https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py
    for the followed template. We use https://huggingface.co/docs/datasets/loading_datasets.html# to load our own
    dataset. We only consider cases from 2021.
    """
    # Choose the data to load; is it a cluster or the complete dataset?
    # Choose from {'full', '0', '1', '2', '3', '4', '5'}
    dataset_name = '0'    
    
    # Create folder names to store logs and checkpoints (used in the training config for example)
    date_and_time = time.strftime("%d-%m_%H-%M-%S", time.localtime())

    checkpoint_dir = f'./sum_{dataset_name}_bart_nl_{date_and_time}/checkpoints'
    logging_dir = f'./sum_{dataset_name}_bart_nl_{date_and_time}/logs'
    best_model_dir = f'./sum_{dataset_name}_bart_nl_{date_and_time}/best_model'

    # Load the dataset, depending on whether we want to train on a cluster or the whole dataset 
    if dataset_name == 'full':
        train_file_name = 'all_cases_1024/train_rechtspraak.parquet'
        val_file_name = 'all_cases_1024/val_rechtspraak.parquet'
    else:  # that is, we have a cluster number (from 0 to 5)
        train_file_name = f'all_cases_1024/cluster_subsets/{dataset_name}_train_rechtspraak.parquet'
        val_file_name = f'all_cases_1024/cluster_subsets/{dataset_name}_val_rechtspraak.parquet'

    raw_dataset = load_dataset(path='parquet',
                               data_files={'train': [train_file_name],
                                           'val': [val_file_name]})

    print(raw_dataset)

    # Dit is het laatste checkpoint dat opgeslagen is voor de crash
    model = BartForConditionalGeneration.from_pretrained("bart_pt/pt_22-04_16-39-16/checkpoints/checkpoint-475000") 
    tokenizer = BartTokenizerFast.from_pretrained("bart_nl_tiny_tk")

    # Inspect the config of the model if necessary
    #print(model.config)

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["description"], max_length=1024, padding="max_length", truncation=True)
        labels = tokenizer(examples["summary"], max_length=256, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # Tokenize the dataset 
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

    # # This configuration evaluates and logs depending on the number of steps
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir=checkpoint_dir,
    #     do_train=True,
    #     do_eval=True,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=8,
    #     evaluation_strategy="steps",
    #     eval_steps=1000,
    #     learning_rate=2e-5,
    #     num_train_epochs=10,#3,
    #     save_total_limit=3,
    #     save_steps=1000,
    #     logging_dir=logging_dir, # Tensorboard needs to be installed! pip install tensorboard
    #     logging_first_step=True,
    #     logging_steps=250, 
    #     load_best_model_at_end=True,
    #     metric_for_best_model='eval_loss',
    #     greater_is_better=False    
    # )

    # This is the same config as above, but with epoch-dependent evaluation, logging, etc
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        #eval_steps=1000,
        learning_rate=2e-5,
        num_train_epochs=10,  # 10, 43 is temp om te kijken wat er gebeurt als ik het model even lang laat traine als full model (in stappen)
        save_strategy="epoch",
        save_total_limit=3,
        #save_steps=1000,
        logging_strategy="epoch",
        logging_dir=logging_dir, # Tensorboard needs to be installed! pip install tensorboard
        logging_first_step=True,
        #logging_steps=250, 
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False    
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset["val"],
        # compute_metrics=compute_rouge  # Right now, not only is very slow, but also causes a cpu memory alloc. error 
    )

    trainer.train()

    # Save the best model that was loaded at the end (look at load_best_model_at_end) 
    trainer.save_model(best_model_dir)


if __name__ == '__main__':
    main()