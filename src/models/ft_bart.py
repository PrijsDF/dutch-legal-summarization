import time 

from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

from src.utils import DATA_DIR, MODELS_DIR


def main():
    """ We use https://huggingface.co/docs/datasets/loading_datasets.html# to load our own dataset. Choose for which
    data you want to fine-tune the BART model by changing the 'dataset_name' var. The right dataset will be loaded
    automatically. Fine-tuned models are stored in the models/ dir
    """
    # Choose the data to load; is it a cluster or the complete dataset?
    # Choose from {'full', '0', '1', '2', '3', '4', '5'}
    dataset_name = 'full'
    
    # Create folder names to store logs and checkpoints (used in the training config for example)
    date_and_time = time.strftime("%d-%m_%H-%M-%S", time.localtime())

    checkpoint_dir = MODELS_DIR / f"sum_{dataset_name}_bart_nl_{date_and_time}/checkpoints"
    logging_dir = MODELS_DIR / f"sum_{dataset_name}_bart_nl_{date_and_time}/logs"
    best_model_dir = MODELS_DIR / f"sum_{dataset_name}_bart_nl_{date_and_time}/best_model"

    # Load the dataset, depending on whether we want to train on a cluster or the whole dataset 
    if dataset_name == 'full':
        train_file_name = DATA_DIR / 'processed/train_rechtspraak.parquet'
        val_file_name = DATA_DIR / 'processed/val_rechtspraak.parquet'
    else:  # that is, we have a cluster number (from 0 to 5)
        train_file_name = DATA_DIR / f'processed/cluster_subsets/{dataset_name}_train_rechtspraak.parquet'
        val_file_name = DATA_DIR / f'processed/cluster_subsets/{dataset_name}_val_rechtspraak.parquet'

    raw_dataset = load_dataset(path='parquet',
                               data_files={'train': [train_file_name],
                                           'val': [val_file_name]})

    # print(raw_dataset)

    # Dit is het laatste checkpoint dat opgeslagen is voor de crash
    # Note: edit the model path to match your own model
    model = BartForConditionalGeneration.from_pretrained(
        MODELS_DIR / f"bart_pt/pt_22-04_16-39-16/checkpoints/checkpoint-475000"
    )
    tokenizer = BartTokenizerFast.from_pretrained(MODELS_DIR / f"bart_nl_tiny_tk")

    # Inspect the config of the model if necessary
    # print(model.config)

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["description"], max_length=1024, padding="max_length", truncation=True)
        labels = tokenizer(examples["summary"], max_length=256, padding="max_length", truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # Tokenize the dataset 
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)

    # Configure the training process
    log_interval = 'epoch'  # Make 'steps' if every n steps eval/saving/logging need to happen (e.g. logging_steps)
    # This is the same config as above, but with epoch-dependent evaluation, logging, etc
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy=log_interval,
        eval_steps=1000,
        learning_rate=2e-5,
        num_train_epochs=10,  # optionally, use 43 to train the 'Cluster 0 model' for the same number of iterations
        save_strategy=log_interval,
        save_total_limit=3,
        save_steps=1000,
        logging_strategy=log_interval,
        logging_dir=logging_dir,  # Tensorboard needs to be installed! pip install tensorboard
        logging_first_step=True,
        logging_steps=250,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False    
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset["val"]
    )

    # Start fine-tuning
    trainer.train()

    # Save the best model that was loaded at the end (look at load_best_model_at_end) 
    trainer.save_model(best_model_dir)


if __name__ == '__main__':
    main()
