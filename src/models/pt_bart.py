import time 

from transformers import BartTokenizerFast
from transformers import BartForConditionalGeneration, BartConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

from src.utils import DATA_DIR, MODELS_DIR


def main():
    """We pretrain a Dutch BART model using the datasets and transformers libraries. We first load a relatively small
    split of the the mc4_nl dataset. Then we load our customly trained tokenizer (see train_tokenizer.py) and we
    initialize the BART model using the config that was also used for bart-base by Facebook. Training is automatically
    logged using tensorboard; make sure that it is installed, otherwise training will still start but without any logs.
    """
    # See https://huggingface.co/datasets/yhavinga/mc4_nl_cleaned for details on the dataset
    dataset = load_dataset('yhavinga/mc4_nl_cleaned', 'tiny')

    config = BartConfig.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration(config)  # BartModel(config)
    tokenizer = BartTokenizerFast.from_pretrained(MODELS_DIR / f"bart_nl_tiny_tk")

    # print(model.config)

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["text"], max_length=1024, padding="max_length", truncation=True)

        return model_inputs

    # Tokenize the dataset 
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # print(tokenized_dataset)

    date_and_time = time.strftime("%d-%m_%H-%M-%S", time.localtime())

    # Note; tensorboard must be installed in order for logging to happen!
    output_dir = MODELS_DIR / f'bart_pt/pt_{date_and_time}/checkpoints'
    logging_dir = MODELS_DIR / f'bart_pt/pt_{date_and_time}/logs'

    # defining training related arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        # One eval iteration takes ~12 minutes for the tiny split. For 786k train iterations, this means an 60 hours
        # of evaluation are needed if eval_steps=2500 like logging_steps
        eval_steps=25000,
        learning_rate=5e-5,
        num_train_epochs=1,
        save_total_limit=3,
        save_steps=25000,  # 2000
        # Tensorboard needs to be installed, otherwise no training logs are saved! pip install tensorboard
        logging_dir=logging_dir,
        logging_first_step=True,
        logging_steps=2500,  # 2500
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False    
    )

    # The data collator is responsible for masking the inputs; ie creating the training objectives
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation']
    )

    # Start training 
    trainer.train()

    # Save the best model that was loaded at the end (look at load_best_model_at_end) 
    output_dir = MODELS_DIR / f'bart_pt/pt_{date_and_time}/best_model'
    trainer.save_model(output_dir)


if __name__ == '__main__':
    main()
