import time 

from transformers import BartTokenizerFast
from transformers import BartModel, BartForCausalLM, BartForConditionalGeneration, BartConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
from torch.utils.data import random_split


def main():
    # See https://huggingface.co/datasets/yhavinga/mc4_nl_cleaned for details on the dataset
    # Inladen kan evt. ook met streaming=true, maar dit is alleen aan te raden wanneer opslag 
    # van de dataset geen optie is. De dataset hoeft niet volledig in ram te passen
    dataset = load_dataset('yhavinga/mc4_nl_cleaned', 'tiny') #'tiny') 

    config = BartConfig.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration(config)  # BartModel(config)
    tokenizer = BartTokenizerFast.from_pretrained("./bart_nl_tiny_tk") # , max_len=256, additional_special_tokens=['[CH]', '[OTHER]', '[VAR]', '[NUM]'])

    # print(model.config)

    def preprocess_function(examples):
            model_inputs = tokenizer(examples["text"], max_length=1024, padding="max_length", truncation=True)

            return model_inputs

    # Tokenize the dataset 
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # print(tokenized_dataset)

    date_and_time = time.strftime("%d-%m_%H-%M-%S", time.localtime())

    # Note; tensorboard must be installed in order for logging to happen!
    output_dir = f'./bart_pt/pt_{date_and_time}/checkpoints'
    logging_dir = f'./bart_pt/pt_{date_and_time}/logs'

    # defining training related arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="steps",
        # één evaluatie voor tiny duurt ~12 min 
        # Voor 786k iteraties betekent dat 60 extra uur aan evaluatie als eval_steps=2500
        eval_steps=25000,
        learning_rate=5e-5,
        num_train_epochs=1,
        save_total_limit=3,
        save_steps=25000, # 2000
		# Tensorboard needs to be installed, otherwise no training logs are saved! pip install tensorboard
        logging_dir=logging_dir,
        logging_first_step=True,
        logging_steps=2500, # 2500
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False    
    )

    # The data collator is responsible for masking the inputs; ie creating the training objectives
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

    # defining trainer using 🤗
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
    output_dir = f'./bart_pt/pt_{date_and_time}/best_model'
    trainer.save_model(output_dir)


if __name__ == '__main__':
    main()