from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import numpy as np
from datasets import load_metric
import torch

def main():
    """ This is a simple fine-tuning script.See https://huggingface.co/docs/transformers/training for the
    followed tutorial.
    """
    raw_dataset = load_dataset("imdb")
    # print(raw_datasets['train'][0])

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    # print(tokenized_dataset['train'][0])

    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(100))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(100))
    # full_train_dataset = tokenized_dataset["train"]
    # full_eval_dataset = tokenized_dataset["test"]

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    print(torch.cuda.is_available())
    training_args = TrainingArguments("test_trainer", num_train_epochs=3, evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    #trainer.evaluate()


if __name__ == '__main__':
    main()
