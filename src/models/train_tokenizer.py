from transformers import BartTokenizerFast
from datasets import load_dataset

from src.utils import DATA_DIR, MODELS_DIR


def main():
    """We train a Dutch BART tokenizer using the mc4_nl tiny dataset. This tokenizer is used to pretrain and fine-tune
    the models later on."""
    dataset = load_dataset('yhavinga/mc4_nl_cleaned', 'tiny', streaming=True)

    # We create an iterator from the generator above
    def dataset_iterator(ds):
        for i in ds:
            yield i["text"]

    dataset_iter = dataset_iterator(dataset['train'])

    # Load a pretrained tokenizer; we will overwrite it later
    bart_large_tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-large')

    # Train from scratch using our own data
    bart_nl_tokenizer = bart_large_tokenizer.train_new_from_iterator(
        dataset_iter,
        vocab_size=25000
    )

    # Save the group of files that constitute the tokenizer
    bart_nl_tokenizer.save_pretrained(MODELS_DIR / f'bart_nl_tiny_tk')


if __name__ == '__main__':
    main()
