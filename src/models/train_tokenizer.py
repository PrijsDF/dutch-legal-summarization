from transformers import AutoTokenizer
from transformers import BartTokenizerFast
from datasets import load_dataset


dataset = load_dataset('yhavinga/mc4_nl_cleaned', 'tiny', streaming=True)

# We create an iterator from the generator above (?)
def dataset_iterator(dataset):
    for i in dataset:
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
bart_nl_tokenizer.save_pretrained("bart_nl_tiny_tk")