import time
from pprint import pprint
import sys

import pandas as pd
import spacy
import gensim
import gensim.corpora as corpora

from src.utils import DATA_DIR, MODELS_DIR, load_dataset

# Some pandas options that allow to view all collumns and rows at once
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 400)
pd.options.display.width = None
# This will prevent a warning from happening during the interpunction removal in the LDA function
pd.options.mode.chained_assignment = None

# Create the spacy nlp object; we disable those features that are not needed
# Download, if needed, using python -m spacy download nl_core_news_sm
nlp = spacy.load("nl_core_news_sm", exclude=[
    'tok2vec',
    'morphologizer',
    'tagger',
    'parser',
    'attribute_ruler',
    'lemmatizer',
    'ner'])
nlp.add_pipe('sentencizer')  # Because we removed the parser, we need to manualy add sentencizer back
nlp.max_length = 1500000  # Otherwise the limit will be 1000000 which is too little


def main():
    """View Open Rechtspraak dataset with pandas."""
    # Load the raw dataset
    all_cases = load_dataset(DATA_DIR / 'open_data_uitspraken/interim', use_dask=True)
    # print(all_cases)

    # Drop the unneeded columns
    all_cases = all_cases.drop(columns=['identifier', 'summary'])

    create_lda_model_dask(all_cases, save_path=MODELS_DIR / 'lda_full_temp')

    # # Delete the summary column as we don't need it
    # all_cases = all_cases.drop(columns=['summary'])
    # print(f'Size of dataset as df: {round(sys.getsizeof(all_cases) / 1024, 2)} kb')
    # # Convert the pandas df to a list of dicts for more efficient processing
    # cases_dict_list = all_cases.to_dict('records')
    #
    # # Temp; testing with batches
    # # print(len(cases_dict_list))
    # cases_dict_list = cases_dict_list
    # # print(len(cases_dict_list))
    #
    # del all_cases
    #
    # # 1. Remove all |'s that were added during data collection
    # remove_pipes(cases_dict_list)
    #
    # # Create LDA model and save it in reports
    # print(f'starting LDA using {len(cases_dict_list)} cases.')
    # create_lda_model(cases_dict_list, save_path=MODELS_DIR / 'lda_full/lda_model')


def remove_pipes(list_of_dicts):
    """The prior-added pipes (|) will be filtered out."""
    for i in range(len(list_of_dicts)):
        list_of_dicts[i]['description'] = list_of_dicts[i]['description'].replace("|", " ")

    return list_of_dicts


def tokenize_text(doc, rm_stop_words=False):
    """Expects a spacy doc. Returns list containing the lowercased non-punctuation tokens of more than 1 non-whitespace
    char, excluding stop words."""
    if rm_stop_words:
        tokenized_text = [token.lower_ for token in doc if len(token.text.strip()) > 1
                          and not token.is_stop
                          and not token.is_punct]
    else:
        tokenized_text = [token.lower_ for token in doc if len(token.text.strip()) > 1 and not token.is_punct]

    # Experimental; we put a bound on the text length as big length might cause memory errors
    # if len(tokenized_text) > 6000:
    #     tokenized_text = tokenized_text[:6000]

    return tokenized_text


def create_lda_model_dask(dataset, save_path):
    """Applies LDA as described in Blei (2003), Latent dirichlet allocation. Used https://towardsdatascience.com/end
    -to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0 as a guide. We follow
    Bommasani & Cardie (2020), Intrinsic Evaluation of Summarization Datasets, in only using the source texts to
    train the LDA model, and not the summaries."""
    start = time.time()

    # We use the first partition of the dask dataset to create the base LDA model
    # The rest of the partitions are used to train the model in batches (otherwise too large for memory)
    base_size = 20
    base_partitions = dataset.partitions[-base_size:]
    batch_partitions = dataset.partitions[:-base_size]

    print(f'Starting processing {base_partitions.npartitions} base partitions, '
          f'and {batch_partitions.npartitions} batch partitions')

    # We use the first partition of the dask dataset to create the base LDA model
    # Preprocess + tokenize the texts + add them to the corpus, using a generator expression
    base_cases = base_partitions.compute().to_dict('records')

    # Remove pipes from the case texts
    processed_texts = remove_pipes(base_cases)

    # Remove the pipes from the texts and tokenize the texts in the partition
    tokenized_texts = [tokenize_text(nlp(text['description']), rm_stop_words=True) for text in processed_texts]

    # Create a dictionary of all the words contained in the corpus, this dictionary will also be used in later batches
    corpus_dictionary = corpora.Dictionary(tokenized_texts)

    # Derive the Term Document Frecuencies for each text (again using generator expression)
    corpus_tdf = [corpus_dictionary.doc2bow(text) for text in tokenized_texts]

    # Train the base model
    num_topics = 5
    lda_model = gensim.models.LdaModel(corpus=corpus_tdf,
                                       id2word=corpus_dictionary,
                                       num_topics=num_topics)

    print(f'Trained the base model on the first partition ({len(corpus_tdf)} cases). '
          f'Batch training the other cases starts now.')
    del base_cases, processed_texts, tokenized_texts, corpus_tdf

    # Now we will iterate over the other partitions and incrementally extend the LDA model
    batches = batch_partitions.npartitions
    for i in range(2):#batches):
        # We follow the same steps as for the base partition above
        batch_cases = batch_partitions.partitions[i].compute().to_dict('records')
        processed_texts = remove_pipes(batch_cases)
        tokenized_texts = [tokenize_text(nlp(text['description']), rm_stop_words=True) for text in processed_texts]
        batch_tdf = [corpus_dictionary.doc2bow(text) for text in tokenized_texts]

        # Finally, update the base-trained LDA (using the base-obtained dictionary)
        lda_model.update(batch_tdf)

        print(f'finished batch {i+1}/{batches}')

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())

    # Save model, and the used dictionary (!), to disk. Note; gensim does not recognize path objects
    lda_model.save(str(save_path / 'lda_model'))
    corpus_dictionary.save_as_text(str(save_path / 'lda_dictionary'))

    print(f'Total time taken to perform LDA on dataset: {round(time.time() - start, 2)} seconds')


if __name__ == '__main__':
    main()
