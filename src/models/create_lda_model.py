import time
from pprint import pprint
import sys

import pandas as pd
import spacy
import gensim
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath
import gensim.corpora as corpora
import nltk
from nltk.corpus import stopwords  # nltk.download('stopwords')

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
    all_cases = load_dataset(DATA_DIR / 'open_data_uitspraken/interim')

    # Delete the summary column as we don't need it
    all_cases = all_cases.drop(columns=['summary'])
    print(f'Size of dataset as df: {round(sys.getsizeof(all_cases) / 1024, 2)} kb')
    # Convert the pandas df to a list of dicts for more efficient processing
    cases_dict_list = all_cases.to_dict('records')

    # Temp; testing with batches
    # print(len(cases_dict_list))
    cases_dict_list = cases_dict_list#[:10000]
    # print(len(cases_dict_list))

    del all_cases

    # 1. Remove all |'s that were added during data collection
    remove_pipes(cases_dict_list)

    # Create LDA model and save it in reports
    print(f'starting LDA using {len(cases_dict_list)} cases.')
    create_lda_model(cases_dict_list, save_path=MODELS_DIR / 'lda_full/lda_model')


def remove_pipes(list_of_dicts):
    """The prior-added pipes (|) will be filtered out."""
    start = time.time()
    for i in range(len(list_of_dicts)):
        list_of_dicts[i]['description'] = list_of_dicts[i]['description'].replace("|", " ")

    print(f'Time taken to remove pipes from dataset: {round(time.time() - start, 2)} seconds')


def tokenize_text(doc, rm_stop_words=False):
    """Expects a spacy doc. Returns list containing the lowercased non-punctuation tokens of more than 1 non-whitespace
    char, excluding stop words."""
    if rm_stop_words:
        tokenized_text = [token.lower_ for token in doc if len(token.text.strip()) > 1
                          and not token.is_stop
                          and not token.is_punct]
    else:
        tokenized_text = [token.lower_ for token in doc if len(token.text.strip()) > 1 and not token.is_punct]

    return tokenized_text


def create_lda_model(dataset, save_path):
    """Applies LDA as described in Blei (2003), Latent dirichlet allocation. Used https://towardsdatascience.com/end
    -to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0 as a guide. We follow
    Bommasani & Cardie (2020), Intrinsic Evaluation of Summarization Datasets, in only using the source texts to
    train the LDA model, and not the summaries."""
    start = time.time()

    #
    # Batch implementation
    #
    batch_size = round(len(dataset)/10)
    print(f'Total cases: {len(dataset)}, batch size: {batch_size}')

    # First create the base model, with batch 1. We use a dictionary created using only batch one
    base_corpus = []
    for case in dataset:
        base_corpus.append(tokenize_text(nlp(case['description']), rm_stop_words=True))

    corpus_dictionary = corpora.Dictionary(base_corpus)

    for obj in locals():
        print(f'{obj} has size: {round(sys.getsizeof(obj) / 1024, 2)} kb')

   # base_tfidf = [corpus_dictionary.doc2bow(text) for text in base_corpus]
   # print(base_corpus[:2])
   # num_topics = 5
   # lda_model = gensim.models.LdaModel(corpus=base_corpus,
     #                                  id2word=corpus_dictionary,
    #                                   num_topics=num_topics)

    #
    #
    #

    # # Preprocess + tokenize the texts + add them to the corpus, using a generator expression
    # start_prep = time.time()
    # corpus = [tokenize_text(nlp(case['description']), rm_stop_words=True) for case in dataset]
    #
    # print(f'Total time taken to preprocess the cases '
    #       f'and combining them into a corpus: {round(time.time() - start_prep, 2)} seconds')
    # del dataset
    #
    # # Create a dictionary of all the words contained in the corpus
    # corpus_dictionary = corpora.Dictionary(corpus)
    #
    # # Derive the Term Document Frecuencies for each text (again using generator expression)
    # start_tfidf = time.time()
    # corpus_tdf = [corpus_dictionary.doc2bow(text) for text in corpus]
    # print(f'Total time taken to compute tf-idf for all cases: {round(time.time() - start_tfidf, 2)} seconds')
    # del corpus
    #
    # # Learn the LDA model, we use 10 batches
    # # Create the base model with the first 10% of the cases
    # #corrrir = corpus_tdf
    # #total_cases = sum(1 for x in corrrir)
    # #print(f'total cases: {total_cases}, len of base batch: {round(total_cases/10)}')
    # #base_split =
    # #for i in range(len(10-1)):
    #
    # num_topics = 5
    # print('creating LDA model')
    # lda_model = gensim.models.LdaModel(corpus=corpus_tdf,
    #                                    id2word=corpus_dictionary,
    #                                    num_topics=num_topics)
    #
    # Print the Keyword in the 10 topics
    #pprint(lda_model.print_topics())
    #
    # # Save model, and the used dictionary (!), to disk. Note; gensim does not recognize path objects
    #lda_model.save(str(save_path))
    # corpus_dictionary.save_as_text(str(save_path))

    print(f'Total time taken to perform LDA on dataset: {round(time.time() - start, 2)} seconds')


if __name__ == '__main__':
    main()
