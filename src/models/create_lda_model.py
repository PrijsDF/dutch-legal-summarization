import time
import re
from pprint import pprint

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

    # Convert the pandas df to a list of dicts for more efficient processing
    cases_dict_list = all_cases.to_dict('records')

    # 1. Remove all |'s that were added during data collection
    cases_dict_list = remove_pipes(cases_dict_list)

    # Create LDA model and save it in reports
    create_lda_model(cases_dict_list)


def remove_pipes(list_of_dicts):
    """The prior-added pipes (|) will be filtered out."""
    for i in range(len(list_of_dicts)):
        list_of_dicts[i]['summary'] = list_of_dicts[i]['summary'].replace("|", " ")
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

    return tokenized_text


def create_lda_model(dataset):
    """Applies LDA as described in Blei (2003), Latent dirichlet allocation. Used https://towardsdatascience.com/end
    -to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0 as a guide. We follow
    Bommasani & Cardie (2020), Intrinsic Evaluation of Summarization Datasets, in only using the case descriptions to
    train the LDA model."""
    start = time.time()

    # Loop over the cases and preprocess + tokenize the texts + add them to the corpus
    corpus = []
    for i in range(len(dataset)):
        # Create the spacy documents; these can be used to tokenize and sentencize the text
        summary_doc = nlp(dataset[i]['summary'])
        text_doc = nlp(dataset[i]['description'])

        # We add both the texts and the summaries together in 1 corpus
        corpus.append(tokenize_text(summary_doc, rm_stop_words=True))
        corpus.append(tokenize_text(text_doc, rm_stop_words=True))

    # Create a dictionary of all the words contained in the corpus
    corpus_dictionary = corpora.Dictionary(corpus)
    # print(len(corpus_dictionary))

    # Derive the Term Document Frecuencies for each text
    corpus_tdf = [corpus_dictionary.doc2bow(text) for text in corpus]

    # Learn the LDA model
    num_topics = 5
    lda_model = gensim.models.LdaMulticore(corpus=corpus_tdf,
                                           id2word=corpus_dictionary,
                                           num_topics=num_topics)

    # Print the Keyword in the 10 topics
    # pprint(lda_model.print_topics())

    # Save model, and the used dictionary (!), to disk.
    lda_model.save(str(MODELS_DIR / 'lda_model'))
    corpus_dictionary.save_as_text(str(MODELS_DIR / 'lda_dictionary'))

    print(f'Time taken to perform LDA on dataset: {round(time.time() - start, 2)} seconds')


if __name__ == '__main__':
    main()
