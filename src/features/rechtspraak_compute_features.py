import time
import re
from pprint import pprint

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import spacy
import gensim
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import nltk
from nltk.corpus import stopwords  # nltk.download('stopwords')
from rouge import Rouge

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
    """Compute all features of the dataset following Bommasani and Cardie (2020). I will use these to compare the
    dataset to existing benchmarks for summarization.

    https://stackoverflow.com/a/56746204; herschrijven om list te gebruiken en niet pandas append."""
    # Load the interim dataset
    all_cases = load_dataset(DATA_DIR / 'open_data_uitspraken/interim')

    # 1. Create df with placeholders for the features (4 simple features and 6 complex features
    features_df = create_features_df()

    # 1.b. preprocess the data (for now only removes the pipes "|")
    processed_cases = preproc_data(all_cases)

    # 2. Compute the simple features + word_compression and sentence_compression
    features_df = compute_simple_features(features_df, processed_cases)
    # print(features_df)

    # 5. Compute topic_similarity
    compute_topic_similarity(all_cases[:40])

    # 6. Compute abstractivity
    # compute_fragments(dataset_dir)

    # 7. Compute redundancy

    # 8. Compute semantic coherence

    # 9. Average the computed features for the whole dataset
    # agg_cases_features = create_agg_df(REPORTS_DIR / 'dataset_metrics.csv')
    # print(agg_cases_features)


def create_features_df():
    """This df will hold all the features that together characterize the dataset."""
    features_df = pd.DataFrame(columns=[
        'identifier',
        'sum_words',
        'sum_sents',
        'desc_words',
        'desc_sents',
        'cmp_words',
        'cmp_sents',
        'topic_similarity',
        'abstractivity',
        'redundancy',
        'semantic_coherence'
    ])

    return features_df


def preproc_data(df):
    """Most interpunction will be removed, words are lowercased and the prior-added pipes (|) will be filtered out."""
    # First remove interpunction etc.; do this with spacy later
    # df['description'] = df['description'].map(lambda x: re.sub('[,/\\[\\].!?;:(){}]', '', x))
    # df['description'] = df['description'].map(lambda x: x.lower())
    df['description'] = df['description'].map(lambda x: x.replace("|", " "))

    return df


def tokenize_list(text_list):
    """Expects a list of descriptions. Gensim is used to return a list that contains a list of tokense for each
     text."""
    return [gensim.utils.simple_preprocess(str(text), deacc=True) for text in text_list]


def tokenize_text(doc):
    """Expects a spacy doc. Returns list containing the tokens of more than 1 non-whitespace char."""
    return [token for token in doc if len(token.text.strip()) > 1]


def sentencize_text(doc):
    """Expects a spacy doc. Returns list containing the sentences."""
    return [sent for sent in doc.sents]


def compute_simple_features(features_df, cases):
    """Compute the simple features and two complex features, namely the compressions. Currently, stop words included.
    The function will return the features_df after filling in the features for each case."""
    # texts = cases['description'].values.tolist()
    # summaries = cases['summary'].values.tolist()

    # tokenized_texts = tokenize_list(texts)
    # tokenized_summaries = tokenize_list(summaries)

    start = time.time()
    for i in range(len(cases[:4])):
        if (i + 1) % 500 == 0:
            print(f'{len(cases) - (i + 1)} cases left. '
                  f'Time elapsed since start: {round(time.time() - start, 2)} seconds')

        identifier = cases['identifier'].values[i]
        summary = cases['summary'].values[i]
        text = cases['description'].values[i]

        # Tokenize the text and summary
        # text_tokens = tokenize_text(text)
        # summary_tokens = tokenize_text(summary)

        # Create the spacy documents; these can be used to tokenize and sentencize the text
        summary_doc = nlp(summary)
        text_doc = nlp(text)

        text_word_length = len(tokenize_text(text_doc))
        text_sent_length = len(sentencize_text(text_doc))

        summary_word_length = len(tokenize_text(summary_doc))
        summary_sent_length = len(sentencize_text(summary_doc))

        # Compute the compression scores
        if summary_word_length > 0 and text_word_length > 0:
            cmp_words = 1 - summary_word_length / text_word_length
        else:
            cmp_words = 999

        if summary_sent_length > 0 and text_sent_length > 0:
            cmp_sents = 1 - summary_sent_length / text_sent_length
        else:
            cmp_sents = 999

        # Scores for current case; we add the other ones later (make them 999 for now)
        case_features = {
            'identifier': identifier,
            'sum_words': summary_word_length,
            'sum_sents': summary_sent_length,
            'desc_words': text_word_length,
            'desc_sents': text_sent_length,
            'cmp_words': cmp_words,
            'cmp_sents': cmp_sents,
            'topic_similarity': 999,
            'abstractivity': 999,
            'redundancy': 999,
            'semantic_coherence': 999,
        }

        # Append row to the dataframe
        features_df = features_df.append(case_features, ignore_index=True)

    print(f'Total time taken to compute metrics of dataset: {round(time.time() - start, 2)} seconds')
    return features_df


def compute_topic_similarity(dataset):
    """First, we load the lda model that we generated in models/train_lda_model.py. Then, we use this model to
    give the topic distribution for both a case and its summary. Finally, these distributions are compared using
    the Jensen-Shannon distance, implemented using scipy.

    Importantly, the same preprocessing steps and stop word filters need to be applied here as to when the lda model
    was trained"""
    start = time.time()

    # Temp. Take a subset of data
    # dataset = dataset[:100]

    # First remove interpunction etc.; do this with spacy later
    dataset['proc_description'] = dataset['description'].map(lambda x: re.sub('[,/\\[\\].!?;:(){}]', '', x))
    dataset['proc_description'] = dataset['proc_description'].map(lambda x: x.lower())
    dataset['proc_description'] = dataset['proc_description'].map(lambda x: x.replace("|", " "))

    # Stop words to be removed
    stop_words = stopwords.words('dutch')
    stop_words.extend(['ter', 'ten'])

    # Tokenization using simple_preprocess()
    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)

    # remove stop words
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    data = dataset['proc_description'].values.tolist()
    # temp
    # print(len(data))

    data_words = remove_stopwords(list(sent_to_words(data)))
    # print(data_words[:1][0][:30])

    # Create Dictionary
    id2word = corpora.Dictionary(data_words)

    # Create Corpus
    texts = data_words

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # print(corpus[:1][0][:30])

    # Load LDA model
    lda_model = LdaModel.load(str(MODELS_DIR / 'lda'))

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())

    print(f'Time taken to perform LDA on dataset: {round(time.time() - start, 2)} seconds')


def compute_fragments():
    """See the paper by Grusky et al. (2018) for a textual description of the algorithm"""

    summ = "Table 1 shows the aggregate statistics about our corpus. The whole list of texts consists of more than " \
           "27 million words, of which 320302 are distinct."
    text = "Table 1 shows the basic statistics about our corpus. We obtained more than 3 million segments from 30554 " \
           "documents, and we have 362811 segments in the gist sections. Overall, 11.54% of the segments were " \
           "chosen as gist statements. The whole corpus contains more than 27 million words, among which 320302 are " \
           "distinct."

    s_tokens = summ.replace(".", "").replace(",", "").split()
    a_tokens = text.replace(".", "").replace(",", "").split()

    s_len = len(s_tokens)
    a_len = len(a_tokens)

    fragments = []
    i = 0
    j = 0

    # See the paper of Grusky et al. (2018) for a textual description
    while i < s_len:
        fragment = []

        while j < a_len:
            if s_tokens[i] == a_tokens[j]:
                i_spar = i
                j_spar = j
                # print(s_tokens[i], a_tokens[j])

                while s_tokens[i_spar] == a_tokens[j_spar]:
                    i_spar += 1
                    j_spar += 1

                    if j_spar == a_len or i_spar == s_len:
                        break

                if len(fragment) < (i_spar - i):
                    fragment = []
                    for r in range(i, i_spar):
                        fragment.append(s_tokens[r])
                else:
                    print(f"No new fragment token to add in i: {i}")

                j = j_spar
            else:
                j += 1
        i += max(len(fragment), 1)
        j = 0

        if fragment:
            fragments.append(fragment)

    print(fragments)


if __name__ == '__main__':
    main()
