import time
import re
from pprint import pprint

import pandas as pd
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


def main():
    """View Open Rechtspraak dataset with pandas."""
    # Load the raw dataset
    all_cases = load_dataset(DATA_DIR / 'open_data_uitspraken/interim')
    # print(all_cases)

    # Create LDA model and save it in reports
    create_lda_model(all_cases[:40])


def create_lda_model(dataset):
    """Applies LDA as described in Blei (2003), Latent dirichlet allocation. Used https://towardsdatascience.com/end
    -to-end-topic-modeling-in-python-latent-dirichlet-allocation-lda-35ce4ed6b3e0 as a guide. We follow
    Bommasani & Cardie (2020), Intrinsic Evaluation of Summarization Datasets, in only using the case descriptions to
    train the LDA model."""
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

    # Build LDA model
    num_topics = 5
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())

    # Save model to disk.
    lda_model.save(str(MODELS_DIR / 'lda'))

    print(f'Time taken to perform LDA on dataset: {round(time.time() - start, 2)} seconds')


if __name__ == '__main__':
    main()
