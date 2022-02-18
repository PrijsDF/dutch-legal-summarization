import time

import pandas as pd
import spacy
from gensim.models import LdaModel
import gensim.corpora as corpora
from rouge import Rouge
from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer

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

# load pretrained mBERT and a pretrained tokenizer for Semantic Coherence prediction
bert_model = BertForNextSentencePrediction.from_pretrained('bert-base-multilingual-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Initialize ROUGE, to be used in the redundancy computation
rouge = Rouge()


def main():
    """Currently this file practically is an adjusted copy of the 'rechtspraak_compute_features.py' file. Here,
    we compute features derived from Bommasani and Cardie (2020) using the cases' texts. These features, later, will be
    used for clustering of cases."""
    # Load the interim dataset
    all_cases = load_dataset(DATA_DIR / 'open_data_uitspraken/interim')

    # Convert the pandas df to a list of dicts for more efficient processing
    cases_dict_list = all_cases.to_dict('records')

    # 1. Remove all |'s that were added during data collection
    cases_dict_list = remove_pipes(cases_dict_list)

    # 2. Add placeholders to the cases list's dicts for each of the features (4 simple features and 6 complex features)
    # This will hold all the cases' feature values; only at the end, we will convert this into a df
    features_dict = {
        'topic_class': -1,
        'redundancy': -1,
        'semantic_coherence': -1
    }
    cases_dict_list = [{**case, **features_dict} for case in cases_dict_list]

    # Make this var true, to load in a previous checkpoint. This can only be done after running it at least once
    checkpoint_exists = False
    if checkpoint_exists:
        # Now, optionally, load in a previous check-point containing the cases and features that are already computed
        cases_checkpoint_df = pd.read_csv(DATA_DIR / 'open_data_uitspraken/features/descriptive.csv')
        cases_checkpoint_dict_list = cases_checkpoint_df.to_dict('records')

        # Provide feedback
        print(f'Cases processed in checkpoint: {len(cases_checkpoint_dict_list)}. '
              f'Cases to go: {len(all_cases) - len(cases_checkpoint_dict_list)}')

        # Combine the list with dicts of cases and the checkpoint list of dicts
        for case in cases_checkpoint_dict_list:
            # We match on this id
            identifier = case['identifier']

            # Find the index of the case
            index = next((i for i, item in enumerate(cases_dict_list) if item["identifier"] == identifier), None)

            # Change the values of the case in the complete list
            cases_dict_list[index] = {**cases_dict_list[index], **case}

    # for c in cases_dict_list[:20]:
    #     del c['summary']
    #     del c['description']
    #
    #     print(c)

    # Now for each case we want to compute each of the features for all cases that haven't been done yet
    start = time.time()
    for i in range(len(cases_dict_list)):
        # Only do something if the cases has not been handled in the checkpoint yet
        if cases_dict_list[i]['topic_class'] == -1:
            # Check time and save checkpoint
            if i % 100 == 0 and i > 0:
                print(f'{len(cases_dict_list) - (i + 1)} cases left. '
                      f'Time elapsed since start: {round((time.time() - start) / 60, 4)} minutes')

                # Make a pd df from the cases that we derived features for
                df_from_dict = pd.DataFrame.from_records([case for case in cases_dict_list
                                                          if case['topic_class'] != -1],
                                                         exclude=['summary', 'description'])

                # Save the checkpoint
                df_from_dict.to_csv(DATA_DIR / 'open_data_uitspraken/features/descriptive.csv', index=False)

                # print(df_from_dict)

            # Create the spacy documents; these can be used to tokenize and sentencize the text
            summary_doc = nlp(cases_dict_list[i]['summary'])
            text_doc = nlp(cases_dict_list[i]['description'])

            # 3 Compute the simple features + word_compression and sentence_compression
            #start = time.time()
            simple_features = compute_simple_features(text_doc)
            #print(f'Simple features took {round(time.time() - start,2)} seconds')

            # Combine the computed simple features and the cases' components
            cases_dict_list[i] = {**cases_dict_list[i], **simple_features}

            # 4. Compute the biggest topic (class) of the text
            #start = time.time()
            cases_dict_list[i]['topic_class'] = compute_topic_class(summary_doc, text_doc)
            #print(f'Topic compute took {round(time.time() - start,2)} seconds')

            # 6. Compute Semantic Coherence
            #start = time.time()
            cases_dict_list[i]['semantic_coherence'] = compute_semantic_coherence(text_doc)
            #print(f'Semantic Coherence took {round(time.time() - start,2)} seconds')

            # 7. Compute Redundancy
            #start = time.time()
            cases_dict_list[i]['redundancy'] = compute_redundancy(text_doc)
            #print(f'Redundancy took {round(time.time() - start,2)} seconds')

            # Make case's text and summary None to boost further speed
            cases_dict_list[i]['summary'] = None
            cases_dict_list[i]['description'] = None

            print(f'Finished iteration {i}')

    # print(cases_dict_list)
    print(f'Total time taken to compute metrics of dataset: {round(time.time() - start, 2)} seconds')

    # 9. Average the computed features for the whole dataset
    # agg_cases_features = create_agg_df(REPORTS_DIR / 'dataset_metrics.csv')
    # print(agg_cases_features)


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


def sentencize_text(doc):
    """Expects a spacy doc. Returns list containing the sentences as spacy Span containers (!)."""

    return [sent for sent in doc.sents]


def compute_simple_features(text_doc):
    """Compute the simple features and two complex features, namely the compressions. Currently, stop words included.
    The function will return the features_df after filling in the features for each case."""
    text_word_length = len(tokenize_text(text_doc))
    text_sent_length = len(sentencize_text(text_doc))

    # Scores for current case; we add the other ones later (make them 999 for now)
    simple_features = {
        'desc_words': text_word_length,
        'desc_sents': text_sent_length,
    }

    return simple_features


def compute_topic_class(summary_doc, text_doc):
    """We compute the topic_class by queuring the LDA model that we learned earlier.

    First, we load the lda model that we generated in models/train_lda_model.py. Then, we use this model to
    give the topic class for the case text. This is the topic class that the case text is most associated with.

    Importantly, the same preprocessing steps and stop word filters need to be applied here as to when the lda model
    was trained. I.e. we need to remove punctuation and make the texts lowercase."""
    # Load LDA model and the corresponding dictionary
    lda_model = LdaModel.load(str(MODELS_DIR / 'lda_model'))
    corpus_dictionary = corpora.Dictionary.load_from_text(str(MODELS_DIR / 'lda_dictionary'))

    # Tokenize the two spacy docs; we exclude stop words
    text_tokenized = tokenize_text(text_doc, rm_stop_words=True)

    # Create bags of ids of the text and summary, using the dictionary
    text_bow = corpus_dictionary.doc2bow(text_tokenized)

    # Get the topic class of the case text
    text_topic_dis = lda_model.get_document_topics(text_bow, minimum_probability=0)
    # print(text_topic_dis)

    topic_class = max(text_topic_dis, key=lambda item: item[1])[0]
    # print(topic_class)

    return topic_class


def compute_semantic_coherence(summary_doc):
    """Compute the semantic coherence score by averaging the BERT next sentence probability predicted for each adjacent
     pair of sentences. Code for next sentence probability computation from https://stackoverflow.com/a/60433070."""
    # For some reason using two sentences gives less UNK tokens than using two lists of tokens...
    # tseq_A = ['Hallo', 'hoe', 'gaat', 'het']
    # tseq_B = ['Goed', 'hoor']
    # seq_A = 'Hallo hoe gaat het'
    # seq_B = 'Goed hoor'

    summary_sents = sentencize_text(summary_doc)

    # Loop over the sentence pairs and compute the probabilities, then add the proportional score to the sc_probability
    sc_probability = 0
    for i in range(len(summary_sents) - 1):
        sentence_a = summary_sents[i].text
        sentence_b = summary_sents[i+1].text

        # encode the two sequences. Particularly, make clear that they must be
        # encoded as "one" input to the model by using 'seq_B' as the 'text_pair'
        encoded = tokenizer.encode_plus(sentence_a, text_pair=sentence_b, return_tensors='pt'
                                        , max_length=512, truncation=True)
        # print(encoded)
        # for ids in encoded["input_ids"]:
        #    print(tokenizer.decode(ids))

        # a model's output is a tuple, we only need the output tensor containing
        # the relationships which is the first item in the tuple
        seq_relationship_logits = bert_model(**encoded)[0]

        # we still need softmax to convert the logits into probabilities
        # index 0: sequence B is a continuation of sequence A
        # index 1: sequence B is a random sequence
        probs = softmax(seq_relationship_logits, dim=1)

        # print(seq_relationship_logits)
        # print(probs)

        # tensor([[9.9993e-01, 6.7607e-05]], grad_fn=<SoftmaxBackward>)
        # very high value for index 0: high probability of seq_B being a continuation of seq_A

        sc_sent_probability = probs[0][0].item() / (len(summary_sents) - 1)
        # print(f'The probability that sequence B follows sequence A is: {sc_sent_probability}')

        sc_probability += sc_sent_probability

    # print(round(sc_sent_probability, 4))
    return round(sc_probability, 4)


def compute_redundancy(summary_doc):
    """Measure redundancy by computing the ROUGE-L F-Score for each pair of distinct sentences in the summary."""
    # First, sentencize the document
    summary_sents = sentencize_text(summary_doc)

    # If there are no found sentences, give the redundancy a score of 999 for post-processing
    if len(summary_sents) > 0:
        # Iterate over all sentence pairs that are not identical and add the computed ROUGE-L score for the pair to the
        # total redundancy score. Then, average by dividing with the number of combinations that we checked
        redundancy = 0
        total_combinations_done = 0
        for sen_a in summary_sents:
            for sen_b in summary_sents:
                if sen_a.text != sen_b.text:
                    total_combinations_done += 1

                    # Compute the rouge scores
                    rouge_scores = rouge.get_scores(sen_a.text, sen_b.text)
                    rouge_l_fscore = rouge_scores[0]['rouge-l']['f']

                    redundancy += rouge_l_fscore

        # Take the mean of the total redundancy; if there were not sent combinations then give 998 as something was
        # wrong
        if total_combinations_done > 0:
            redundancy = round(redundancy / total_combinations_done, 4)
        else:
            redundancy = 998
        # print(f'Total sent combinations:{total_combinations_done}, Found redundancy: {redundancy}')
    else:
        redundancy = 999

    return redundancy


if __name__ == '__main__':
    main()
