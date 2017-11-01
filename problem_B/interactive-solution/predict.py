# coding: utf-8
# Baseline: MaxMatch
import pandas as pd
import tqdm
import os
import re


def normalize_answer(text):
    """Lower text and remove punctuation and extra whitespace."""
    return ' '.join(re.findall(r"\w+", text)).lower()


def sentence_to_word(sentences):
    sentences_in_words = list()
    for sentence in sentences:
        sentences_in_words.append(normalize_answer(sentence).split())
    return sentences_in_words


def text_to_sentence(text):
    sentences = text.split(".")
    return [s.strip() for s in sentences if s.strip() != '']


def get_max_match_sentance(data_row):
    sentences = text_to_sentence(data_row["paragraph"])
    sentences_in_words = sentence_to_word(sentences)
    question_in_words = sentence_to_word([data_row["question"]])[0]

    max_overlap = None
    max_match_sentance_id = None

    question_words = set(question_in_words)
    for sentance_id in range(len(sentences_in_words)):
        sentence_words = set(sentences_in_words[sentance_id])
        overlap = len(sentence_words.intersection(question_words))
        if max_overlap is None or overlap > max_overlap:
            max_overlap = overlap
            max_match_sentance_id = sentance_id

    return sentences[max_match_sentance_id]


if __name__ == '__main__':
    DATA_FILE = os.environ.get('INPUT')
    PREDICTION_FILE = os.environ.get('OUTPUT')

    df = pd.DataFrame.from_csv(DATA_FILE, sep=',', index_col=None)
    df = df[['paragraph_id', 'question_id', 'paragraph', 'question']]

    df['predictions'] = None
    for data_ind in tqdm.tqdm(df.index.values):
        full_sentance = get_max_match_sentance(df.loc[data_ind])
        df.loc[data_ind, 'predictions'] = full_sentance

    df['answer'] = df['predictions']
    df.set_index(['paragraph_id', 'question_id'])['answer'].to_csv(PREDICTION_FILE, header=True)
