import re
import numpy as np
import pandas as pd
from collections import Counter
import functools
import tqdm
import pymorphy2


def normalize_answer(text):
    """Lower text and remove punctuation and extra whitespace."""
    return ' '.join(re.findall(r"\w+", text)).lower()


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = len(common)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def sentence_to_word(sentences):
    sentences_in_words = list()
    for sentence in sentences:
        sentences_in_words.append(tuple(normalize_answer(sentence).split()))
    return sentences_in_words


def text_to_sentence(text):
    sentences = text.split(".")
    return [s.strip() for s in sentences if s.strip() != '']


def uniq_words(text):
    return set(re.findall("\w+", text))


def calculate_idfs(data):
    counter_paragraph = Counter()
    uniq_paragraphs = data['paragraph'].unique()
    for paragraph in tqdm.tqdm(uniq_paragraphs, desc="calc idf"):
        set_words = uniq_words(paragraph)
        counter_paragraph.update(set_words)

    num_docs = uniq_paragraphs.shape[0]
    idfs = {}
    for word in counter_paragraph:
        idfs[word] = np.log(num_docs / counter_paragraph[word])
    return idfs


class FeatureMaker(object):

    """
    Класс для построения признаков из текста
    Если есть колонка answer, то добавляется колонка target, если нет, то не добавляется.
    """

    def __init__(self, dataframe, idfs):
        self.data = dataframe.sort_values('paragraph')
        self.idfs = idfs

        self.result = {}
        self.keys = {}
        self.rev_keys = {}
        self.max_key = 0

        self.morph = pymorphy2.MorphAnalyzer()
        # Максимальная длинна в словах кандидатов
        self.MAX_SPAN_LEN = 5
        self.MORPH_TAGS = {
            'NOUN', 'VERB', 'ADJF', 'ADJS', 'NUMR', 'PREP', 'CONJ', 'PRCL',
            'INTJ', 'GRND', 'COMP', 'INFN', 'PRTF', 'PRTS', 'ADVB', 'NPRO',
            'PRED', 'LATN', 'ROMN', 'UNKN'
        }
        columns = ['ADJF', 'ADJS', 'ADVB', 'COMP', 'CONJ', 'GRND', 'IDF (Span)', 'INFN',
                   'INTJ', 'LATN', 'Left Length', 'Match IDF  (Span)',
                   'Match IDF (Right of Span)', 'Match IDF (Whole Sentence)',
                   'Match TF IDF (Left of Span)', 'NOUN', 'NPRO', 'NUMR', 'PRCL', 'PRED',
                   'PREP', 'PRTF', 'PRTS', 'ROMN', 'Right Length', 'Sentence Length',
                   'Span Length', 'UNKN', 'VERB', 'question_id', 'paragraph_id']
        if 'answer' in self.data.columns:
            columns += ['target']
        self.columns = columns
        # preallocate data
        for key in columns:
            self.result[key] = [None] * self.data.shape[0] * 100

    def make(self):
        for data_ind in tqdm.tqdm(self.data.index.values, desc="calc features"):
            self.generate_candidates_with_features(self.data.loc[data_ind])
        for col_name in self.result:
            self.result[col_name] = self.result[col_name][:self.max_key]
        return pd.DataFrame.from_dict(self.result).fillna(0, inplace=False)

    def generate_candidates_with_features(self, data_row):
        paragraph = data_row["paragraph"]
        question = data_row["question"]
        if 'answer' in data_row:
            answer = data_row["answer"]
        else:
            answer = None
        question_id = data_row["question_id"]
        paragraph_id = data_row["paragraph_id"]
        sentences = text_to_sentence(paragraph)
        sentences_in_words = sentence_to_word(sentences)
        question_in_words = sentence_to_word([question])[0]

        # определение предложения для поиска ответа
        # выбираем такое предложение, которое сильнее всего пересекается с вопросом
        # это эвристика для уменьшения числа кандидатов
        sentence_id = self.get_max_match_sentance_id(sentences_in_words,
                                                     question_in_words)
        sentence = sentences_in_words[sentence_id]
        sentenceMorphy = self.convert_morphy(sentence)

        # генерируем все воможные кандидаты для выбранного предложения и строим для них признаки
        for i in range(len(sentence)):
            for j in range(i + 1, min(i + 1 + self.MAX_SPAN_LEN, len(sentence) + 1)):
                span_hash = self.calculate_key(question_id, paragraph_id, sentence_id, (i, j))
                self.generate_features_for_span(
                    span_hash, sentence, question_in_words, (i, j),
                    sentenceMorphy, question_id, answer, paragraph_id)

        # добавляем еще одного кандидата - все предложение
        span = (0, len(sentence))
        span_hash = self.calculate_key(question_id, paragraph_id, sentence_id, span)
        self.generate_features_for_span(span_hash, sentence, question_in_words,
                                        span, sentenceMorphy, question_id, answer, paragraph_id)

    def generate_features_for_span(self, key, sentence, question, span,
                                   sentenceMorphy, question_id, answer, paragraph_id):
        """
        Метод для генерации признаков для ответа-кандидата
        """
        self.add_column(key, 'Left Length', span[0])
        self.add_column(key, 'Right Length', len(sentence) - span[1])
        self.add_column(key, 'Sentence Length', len(sentence))
        self.add_column(key, 'Span Length', span[1] - span[0])

        self.add_idf_feature(key, 'IDF (Span)', sentence, span)

        self.add_idf_match_feature(key, 'Match TF IDF (Left of Span)',
                                   sentence, question, (0, span[0]))
        self.add_idf_match_feature(key, 'Match IDF (Right of Span)', sentence,
                                   question, (span[1], len(sentence)))
        self.add_idf_match_feature(key, 'Match IDF (Whole Sentence)', sentence,
                                   question, (0, len(sentence)))
        self.add_idf_match_feature(key, 'Match IDF  (Span)', sentence,
                                   question, span)

        # добавляем признаки на основе частей речи слов
        self.add_morph_tag_feature(key, sentenceMorphy, span)

        self.add_column(key, 'question_id', question_id)
        self.add_column(key, 'paragraph_id', paragraph_id)
        if answer is not None:
            self.add_column(key, 'target', f1_score(' '.join(sentence[span[0]:span[1]]), answer))

    def get_max_match_sentance_id(self, sentences_in_words, question_in_words):
        """
        поиск предложения в параграфе, которое сильнее всего пересекается с вопросом
        """
        max_overlap = -1
        max_match_sentance_id = None

        question_words = set(question_in_words)
        for sentance_id in range(len(sentences_in_words)):
            overlap = len(set(sentences_in_words[sentance_id]) & question_words)
            if overlap > max_overlap:
                max_overlap = overlap
                max_match_sentance_id = sentance_id

        return max_match_sentance_id

    def add_column(self, key, name, value):
        """
        метод для добавления колонки в результирующие данные
        """
        if len(self.result[name]) <= key:
            self.result[name] += [None] * (key + 1 - len(self.result[name]))
        self.result[name][key] = value

    def add_idf_feature(self, key, name, sentence, span):
        """
        добавление признаков вида \sum_{w in span} idf[w]
        """
        self.add_column(key, name, self.calculate_sum_idf_sentence(sentence, span))

    @functools.lru_cache(maxsize=2 ** 14)
    def calculate_sum_idf_sentence(self, sentence, span):
        sum_idf = 0.0
        for w in sentence[span[0]:span[1]]:
            if w in self.idfs:
                sum_idf += self.idfs[w]
        return sum_idf

    @functools.lru_cache(maxsize=2 ** 14)
    def calculate_sum_idf_sentence_question(self, sentence, question, span):
        sum_idf = 0.0
        for w in sentence[span[0]:span[1]]:
            if w in question and w in self.idfs:
                sum_idf += self.idfs[w]
        return sum_idf

    def add_idf_match_feature(self, key, name, sentence, question, span):
        """
        добавление признаков вида \sum_{w in span and w in question} idf[w]
        """
        self.add_column(key, name,
                        self.calculate_sum_idf_sentence_question(sentence, question, span))

    def calculate_key(self, question_id, paragraph_id, sentence_id, span):
        """
        метод для генерации уникального ключа в результирующей таблице
        """
        key_tuple = (question_id, paragraph_id, sentence_id, span[0], span[1])
        if key_tuple not in self.keys:
            self.keys[key_tuple] = self.max_key
            self.rev_keys[self.max_key] = key_tuple
            self.max_key += 1
        return self.keys[key_tuple]

    @functools.lru_cache(maxsize=2 ** 17)
    def morphize(self, word):
        return self.morph.parse(word)[0].tag.grammemes & self.MORPH_TAGS

    @functools.lru_cache(maxsize=2 ** 4)
    def convert_morphy(self, sentence):
        return tuple((self.morphize(word) for word in sentence))

    @functools.lru_cache(maxsize=2 ** 8)
    def get_morph_counter(self, sentenceMorphy, span):
        count_grammes = Counter((grammem for grammems in sentenceMorphy[span[0]:span[1]]
                                 for grammem in grammems))
        return count_grammes

    def add_morph_tag_feature(self, key, sentenceMorphy, span):
        count_grammes = self.get_morph_counter(sentenceMorphy, span)
        for k in self.MORPH_TAGS:
            self.add_column(key, k, count_grammes.get(k, 0))

    def get_span(self, key):
        """
        reverse opeartion for found selected answer
        """
        question_id, paragraph_id, sentence_id, i, j = self.rev_keys[key]
        paragraph = self.data[(self.data["question_id"] == question_id) & (self.data['paragraph_id'] == paragraph_id)].iloc[0]["paragraph"]

        sentences = text_to_sentence(paragraph)
        sentences_in_words = sentence_to_word(sentences)
        return ' '.join(sentences_in_words[sentence_id][i:j])
