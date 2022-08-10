from constants import *
import re
from nltk import FreqDist
import numpy as np
import sqlite3


def remove_chars(text):
    """
    Preformats input text to remove all special characters, split punctuation, remove double lines.
    """
    for c in ['\"', '\'', ':', 'ã', '¯', 'â', '¿', 'â', '½', ',', '.', '?', '!', ';']:
        text = text.replace(c, "")
    text = re.sub("[\(\[].*?[\)\]]", "", text)
    for i in range(3):
        text = text.replace('\n\n\n', '\n\n')
    text = text.replace('\n', ' \n ')
    text = text.replace("  ", " ")
    text = text.lower()
    text = text.replace("  ", " ")
    text = text.replace('embed', " ")
    return text


def preformat(filename):
    """
    Preformats data into three lists: list of all sequential words in the corpus, list of all unique common words,
    list of all unique uncommon words.
    """
    UNK_CUTOFF = 5  # number of times a word must appear to be included in corpus
    conn = sqlite3.connect(filename)
    curr = conn.cursor()
    curr.execute("""SELECT * FROM lyrics""")
    songs_list = curr.fetchall()  # list of all songs fetched from sql database
    conn.close()
    songs_as_text = ""
    for song in songs_list:
        songs_as_text += song[0] + "\n\n"  # need the 0th index because each element is a tuple from fetchall()
    corpus = remove_chars(songs_as_text)
    corpus = corpus.split(' ')
    frequency_dist = FreqDist(corpus)
    most_common = frequency_dist.most_common()
    common = list()
    uncommon = list()
    for i, word in enumerate(most_common):
        if word[1] >= UNK_CUTOFF:
            common.append(word[0])
        else:
            uncommon.append(word[0])
    common = sorted(list(set(common)))
    uncommon = sorted(list(set(uncommon)))
    return corpus, common, uncommon


def preprocess(filename, pred_len):
    """
    Preprocesses data into sequence/label (or input/label) pairs. Ignores all uncommon words.
    Main endpoint for preprocessing.
    """
    corpus, common, uncommon = preformat(filename)
    corpus_len = len(corpus)

    dictionary = dict((key, value) for value, key in enumerate(common))

    inputs = list()
    labels = list()
    # this preprocessed for an ngram model, specifically where n=pred_len
    for i in range(corpus_len - pred_len):
        if len(set(corpus[i:i + pred_len + 1]).intersection(uncommon)) == 0:  # ensures no uncommon words
            inputs.append(corpus[i:i + pred_len])
            labels.append(corpus[i + pred_len])

    # shuffling data
    elements = len(inputs)
    inputs, labels = np.array(inputs), np.array(labels)
    ids = np.random.permutation(elements)
    inputs = inputs[ids]
    labels = labels[ids]

    return inputs, labels, dictionary


if __name__ == "__main__":
    print(preprocess(DATA_LOCATION, PRED_LEN)[0])
