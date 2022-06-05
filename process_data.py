from constants import *
import re
from nltk import FreqDist
import numpy as np
from itertools import chain


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
    with open(filename, 'r') as f:
        corpus = remove_chars(f.read())
    corpus = corpus.split(' ')
    # print("Total Words: " + str(len(corpus)))
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
    # print(f'Unique UNCOMMON words (LESS than {UNK_CUTOFF} appearances): {len(uncommon)}')
    # print(f'Unique COMMON words (MORE than {UNK_CUTOFF} appearances): {len(common)}')
    return corpus, common, uncommon


def preprocess(filename, pred_len=5):
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

    # print(f'Number of text input-label pairs: {len(inputs)}')

    # uses 3% of dataset for model validation
    train_I, test_I, train_L, test_L = train_test_split(inputs, labels, random_seed=69)

    return train_I, test_I, train_L, test_L, dictionary


# method from stackoverflow to reduce package size from scikit learn
def _indexing(x, indices):
    """
    :param x: array from which indices has to be fetched
    :param indices: indices to be fetched
    :return: sub-array from given array and indices
    """
    # np array indexing
    if hasattr(x, 'shape'):
        return x[indices]

    # list indexing
    return [x[idx] for idx in indices]


# method from stackoverflow to reduce package size from scikit learn
def train_test_split(*arrays, test_size=VALIDATION_SZ, shufffle=True, random_seed=1):
    """
    splits array into train and test data.
    :param arrays: arrays to split in train and test
    :param test_size: size of test set in range (0,1)
    :param shufffle: whether to shuffle arrays or not
    :param random_seed: random seed value
    :return: return 2*len(arrays) divided into train ans test
    """
    # checks
    assert 0 < test_size < 1
    assert len(arrays) > 0
    length = len(arrays[0])
    for i in arrays:
        assert len(i) == length

    n_test = int(np.ceil(length * test_size))
    n_train = length - n_test

    if shufffle:
        perm = np.random.RandomState(random_seed).permutation(length)
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
    else:
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, length)

    return list(chain.from_iterable((_indexing(x, train_indices), _indexing(x, test_indices)) for x in arrays))


if __name__ == "__main__":
    preprocess('data_dump/all_lyrics.txt')
