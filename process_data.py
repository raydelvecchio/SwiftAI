import lyricsgenius
from constants import *
import json
import re
from nltk import FreqDist
import sklearn.model_selection as skm


def download_lyrics():
    """
    Downloads all lyrics from all albums into JSON files from GENIUS API. Should be called only once.
    Can also download all lyrics from every Taylor Swift song, but I opted for just the albums.
    """
    genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
    album_data = list()
    for album in ALBUMS:
        album_data.append(genius.search_album(album, 'Taylor Swift'))
    for jsn in album_data:
        jsn.save_lyrics()


def txt_dump():
    """
    Writes all lyrics into one large text file.
    """
    filenames = [f'data_dump/Lyrics_{i.replace(" ", "")}.json' for i in ALBUMS]
    with open('data_dump/all_lyrics.txt', 'w') as txt:
        for name in filenames:
            file = open(name)
            data = json.load(file)
            for song in data['tracks']:
                song = song['song']
                lyrics = song['lyrics']
                txt.write(lyrics)
            file.close()


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
    Preprocesses data into sequence/label (or input/label) pairs. Main endpoint for preprocessing.
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

    train_I, test_I, train_L, test_L = skm.train_test_split(inputs, labels, test_size=0.05, random_state=69)

    return train_I, test_I, train_L, test_L, dictionary


if __name__ == "__main__":
    preprocess('data_dump/all_lyrics.txt')
