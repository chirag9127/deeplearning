import numpy as np

from os import listdir
from os.path import isfile, join
from sklearn.cross_validation import train_test_split


POS_PATH = '../data/aclImdb/train/pos/'
NEG_PATH = '../data/aclImdb/train/neg/'


def load_corpus():
    """
    Load the positive and negative reviews from ../data/aclImdb/
    """
    pos_files = [f for f in listdir(POS_PATH) if isfile(join(POS_PATH, f))]
    neg_files = [f for f in listdir(NEG_PATH) if isfile(join(NEG_PATH, f))]
    pos_reviews = []
    neg_reviews = []
    for filename in pos_files:
        with open('{0}/{1}'.format(POS_PATH, filename)) as f:
            for line in f:
                pos_reviews.append(line)
    for filename in neg_files:
        with open('{0}/{1}'.format(NEG_PATH, filename)) as f:
            for line in f:
                neg_reviews.append(line)
    return pos_reviews, neg_reviews


def clean_corpus(corpus):
    """
    Cleaning the corpus. Removing \n, \t and breaks. This also isolates the
    punctuation marks so they can be used as a feature.
    """
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '').replace('\t', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    return corpus


def create_train_and_test_sets():
    """
    Creates the train and test splits and stores them in the processed folder.
    """
    pos_reviews, neg_reviews = load_corpus()
    y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    x_train, x_test, y_train, y_test = train_test_split(
        np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)
    x_train = clean_corpus(x_train)
    x_test = clean_corpus(x_test)
    with open('../data/processed/train.tsv', 'wb') as fo:
        for train, label in zip(x_train, y_train):
            fo.write('{0}\t{1}\n'.format(train, label))
    with open('../data/processed/test.tsv', 'wb') as fo:
        for test, label in zip(x_test, y_test):
            fo.write('{0}\t{1}\n'.format(test, label))


def load_sets():
    x_train, x_test, y_train, y_test = [], [], [], []
    with open('../data/processed/train.tsv') as f:
        for line in f:
            line = line.strip().split('\t')
            train, label = line[0], float(line[1])
            x_train.append(train)
            y_train.append(label)
    with open('../data/processed/test.tsv') as f:
        for line in f:
            line = line.strip().split('\t')
            test, label = line[0], float(line[1])
            x_test.append(test)
            y_test.append(label)
    return x_train, x_test, y_train, y_test
