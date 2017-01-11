import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

from helpers import load_sets


def tokenize(corpus):
    """
    Using split as tokenizer for now.
    """
    new_corpus = []
    for item in corpus:
        item = item.split()
        new_corpus.append(item)
    return new_corpus


def word_features(example):
    """
    Converting an example to features
    """
    return dict([(word, True) for word in example])


def featurize():
    """
    1. Load the training and test sets
    2. Tokenize the examples
    3. Create word features
    """
    x_train, x_test, y_train, y_test = load_sets()
    x_train = tokenize(x_train)
    x_test = tokenize(x_test)
    x_train = [word_features(item) for item in x_train]
    x_test = [word_features(item) for item in x_train]
    return x_train, x_test, y_train, y_test


def build_naive_bayes():
    x_train, x_test, y_train, y_test = featurize()
    train_set = [(x, y) for x, y in zip(x_train, y_train)]
    test_set = [(x, y) for x, y in zip(x_test, y_test)]
    classifier = NaiveBayesClassifier.train(train_set)
    print 'accuracy:', nltk.classify.util.accuracy(classifier, test_set)
    classifier.show_most_informative_features()


build_naive_bayes()
