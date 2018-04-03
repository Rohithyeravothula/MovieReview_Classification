from nbclassify import nb_learn, nb_dev_test_sentiment, nb_dev_test_authenticity, \
    BINARY_AUTH_CLASS_GROUP, BINARY_SENT_CLASS_GROUP, nb_test, store_models, read_models
from util import pprint, dev_data_filename, dev_data_key_filename, train_data_filename, \
    nbmodel_filename, output_filename

import os
import sys

def dev_learn():
    sent_model, auth_model = nb_learn(train_data_filename)
    store_models(nbmodel_filename, sent_model, auth_model)


def dev_test():
    sent_model, auth_model = read_models(nbmodel_filename)
    nb_dev_test_sentiment(sent_model)
    nb_dev_test_authenticity(auth_model)


def dev_sentiment():
    sent_model, auth_model = read_models(nbmodel_filename)
    nb_dev_test_sentiment(sent_model)


def test_nb_test():
    sent_model, auth_model = read_models(nbmodel_filename)
    nb_test(dev_data_filename, sent_model, auth_model, output_filename)


if __name__ == '__main__':
    dev_learn()
    dev_sentiment()
    # nb_test()
    # dev_test_autheiticity()
    # dev_test_sentiment()
