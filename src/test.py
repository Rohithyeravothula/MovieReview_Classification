from nbclassify import nb_learn, nb_dev_test_sentiment, nb_dev_test_authenticity, \
    BINARY_AUTH_CLASS_GROUP, BINARY_SENT_CLASS_GROUP, nb_test, store_models, read_models
from util import pprint, dev_data_filename, dev_data_key_filename, train_data_filename, \
    nbmodel_filename, output_filename

import os
import sys

def dev_test_sentiment():
    model = nb_learn(train_data_filename,  "sent", BINARY_SENT_CLASS_GROUP)
    # print(model.counter)
    nb_dev_test_sentiment(model)


def dev_test_autheiticity():
    model = nb_learn(train_data_filename, "auth", BINARY_AUTH_CLASS_GROUP)
    nb_dev_test_authenticity(model)


def dev_learn():
    sent_model = nb_learn(train_data_filename, "sent", BINARY_SENT_CLASS_GROUP)
    auth_model = nb_learn(train_data_filename, "auth", BINARY_AUTH_CLASS_GROUP)
    store_models(nbmodel_filename, sent_model, auth_model)
    print(sent_model.classes_prior_counts)

    # print(auth_model.class_prior_prob)


def dev_test():
    dev_test_sentiment()
    dev_test_autheiticity()


def nb_test():
    sent_model, auth_model = read_models(nbmodel_filename)
    nb_test(dev_data_filename, sent_model, auth_model, output_filename)


def submission_learn():
    pwd = os.getcwd()
    train_filename = sys.argv[1]
    sent_model = nb_learn(train_filename, "sent", BINARY_SENT_CLASS_GROUP)
    auth_model = nb_learn(train_filename, "auth", BINARY_AUTH_CLASS_GROUP)
    store_models(os.path.join(pwd, nbmodel_filename), sent_model=sent_model, auth_model=auth_model)


def submission_test():
    pwd = os.getcwd()
    output_filename = os.path.join(pwd, output_filename)
    test_filename = sys.argv[1]
    sent_model, auth_model = read_models(nbmodel_filename)
    nb_test(test_filename, sent_model=sent_model, auth_model=auth_model, output_filename = output_filename)


if __name__ == '__main__':
    dev_learn()
    dev_test()
    # nb_test()
    # dev_test_autheiticity()
    # dev_test_sentiment()
