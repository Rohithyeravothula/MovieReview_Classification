from nbclassify import nb_learn, nb_dev_test_sentiment, nb_dev_test_authenticity, \
    BINARY_AUTH_CLASS_GROUP, BINARY_SENT_CLASS_GROUP, nb_test, store_models, read_models
from util import pprint, dev_data_filename, dev_data_key_filename, train_data_filename, \
    nbmodel_filename, output_filename

import os
import sys

def dev_learn():
    sent_model, auth_model = nb_learn(train_data_filename)
    words = list(sent_model.counter.items())
    words.sort(key=lambda x:sum(x[1]), reverse=True)
    # pprint(words[:30])
    store_models(nbmodel_filename, sent_model, auth_model)


def dev_test():
    sent_model, auth_model = read_models(nbmodel_filename)
    nb_dev_test_sentiment(sent_model)
    nb_dev_test_authenticity(auth_model)

def dev_auth():
    sent_mode, auth_model = read_models(nbmodel_filename)
    nb_dev_test_authenticity(auth_model)

def dev_sentiment():
    sent_model, auth_model = read_models(nbmodel_filename)
    print(nb_dev_test_sentiment(sent_model))


def test_nb_test():
    sent_model, auth_model = read_models(nbmodel_filename)
    nb_test(dev_data_filename, sent_model, auth_model, output_filename)


def funny(text):
    print(text)
    import string
    punctuations = set(string.punctuation)
    text = [letter for letter in text if letter not in punctuations]
    print("".join(text))

if __name__ == '__main__':
    # funny("wow, this is ? a test!! onlye. To see if, tihs works")
    dev_learn()
    # dev_sentiment()
    # dev_auth()
    # dev_test()
    sent_model, auth_model = read_models(nbmodel_filename)
    nb_test(dev_data_filename, sent_model, auth_model, output_filename)
    # dev_test_autheiticity()
    # dev_test_sentiment()




def nbclassify():
    test_filename = sys.argv[1]
    sent_model, auth_model = read_models(nbmodel_filename)
    nb_test(test_filename, sent_model=sent_model, auth_model=auth_model, output_filename=output_filename)


def nblearn():
    import os
    train_filename = sys.argv[1]
    sent_model, auth_model = nb_learn(train_filename)
    store_models(os.path.join(os.getcwd(), nbmodel_filename), sent_model=sent_model, auth_model=auth_model)
