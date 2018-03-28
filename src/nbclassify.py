train_data_filename = "../data/train-labeled.txt"

dev_data_filename = "../data/dev-text.txt"
dev_data_key_filename = "../data/dev-key.txt"

"""
each classification will be of type
[id, text, pos/neg, real/fake]
"""


def pprint(list_input):
    for line in list_input:
        print(line)


def break_train_data_line(text):
    if not text:
        return None
    split = text.split(" ")
    review_id, authenticity, sentiment = split[:3]
    review_text = " ".join(split[3:])
    return [review_id, review_text, authenticity, sentiment]


def read_train_data():
    train_data = []
    fp = open(train_data_filename, 'r')
    for text_line in fp.read().splitlines():
        train_data.append(break_train_data_line(text_line))
    fp.close()
    return train_data


def read_test_data(filename):
    fp = open(filename, 'r')
    fp.close()
    return fp.read().splitlines()


def read_dev_data():
    dev_data = []
    fp = open(dev_data_filename, 'r')
    for line in fp.read().splitlines():
        review_id, *review_text = line.split(" ")
        dev_data.append([review_id, " ".join(review_text)])
    fp.close()
    return dev_data


def read_dev_key_data():
    fp = open(dev_data_key_filename)
    dev_key_map = []
    for line in fp.read().splitlines():
        dev_key_map.append(line.split(" "))
    fp.close()
    return dev_key_map


def clean_text():
    """
    remove stop words, pronouns, convert case and return
    :return:
    """
    pass


def compute_probabilities(text):
    """
    returns json of words, and their probabilities
    { "word":[0,0,0,0]} => contains probabilities of word against each class
    :param text:
    :return:
    """
    pass


def classify(input_text, probabilities):
    pass


def compute_error(text_classification):
    pass


if __name__ == '__main__':
    # pprint(read_train_data())
    # pprint(read_dev_data())
    # pprint(read_dev_key_data())
