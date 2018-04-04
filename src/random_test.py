from util import read_test_data, read_train_data, read_dev_data, read_dev_key_data, \
    train_data_filename, dev_data_filename, dev_data_key_filename, break_train_data_line, full_filename
from random import shuffle, randint
from nbclassify import nb_learn_data, nb_test, nb_dev_test_sentiment_data, nb_dev_test_authenticity_data
from statistics import mean


def write_full_data():
    test_data = read_test_data(train_data_filename)
    dev_data = read_dev_data(dev_data_filename)
    dev_key_data = read_dev_key_data(dev_data_key_filename)


    dev_lines = []
    for (review_id, review_text) in dev_data:
        auth, sent = dev_key_data[review_id]
        dev_lines.append((review_id, "{} {} {} ".format(auth, sent, review_text)))


    total_data = test_data + dev_lines


    data = "\n".join([" ".join([review_id, text]) for (review_id, text) in total_data])

    with open("/tmp/full_data.txt", 'w') as fp:
        fp.write(data)


def read_full_data(filename):
    full_data = []
    with open(filename, 'r') as fp:
        for line in fp.readlines():
            full_data.append(break_train_data_line(line))

    return full_data


def random_split_test_train(data):
    shuffle(data)
    data_length = len(data)
    split_index = int(0.8*data_length)
    test_data, test_data_key = convert_to_test(data[split_index:])
    return data[0:split_index], test_data, test_data_key

def convert_to_test(data):
    test_data = []
    test_data_key = {}
    for line in data:
        # print(line)
        review_id, review_text, auth, sent = line
        test_data.append((review_id, review_text))
        test_data_key[review_id] = (auth, sent)
    return test_data, test_data_key

def test_random_patch(data):
    train_data, test_data, test_data_key = random_split_test_train(data)
    sent_model, auth_model = nb_learn_data(train_data)
    (sp1, sr1, sf11), (sp2, sr2, sf12) = nb_dev_test_sentiment_data(sent_model, test_data, test_data_key)
    (ap1, ar1, af11), (ap2, ar2, af12) = nb_dev_test_authenticity_data(auth_model, test_data, test_data_key)
    f1_scores = [sf11, sf12, af11, af12]
    f1_avg = mean(f1_scores)
    print(f1_avg)
    return f1_avg




if __name__ == '__main__':
    full_data = read_full_data(full_filename)
    f1_scores = []
    for i in range(0, 10):
        f1_scores.append(test_random_patch(full_data))
    print("final average:" , mean(f1_scores))