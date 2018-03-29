
train_data_filename = "../data/train-labeled.txt"
# train_data_filename = "../data/sample-train-labeled.txt"

dev_data_filename = "../data/dev-text.txt"
dev_data_key_filename = "../data/dev-key.txt"

stop_words = {}


def pprint(collection):
    if isinstance(collection, dict):
        for line in collection.items():
            print(line)

    for line in collection:
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
    dev_key_map = {}
    for line in fp.read().splitlines():
        review_id, authentic, sentiment = line.split(" ")
        dev_key_map[review_id] = [authentic, sentiment]
    fp.close()
    return dev_key_map


def clean_text(text):
    """
    remove stop words, pronouns, convert case
    :return:
    """
    return [word.strip().lower() for word in text.split(" ") if word not in stop_words]


def get_word_features(text):
    """
    return words that are considered to be features
    """
    return clean_text(text)


if __name__ == '__main__':
    ""
    # pprint(read_train_data())
    # pprint(read_dev_data())
    # pprint(read_dev_key_data())
