from collections import Counter, defaultdict
import json
import math
from operator import add
import sys
import os

"""
each classification will be of type
[id, text, pos/neg, real/fake]
"""
BINARY_AUTH_CLASS_GROUP = ["True", "Fake"]
BINARY_SENT_CLASS_GROUP = ["Pos", "Neg"]
ASSIGNMENT_CLASSES = BINARY_AUTH_CLASS_GROUP + BINARY_SENT_CLASS_GROUP




from typing import List, Dict, Set, Tuple
import string

train_data_filename = "../data/train-labeled.txt"
# train_data_filename = "../data/sample-train-labeled.txt"

dev_data_filename = "../data/dev-text.txt"
dev_data_key_filename = "../data/dev-key.txt"

nbmodel_filename = "nbmodel.txt"
output_filename = "nboutput.txt"



stop_words = {'i', 'or', 'besides', 'six', 'whom', 'either', 'being', 'when', 'always', 'even',
              'amongst', 'on', 'all', 'over', 'eight', 'back', 'has', 'have', 'less', 'ourselves',
              'a', 'about', 'my', 'seems', 'until', 'keep', 'toward', 'anyway', 'to', 'around',
              'beforehand', 'cannot', 'could', 'does', 'had', 'somehow', 'thus', 'am', 'if',
              'their', 'front', 'who', 'once', 'put', 'is', 'some', 'under', 'whole', 'well',
              'beyond', 'often', 'onto', 'see', 'sometimes', 'by', 'forty', 'fifteen', 'part',
              'everyone', 'than', 'these', 'can', 'twelve', 'another', 'been', 'next', 'same',
              'seeming', 'further', 'used', 'might', 'become', 'himself', 'our', 'twenty', 'ours',
              'us', 'thereafter', 'else', 'few', 'after', 'therefore', 'was', 'various', 'move',
              'several', 'elsewhere', 'would', 'among', 'so', 'nor', 'this', 'becoming', 'yourself',
              'each', 'also', 'mine', 'everything', 'for', 'done', 'empty', 'per', 'whereafter',
              'please', 'together', 'then', 'unless', 'full', 'however', 'give', 'no', 'below',
              'since', 'whereby', 'already', 'that', 'must', 'between', 'seemed', 'hereupon',
              'because', 'down', 'every', 'made', 'as', 'thru', 'neither', 'least', 'wherein',
              'both', 'here', 'and', 'indeed', 'therein', 'bottom', 'throughout', 'yourselves',
              'regarding', 'ca', 'ever', 'call', 'somewhere', 'there', 'whoever', 'whence', 'in',
              'serious', 'such', 'latterly', 'last', 'only', 'top', 'against', 'out', 'name',
              'much', 'along', 'herein', 'from', 'hers', 'two', 'into', 'while', 'without',
              'whether', 'became', 'anyhow', 'where', 'within', 'enough', 'hereby', 'four', 'very',
              'whatever', 'myself', 'again', 'alone', 'yours', 'should', 'them', 'nobody', 'nine',
              'nevertheless', 'three', 'up', 'moreover', 'why', 'afterwards', 'not', 'his',
              'sometime', 'first', 'never', 'go', 'otherwise', 'third', 'via', 'will', 'herself',
              'at', 'becomes', 'before', 'him', 'themselves', 'amount', 'your', 'did', 'are',
              'what', 'more', 'namely', 'perhaps', 'whenever', 'do', 'hereafter', 'just',
              'thereupon', 'too', 'anywhere', 'you', 'be', 'sixty', 'most', 'behind', 'mostly',
              'other', 'something', 'during', 'meanwhile', 'seem', 'though', 'although', 'latter',
              'get', 'anyone', 'itself', 'they', 'of', 'take', 'show', 'whither', 'none', 'yet',
              'she', 'wherever', 'ten', 'upon', 'beside', 'an', 'any', 'but', 'make', 'hence',
              'off', 'one', 'own', 'rather', 'someone', 'using', 'it', 'anything', 'may', 'others',
              'nothing', 'really', 'we', 'due', 'me', 'whose', 'everywhere', 're', 'former',
              'fifty', 'above', 'say', 'the', 'doing', 'still', 'thence', 'eleven', 'five', 'her',
              'quite', 'thereby', 'whereupon', 'many', 'almost', 'except', 'hundred', 'nowhere',
              'whereas', 'none', 'with', 'across', 'which', 'those', 'towards', 'how', 'side',
              'he', 'were', 'its', 'formerly', 'now', 'through'}


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

def break_test_data_line(text: str):
    if not text:
        return None
    review_id, *review_text = text.split(" ")
    return review_id, " ".join(review_text)


def read_train_data(filename: str) -> List[List[str]]:
    train_data = []
    fp = open(filename, 'r')
    for text_line in fp.read().splitlines():
        train_data.append(break_train_data_line(text_line))
    fp.close()
    return train_data


def read_test_data(filename):
    fp = open(filename, 'r')
    test_data = []
    for line in fp.read().splitlines():
        test_data.append(break_test_data_line(line))
    fp.close()
    return test_data


def read_dev_data(filename: str) -> List[Tuple[str, str]]:
    dev_data = []
    fp = open(filename, 'r')
    for line in fp.read().splitlines():
        review_id, *review_text = line.split(" ")
        dev_data.append((review_id, " ".join(review_text)))
    fp.close()
    return dev_data


def read_dev_key_data(filename) -> Dict[str, Tuple[str, str]]:
    fp = open(filename)
    dev_key_map = {}
    for line in fp.read().splitlines():
        review_id, authentic, sentiment = line.split(" ")
        dev_key_map[review_id] = (authentic, sentiment)
    fp.close()
    return dev_key_map


def write_predictions(predictions, filename):
    buffer = []
    for (review_id, auth, sent) in predictions:
        buffer.append(" ".join([review_id, auth, sent]))

    with open(filename, 'w') as fp:
        fp.write("\n".join(buffer))


def get_clean_text(text: str) -> str:
    """
    remove stop words, pronouns, convert case
    :return:
    """
    unigrams = text.split(" ")
    return " ".join(word.strip().lower() for word in unigrams if word not in stop_words)



# def remove_punctuations(text: str) -> str:
#     symbols = {"?"}
#     unigrams = text.split(" ")
#     clean = []
#     for word in unigrams:
#         clean_word = "".join(letter for letter in list(word) if letter not in symbols)
#         if clean_word not in stop_words:
#             clean.append(clean_word)
#
#     return " ".join(clean)

def identify_negations(text):
    """
    identifies negations in text, replaces "not good" with "not_good"
    :param text: lower case representation of text
    :return:
    """
    negation = False
    unigrams = text.split(" ")
    words = []
    for word in unigrams:
        if negation:
            words.append("not_{}".format(word))
        elif word.lower() in {"not", "n't"}:
            negation = not negation
        elif len(set(word).intersection(set(string.punctuation))) > 0:
            negation = False
    return text + " ".join(words)


def get_ngrams(text, n):
    unigrams = text.split(" ")
    return list(set([" ".join(unigrams[i:i + n]) for i in range(0, len(unigrams))]))


def get_sentiment_word_features(text):
    """
    return words that are considered to be features
    """
    return get_ngrams(get_clean_text(text), 1)


def get_authenticity_word_features(text):
    return get_ngrams(get_clean_text(text), 1)






class NaiveBayesModel:
    def __init__(self, classes):
        self.unknown_word_prob = {}
        self.counter = {}
        self.classes = classes
        self.class_indices = self.build_class_index()
        self.probabilities = {}
        self.classes_prior_counts = defaultdict(int)
        self.class_prior_prob = defaultdict(int)

    def __str__(self):
        return self.counter

    def __repr__(self):
        return self.counter

    def build_class_index(self):
        # {"True": 0, "Fake": 1, "Pos": 2, "Neg": 3}
        return {self.classes[i]: i for i in range(0, len(self.classes))}

    def get_class_index(self, classification):
        return self.class_indices[classification]

    def add_feature(self, word, freq, classification):
        # ToDo: for some reason, removing empty token reduces the accuracy by 2%
        if word in self.counter and classification in self.classes:
            self.counter[word][self.get_class_index(classification)] += freq
        else:
            self.counter[word] = [freq] * len(self.classes)

    def store_model(self):
        model = self.model_repr()
        with open("model.txt", 'w') as fp:
            json.dump(model, fp, indent=1)

    def read_model(self):
        if not self.counter:
            print("non empty model being replaces")
        with open("model.txt", 'r') as fp:
            model = json.load(fp)
            self.builder(model)

    def model_repr(self):
        model = {"prior": self.class_prior_prob,
                 "prob": self.probabilities,
                 "unknown": self.unknown_word_prob}
        return model

    def builder(self, model):
        self.probabilities = model["prob"]
        self.class_prior_prob = model["prior"]
        self.unknown_word_prob = model["unknown"]

    def get_class_total_feq(self, classification):
        return sum([prob[self.get_class_index(classification)]
                    for prob in self.counter.values()])

    def get_classification_freq(self):
        return {classification: self.get_class_total_feq(classification)
                for classification in self.class_indices}

    def compute_feature_probabilities(self):
        classification_freq = self.get_classification_freq()
        for word, counts in self.counter.items():
            probs = []
            for cls, index in self.class_indices.items():
                # probs.append(counts[index])
                # probs.append(classification_freq[cls])
                probs.append(math.log10(counts[index] / classification_freq[cls]))
            self.probabilities[word] = probs

    def add_prior(self, cls):
        self.classes_prior_counts[cls] += 1

    def compute_prior_probabilities(self):
        total = sum(self.classes_prior_counts.values())
        for cls in self.classes:
            confidence = self.classes_prior_counts[cls] / total
            if confidence:
                self.class_prior_prob[cls] = math.log10(confidence)

    def compute_model(self):
        # ToDo: number is hyperparam, test this out
        self.remove_top_words(0)
        self.add_one_smoothing()
        self.compute_feature_probabilities()
        self.compute_prior_probabilities()
        self.smooth_unknown_words()

    def get_class_confidence(self, input_text, feature_function):
        # features = get_word_features(input_text)
        features = feature_function(input_text)
        class_confidence = {}
        for cls, index in self.class_indices.items():
            confidence = self.class_prior_prob[cls]
            for word in features:
                if word in self.probabilities:
                    confidence += self.probabilities[word][index]
                else:
                    confidence += self.unknown_word_prob[cls]
            class_confidence[cls] = confidence
        return class_confidence

    def add_one_smoothing(self):
        for word in self.counter:
            for i in range(0, len(self.classes)):
                self.counter[word][i] += 1

    def smooth_unknown_words(self):
        low_frequent_words = [0]*len(self.classes)
        for word in self.counter:
            if sum(self.counter[word]) < 10:
                low_frequent_words = list(map(add, low_frequent_words, self.counter[word]))

        total = sum(low_frequent_words)
        # print(low_frequent_words)
        for cls, index in self.class_indices.items():
            if low_frequent_words[index]:
                self.unknown_word_prob[cls] = math.log(low_frequent_words[index]/total)
            else:
                self.unknown_word_prob[cls] = 0

    def remove_top_words(self, limit):
        all_words = list(self.counter.items())
        all_words.sort(key = lambda item: sum(item[1]), reverse=True)
        top_words = all_words[:limit+1]
        for (word, _) in top_words:
            del self.counter[word]



def build_sentiment_model(train_data, classes=None):
    """
    returns json of words, and their probabilities
    { "word":[0,0,0,0]} => contains probabilities of word against each class
    :param train_data: [review_id, review_text, True/Fake, Pos/Neg]
    return: NaiveBayesModel, to add probabilities
    """
    model = NaiveBayesModel(classes)
    for (review_id, review_text, authenticity, sentiment) in train_data:
        features = Counter(get_sentiment_word_features(review_text))
        for word, freq in features.items():
            model.add_feature(word, freq, sentiment)
            model.add_prior(sentiment)
    model.compute_model()
    return model

def build_authenticity_model(train_data, classes=None):
    """
    refer to build_sentiment_model, very much similar
    """
    model = NaiveBayesModel(classes)
    for (review_id, review_text, authenticity, sentiment) in train_data:
        features = Counter(get_authenticity_word_features(review_text))
        for word, freq in features.items():
            model.add_feature(word, freq, authenticity)
            model.add_prior(authenticity)
    model.compute_model()
    return model


def nb_learn(train_filename: str, class_name: str, classes: List[str]=None):
    train_data = read_train_data(train_filename)
    if class_name.strip().lower() == "sent":
        nb_model = build_sentiment_model(train_data, classes)
    elif class_name.strip().lower() == "auth":
        nb_model = build_authenticity_model(train_data, classes)
    else:
        raise Exception("no model found")
    # pprint(nb_model.counter)
    # nb_model.store_model()
    return nb_model


def nb_predict_sentiment(model: NaiveBayesModel, input_text: str) -> str:
    cls_confidence = model.get_class_confidence(input_text, get_sentiment_word_features)
    sent_class = get_prediction(cls_confidence, BINARY_SENT_CLASS_GROUP)
    return sent_class


def nb_predict_authenticity(model: NaiveBayesModel, input_text: str) -> str:
    cls_confidence = model.get_class_confidence(input_text, get_authenticity_word_features)
    auth_class = get_prediction(cls_confidence, BINARY_AUTH_CLASS_GROUP)
    return auth_class


def get_prediction(cls_confidence, input_classes):
    input_cls_conf = [(cls_confidence[key], key) for key in input_classes if key in cls_confidence]
    input_cls_conf.sort(reverse=True)
    if input_cls_conf:
        return input_cls_conf[0][1]
    return input_classes[0]


def nb_dev_test_sentiment(model: NaiveBayesModel):
    dev_data = read_dev_data()
    dev_key = read_dev_key_data()
    prediction = []
    gold = []
    for (review_id, review_text) in dev_data:
        sent_class = nb_predict_sentiment(model, review_text)
        prediction.append(sent_class)
        gold.append(dev_key[review_id][1])
    print(prediction)
    print(gold)
    print(get_performance_measure(prediction, gold, "Pos"))
    print(get_performance_measure(prediction, gold, "Neg"))


def nb_dev_test_authenticity(model: NaiveBayesModel):
    dev_data = read_dev_data()
    dev_key = read_dev_key_data()
    prediction = []
    gold = []
    for (review_id, review_text) in dev_data:
        auth_class = nb_predict_authenticity(model, review_text)
        prediction.append(auth_class)
        gold.append(dev_key[review_id][0])
    print(prediction)
    print(gold)
    print(get_performance_measure(prediction, gold, "True"))
    print(get_performance_measure(prediction, gold, "Fake"))


def nb_test(filename: str, sent_model: NaiveBayesModel, auth_model: NaiveBayesModel, output_filename: str):
    test_data = read_test_data(filename)
    predictions = []
    for (review_id, review_text) in test_data:
        sent_class = nb_predict_sentiment(sent_model, review_text)
        auth_class = nb_predict_authenticity(auth_model, review_text)
        predictions.append((review_id, auth_class, sent_class))
    write_predictions(predictions, output_filename)



def store_models(filename: str, sent_model: NaiveBayesModel, auth_model: NaiveBayesModel):
    model = {"sent": sent_model.model_repr(), "auth": auth_model.model_repr()}
    with open(filename, 'w') as fp:
        json.dump(model, fp)


def read_models(filename: str) -> Tuple[NaiveBayesModel, NaiveBayesModel]:
    sent_model = NaiveBayesModel(BINARY_SENT_CLASS_GROUP)
    auth_model = NaiveBayesModel(BINARY_AUTH_CLASS_GROUP)
    with open(filename, 'r') as fp:
        model = json.load(fp)
        sent_model.builder(model["sent"])
        auth_model.builder(model["auth"])
        return sent_model, auth_model




def get_performance_measure(prediction, dev_gold, cls):
    # print(prediction)
    # print(dev_gold)
    total = len(prediction)
    pos_cls_gold = len(list(filter(lambda x: x == cls, dev_gold)))
    pos_cls_pred = len(list(filter(lambda x: x == cls, prediction)))
    true_positive = 0
    for pred, gold in zip(prediction, dev_gold):
        if pred==cls and pred == gold:
            true_positive+=1
    pos_cls_pred = max(1, pos_cls_pred)
    print(cls, true_positive, pos_cls_pred, pos_cls_gold)
    precision = true_positive/pos_cls_pred
    recall = true_positive/pos_cls_gold
    f1 = 2/((1/precision) + (1/recall))
    return precision, recall, f1



if __name__ == '__main__':
    pwd = os.getcwd()
    output_filename = os.path.join(pwd, output_filename)
    test_filename = sys.argv[1]
    sent_model, auth_model = read_models(nbmodel_filename)
    nb_test(test_filename, sent_model=sent_model, auth_model=auth_model, output_filename = output_filename)
