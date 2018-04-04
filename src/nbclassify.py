from util import read_train_data, read_test_data, read_dev_key_data, read_dev_data, \
    get_sentiment_word_features, get_authenticity_word_features, write_predictions, \
    dev_data_key_filename, dev_data_filename
from typing import List, Dict, DefaultDict, Set, Tuple
from collections import Counter, defaultdict
import json
import math
from operator import add
import sys


"""
each classification will be of type
[id, text, pos/neg, real/fake]
"""
BINARY_AUTH_CLASS_GROUP = ["True", "Fake"]
BINARY_SENT_CLASS_GROUP = ["Pos", "Neg"]
ASSIGNMENT_CLASSES = BINARY_AUTH_CLASS_GROUP + BINARY_SENT_CLASS_GROUP

class NaiveBayesModel:
    def __init__(self, classes):
        self.unknown_word_prob = {}
        self.counter = {}
        self.classes = classes
        self.class_indices = self.build_class_index()
        self.probabilities = {}
        self.classes_prior_counts = defaultdict(int)
        self.class_prior_prob = defaultdict(float)

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
        # ToDo: items will not return in order
        classification_freq = self.get_classification_freq()
        for word, counts in self.counter.items():
            index_probs = []
            for cls, index in self.class_indices.items():
                index_probs.append((index, math.log10(counts[index] / classification_freq[cls])))
            index_probs.sort()
            self.probabilities[word] = [prob for (index, prob) in index_probs]

    def add_prior(self, cls):
        self.classes_prior_counts[cls] += 1

    def compute_prior_probabilities(self):
        total = sum(self.classes_prior_counts.values())
        for cls in self.classes:
            confidence = self.classes_prior_counts[cls] / total
            if confidence:
                self.class_prior_prob[cls] = math.log10(confidence)

    def compute_model(self, frequency_threshold):
        # ToDo: number is hyperparam, test this out
        self.remove_top_words(0)
        self.remove_low_frequent_words(frequency_threshold)
        self.add_one_smoothing()
        self.compute_feature_probabilities()
        self.compute_prior_probabilities()
        self.smooth_unknown_words()

    def get_class_confidence(self, input_text, feature_function):
        # features = get_word_features(input_text)
        # ToDo: items willnot result in ordered map
        features = feature_function(input_text)
        class_confidence = {}
        for cls, index in self.class_indices.items():
            confidence = self.class_prior_prob[cls]
            for word in features:
                if word in self.probabilities:
                    confidence += self.probabilities[word][index]
                else:
                    # add only unigram probability
                    if len(word.split(" ")) == 1:
                        confidence += self.unknown_word_prob[cls]
                    else:
                        confidence += 0
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
        # ToDo: items will not result in ordered pair
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

    def remove_low_frequent_words(self, frequency):
        words = list(self.counter.keys())
        for key in words:
            if sum(self.counter[key]) < frequency:
                del self.counter[key]



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


def build_model(train_data: List[Tuple[str, str]], classes: List[str], feature_function) -> NaiveBayesModel:
    model = NaiveBayesModel(classes)

    for (review_text, class_name) in train_data:
        features = Counter(feature_function(review_text))
        for (feature, freq) in features.items():
            model.add_feature(feature, freq, class_name)
        model.add_prior(class_name)
    if classes == BINARY_SENT_CLASS_GROUP:
        model.compute_model(3)
    elif classes == BINARY_AUTH_CLASS_GROUP:
        model.compute_model(0)
    return model


def nb_learn_data(train_data):
    train_sentiment = [(review_text, sent) for (_, review_text, _, sent) in train_data]
    train_auth = [(review_text, auth) for (_, review_text, auth, _) in train_data]
    sent_model = build_model(train_sentiment, BINARY_SENT_CLASS_GROUP, get_sentiment_word_features)
    auth_model = build_model(train_auth, BINARY_AUTH_CLASS_GROUP, get_authenticity_word_features)
    return sent_model, auth_model

def nb_learn(train_data_file: str):
    train_data = read_train_data(train_data_file)
    return nb_learn_data(train_data)


def nb_predict(model: NaiveBayesModel, input_text: str, feature_function):
    cls_confidence = model.get_class_confidence(input_text, feature_function)
    classification = get_prediction(cls_confidence, model.classes)
    return classification


def nb_predict_sentiment(model: NaiveBayesModel, input_text: str) -> str:
    return nb_predict(model, input_text, get_sentiment_word_features)


def nb_predict_authenticity(model: NaiveBayesModel, input_text: str) -> str:
    return nb_predict(model, input_text, get_authenticity_word_features)


def get_prediction(cls_confidence, input_classes):
    input_cls_conf = [(cls_confidence[key], key) for key in input_classes if key in cls_confidence]
    input_cls_conf.sort(reverse=True)
    if input_cls_conf:
        return input_cls_conf[0][1]
    return input_classes[0]


def nb_dev_test_sentiment_data(model: NaiveBayesModel, dev_data, dev_key):
    prediction = []
    gold = []
    for (review_id, review_text) in dev_data:
        sent_class = nb_predict_sentiment(model, review_text)
        prediction.append(sent_class)
        gold.append(dev_key[review_id][1])
    # print(prediction)
    # print(gold)
    # print(get_performance_measure(prediction, gold, "Pos"))
    # print(get_performance_measure(prediction, gold, "Neg"))
    return get_performance_measure(prediction, gold, "Pos"), get_performance_measure(prediction, gold, "Neg")


def nb_dev_test_sentiment(model: NaiveBayesModel):
    dev_data = read_dev_data(dev_data_filename)
    dev_key = read_dev_key_data(dev_data_key_filename)
    return nb_dev_test_sentiment_data(model, dev_data, dev_key)


def nb_dev_test_authenticity_data(model: NaiveBayesModel, dev_data, dev_key):
    prediction = []
    gold = []
    for (review_id, review_text) in dev_data:
        auth_class = nb_predict_authenticity(model, review_text)
        prediction.append(auth_class)
        gold.append(dev_key[review_id][0])
    # print(prediction)
    # print(gold)
    # print(get_performance_measure(prediction, gold, "True"))
    # print(get_performance_measure(prediction, gold, "Fake"))
    return get_performance_measure(prediction, gold, "True"), get_performance_measure(prediction, gold, "Fake")


def nb_dev_test_authenticity(model: NaiveBayesModel):
    dev_data = read_dev_data(dev_data_filename)
    dev_key = read_dev_key_data(dev_data_key_filename)
    return nb_dev_test_authenticity_data(model, dev_data, dev_key)

def nb_test_data(test_data, sent_model: NaiveBayesModel, auth_model: NaiveBayesModel, output_filename: str):
    predictions = []
    for (review_id, review_text) in test_data:
        sent_class = nb_predict_sentiment(sent_model, review_text)
        auth_class = nb_predict_authenticity(auth_model, review_text)
        predictions.append((review_id, auth_class, sent_class))
    write_predictions(predictions, output_filename)

def nb_test(filename: str, sent_model: NaiveBayesModel, auth_model: NaiveBayesModel, output_filename: str):
    test_data = read_test_data(filename)
    nb_test_data(test_data, sent_model, auth_model, output_filename)


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
    # print(cls, true_positive, pos_cls_pred, pos_cls_gold)
    precision = true_positive/pos_cls_pred
    recall = true_positive/pos_cls_gold
    f1 = 2/((1/precision) + (1/recall))
    return precision, recall, f1



