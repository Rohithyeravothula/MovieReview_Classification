from util import *
from collections import Counter, defaultdict
import json
import math

"""
each classification will be of type
[id, text, pos/neg, real/fake]
"""
BINARY_AUTH_CLASS_GROUP = ["True", "Fake"]
BINARY_SENT_CLASS_GROUP = ["Pos", "Neg"]
ASSIGNMENT_CLASSES = BINARY_AUTH_CLASS_GROUP + BINARY_SENT_CLASS_GROUP

class NaiveBayesModel:
    def __init__(self, classes=None):
        if classes is None:
            classes = ASSIGNMENT_CLASSES
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
        if word in self.counter:
            self.counter[word][self.get_class_index(classification)] += freq
        else:
            self.counter[word] = [freq] * 4

    def store_model(self):
        with open("model.txt", 'w') as fp:
            json.dump(self.probabilities, fp, indent=1)

    def read_model(self):
        if not self.counter:
            print("non empty model being replaces")
        with open("model.txt", 'r') as fp:
            self.counter = json.load(fp)

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
        self.add_one_smoothing()
        self.compute_feature_probabilities()
        self.compute_prior_probabilities()

    def get_class_confidence(self, input_text):
        features = get_word_features(input_text)
        class_confidence = {}
        for cls, index in self.class_indices.items():
            confidence = self.class_prior_prob[cls]
            for word in features:
                if word in self.probabilities:
                    confidence += self.probabilities[word][index]
                else:
                    "handle unknown words"
            class_confidence[cls] = confidence
        return class_confidence

    def add_one_smoothing(self):
        for word in self.counter:
            for i in range(0, len(self.classes)):
                self.counter[word][i] += 1


def build_model(train_data):
    """
    returns json of words, and their probabilities
    { "word":[0,0,0,0]} => contains probabilities of word against each class
    :param train_data: [review_id, review_text, True/Fake, Pos/Neg]
    return: NaiveBayesModel, to add probabilities
    """
    model = NaiveBayesModel()
    for (review_id, review_text, authenticity, sentiment) in train_data:
        features = Counter(get_word_features(review_text))
        for word, freq in features.items():
            model.add_feature(word, freq, authenticity)
            model.add_feature(word, freq, sentiment)
            model.add_prior(authenticity)
            model.add_prior(sentiment)
    model.compute_model()
    return model


def compute_error(text_classification):
    pass


def nb_learn():
    train_data = read_train_data()
    nb_model = build_model(train_data)
    # pprint(nb_model.counter)
    nb_model.store_model()
    return nb_model


def nb_predict(model: NaiveBayesModel, input_text):
    cls_confidence = model.get_class_confidence(input_text)
    auth_class_conf = [(cls, conf) for cls, conf in cls_confidence.items()
                       if cls in BINARY_AUTH_CLASS_GROUP]
    auth_class = max(auth_class_conf, key=lambda x: x[1])[0]

    sent_class_conf = [(cls, conf) for cls, conf in cls_confidence.items()
                       if cls in BINARY_SENT_CLASS_GROUP]
    sent_class = max(sent_class_conf, key = lambda x: x[1])[0]

    return (auth_class, sent_class)


def nb_dev_test(model):
    dev_data = read_dev_data()
    dev_key = read_dev_key_data()
    dev_gold = []
    prediction = []
    for (review_id, review_text) in dev_data:
        auth_class, sent_class = nb_predict(model, review_text)
        prediction.append([auth_class, sent_class])
        dev_gold.append(dev_key[review_id])
    # print(prediction)
    # print(list(map(lambda x: [x[1], x[2]], dev_key)))
    helper(prediction, dev_gold)

def helper(predictions, gold_key):
    """lol logic, be careful with this"""
    measuers = []
    f1_scores = []
    for idx, cls in enumerate(ASSIGNMENT_CLASSES):
        true_cls = [p[idx//2] for p in predictions]
        true_gold = [g[idx//2] for g in gold_key]
        precision, recall, f1 = get_performance_measure(true_cls, true_gold, cls)
        measuers.append((cls,  [precision, recall, f1]))
        f1_scores.append(f1)
    pprint(measuers)
    print(sum(f1_scores)/len(f1_scores))


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
    # print(cls, true_positive, pos_cls_pred, pos_cls_gold)
    precision = true_positive/pos_cls_pred
    recall = true_positive/pos_cls_gold
    f1 = 2/((1/precision) + (1/recall))
    return precision, recall, f1
