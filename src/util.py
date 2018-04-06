from typing import List, Dict, Set, Tuple
import string

train_data_filename = "../data/train-labeled.txt"
# train_data_filename = "../data/sample-train-labeled.txt"

dev_data_filename = "../data/dev-text.txt"
dev_data_key_filename = "../data/dev-key.txt"

nbmodel_filename = "nbmodel.txt"
output_filename = "nboutput.txt"

full_filename = "../data/full_data.txt"


punctuations = set(string.punctuation)

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


# stop_words = set(['a', 'able', 'about', 'above', 'across', 'again', "ain't", 'all', 'almost', 'along', 'also', 'am', 'among', 'amongst', 'an', 'and', 'anyhow', 'anyone', 'anyway', 'anyways', 'appear', 'are', 'around', 'as', "a's", 'aside', 'ask', 'asking', 'at', 'away', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'behind', 'below', 'beside', 'besides', 'between', 'beyond', 'both', 'brief', 'but', 'by', 'came', 'can', 'come', 'comes', 'consider', 'considering', 'corresponding', 'could', 'do', 'does', 'doing', 'done', 'down', 'downwards', 'during', 'each', 'edu', 'eg', 'eight', 'either', 'else', 'elsewhere', 'etc', 'even', 'ever', 'every', 'ex', 'few', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'from', 'further', 'furthermore', 'get', 'gets', 'getting', 'given', 'gives', 'go', 'goes', 'going', 'gone', 'got', 'gotten', 'happens', 'has', 'have', 'having', 'he', 'hed', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', "here's", 'hereupon', 'hers', 'herself', "he's", 'hi', 'him', 'himself', 'his', 'how', 'hows', 'i', "i'd", 'ie', 'if', "i'll", "i'm", 'in', 'inc', 'indeed', 'into', 'inward', 'is', 'it', "it'd", "it'll", 'its', "it's", 'itself', "i've", 'keep', 'keeps', 'kept', 'know', 'known', 'knows', 'lately', 'later', 'latter', 'latterly', 'lest', 'let', "let's", 'looking', 'looks', 'ltd', 'may', 'maybe', 'me', 'mean', 'meanwhile', 'might', 'most', 'my', 'myself', 'name', 'namely', 'nd', 'near', 'nearly', 'need', 'needs', 'neither', 'next', 'nine', 'no', 'non', 'now', 'nowhere', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'other', 'others', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'per', 'placed', 'que', 'quite', 're', 'regarding', 'said', 'same', 'saw', 'say', 'saying', 'says', 'second', 'secondly', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sensible', 'sent', 'seven', 'several', 'she', "she'd", "she'll", "she's", 'since', 'six', 'so', 'some', 'somebody', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'specified', 'specify', 'specifying', 'still', 'sub', 'such', 'sup', 'sure', 'take', 'taken', 'tell', 'tends', 'th', 'than', 'that', 'thats', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'theres', "there's", 'thereupon', 'these', 'they', "they'd", "they'll", "they're", "they've", 'think', 'third', 'this', 'those', 'though', 'three', 'through', 'thru', 'thus', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', "t's", 'twice', 'two', 'un', 'under', 'up', 'upon', 'us', 'use', 'used', 'uses', 'using', 'usually', 'value', 'various', 'very', 'via', 'viz', 'vs', 'want', 'wants', 'was', "wasn't", 'way', 'we', "we'd", "we'll", 'went', 'were', "we're", "weren't", "we've", 'what', 'whatever', "what's", 'when', 'whence', 'whenever', "when's", 'where', 'whereafter', 'whereas', 'whereby', 'wherein', "where's", 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', "who's", 'whose', 'why', "why's", 'will', 'willing', 'wish', 'with', 'within', 'without', "won't", 'would', "wouldn't", 'yes', 'yet', 'you', "you'd", "you'll", 'your', "you're", 'yours', 'yourself', 'yourselves', "you've"])



def pprint(collection):
    if isinstance(collection, dict):
        for line in collection.items():
            print(line)
    else:
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
    result = text + " ".join(words)
    print(text)
    print(result)


def get_ngrams(text, n):
    unigrams = text.split(" ")
    return list(set([" ".join(unigrams[i:i + n]) for i in range(0, len(unigrams))]))


def get_sentiment_word_features(text):
    """
    return words that are considered to be features
    # no change with removing numeric characters

    lower case 20 iter => 93.6
    without lower case => 93.7
    """
    unigrams = text.split(" ")

    bigrams = []
    sentences = text.split(".")
    for sent in sentences:
        sent_unigrams = sent.split(" ")
        for (u1, u2) in zip(sent_unigrams, sent_unigrams[1:]):
            if u1 not in stop_words and u2 not in stop_words:
                bigrams.append("{} {}".format(u1, u2))

    stop_less = []
    for word in unigrams:
        if word.lower().strip() not in stop_words:
            stop_less.append(word.lower())

    stop_less = " ".join(stop_less)

    negative_words = []
    negation = False
    for word in unigrams:
        if negation:
            negative_words.append("not_{}".format(word))
        elif word.lower() in {"not", "n't"}:
            negation = not negation
        elif len(set(word).intersection(set(string.punctuation))) > 0:
            negation = False
    # print(negative_words)

    return get_ngrams(stop_less, 1) + bigrams


def get_authenticity_word_features(text):
    # using lower case decreases the f1 score
    # no bigrams as well

    unigrams = text.split(" ")
    stop_less = []
    for word in unigrams:
        if word.lower().strip() not in stop_words:
            stop_less.append(word.strip().lower())
    stop_less = " ".join(stop_less)

    # stop_less = " ".join(
    #     word.strip().lower() for word in unigrams if word not in stop_words)
    return get_ngrams(stop_less, 1)


