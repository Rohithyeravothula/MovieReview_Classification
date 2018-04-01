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

