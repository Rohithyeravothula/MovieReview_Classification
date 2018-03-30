from nbclassify import nb_learn, nb_predict, nb_dev_test
from util import pprint


def test_prediction():
    model = nb_learn()
    c1, c2 = nb_predict(model, "mouth and began bleeding")
    print(c1, c2)

def dev_test():
    model = nb_learn()
    nb_dev_test(model)


def experiment():
    model = nb_learn()
    words = []
    for word in model.counter:
        words.append([word, sum(model.counter[word])])
    words.sort(key=lambda x:x[1], reverse=True)
    pprint(words)


if __name__ == '__main__':
    # experiment()
    dev_test()
