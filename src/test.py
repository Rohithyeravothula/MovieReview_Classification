from nbclassify import nb_learn, nb_predict, nb_dev_test

def test_prediction():
    model = nb_learn()
    c1, c2 = nb_predict(model, "mouth and began bleeding")
    print(c1, c2)

def dev_test():
    model = nb_learn()
    nb_dev_test(model)


if __name__ == '__main__':
    dev_test()