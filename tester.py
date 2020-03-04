import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from Network import Net
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits


########################
#   START TESTING
#
def main():
    # hyperparams
    epochs = 1000
    lr = 0.02
    ls = 128
    size = 1000

    # collect data and split with sklearn
    # data types: "moons", "multi", "diabetes", "digit"
    data_type = "digit"
    features, labels = get_data(size, data_type)
    one_hot_target = pd.get_dummies(labels)
    train_x, x_val, train_y, y_val = train_test_split(features, one_hot_target, test_size=0.1, random_state=20)
    train_y = np.array(train_y)
    y_val = np.array(y_val)

    # training
    model = Net(train_x, train_y, epochs, ls, lr)
    model.train()

    # testing
    if data_type != "digit":
        plt.subplot(2,1,1)
        plt.title('Training Batch')
    print("Training accuracy: ", test(model, train_x, train_y, data_type))
    if data_type != "digit":
        plt.tight_layout(pad=3.0)
        plt.subplot(2,1,2)
    print("Test accuracy: ", test(model, x_val, np.array(y_val), data_type))
    if data_type != "digit":
        plt.title('Testing Batch')
        plt.show()
#
#   END TESTING
#########################


########################
#   START DATA OPTIONS
#
def get_data(size, data_type):
    if data_type == "moons":
        feature_set, labels = datasets.make_moons(size, noise=0.1)
        labels = labels.reshape(size, 1)
        labels = [ls[0] for ls in labels]
        labels = np.array(labels)

    elif data_type == "digit":
        dig = load_digits()
        feature_set = dig.data
        labels = dig.target

    elif data_type == "multi":
        class1 = np.random.randn(700, 2) + np.array([0, -3])
        class2 = np.random.randn(700, 2) + np.array([3, 3])
        class3 = np.random.randn(700, 2) + np.array([-3, 3])
        feature_set = np.vstack([class1, class2, class3])
        labels = np.array([0] * 700 + [1] * 700 + [2] * 700)

    elif data_type == "diabetes":
        feature_set, labels = datasets.load_diabetes(return_X_y=True)
        labels = labels.reshape(len(feature_set), 1)
        labels = [label[0] for label in labels]
        labels = np.array(labels)

    else:
        print("Invalid datatype")
        exit(0)

    return feature_set, labels
#
#   END DATA OPTIONS
#########################


########################
#   START TEST DISPLAY OPTIONS
#
def test(model, x, y, data_type):
    acc = 0
    ind = 0
    for xx, yy in zip(x, y):
        s = model.predict(xx)
        if data_type == "moons":
            if s == np.argmax(yy):
                if s > 0.5:
                    plt.scatter(xx[0], xx[1], c='b')
                else:
                    plt.scatter(xx[0], xx[1], c='g')
                acc += 1
            else:
                plt.scatter(xx[0], xx[1], c='r')

        elif data_type == "digit":
            if s == np.argmax(yy):
                acc += 1

        elif data_type == "diabetes":
            ind += 1
            if s == np.argmax(yy):
                plt.scatter(ind, s, c='b')
                acc += 1
            else:
                plt.scatter(ind, s, c='r')
        elif data_type == "multi":
            if s == np.argmax(yy):
                if s == 0:
                    plt.scatter(xx[0], xx[1], c='b')
                elif s == 1:
                    plt.scatter(xx[0], xx[1], c='g')
                elif s == 2:
                    plt.scatter(xx[0], xx[1], c='y')
                acc += 1
            else:
                plt.scatter(xx[0], xx[1], c='r')

    return acc / len(x) * 100
#
#   END TEST DISPLAY OPTIONS
########################


main()
