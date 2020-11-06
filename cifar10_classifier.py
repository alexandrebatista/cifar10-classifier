import numpy as np

import wisardpkg as wp

from load_cifar10 import load_cifar_10_data


def evaluation(estimations, labels):
    hits = 0
    for index in range(0, len(estimations)):
        if str(estimations[index]) == str(labels[index]):
            hits = hits + 1

    acc = float(hits) / len(labels)
    print('Acc: ' + str(acc))


def array_filter(array, value):
    result = []
    for i in range(0, len(array)):
        if array[i] == value:
            result[i] = 1
        else:
            result = -1
    return result


class Net:
    def __init__(self, address_size, classes_number):
        ignoreZero = False
        self.models = []
        for i in range(0, classes_number):
            self.models[i] = wp.Wisard(address_size, ignoreZero=ignoreZero, verbose=True, returnConfidence=True)

    def train(self, x, y):
        for i in range(0, len(self.models)):
            filtered_y = array_filter(y, str(i))
            self.models[i].train(x, filtered_y)

    def classify(self, x):
        models_quantity = len(self.models)
        samples_quantity = len(x)
        outcome = np.empty([models_quantity, samples_quantity], dtype=int)
        for i in range(0, len(self.models)):
            outcome = np.append(outcome, [self.models[i].classify(x)])

        return outcome


def flatten(char):
    return np.reshape(char, (-1,)).tolist()


# ----------------------------------------------------------------------------------------------------------------------
# Load data
cifar_10_dir = 'cifar-10-batches-py'

train_data, train_labels, test_data, test_labels, label_names = load_cifar_10_data(cifar_10_dir)

# ----------------------------------------------------------------------------------------------------------------------
# Preprocessing data

# Converting labels from int to string
y_train = [str(label) for label in train_labels]
y_test = [str(label) for label in test_labels]

threshold = 125
X_train = np.where(train_data > threshold, 1, 0)
X_test = np.where(test_data > threshold, 1, 0)

X_train = [flatten(sample) for sample in X_train]
X_test = [flatten(sample) for sample in X_test]

# ----------------------------------------------------------------------------------------------------------------------
# Setting model
net = Net(28, len(label_names))

# ----------------------------------------------------------------------------------------------------------------------
# Training
net.train(X_train, y_train)

# ----------------------------------------------------------------------------------------------------------------------
# Testing
out = net.classify(X_test)

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation
evaluation(out, y_test)
