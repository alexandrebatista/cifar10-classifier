import numpy as np

import wisardpkg as wp

from load_cifar10 import load_cifar_10_data


def flatten(char):
    return np.reshape(char, (-1,)).tolist()


# ----------------------------------------------------------------------------------------------------------------------
# Load data

cifar_10_dir = 'cifar-10-batches-py'

train_data, train_labels, test_data, test_labels, label_names = load_cifar_10_data(cifar_10_dir)

# ----------------------------------------------------------------------------------------------------------------------
# Preprocessing data

# Converting labels from int to string
train_labels = [str(label) for label in train_labels]
test_labels = [str(label) for label in test_labels]

threshold = 115
X = np.where(train_data > threshold, 1, 0)
y = np.where(test_data > threshold, 1, 0)
# trainingSet, validationSet, testSet = np.split(X, [int(len(X) * 0.8), int(len(X) * 0.9)])
# trainingSetY, validationSetY, testSetY = np.split(y, [int(len(y) * 0.8), int(len(y) * 0.9)])

X = [flatten(sample) for sample in X]
y = [flatten(sample) for sample in y]

# ----------------------------------------------------------------------------------------------------------------------
# Setting model

addressSize = 15
ignoreZero = False

model = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=True)

# ----------------------------------------------------------------------------------------------------------------------
# Training
model.train(X, train_labels)

# ----------------------------------------------------------------------------------------------------------------------
# Testing
out = model.classify(y)

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation
hits = 0
for index in range(0, len(out)):
    if str(out[index]) == str(test_labels[index]):
        hits = hits + 1

acc = float(hits) / len(test_labels)
print('Acc: ' + str(acc))
