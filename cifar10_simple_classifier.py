import numpy as np

import wisardpkg as wp

from load_cifar10 import load_cifar_10_data
from tools import evaluation, flatten

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
X_train = np.where(train_data > threshold, 1, 0)
X_test = np.where(test_data > threshold, 1, 0)

vehicle_labels = ['0', '1', '8', '9']

vehicles_train = ['1' if label in vehicle_labels else '0' for label in train_labels]
vehicles_test = ['1' if label in vehicle_labels else '0' for label in test_labels]

X_train = [flatten(sample) for sample in X_train]
X_test = [flatten(sample) for sample in X_test]

# ----------------------------------------------------------------------------------------------------------------------
# Setting model

addressSize = 15
ignoreZero = False

model = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=True)

# ----------------------------------------------------------------------------------------------------------------------
# Training
model.train(X_train, train_labels)

# ----------------------------------------------------------------------------------------------------------------------
# Testing
out = model.classify(X_test)

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation
evaluation(out, test_labels)
