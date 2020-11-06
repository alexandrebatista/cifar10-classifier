import wisardpkg as wp

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)


def flatten(char):
    return np.reshape(char, (-1,)).tolist()


# ----------------------------------------

threshold = 125
X = np.where(X > threshold, 1, 0)
trainingSet, validationSet, testSet = np.split(X, [int(len(X) * 0.8), int(len(X) * 0.9)])
trainingSetY, validationSetY, testSetY = np.split(y, [int(len(y) * 0.8), int(len(y) * 0.9)])

addressSize = 28
ignoreZero = False

model = wp.Wisard(addressSize, ignoreZero=ignoreZero, verbose=False)

flattenTrainingSet = [flatten(number) for number in trainingSet]
flattenValidationSet = [flatten(number) for number in validationSet]
flattenTestSet = [flatten(number) for number in testSet]

model.train(flattenTrainingSet, trainingSetY)
out = model.classify(flattenTestSet)

hits = 0
for index in range(0, len(out)):
    if str(out[index]) == str(testSetY[index]):
        hits = hits + 1

acc = float(hits) / len(testSetY)
print('Acc: ' + str(acc))
