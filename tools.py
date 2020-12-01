import numpy as np
import matplotlib.pyplot as plt


def print_images(images, grayscale=False):
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 4
    for i in range(1, columns * rows + 1):
        if (i - 1 < len(images)):
            fig.add_subplot(rows, columns, i)
            if grayscale:
                plt.imshow(images[i - 1], cmap=plt.cm.gray)
            else:
                plt.imshow(images[i - 1])
            plt.axis('off')
    plt.show()


def flatten(char):
    return np.reshape(char, (-1,)).tolist()

def filter_data(data, data_labels, filter_labels):
    filtered_data = []
    filtered_data_labels = []
    for i in range(0, len(data)):
        if str(data_labels[i]) in filter_labels:
            filtered_data.append(data[i])
            filtered_data_labels.append(data_labels[i])
    return filtered_data, filtered_data_labels

def evaluation(estimations, labels):
    hits = 0
    for index in range(0, len(estimations)):
        if str(estimations[index]) == str(labels[index]):
            hits = hits + 1

    acc = float(hits) / len(labels)
    print('Acc: ' + str(acc))
