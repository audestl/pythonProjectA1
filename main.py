

import sklearn
import numpy as np
from sklearn import datasets, svm, metrics

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv

#
#
# dataset = np.loadtxt("train_1.csv", delimiter=",")


# PLOTTING
# character = []
# characterClass = [0]*26
# value = [0]*26
# for cell in range(len(value)):
#     value[cell] = cell
#
#
# for cell in range(len(dataset)):
#     character.append(dataset[cell][-1])
#
# for cell in range(len(character)):
#     num = int(character[cell])
#     characterClass[num] = characterClass[num] + 1.0
#
#
# plt.plot(value, characterClass, "ob")
# plt.ylabel('Number of occurrence of the letter')
# plt.xlabel('All the letters indices')
# plt.show()


#Gaussian Naive Bayes model


with open('train_1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    features = []
    letters = []
    for row in readCSV:
        features.append(row[:-1])
        letters.append(row[-1])
    # for row in features:
    #     print(row)
    # print("List of outcomes: ", letters)

with open('test_with_label_1.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    features2 = []
    letters2 = []
    for row in readCSV:
        features2.append(row[:-1])
        letters2.append(row[-1])
    # for row in features2:
    #     print(row)
    # print("List of outcomes: ", letters2)

    npFeatures = np.array(features)
    npFeatures = npFeatures.astype(np.float64)
    npLetters = np.array(letters)
    npLetters = npLetters.astype(np.float64)

    npFeatures2 = np.array(features2)
    npFeatures2 = npFeatures2.astype(np.float64)

    clf = GaussianNB()
    clf.fit(npFeatures, npLetters)
    y_pred = clf.predict(npFeatures2)
    print(y_pred)









