

import sklearn
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
import csv
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import GridSearchCV
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
    print("List of outcomes: ", letters2)

    trainFeatures = np.array(features)
    trainFeatures = trainFeatures.astype(np.float64)
    trainLabels = np.array(letters)
    trainLabels = trainLabels.astype(np.float64)

    testFeatures = np.array(features2)
    testFeatures = testFeatures.astype(np.float64)
    testLabels = np.array(letters2)
    testLabels = testLabels.astype(np.float64)

    # Base Decision Tree Experiment
    clf = tree.DecisionTreeClassifier()
    clf.fit(trainFeatures, trainLabels)
    prediction = clf.predict(testFeatures)
    print("Base DT applied:")
    print(prediction)

    # Best Decision Tree Experiment
    clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=10, min_impurity_split=3, class_weight="balanced")
    clf.fit(trainFeatures, trainLabels)
    prediction = clf.predict(testFeatures)
    print("Best DT applied:")
    print(prediction)

    # precision = precision_score(testLabels, prediction, average =None)
    # recall = recall_score(testLabels, prediction, average =None)
    # f1 = f1_score(testLabels, prediction, average =None)
    # accuracy = accuracy_score(testLabels, prediction)
    # f1_macro = f1_score(testLabels, prediction, average ='macro')
    # f1_weighted = f1_score(testLabels, prediction, average ='weighted')

    # Best Decision Tree


# -----------------OUTPUT-----------------------
#
# # Confusion matrix
# np.savetxt("Base-DT-DS1.csv", confusion_matrix(testLabels, prediction), delimiter=",", fmt='%d')
# # print(testLabels)
#
# with open('Base-DT-DS1.csv', mode='r+', newline='') as output_file:
#     output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     content = output_file.read()
#     # output_file.seek(0,0) (To put pred at beginning of file, but it deletes some part of confusion matrix, to check)
#
#     # Predictions
#     output_writer.writerow("")
#     output_writer.writerow(["Predictions"])
#     for x in range(1, 81):
#         output_writer.writerow((x, int(prediction[x-1])))
#
#     # Precision, recall and f1
#     output_writer.writerow("")
#     output_writer.writerow(["Class,Precision,Recall,F1-measure"])
#     for x in range(1, 27):
#         output_writer.writerow((x, np.around(precision[x - 1], 2), np.around(recall[x - 1],2 ), np.around(f1[x - 1], 2)))
#
#     # Accuracy, macro-average f1 and weighted-average f1
#     output_writer.writerow("")
#     output_writer.writerow(("Accuracy", np.around(accuracy, 3)))
#     output_writer.writerow("")
#     output_writer.writerow(("Macro-average F1-measure", np.around(f1_macro, 3)))
#     output_writer.writerow("")
#     output_writer.writerow(("Weighted-average F1-measure", np.around(f1_weighted, 3)))
#












