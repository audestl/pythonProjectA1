import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import csv

for count in range(2):

    if count == 0:
        train_set_name = 'train_1.csv'
        validation_set_name = 'val_1.csv'
        test_set_name = 'test_with_label_1.csv'
        output_file_name = 'Best-MLP-DS1.csv'
        class_size = 26
        validation_size = 239
        dataset_size = 80
    else:
        train_set_name = 'train_2.csv'
        validation_set_name = 'val_2.csv'
        test_set_name = 'test_with_label_2.csv'
        output_file_name = 'Best-MLP-DS2.csv'
        class_size = 10
        validation_size = 1560
        dataset_size = 520

    with open(train_set_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        features = []
        letters = []
        for row in readCSV:
            features.append(row[:-1])
            letters.append(row[-1])

    with open(validation_set_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        features2 = []
        letters2 = []
        for row in readCSV:
            features2.append(row[:-1])
            letters2.append(row[-1])

    with open(test_set_name) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        features3 = []
        letters3 = []
        for row in readCSV:
            features3.append(row[:-1])
            letters3.append(row[-1])

        trainFeatures = np.array(features)
        trainFeatures = trainFeatures.astype(np.float64)
        trainLabels = np.array(letters)
        trainLabels = trainLabels.astype(np.float64)

        validateFeatures = np.array(features2)
        validateFeatures = validateFeatures.astype(np.float64)
        validateLabels = np.array(letters2)
        validateLabels = validateLabels.astype(np.float64)

        testFeatures = np.array(features3)
        testFeatures = testFeatures.astype(np.float64)
        testLabels = np.array(letters3)
        testLabels = testLabels.astype(np.float64)

        hyperParams = {"activation": ['identity', 'logistic', 'tanh', 'relu'],
                       "hidden_layer_sizes": [(30, 50), (10, 10, 10)],
                       "solver": ['adam', 'sgd']}

        # Best MLP
        clf = MLPClassifier()
        clf_cv = GridSearchCV(clf, hyperParams, cv=5, n_jobs=-1)
        clf_cv.fit(trainFeatures, trainLabels)
        print(clf_cv.best_params_)
        print(clf_cv.best_score_)

        prediction = clf_cv.predict(testFeatures)

        # Evaluation
        precision = precision_score(testLabels, prediction, average=None, zero_division=0)
        recall = recall_score(testLabels, prediction, average=None, zero_division=0)
        f1 = f1_score(testLabels, prediction, average=None, zero_division=0)
        accuracy = accuracy_score(testLabels, prediction)
        f1_macro = f1_score(testLabels, prediction, average='macro', zero_division=0)
        f1_weighted = f1_score(testLabels, prediction, average='weighted', zero_division=0)

    # -----------------OUTPUT-----------------------

    # Confusion matrix
    np.savetxt(output_file_name, confusion_matrix(testLabels, prediction), delimiter=",", fmt='%d')

    with open(output_file_name, mode='r+', newline='') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        content = output_file.read()
        # output_file.seek(0,0) (To put pred at beginning of file, but it deletes some part of confusion matrix, to check)

        # Predictions
        output_writer.writerow("")
        output_writer.writerow(["Predictions"])
        for x in range(0, dataset_size):
            output_writer.writerow(((x + 1), int(prediction[x])))

        # Precision, recall and f1
        output_writer.writerow("")
        output_writer.writerow(["Class,Precision,Recall,F1-measure"])
        for x in range(0, class_size):
            output_writer.writerow \
                ((x, np.around(precision[x], 2), np.around(recall[x], 2), np.around(f1[x], 2)))

        # Accuracy, macro-average f1 and weighted-average f1
        output_writer.writerow("")
        output_writer.writerow(("Accuracy", np.around(accuracy, 3)))
        output_writer.writerow("")
        output_writer.writerow(("Macro-average F1-measure", np.around(f1_macro, 3)))
        output_writer.writerow("")
        output_writer.writerow(("Weighted-average F1-measure", np.around(f1_weighted, 3)))
        output_writer.writerow("")
        output_writer.writerow(("Best hyper parameter values: ", clf_cv.best_params_))
