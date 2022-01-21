import math
import statistics
import numpy
import scikitplot as skplt
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def knn(X_train,X_test,y_train,y_test, k):
    start_time = time.time()

    X_train = numpy.array(X_train)
    X_test = numpy.array(X_test)
    y_train = numpy.array(y_train)
    y_test = numpy.array(y_test)

    y_pred = []
    y_prob = []
    num_test = len(X_test)
    dist = numpy.zeros(num_test)
    errors = numpy.zeros(num_test)
    for i in range(num_test):
        dist = numpy.sum((X_train - X_test[i, :]) ** 2, axis = 1) ** 0.5
        sortIndex = numpy.argsort(dist)
        bestLabels = y_train[sortIndex[0:k]]
        y_prob.append([1-sum(bestLabels)/k, sum(bestLabels)/k])
        prediction = (sum(bestLabels) > k / 2.0) * 1.0
        y_pred.append(prediction)
        errors[i] = (y_test[i] != prediction) * 1.0


    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    num_errors = cm[0][1] + cm[1][0]
    print("total errors = ",num_errors )

    print("---KNN Computational Time for testing time used:  %.3f milliseconds ---" % ((time.time() - start_time) * 1000))

    # detil report
    print(classification_report(y_test, y_pred))

    skplt.metrics.plot_roc(y_pred, y_prob)
    plt.show()

    return num_errors


def native_bayes(data_train,data_test, y_test):
    start_time = time.time()
    # Split the dataset by class values, returns a dictionary
    def separate_by_class(dataset):
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated

    # Calculate the mean, stdev and count for each column in a dataset
    def summarize_dataset(dataset):
        summaries = [(statistics.mean(column), statistics.stdev(column), len(column)) for column in zip(*dataset)]
        del (summaries[-1])
        return summaries

    # Split dataset by class then calculate statistics for each row
    def summarize_by_class(dataset):
        separated = separate_by_class(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = summarize_dataset(rows)
        return summaries

    # Calculate the Gaussian probability distribution function for x
    def calculate_probability(x, mean, stdev):
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    # Calculate the probabilities of predicting each class for a given row
    def calculate_class_probabilities(summaries, row):
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
        return probabilities

    def predict(summaries, row):
        probabilities = calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return [best_label,  [probabilities[0], probabilities[1]]]



    summaries = summarize_by_class(data_train)
    print("---Native Bayes  Computational Time for training:  %.3f milliseconds ---" % ((time.time() - start_time)*1000))

    start_time = time.time()
    y_pred = []
    y_prob = []
    for i in range(len(data_test)):
        prediction = predict(summaries, data_test[i])[0]
        probs = predict(summaries, data_test[i])[1]
        y_pred.append(prediction)
        y_prob.append(probs)

    # confusion matrix (true positive/ false positive/ 2*2)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    num_errors = cm[0][1] + cm[1][0]
    print("total errors = ", num_errors)

    print("---Native Bayes Computational Time for testing:  %.3f milliseconds ---" % ((time.time() - start_time) * 1000))

    # detil report
    print(classification_report(y_test, y_pred))

    skplt.metrics.plot_roc(y_pred, y_prob)
    plt.show()

    return num_errors


def svm(X_train,X_test,y_train,y_test):
    start_time = time.time()

    X_train = numpy.array(X_train)
    X_test = numpy.array(X_test)
    y_train = numpy.array(y_train)
    y_test = numpy.array(y_test)

    #use liner kernel
    svclassifier = SVC(kernel='linear',probability=True)
    svclassifier.fit(X_train, y_train)

    print("---SVM Computational Time for training:  %.3f milliseconds ---" % ((time.time() - start_time)*1000))

    # start counting time
    start_time = time.time()

    y_pred = svclassifier.predict(X_test)

    y_prob = svclassifier.predict_proba(X_test)

    # confusion matrix (true positive/ false positive/ 2*2)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    num_errors = cm[0][1] + cm[1][0]
    print("total errors = ", num_errors)

    print("---SVM Computational Time for testing:  %.3f milliseconds ---" % ((time.time() - start_time)*1000))

    # detil report
    print(classification_report(y_test,y_pred))


    skplt.metrics.plot_roc(y_pred, y_prob)
    plt.show()
    return num_errors



def read_data():
    fname = './banknotes.txt'
    file = open(fname, 'r')
    data = []
    label = []
    data_with_label = []

    for line in file:
        v = line.split(',')[0]
        w = line.split(',')[1]
        x = line.split(',')[2]
        y = line.split(',')[3]
        z = line.split(',')[4].rstrip('\n')

        data.append([float(v), float(w), float(x), float(y)])
        label.append(int(z))
        data_with_label.append([float(v), float(w), float(x), float(y), int(z)])

    data_package = [data, label, data_with_label]
    return data_package

# split data into four sections, prepare four train-test pairs, for cross validations
def prepare_train_test_dataset_pair(data):
    kf4 = KFold(n_splits=4, shuffle=False)
    dataset_pair = []
    for train_index, test_index in kf4.split(data):
        data_train = [data[i] for i in train_index]
        data_test = [data[i] for i in test_index]
        dataset_pair.append([data_train, data_test])

    return dataset_pair



def knn_main(data, k):
    print("\n///////////////////////////////////////////////////////////////////////////////////////\n"
          "/////////////////////////////////        KNN, k = {}      ///////////////////////////////\n"
          "///////////////////////////////////////////////////////////////////////////////////////\n".format(k))
    X_pairs = prepare_train_test_dataset_pair(data[0])
    y_pairs = prepare_train_test_dataset_pair(data[1])

    knn_errors = []
    for i in range(len(X_pairs)):
        print("------------------------------------KNN (k = {})  test #{}------------------------------------".format(k,
                                                                                                                      i + 1))

        result = knn(X_pairs[i][0], X_pairs[i][1], y_pairs[i][0], y_pairs[i][1], k)
        knn_errors.append(result)
    avg_errors = np.average(knn_errors)

    print("----- KNN, k = {}, Average number of errors from cross validations = {}".format(k, avg_errors))
    return avg_errors

def native_bayes_main(data):
    print("\n///////////////////////////////////////////////////////////////////////////////////////\n"
          "////////////////////////////        Native Bayes        //////////////////////////////\n"
          "///////////////////////////////////////////////////////////////////////////////////////\n")

    y_pairs = prepare_train_test_dataset_pair(data[1])
    data_pairs = prepare_train_test_dataset_pair(data[2])

    nb_errors = []
    for i in range(len(y_pairs)):
        print("------------------------------------Native Bayes  test #{}------------------------------------".format(i + 1))

        result = native_bayes(data_pairs[i][0], data_pairs[i][1] , y_pairs[i][1])
        nb_errors.append(result)

    print("----- Native Bayes  Average number of errors from cross validations = ", np.average(nb_errors))
    return nb_errors

def svm_main(data):
    print("\n///////////////////////////////////////////////////////////////////////////////////////\n"
          "/////////////////////////////////        SVM        ///////////////////////////////////\n"
          "///////////////////////////////////////////////////////////////////////////////////////\n")
    X_pairs = prepare_train_test_dataset_pair(data[0])
    y_pairs = prepare_train_test_dataset_pair(data[1])

    svm_errors = []
    for i in range(len(X_pairs)):
        print("------------------------------------SVM test #{}------------------------------------".format(i + 1))

        result = svm(X_pairs[i][0], X_pairs[i][1], y_pairs[i][0], y_pairs[i][1])
        svm_errors.append(result)

    print("----- SVM Average number of errors from cross validations = ", np.average(svm_errors))
    return svm_errors


def main():
    dt = read_data()
    # cut number of data to 1200, for easier cross validation
    for i in range(3):
        dt[i] = dt[i][:1200]
    # shuffle the dataset to make data of two classes mixed together
    dt[0], dt[1], dt[2] = shuffle(dt[0], dt[1], dt[2])

    #KNN
    avg_errors = dict()
    for k in range(5, 31):
        avg_error = knn_main(dt, k)
        avg_errors[k] = avg_error

    print("k and it's avg number of errors:", avg_errors)

    k_smallest_avg_errors = min(avg_errors, key=avg_errors.get)
    print("----------    When k = {}, KNN has the smallest avg number of errors = {}    ----------".format(
        k_smallest_avg_errors, avg_errors[k_smallest_avg_errors]))

    #Native Bayes
    native_bayes_main(dt)

    #SVM
    svm_main(dt)

    return


if __name__ == "__main__":
    main()

