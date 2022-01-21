# Classifiers_Comparison

In this project, three algorithms: K Nearest Neighbors, Naïve Bayes and Support Vector Machine are applied to a larger scale data classification task. 

The performance of three algorithms are compared by following analyses:

    Computational Times for both training and testing
    Confusion matrixs
    ROC (Receiver Operating Characteristic) curves

Cross Validation were applied for improving generalization.

## Data source

The data team used is called “banknote authentication Data Set” derived from URL: http://archive.ics.uci.edu/ml/datasets/banknote+authentication. This dataset was extracted from images that were taken from genuine and forged banknote-like specimens. And the variance of Wavelet Transformed image, skewness of Wavelet Transformed image, curtosis of Wavelet Transformed image (continuous) and entropy of image, a total of four features are used to distinguish it’s a real or fake banknote. The output 0 is considered as fake banknote and 1 as a real banknote. The dataset contains 1372 instances and the data is multivariate. 

## Organize data 
For easier cross-validations, the number of data has been cut to 1200, and use shuffle method to shuffle the dataset to make data of two classes mixed together. Then, use KFold method to prepare Four pairs of train-test data pairs for cross-validations. Each training data contains 900 data points and 300 testing data points.


## Result observation and comparison:

### K Nearest Neighbors:
Tests were run for k equals from 5 to 30. The avg number range from 0 to 3.5. When k is 6, it gives the smallest avg number of errors which is 0. The avg number of errors is 1.57.

KNN doesn't have a training process and its total computing time is around 37 milliseconds.

### Naïve Bayes:
In the four tests, the total number of errors is 41, 48, 42 and 58 respectively and the average is 47.25. 

Its computational time for training is around 22 milliseconds and computational time for testing is 7 milliseconds.

### Support Vector Machine:
In the four tests, the total number of errors is 4, 5, 4 and 2 respectively and the average is 3.75. 

Its computational time for training is around 27 milliseconds and computational time for testing is 7 milliseconds.

## Conclusion
Observed from both the results above and the ROC curves, KNN and SVM have relatively high accuracy which is about 99% and Naïve Bayes can only provide 84% accuracy. In terms of total computational time, these three algorithms are very close, all of them need around 33 milliseconds to run one test.