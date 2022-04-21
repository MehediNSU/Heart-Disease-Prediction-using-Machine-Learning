# Heart-Disease-Prediction-using-Machine-Learning

## Introduction

The term "heart disease may refer to numerous types of heart conditions. The most common type is coronary heart disease, which can affect the blood flow to the heart, which in turn can cause a heart attack. Sometimes, heart diseases may be "silent", and not diagnosed until a person experiences signs or symptoms of a heart attack, heart failure, or an arrhythmia.

Heart disease continues to remain the leading cause of death globally, according to the annual heart disease and stroke statistics update from the American Heart Association. In 2019, nearly 18.6 million people worldwide died from cardiovascular diseases(CVD) according to the update, which is a 17.1\% increase since the past decade. In addition to that, the researchers wrote that 523.2 million people were diagnosed with CVD in 2019, which is an increase of 26.6\% from 2010.

The main problem of heart diseases is that it cannot be cured and neither reversed, and requires treatment and careful monitoring for an extensive period of time. As a result, early prediction is very crucial. Once a person is diagnosed with the symptoms of a heart disease, it can treated with medications, procedures and lifestyle changes.

In recent times, machine learning has been found to be very fruitful in detecting different types of diseases in general, as well as heart diseases. For my research, I applied several machine learning classifiers and techniques for the prediction of heart diseases. Dependable prediction of heart diseases will be critical in detection of the disease at an early stage. The dataset which I have used for the training and testing purposes has been obtained from Jono Seba Medical, Dinajpur. Then I applied many available python ML frameworks for data visualization as well as exploratory data analysis(EDA). Then, I went through the phase of pre-processing, which involved the scaling of different features using min-max scaler as well as the transformation of the categorical values in the dataset to numerical values. After that I went through the process of feature selection to select the features of greater importance, and discarding the lesser important features. The next step involved the splitting of the dataset into train and test sets, into percentages of 80% and 20% respectively. My research involved the application of total six different classifiers on the train set. I assessed each of these classifiers using four different evaluation metrics- accuracy, precision, recall and f-1 score.

## Methodology
Figure 1 illustrates the methodology I employed in my study

<img src="https://raw.githubusercontent.com/Maaher01/Heart-Disease-Prediction-using-Machine-Learning/main/Pictures/MethodologyDiagram.png" width="700" height="300" />

The dataset consisted of a total of 226 instances and 17 features. The relationship among the features in the dataset was computed through a heatmap using Pearson's correlation, shown in figure 2. The values inside of a heatmap ranges from -1 to +1. The closer the value is compared to 0, the lesser linearly dependency among the two variables. Values closer to 1 indicate significant positive connection. On the other hand, values closer to -1 illustrates that the variables are strongly negatively connected.

<img src="https://raw.githubusercontent.com/Maaher01/Heart-Disease-Prediction-using-Machine-Learning/main/Pictures/Heatmap.png" width="700" height="600" />

### I. Preprocessing

To convert raw data into a format appropriate for machine learning tasks, preprocessing is very important. All the categorical data were converted into numerical data. In the end, feature scaling was used on all the numeric attributes, so all of their values lie between 0 and 1. I used min-max scaler for feature scaling, which is represented by the following equation:

$$X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

$$x = \frac{\sum{x_{i}}}{n}$$

## Machine Learning Algorithm application

In my research, I applied a total of six different classifiers. They are as follows:

### ZeroR Classifier

ZeroR classifier is the simplemost classifier algorithm, which depends on the target, and ignores all other predictors. In other words, ZeroR classifier simply predicts the majority class. Though, it is of very low predictable power, it is useful in finding out a baseline performance as a benchmark for other classification methods.

### K-Nearest Neighbour

K-Nearest neighbour(KNN) classifier determines the distances of each instance from the query instance. With regard to this, the classifier picks out the k nearest instances. Each of the instances vote for the output. Using the voting method, the predicted output of the query instance is found and returned. Both Euclidean and Manhattan distance metric were used as means of calculating the distance.

### Decision Tree Classifier

The main motive of a Decision Tree is to develop a model that predicts classes or values of target variables using a decision tree, drawn from the training set of the data. This approach splits the dataset into divisions by starting with the most suitable attribute. The splitting procedure is repeated until all of the divisions are homogenous. In my research, the highest accuracy was obtained using Decision Tree Classifier.

### Support Vector Machine

This is a supervised ML algorithm which inspects data for classification and regression problems. This algorithm is vastly used because of the superior accuracy it usually yields

### Logistic Regression

Logistic regression is a common classifier for solving binary classification problems. It can also be extended to solve multi-class classification problems. Logistic regression uses the idea of linear regression, and applies sigmoid function on top of it. In my research, Logistic Regression Classifier, also yielded superior results.

### Multinomial Naive Bayes

Multinomial Naive Bayes algorithm is a collection of many algorithms, all of which share one common principle, which is, each feature being classified, is not related to any other feature. The presence or absence of a feature does not affect the presence or absence of another feature. It is a probabilistic learning method.

## Experimental Results

After the pre-processing phase, the classifiers mentioned above were used to calculate the best model for heart disease prediction. The performance of the classifier was evaluated using the four evaluation metrics mentioned below:

Considering,

TP = True Positive

TN = True Negative

FP = False Positive

FN = False Negative

### Accuracy

The number of times the appropriate estimates are made is known as accuracy. Accuracy can be calculated using the subsequent equation:

$$accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

### Precision

Precision refers to how accurately the model predicts positive values among all positive values predicted by the model.

$$precision = \frac{TP}{TP + FP}$$

### Recall

Recall can be used to measure the degree to which the model detects true positives. On the other hand, if the value of the recall is low, it shows that the model has faced a lot of false negatives.

$$recall = \frac{TP}{TP + FN}$$

### F-1 Score

The harmonic mean of the precision and recall values is called the f1-score. It can be calculated using the following equation:

$$F_{1} = 2 * \frac{precision}{precision + recall}$$

Table 1 shows the obtained results after applying the algorithms.

<img src="https://raw.githubusercontent.com/Maaher01/Heart-Disease-Prediction-using-Machine-Learning/main/Pictures/table.png" width="500" height="300" />

## Conclusion

Heart diseases cannot be cured and neither reversed, and as a result, early prediction is very crucial. The main aim of this research was to find a dependable and precise way of predicting heart diseases. We used a total of six different classifiers in our research. The achieved results are very much reliable and accurate. Between them, both the Decision Tree classifier and Logistic Regression accomplished the best accuracy, though Logistic Regression achieved superior results in the three other metrics (precision, recall and f1-score).






