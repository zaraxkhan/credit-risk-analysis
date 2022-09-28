# Credit-Risk-Analysis
With the use of Python and Machine Learning, I will use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. 

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, I’ll use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Finally, I’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

# Overview
I evaluated three machine learning models by using resampling to determine which is better at predicting credit risk. First, I used the oversampling RandomOverSampler and SMOTE algorithms, and then the undersampling ClusterCentroids algorithm. Using these algorithms, I resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

I also used a combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from the first part of this analysis. Using the SMOTEENN algorithm, I resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

Lastly, I trained and compared two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluate each model. Using both algorithms, I resampled the dataset, viewed the count of the target classes, trained the ensemble classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

# Results 

## Random Oversampling

## SMOTE Oversampling

## Random Undersampling

## SMOTEENN 

## Balanced Random Forest Classifier

## Easy Endemble AdaBoost Classifier

# Summary
