# Credit-Risk-Analysis
With the use of Python and Machine Learning, I will use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. 

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, I’ll oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, I’ll use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, I’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Finally, I’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

# Overview
I evaluated three machine learning models by using resampling to determine which is better at predicting credit risk. First, I used the oversampling RandomOverSampler and SMOTE algorithms, and then the undersampling ClusterCentroids algorithm. Using these algorithms, I resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

I also used a combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from the first part of this analysis. Using the SMOTEENN algorithm, I resampled the dataset, viewed the count of the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

Lastly, I trained and compared two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluate each model. Using both algorithms, I resampled the dataset, viewed the count of the target classes, trained the ensemble classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report.

# Results 
I will be going over the accuracy score, precision score, and recall score for all 6 of the analysis. These scores are important when looking at how accurately our models were at predicting the loan status for each credit risk individual. 

## Random Oversampling
For this first analysis, after resampling the training data, I used that data to train the logistic regression model. From this model;
- the balanced accuracy score was 0.644 or around 64%
 
 ![RO_AS](https://user-images.githubusercontent.com/105755095/192882395-36c68a17-159c-47ae-a1b7-e750d5a737d3.png)

- The high risk presicion rate was only .01 or 1% and the recall was .66 or 66%
- the low risk precision rate was at a whole 1.0 or 100% while the recall was .62 or 62%

 ![RO_PR](https://user-images.githubusercontent.com/105755095/192882827-07ffba10-8a0a-4ee4-b085-6fec20dd9136.png)

This model is able to predict about 64% of the data into the correct group. However looking at our precision and recall matrix, it is clear to see that this model does not do a good job at with the precision of high risk loans. It is only able to accurately predict 1% of the high risk individuals wanting a loan, however it is able to predict all the low risk loans. This means it is deciding that most all the individuals are low risk and putting them into the low risk category. This model correctly classifies about 66% of high risk individuals and 62% of low risk individuals per the recall column. 

## SMOTE Oversampling

## Random Undersampling

## SMOTEENN 

## Balanced Random Forest Classifier

## Easy Endemble AdaBoost Classifier

# Summary
