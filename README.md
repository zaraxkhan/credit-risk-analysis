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

- the high risk presicion rate was only .01 or 1% and the recall was .66 or 66%
- the low risk precision rate was at a whole 1.0 or 100% while the recall was .62 or 62%

 ![RO_PR](https://user-images.githubusercontent.com/105755095/192882827-07ffba10-8a0a-4ee4-b085-6fec20dd9136.png)

This model is able to predict about 64% of the data into the correct group. However looking at our precision and recall matrix, it is clear to see that this model does not do a good job at with the precision of high risk loans. It is only able to accurately predict 1% of the high risk individuals wanting a loan, however it is able to predict all the low risk loans. This means it is deciding that most all the individuals are low risk and putting them into the low risk category and also adding many low risk individuals to the high risk category. This model correctly classifies about 66% of high risk individuals and 62% of low risk individuals per the recall column. 

## SMOTE Oversampling
This second analysis used SMOTE Oversampling when training the data.
- the accuracy score was .651 or 65%

 ![S_AS](https://user-images.githubusercontent.com/105755095/192886272-597e5fdf-2b80-4202-a709-92ae4c4e8d15.png)

- the high risk precision score was .01 or 1% and recall was .61 or 61%
- the low risk precision score was 1.00 or 100% and the recall was .69 or 69%

 ![S_PR](https://user-images.githubusercontent.com/105755095/192886309-af89d215-85aa-4bb6-b5b6-fb7d63c63346.png)

This model is able to predict 65% of individuals into the correct loan status group. However, just like the last model, this model does not do too good with identifying high risk individuals. Looking at the high risk precision model with a score of 1%, this model is only correctly identifying 1% of the high risk individuals while correctly identifying 100% of the low risk individuals. This means just like the last model, this model is also predicting most all the data points to be low risk individuals and also adding many low risk individuals to the high risk category. Looking at the recall column, only 61% of the high risk individuals were actually high risk while 69% of low risk indiviudals were actually low risk.

## Random Undersampling
The third analysis undersampled the data points when training. In this model;
- the accuracy score was .574 or 57%
 
 ![U_AS](https://user-images.githubusercontent.com/105755095/192887574-642dbc09-e7cc-4075-9751-9c2199297f54.png)

- the high risk precision score was .01 or 1% and the recall score was .62 or 62%
- the low risk precision score was 1.00 or 100% and the recall score was .53 or 53%
 
 ![U_PR](https://user-images.githubusercontent.com/105755095/192887598-c932b30a-d3fd-47e3-a297-77021831e911.png)

This model is able to predict 57% of individuals into the correct loan status group. However, just like the last model, this model does not do too good with identifying high risk individuals. Looking at the high risk precision model with a score of 1%, this model is only correctly identifying 1% of the high risk individuals while correctly identifying 100% of the low risk individuals. This means just like the last model, this model is also predicting most all the data points to be low risk individuals and also adding many low risk individuals to the high risk category. Looking at the recall column, only 62% of the high risk individuals were actually high risk while only 53% of low risk indiviudals were actually low risk.

## SMOTEENN 
This model was a combination of over sampling and undersampling
- the accuracy score was .644 or 64%
 
 ![SM_AS](https://user-images.githubusercontent.com/105755095/192888206-cb6565ff-42ad-40df-8759-0c67fbc324b1.png)

- the high risk precision score was .01 or 1% and the recall was .72 or 72%
- the low risk precision score was 1.00 or 100% and the recall was .57 or 57%
 
 ![SM_PR](https://user-images.githubusercontent.com/105755095/192888226-c6685605-5dc3-4f3c-8267-317e97779948.png)
 
This model is able to predict 64% of individuals into the correct loan status group. However, just like the models before, this model does not do too good with identifying high risk individuals. Looking at the high risk precision model with a score of 1%, this model is only correctly identifying 1% of the high risk individuals while correctly identifying 100% of the low risk individuals. This means just like the last model, this model is also predicting most all the data points to be low risk individuals and also adding many low risk individuals to the high risk category. Looking at the recall column, about 72% of the high risk individuals were actually high risk while only 57% of low risk indiviudals were actually low risk. This model is able to identify high risk individuals more accurately than the rest of the models. 

## Balanced Random Forest Classifier
This ensemble algorithm had the following scores;
- the accuracy score was .788 or 79%

 ![BRF_AS](https://user-images.githubusercontent.com/105755095/192888889-c396fd81-3afc-4676-88cc-01098fe94c28.png)

- the high risk precision score was .03 or 3% and recall of .7 or 70%
- the low risk precision score was 1.00 or 100% and the recall of .87 or 87%

 ![BRF_PR](https://user-images.githubusercontent.com/105755095/192888902-f3c956a8-5f04-4c4e-8005-4c993de304d8.png)

This model is better than the previous as it is able to accuratly predict 79% of loans in the correct status group. It still did not do as good in predicting high risk individuals from the given variables. The high risk precision score was 3%, better than the other models, but still very low. Only 3% of individuals flagged as high risk were correctly classified while the rest 97% were actually low risk loans. Meanwhile, 100% of low risk individuals were accuratly classified. Just as before, this model is putting most of the individuals in the low risk category and too many in the high risk category. The recall scores are much better in this algorithm. The high risk recall is at 70%. This means 70% of actual high risk loans we correctly classified while a good 87% of low risk loans were correctly classified. 

## Easy Endemble AdaBoost Classifier
This final ensemble algorithm had the following scores;
- the accuracy score was .931 or 93%

 ![EE_AS](https://user-images.githubusercontent.com/105755095/192890109-9c606b2a-91f4-4d35-9863-621a9da97a8a.png)

- the high risk precision score was .09 or 9% with a recall score of .92 or 92%
- the low risk precision score was 1.00 or 100% and had a recall score of .94 or 94%

 ![EE_PR](https://user-images.githubusercontent.com/105755095/192890126-4df45a0f-6581-40b3-b3a0-2f83d6b40c5c.png)
 
 This algorithm preformed the best. The accuracy score was over 93%. This model is able to accuraetly place the 93% of the loans in the correct category. Although, the precision of high risk loans was still low, it had the best score at 9%. That is, 9% of individuals flagged as high risk were actually high risk and 100% of individuals labeled as low risk were actually low risk. The recall on this algorithm has significantly improved from the rest. 92% of actual high risk loans were classified as high risk and 94% of low risk loans were classified as low risk. 

# Summary
All the data sets were able to measure the percentage of individuals flagged as low risk correctly but had a lot of trouble accurately measuring the percentage of high risk individuals flagged as high risk. These models were adding many low risk loans to the high risk category and that is why the precision score was always so low for all the models. However, the algorithms were getting 100% of the low risk individuals as low risk. The recall scores started off very low, but the ensemble algorithms did a great job in correctly classifying the individuals into the correct category. 

If the company had to choose an algorithm, they should definietly go for the Easy Ensemble model. This model did the best in accurately predicting individual credit risks. However, in reality, I would not recommend any of the models because the presicion score for high risk loans was so incredibly low. None of the models even made above 10% in that category. This means a lot of the low risk loans are being categorized as high risk and that is a lot of business that the company could be losing. 
