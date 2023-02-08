# Credit_Risk_Analysis
## Project Overview
Employ different techniques to train and evaluate models with unbalanced classes. For this particular project we are going to use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, the objective of the project is to predict credit risk using machine learning algorithms.

First of all, we will need to oversample the data using he RandomOverSampler and SMOTE algorithms. Likewise, then we will undersample the data using the ClusterCentroids algorithm. After that we’ll use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. 

Next, we’ll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. 

Finally, we’ll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.


## Results
Using different algorithms as mentioned above we will resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report as shown in this section. Therefore, the expected results from this project  using each algorithm are detailed below:

### Resampling Models to Predict Credit Risk

#### Naïve Random Oversampling: Using Oversampling Random Over Sampler
•	Accuracy Score: 62.5%

• Precision High Risk: 1%

• Precision Low Risk: 100%

• Recall High Risk: 60%

• Recall Low Risk: 65%

![image](https://github.com/rdonosob1/Credit_Risk_Analysis/blob/main/Resources/RandomOverSampler.png)

#### SMOTE Oversampling: Using SMOTE Oversampling Algorithm we obtained the following results:
•	Accuracy Score: 65.1%

•	Precision High Risk: 1%

•	Precision Low Risk: 100%

•	Recall High Risk: 64%

•	Recall Low Risk: 66%

![image](https://github.com/rdonosob1/Credit_Risk_Analysis/blob/main/Resources/Smote.png)

### Undersampling Models to Predict Credit Risk

#### Cluster Centroids: Using Undersampling ClusterCentroids algorithm we obtained the following results:
•	Accuracy Score: 65.1%

•	Precision High Risk: 1%

•	Precision Low Risk: 100%

•	Recall High Risk: 59%

•	Recall Low Risk: 43%

![image]()

### Combination (Over and Under) Sampling
#### SMOTEENN: Using SMOTEENN Algorithm we obtained the following results:
•	Accuracy Score: 51%

•	Precision High Risk: 1%

•	Precision Low Risk: 100%

•	Recall High Risk: 70%

•	Recall Low Risk: 57%

![image]()

### Ensemble Learners
#### Balanced Random Forest Classifier: Using this algorithm we obtained the following results:
•	Accuracy Score: 67.2%

•	Precision High Risk: 73%

•	Precision Low Risk: 100%

•	Recall High Risk: 34%

•	Recall Low Risk: 100%

![image](https://github.com/rdonosob1/Credit_Risk_Analysis/blob/main/Resources/Random_Forest.png)

#### Easy Ensemble AdaBoost Classifier: Using this algorithm we obtained the following results:
•	Accuracy Score: 92.5%

•	Precision High Risk: 7%

•	Precision Low Risk: 100%

•	Recall High Risk: 91%

•	Recall Low Risk: 94%

![image](https://github.com/rdonosob1/Credit_Risk_Analysis/blob/main/Resources/Easy%20Ensemble%20Analysis_Confuion%20Matrix.png)

![image](https://github.com/rdonosob1/Credit_Risk_Analysis/blob/main/Resources/Easy%20Ensemble%20Analysis.png)

## Summary

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, the objective of the project is to predict credit risk using machine learning algorithms and therefore, evaluate the results to further recommend the use of any of these techniques.

After analyzing the 6 models, we tried to find the best model that can detect how risky it may be to provide credit to an individual. With that being said, the aim of this project consist in finding a model that best fits the interest of the lender and avoid providing credit to a high risk person  without detecting them first.
That correlating statistic for this is the recall rate for high risk.

The first part of the analysis consisted in use resampling to predict credit risk. In that section we used oversampling (RandomOverSampler and SMOTE)and undersampling (ClusterCentroids) algorithms.

The results shown are as follows:

#### Random Over sampler  
This model on one hand shows an accuracy score of 62.5%, a precision rate avg of 99%, and a recall rate of average 65%. However, 5952 low risk loans are flagged as high risk. This means that using this model, these people would able to get a credit but the lender would deny it. It would make upset to around 5900 people that will go somewhere else for credit and the lender would lose that business.  

#### SMOTE
A similar scenario we can observe after running this algorithm where the results slightly changed, but they are pretty much align to the previous ones, where the accuracy score is 65.1%, the precision rate avg is still 99%, and a recall rate average is now 66%. The scenario keeps showing a consistent as shown with the previous algorithm. 

#### Udersampling 
Next, we tried to Undersample the model using cluster centroigs algorithm. The results, improved a little bit, but not enough to feel confident about using this model where the accuracy score is 65.1%, the avg. precision rate is still the same, but the avg recall rate went down to 44%.


#### Combination (Over-and-Undersampling)
To complete this part of the analysis, we used a combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms. However, we can see a drop on the accuracy score from 65% to 51%, while the precision rate is still the same (99%) and the recall rate improved from the previous one and went up to 57%. Nevertheless, we are not confident at all due to the results haven’t improved yet and aren't good enough. 

#### Easy Ensemble Classifying
Finally, we tried to use ensemble classifiers Easy Ensemble Classifying in order to find a better result for the model to use. In this section we use BalancedRandomForestClassifier algorithm and Easy Ensemble Classifier where we obtained the following results 
•	BalanceRandomFOrest Classifier: Accuracy Score: 67.2%, avg precision 100% and avg recall
•	Easy Ensemble Classifier: This last results show an important improvement comparing to the previous ones where we can see that this model is very reliable with an accuracy score of 92.5%. Likewise the avg precision is 100% and the recall rate is 94%.
After factoring in these three main statistics, the model that I would recommend to use for predicting high risk loans is the Easy Ensemble Classifying model.

