# Credit Card Fraud Detection

## Introduction

Credit card fraud detection is one of the most important issues for credit card companies to deal with in order to earn trust from its customers. As machine learning techniques are robust to many tackle classification problems settings such as image recognition, we aim to explore various machine learning classification algorithms on this particular problem of classifying credit card fraud. This work showcases on how to compare different algorithms and fine-tune them. The dataset mainly contains 492 frauds out of 284,807 transactions. It has 28 principle components, transaction time, and tranaction amount with labels, 0 being non-fraud and 1 being fraud.

## Methodology

The outline of this project mainly includes exploratory data analysis, model selection, and evaluation.

### Exploratory Data Analysis

We have found that the data does not contain any missing data. As shown by `figure 1`, the distribution of the transcation time appears to be bimodel. Also, the more transaction happens within the same period of time, the more likely the transaction amount rises (see `figure 2`). Due to the prior PCA procedure, most predictors appear to be uncorrelated (see `figure 3`)

![](/figures/fig1.png)

*Figure 1: Distribution of the transaction time*

![](/figures/fig2.png)

*Figure 2: Scatterplot of trasaction time and amount*

![](/figures/fig4.png)

*Figure 3: Scatterplot matrix of the dataset*

### Model Selection and Evaluation

Given the umbalanced class setup, we have used precision-recall curve to guide our model selection process and f1-score to be the metric. Moreover, stratified 10-fold cross validation is used to estimate the performance of the algorithms. We have used precision-recall curve to guide our model selection process. We splitted data in 80% training and 20% testing randomly.

#### Baseline setup

We fit the data using both linear algorithms and non linear algorithms. They are logistic regression (LR), linear discriminant analysis (LDA), SGDClassifier (SGDC), linear support vector machines (Linear SVC), classification and regression trees (CART), gaussian naive bayes (NB) and k-nearest neighbors (KNN). We use these algorithms as the baseline. The result is shown by `figure 4` below.

![](/figures/fig5.png)

*Figure 4: Precision-recall curve for all baseline algorithms*

Then, we want to see if standardization can improve some of the algorithm's performance. Indeed, k-nearest neigbors has improved by training on standardized data as shown by `figure 5`.

![](/figures/fig6.png)

*Figure 5: Precision-recall curve for all baseline algorithms based on standardized data*


##### Algorithm Tuning

We then implement grid search on KNN to fine tune the parameters as KNN has the best performance on the precision-recall curve. We have found that k = 3 is an ideal option to give the best f1-score of 0.830783 among all the threshold values we have chosen (1,3,5,6,7,8,9,10,11).  

#### Ensemble Methods

In an attempt to improve model performance, we try using ensemble methods. These methods include baggingclassifier (BR), Random Forest Classifier (RF), AdaBoost Classifier (AB), Gradient Boosting Classifier (GB), Extra Trees Classifier (ET), Voting Classifier (VT). Similarly, we first fit them with the original data, then standardized data. We have found that RF and ET perform the best. We then fine-tune the ET algorithm and realized that setting the parameter `n_estimators` equal to 100 gives us the best performance among all other values we tried (10,50,100,150,200,300,500,1000)

#### Neural Network

(Work in progress)

## Conclusion and discussion

(Work in progress)

## Data Source
[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud). 

## Reference

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon

Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. Credit card fraud detection: a realistic modeling and a novel learning strategy, IEEE transactions on neural networks and learning systems,29,8,3784-3797,2018,IEEE

Dal Pozzolo, Andrea Adaptive Machine learning for credit card fraud detection ULB MLG PhD thesis (supervised by G. Bontempi)

Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Aël; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier

Carcillo, Fabrizio; Le Borgne, Yann-Aël; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing

Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé, Gianluca Bontempi Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection, INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019

Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé, Gianluca Bontempi Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection Information Sciences, 2019


