# Load the packages
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier


seed = 42


# Data exploration
data = pd.read_csv("../creditcard.csv")

# Get familiar with the column names
print(data.columns)

# Check data types
print(data.dtypes)


# Descriptive Statistics
print(data.describe())


# ## What is the dimension of the dataset?
# ### We have 284807 transcations and 30 features with 1 being the output class


# Explore the number of rows and columns
data.shape # We have 284807 transcations and 30 features with 1 being the output class


# ## Are there any missing values?
# ### No missing values

# Check missing values
droppedna = data.dropna() # drop the rows that have at least one element missing
droppedna.shape # Verify the dimension of the data and it is the same as the original dimension


# ## What is the class distribution?
# ### The output are highly imbalanced. Class 1 (fraud cases) is only 0.17% of all the outputs

# class distribution at a glance
print(data.groupby(['Class']).size())
num_class0 = data.groupby(['Class']).size()[0]
num_class1 = data.groupby(['Class']).size()[1]
percentage = num_class1/(data.shape[0]) * 100
print("The percentage of class 1 in the entire dataset is: %5.5f" %  percentage) 


# ## What is the distribution of the transaction time?
# ### It shows a bimodel distribution

sns.distplot(data['Time'])


# ## What is the relationship between time and amount?
# ### We can see that the more transcaction records, the more it is likely to appear a higher amount.

sns.jointplot(x="Time", y="Amount", data=data[["Time","Amount"]]);


# ## The distribution of the data

data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show()

# Scatterplot matrix
sns.set(style="ticks")
sns.pairplot(data)

# Split training and testing dataset
X = data.drop(columns="Class")
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


# ## Evaluation metrics with precision and recall

def evaluate(y_test, y_pred):
    # precision true positive / (true positive + false positive)
    # recall: true positive  / (true positive + false ne)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    # f1 = 2 * (precision * recall) / (precision + recall)
    f1 = f1_score(y_test, y_pred)
    #auc_score = auc(precision, recall)
    return precision, recall, f1

def summarizeGridSearchResult(grid_result):
    """
    A helper function to summarize the grid search result
    
    Parameters:
    
    grid_result: an object returned by the GridSearchCV function. 
    
    Returns:
    
    None
    """
    # summarize results
    print("The Best F1 score is : %f with parameter(s) %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return None


# ## Classic classification algorithms
# * Linear Algorithms: Linear Algorithms: Logistic Regression (LR),  Linear Discriminant Analysis (LDA), SGDClassifier (SGDC), Linear Support Vector Machines (Linear SVC)
# * Nonlinear Algorithms: Classification and Regression Trees (CART), Gaussian Naive Bayes (NB) and k-Nearest Neighbors (KNN).


# Test on more algorithms
models = []
models.append(('LR' , LogisticRegression(random_state = seed))) # Specify 'sag' for a faster performance to deal with large dataset.
models.append(('LDA' , LinearDiscriminantAnalysis()))
models.append(('KNN' , KNeighborsClassifier()))
models.append(('CART' , DecisionTreeClassifier(random_state = seed)))
models.append(('NB' , GaussianNB()))
models.append(('SVM' , LinearSVC(random_state = seed)))
models.append(('SGDC', SGDClassifier()))
# evaluate each model in turn
results = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1 = evaluate(y_test,y_pred)
    results.append((name, precision, recall, f1))


# ## Comparison among all algorithms in terms of AUPRC

# plot the precision-recall curves
for name, precision, recall, f1 in results:
    plt.plot(recall, precision, label=name)
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()


# ## Will standardization improve the algorithm performance?
# ### Yes, KNN has greatly improved its performance based the area under precision and recall curve. 

# Tried standardizing the data to see if there is a performance gain.
std_models = []
std_models.append(('ScaledLR', Pipeline([('Scaler' , StandardScaler()),( 'LR',
LogisticRegression(random_state = seed))]))) 
std_models.append(('ScaledLDA', Pipeline([('Scaler' , StandardScaler()),('LDA',
LinearDiscriminantAnalysis())])))
std_models.append(('ScaledKNN', Pipeline([('Scaler' , StandardScaler()),('KNN',
KNeighborsClassifier())])))
std_models.append(('ScaledCART', Pipeline([('Scaler' , StandardScaler()),('CART',
DecisionTreeClassifier(random_state = seed))])))
std_models.append(('ScaledNB' , Pipeline([('Scaler' , StandardScaler()),('NB',
GaussianNB())])))
std_models.append(('ScaledSVM' , Pipeline([('Scaler' , StandardScaler()),('SVM',
LinearSVC(random_state = seed))])))
std_models.append(('ScaledSGDC', Pipeline([('Scaler' , StandardScaler()),('SVM',
SGDClassifier())])))
# evaluate each model in turn
std_results = []
for name, model in std_models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1 = evaluate(y_test,y_pred)
    std_results.append((name, precision, recall, f1))


# plot the precision-recall curves
for name, precision, recall, f1 in std_results:
    plt.plot(recall, precision, label=name)
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()


# ## Algorithm Tuning for KNN
# ### We see that setting k=3 gives us the best f1 score 

# Tuning scaled KNN
n_neighbors = [1,3,5,6,7,8,9,10,11] # default is 5
param_grid = dict(n_neighbors=n_neighbors)
scoring = 'f1' # We use f1 to be our scoring metrics as it is a measure of how well our model performs based on the precision and recall
knn = KNeighborsClassifier()
grid = GridSearchCV(estimator = knn, param_grid = param_grid, scoring = scoring) # Note the gridsearchCV will use 5-fold CV by default


# Fit the model with standardized data
grid_result = grid.fit(scaledX_train, y_train)
summarizeGridSearchResult(grid_result)


# ## Will ensemble methods improve classification performance?
# ### We see that extra tree and random forest classifier are overall better ensemble methods with 0.86 f1 score 

# Test on more algorithms
ensembles_models = []
ensembles_models.append(('BC' , BaggingClassifier(KNeighborsClassifier(n_neighbors=3),random_state = seed))) # Specify 'sag' for a faster performance to deal with large dataset.
ensembles_models.append(('RF' , RandomForestClassifier(random_state=seed)))
ensembles_models.append(('AB' , AdaBoostClassifier(random_state=seed)))
ensembles_models.append(('GB' , GradientBoostingClassifier(random_state = seed)))
ensembles_models.append(('ET' , ExtraTreesClassifier(random_state = seed)))
# We select the top 3 models based on the original data without standardization for the voting classifier
clf1 = LogisticRegression(random_state = seed)
clf2 = DecisionTreeClassifier(random_state = seed)
clf3 = LinearDiscriminantAnalysis()
ensembles_models.append(('VT', VotingClassifier(estimators=[('LR', clf1), ('CART', clf2), ('LDA', clf3)], voting='hard')))

# evaluate each model in return
ens_results = []
for name, model in ensembles_models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    precision, recall, f1 = evaluate(y_test,y_pred)
    ens_results.append((name, precision, recall, f1))

# plot the precision-recall curves
for name, precision, recall, f1 in ens_results:
    plt.plot(recall, precision, label=name)
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()


# ## Algorithm Tuning for ExtraTreeClassifier

etclf = ExtraTreesClassifier(random_state = seed)
n_estimators = [10,50,100,150,200,300,500,1000] # default is 5
params = dict(n_estimators=n_estimators)
grid_et = GridSearchCV(estimator=etclf, param_grid=params, scoring = 'f1')
grid_et_result = grid_et.fit(X_train, y_train)
summarizeGridSearchResult(grid_et_result)


# ## Will neural network model outperform all algoritmhs used previously?
# ### The simple NN we have doesn't seem to perform better than the previous algorithms we see. There are several reasons for that. One is our model is not complex enough, second is that we need to also fine-tune our hyper-parameters such as number of nodes, number of layers, loss function, and optimizer etc.

# ### Simple neural network


# Building a simple neural network model
def simpleNNModel(trainX,trainY,loss='binary_crossentropy',optimizer='adam'):
    # Define the model
    model = Sequential()
    model.add(Dense(10, input_dim=30, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
    model.fit(trainX, trainY, epochs=10)
    return model

nnmodel = simpleNNModel(X_train,y_train)
y_pred = nnmodel.predict_classes(X_test)
nn_precision = precision_score(y_test, y_pred)
print('Precision: %f' % nn_precision)
nn_recall = recall_score(y_test, y_pred)
print('Recall: %f' % nn_recall)
nn_f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % nn_f1)




