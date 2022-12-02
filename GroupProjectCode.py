# CSU44061 - Machine Learning
# Group 60 
# Matthew Dowse

## Similar to my assignment 4 in parts

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

## Note I am assuming we have combined the 18 csvs into 1 csv
# called hockey.csv 
# in the format of 
# column 0 = Start + Side, column 1 = Finish + Side, GSO = 0 or 1
df = pd.read_csv("hockey.csv")
columns = ['Starting', 'Ending', 'GSO']
df.columns = columns
print(df.head())

#Start with separating data into positive and negative markers
# posList is when GSO = 1, negList when GSO is -1
posList = df[df['GSO'] == 1]
negList =df[df['GSO'] == -1]

#get all the postives and negatives to graph later in different colours.
startingPos=posList.iloc[:,0]
endingPos=posList.iloc[:,1]
startingNeg=negList.iloc[:,0]
endingNeg=negList.iloc[:,1]

#create 2D plot of full dataset 
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(startingNeg, endingNeg, color='black', marker='o', label='y-axis')
ax1.scatter(startingPos, endingPos, color='green', marker='+', label='x-axis')
ax1.legend(['is GSO = -1', 'is GSO = 1'], bbox_to_anchor =(1, 0.5))
plt.title("Scatter plot of full dataset, showing all atacks with y ouput dictating marker")
plt.xlabel("X-axis, Feature 1, X1")
plt.ylabel("Y-axis, Feature 2, X2")
plt.show()

###
# Separate data into two graphs for GSO and not GSO
# fig = plt.figure()
# ax1.scatter(startingPos, endingPos, color='green', marker='+', label='x-axis')
# ax1.scatter(startingPos, endingPos, color='black', marker='o', label='y-axis')
# ax1.legend(['is GSO = 1', 'is GSO = -1'], bbox_to_anchor =(1, 0.5))
# plt.title("Scatter plot of full dataset, showing all atacks with y ouput dictating marker")
# plt.xlabel("X-axis, Feature 1, X1")
# plt.ylabel("Y-axis, Feature 2, X2")
# plt.show()



#Get column data
startingArea = df.iloc[:,0]
endingArea = df.iloc[:,1]
X = np.column_stack((startingArea, endingArea)) 
gso = df.iloc[:,2]

#create 3D plot of full dataset
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:,0], X[:,1], gso)
ax.set_xlabel('X-axis - Starting area', fontweight='bold')
ax.set_ylabel('Y-axis - Ending area', fontweight='bold')
ax.set_zlabel('Z-axis - GSO', fontweight='bold')
plt.title("3D Scatter plot of full dataset")
plt.show()

###################################################################################################
###################################################################################################

#Modelling time 
#Things to consider
# Logistic Regression L1 or L2? 
#
# First use k fold cross validation to
# to tune hyperameter weight C given to the penalty in the cost function

# Find best weight C to avoid overfitting (smallest value for C)
# but not too small so it increases the prediction error 
# Note - penalty for l1 and l2
# using cross val score and f1 scoring for error
mean_error=[]; std_error=[]
Ci_range = [0.1, 0.5, 1, 5, 10, 50, 100]
for Ci in Ci_range:
    from sklearn.linear_model import LogisticRegression
    lRmodel = LogisticRegression(penalty='l2', solver='lbfgs', C=Ci)
    scores=[]
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        lRmodel.fit(X[train], gso[train])
        ypred = lRmodel.predict(X[test])
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(lRmodel, X, gso, cv=5, scoring='f1')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())    

import matplotlib.pyplot as plt
plt.errorbar(Ci_range, mean_error, yerr=std_error)
plt.xlabel('Ci'); plt.ylabel('F1 score') 
plt.title('Error bar graph showing Weights C against F1 Score')  
plt.show()
    
     

    
# C weight should be chosen now

##Logistic regression time. 
## Thoughts: either split 80:20 or try K-fold ? Lets do both since our dataset is small
## l1 or l2 

# 1. 80:20 split, l2 penalty
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, gso, test_size=0.2, random_state=15)  
#fit model with l2 penalty with weight C = 
### ADD C VALUE its 5 right now
from sklearn.linear_model import LogisticRegression
lRmodel = LogisticRegression(penalty='l2', C=5, solver='lbfgs')
#fit the model with the training data
lRmodel.fit(X_train, y_train)
#predictions of y using test data
y_pred = lRmodel.predict(X_test)
print("prediction = ", y_pred)
#Check accuracy between prediction and actual test data
score = lRmodel.score(X_test, y_test)
print("Score = ", score)   

#Baseline Comparison using Dummy Regressor 
from sklearn.dummy import DummyRegressor
dummy_regr = DummyRegressor(strategy="median")
dummy_regr.fit(X_test, y_test) 
dummypred = dummy_regr.predict(X_test)
dummytest = dummy_regr.score(X_test, y_test)   

dummy_regr2 = DummyRegressor(strategy="mean")
dummy_regr2.fit(X_test, y_test) 
dummypred2 = dummy_regr2.predict(X_test)
dummytest2 = dummy_regr2.score(X_test, y_test) 
#print("Score for Dummy Median ",)
#Compare Mean Squared error to see whether constant or regressions has higher error.
from sklearn.metrics import mean_squared_error
#print("square error of median %f %f"%(mean_squared_error(y_test,y_pred),mean_squared_error(y_test,dummypred)))
#print("square error of mean %f %f"%(mean_squared_error(y_test,y_pred),mean_squared_error(y_test,dummypred2)))


#Baseline Comparison using Dummy Classifier
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
dummy_clf.predict(X_train)
clfScore = dummy_clf.score(X_train, y_train)
dummy_clf2 = DummyClassifier(strategy="uniform")
dummy_clf2.fit(X_train, y_train)
dummy_clf2.predict(X_train)
clfScore2 = dummy_clf2.score(X_train, y_train)
dummy_clf3 = DummyClassifier(strategy="prior")
dummy_clf3.fit(X_train, y_train)
dummy_clf3.predict(X_train)
clfScore3 = dummy_clf3.score(X_train, y_train)
print("Classifier Score Most Frequent ",clfScore)
print("Classifier Score Uniform",clfScore2)
print("Classifier Score Prior",clfScore3)

#not sure if needed
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_test[:,0], X_test[:,1], y_pred)
ax.set_xlabel('X-axis - X1', fontweight ='bold')
ax.set_ylabel('Y-axis - X2', fontweight ='bold')
ax.set_zlabel('Z-axis - y', fontweight ='bold')
plt.title("3D scatter plot of testing data with prediction")
plt.show()

## CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
print("Confusion Matrix for LR = " , confusion_matrix(y_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

## ROC CURVE 
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test,lRmodel.decision_function(X_test))
plt.plot(fpr,tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for Logistic Regression Model')
plt.plot([0, 1], [0, 1], color='green',linestyle='--')
plt.show()
 
#################################################################################################################
#################################################################################################################

## K Nearest Neighbour time
# First we will have to see which k is best to use
# I am splitting the data in a 80:20 split
# split X and y into training and testing sets
# performs a train test split. 80 training, 20 test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, gso, test_size=0.2, random_state=15)
k_range = [1,2,3,4,5,6,7,8,9,10]

# choose k using kfold cross validation to create and train a kNN classifier
mean_error=[]
std_error=[]
for k in k_range:
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=k,weights='uniform')
    scores=[]
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model.fit(X[train],gso[train])
        y_pred = model.predict(X_test)
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, gso, cv=5, scoring='f1')    
    mean_error.append(np.array(scores).std())   
    std_error.append(np.array(scores).std()) 
import matplotlib.pyplot as plt
plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=3)   
plt.xlabel('k')
plt.ylabel('f1 score')    
plt.show()   
score = model.score(X_test, y_test)
print("Score = ", score)           
    
# Decide which K is best 

## kNN CLASSIFIER 
# Chose k right now its 9

import matplotlib.pyplot as plt
kNmodel = KNeighborsClassifier(n_neighbors=9,weights='uniform').fit(X_train, y_train)
y_pred = kNmodel.predict(X_test)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train[:,0], X_train[:,1], y_train)
ax.set_xlabel('X-axis - X1', fontweight ='bold')
ax.set_ylabel('Y-axis - X2', fontweight ='bold')
ax.set_zlabel('Z-axis - y', fontweight ='bold')
plt.title("3D scatter plot of training data")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_test[:,0], X_test[:,1], y_pred)
ax.set_xlabel('X-axis - X1', fontweight ='bold')
ax.set_ylabel('Y-axis - X2', fontweight ='bold')
ax.set_zlabel('Z-axis - y', fontweight ='bold')
plt.title("3D scatter plot of testing data with prediciton")
plt.show()

## CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
print("Confusion Matrix for kNN = " , confusion_matrix(y_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

y_scores = kNmodel.predict_proba(X_test)

## ROC CURVE 
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test,y_scores[:,1])
plt.plot(fpr,tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve for kNN Classifier Model')
plt.plot([0, 1], [0, 1], color='green',linestyle='--')
plt.show()