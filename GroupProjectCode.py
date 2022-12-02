# CSU44061 - Machine Learning
# Group 60 
# Matthew Dowse

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

## Note I am assuming we have combined the 18 csvs into 1 csv
# called hockey.csv 
# in the format of 
# column 0 = Start + Side, column 1 = Finish + Side, GSO = 0 or 1
df = pd.read_csv("hockey.csv", index_col=False)
columns = ['Starting', 'Ending', 'GSO']
df.columns = columns
print(df.head())

#Start with separating data into positive and negative markers
# posList is when GSO = 1, negList when GSO is -1
posList = df[df['GSO'] == 1]
negList =df[df['GSO'] == -1]

print(posList)

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

##
#Separate data into two graphs for GSO and not GSO
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(startingPos, endingPos, color='green', marker='+')
ax1.legend(['is GSO = 1'], bbox_to_anchor =(1, 0.5))
plt.title("Scatter plot of full dataset, showing all atacks with y ouput dictating marker")
plt.xlabel("X-axis, Feature 1, X1")
plt.ylabel("Y-axis, Feature 2, X2")
plt.show()

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
# perhaps change to accuracy
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
        scores = cross_val_score(lRmodel, X, gso, cv=5, scoring='accuracy')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())    

import matplotlib.pyplot as plt
plt.errorbar(Ci_range, mean_error, yerr=std_error)
plt.xlabel('Ci'); plt.ylabel('Accuracy score') 
plt.title('Error bar graph showing Weights C against Accuracy Score')  
plt.show()
        
# C weight should be chosen now
# probably 5
# graph was weird

##Logistic regression time. 
## Thoughts: either split 80:20 or try K-fold ? Lets do both since our dataset is small
## l1 or l2 

from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
for train, test in kf.split(X):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2', C=1, solver='liblinear').fit(X[train], gso[train])
    ypred = model.predict(X[test])
    print("prediction = ", ypred)
    score = model.score(X[test], gso[test])
    print("Score = ", score)  
    

# 1. 80:20 split, l2 penalty
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, gso, test_size=0.2, random_state=15)  
#fit model with l2 penalty with weight C = 
### ADD C VALUE its 5 right now
from sklearn.linear_model import LogisticRegression
lRmodel = LogisticRegression(penalty='l2', C=1, solver='liblinear')
#fit the model with the training data
lRmodel.fit(X_train, y_train)
#predictions of y using test data
y_pred = lRmodel.predict(X_test)
print("prediction = ", y_pred)
#Check accuracy between prediction and actual test data
score = lRmodel.score(X_test, y_test)
print("Score = ", score)   

#Logistic 3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X_test[:,0], X_test[:,1], y_pred)
ax.set_xlabel('X-axis - Starting area', fontweight='bold')
ax.set_ylabel('Y-axis - Ending area', fontweight='bold')
ax.set_zlabel('Z-axis - GSO', fontweight='bold')
plt.title("3D scatter plot of testing data with prediction from Logistic Regression")
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
ax.set_xlabel('X-axis - Starting area', fontweight='bold')
ax.set_ylabel('Y-axis - Ending area', fontweight='bold')
ax.set_zlabel('Z-axis - GSO', fontweight='bold')
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

#############################################################################################################
#############################################################################################################

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

##bar chart of area on pitch attack started for GSo==1
left1= [0,0,0,0]
right1= [0,0,0,0]
centre1= [0,0,0,0]

#graph of area of pitch attack starts
for i in df.index:
    if(df['GSO'][i]==1):
        if(df['Starting'][i]==11):
            left1[0]+=1
        if(df['Starting'][i]==12):
            centre1[0]+=1
        if(df['Starting'][i]==13):
            right1[0]+=1

        if(df['Starting'][i]==21):
            left1[1]+=1
        if(df['Starting'][i]==22):
            centre1[1]+=1
        if(df['Starting'][i]==23):
            right1[1]+=1

        if(df['Starting'][i]==31):
            left1[2]+=1
        if(df['Starting'][i]==32):
            centre1[2]+=1
        if(df['Starting'][i]==33):
            right1[2]+=1

        if(df['Starting'][i]==41):
            left1[3]+=1
        if(df['Starting'][i]==42):
            centre1[3]+=1
        if(df['Starting'][i]==43):
            right1[3]+=1

labels = ['Q1', 'Q2', 'Q3', 'Q4']
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots()

width=0.3
rects1 = ax.bar(x - width, left1, width, label='Left')
rects2 = ax.bar(x, centre1, width, label='Centre')
rects3 = ax.bar(x + width, right1, width, label='Right')

ax.set_ylabel('Number of Attacks')
ax.set_xlabel('Quarter Attack Started')
ax.set_title('Areas Attacks Started leading to GSO')
ax.set_xticks(x, labels)
ax.legend(bbox_to_anchor =(1, 0.5))

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

fig.tight_layout()
#plt.show()

#graph of area of pitch attack starts for GSO==-1

left2= [0,0,0,0]
right2= [0,0,0,0]
centre2= [0,0,0,0]

for i in df.index:
    if(df['GSO'][i]==-1):
        if(df['Starting'][i]==11):
            left2[0]+=1
        if(df['Starting'][i]==12):
            centre2[0]+=1
        if(df['Starting'][i]==13):
            right2[0]+=1

        if(df['Starting'][i]==21):
            left2[1]+=1
        if(df['Starting'][i]==22):
            centre2[1]+=1
        if(df['Starting'][i]==23):
            right2[1]+=1

        if(df['Starting'][i]==31):
            left2[2]+=1
        if(df['Starting'][i]==32):
            centre2[2]+=1
        if(df['Starting'][i]==33):
            right2[2]+=1

        if(df['Starting'][i]==41):
            left2[3]+=1
        if(df['Starting'][i]==42):
            centre2[3]+=1
        if(df['Starting'][i]==43):
            right2[3]+=1

labels = ['Q1', 'Q2', 'Q3', 'Q4']
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots()

width=0.3
rects4 = ax.bar(x - width, left2, width, label='Left')
rects5 = ax.bar(x, centre2, width, label='Centre')
rects6 = ax.bar(x + width, right2, width, label='Right')

ax.set_ylabel('Number of Attacks')
ax.set_xlabel('Quarter Attack Started')
ax.set_title('Areas Attacks Started not leading to GSO')
ax.set_xticks(x, labels)
ax.legend(bbox_to_anchor =(1, 0.5))

ax.bar_label(rects4, padding=3)
ax.bar_label(rects5, padding=3)
ax.bar_label(rects6, padding=3)

fig.tight_layout()
#plt.show()


labels = ['Q1', 'Q2', 'Q3', 'Q4']
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots()

width=0.3
left3 = np.add(left1,left2)
centre3 = np.add(centre1,centre2)
right3 = np.add(right1,right2)
rects7 = ax.bar(x - width, left3, width, label='Left')
rects8 = ax.bar(x, centre3, width, label='Centre')
rects9 = ax.bar(x + width, right3, width, label='Right')

ax.set_ylabel('Number of Attacks')
ax.set_xlabel('Quarter Attack Started')
ax.set_title('Areas Attacks Started')
ax.set_xticks(x, labels)
ax.legend(bbox_to_anchor =(1, 0.5))

ax.bar_label(rects7, padding=3)
ax.bar_label(rects8, padding=3)
ax.bar_label(rects9, padding=3)

fig.tight_layout()
plt.show()
