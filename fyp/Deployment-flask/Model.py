import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
import sklearn.model_selection as model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
import pickle


#load dataset
mydata=pd.read_csv("my_newdataset(Ld_State_numerical)(1).csv")
mydata.head();

X_train1 = mydata.sample(frac=1.0, random_state=101)

#stand scaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train1)

#K means clustering
n_samples = 68
random_state = 10
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_train2)
y_pred=[x+1 for x in y_pred]


#build diffrent model

y = y_pred
y_train, y_test = model_selection.train_test_split(y, train_size=0.80, test_size=0.20, random_state=101)
X_test = mydata.sample(frac=0.20, random_state=101)
X_train = mydata.sample(frac=0.80, random_state=101)

#one vs one(svm)
clf = OneVsOneClassifier(LinearSVC(random_state=0)).fit(X_train, y_train)

#one vs rest(svm)
clr = OneVsRestClassifier(SVC()).fit(X_train, y_train)

#KNeighbors
n_neighbors=11
for weights in ['uniform', 'distance']:
 clfknn = neighbors.KNeighborsClassifier(n_neighbors, weights=weights).fit(X_train, y_train)

#logistic reggression 
logreg = LogisticRegression(solver='liblinear', max_iter=1000).fit(X_train, y_train)


#claculate accuracy
clf_pred = clf.predict(X_test)
clr_pred = clr.predict(X_test)
clfknn_pred = clfknn.predict(X_test)
logreg_pred = logreg.predict(X_test)

clf_accuracy = accuracy_score(y_test, clf_pred)
print('clf_accuracy:' , "%.2f" % (clf_accuracy*100))

clr_accuracy = accuracy_score(y_test, clr_pred)
print( 'clr_accuracy: ' ,"%.2f" % (clr_accuracy*100))

clfknn_accuracy = accuracy_score(y_test, clfknn_pred)
print( 'clfknn_accuracy: ' ,"%.2f" % (clfknn_accuracy*100))

logreg_accuracy = accuracy_score(y_test, logreg_pred)
print( 'LogisticRegression: ' ,"%.2f" % (logreg_accuracy*100))


filename = 'finalclf_model.pkl'
pickle.dump(clf, open(filename, 'wb'))

filename = 'finalclr_model.pkl'
pickle.dump(clr, open(filename, 'wb'))

filename = 'finalclfknn_model.pkl'
pickle.dump(clfknn, open(filename, 'wb'))

filename = 'finallogreg_model.pkl'
pickle.dump(logreg, open(filename, 'wb'))
