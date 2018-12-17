# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 21:20:04 2018

@author: PackardBell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

veriler = pd.read_excel("Iris.xls")

x = veriler.iloc[:,0:4].values
y = veriler.iloc[:,4:].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

print(y_test)
print("---------------")

#Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_Train=sc.fit_transform(x_train)
X_Test=sc.transform(x_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
Log_R = LogisticRegression(random_state=0)
Log_R.fit(X_Train,y_train)
print("Logistic Regression Results")
tahmin_LR = Log_R.predict(X_Test)
cm = confusion_matrix(y_test,tahmin_LR)
print(cm)

#KNN 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 3 , metric = "minkowski")
knn.fit(X_Train,y_train)
tahmin_KNN = knn.predict(X_Test)
print("KNN Results")
cm = confusion_matrix(y_test,tahmin_KNN)
print(cm)

#SVC
from sklearn.svm import SVC
svc = SVC(kernel = "linear",C=1e3, gamma=0.2)
svc.fit(X_Train,y_train)
tahmin_SVC = svc.predict(X_Test)
print("SVC Results")
cm = confusion_matrix(y_test,tahmin_SVC)
print(cm)

#Gaussian Naive Bayes(GNB)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_Train,y_train)
tahmin_gnb = gnb.predict(X_Test)
print("GNB Results")
cm = confusion_matrix(y_test,tahmin_gnb)
print(cm)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_Train,y_train)
tahmin_DT = dtc.predict(X_Test)
print("Decision Tree Results")
cm = confusion_matrix(y_test,tahmin_DT)
print(cm)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rdc = RandomForestClassifier(n_estimators=10,criterion="entropy")
rdc.fit(X_Train,y_train)
tahmin_RF = rdc.predict(X_Test)
print("Random Forest Results")
cm = confusion_matrix(y_test,tahmin_RF)
print(cm)

