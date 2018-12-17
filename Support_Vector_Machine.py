# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 19:11:18 2018

@author: PackardBell
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

print(y_test)
print("---------------")


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_Train=sc.fit_transform(x_train)
X_Test=sc.transform(x_test)


from sklearn.svm import SVC
svc = SVC(kernel = "rbf",C=1e3, gamma=0.1)
svc.fit(X_Train,y_train)
tahmin = svc.predict(X_Test)
print(tahmin)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,tahmin)
print(cm)

























