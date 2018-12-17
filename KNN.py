# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:50:05 2018

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

#standartlastırma
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_Train=sc.fit_transform(x_train)
X_Test=sc.transform(x_test)


from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 1 , metric = "minkowski")
knn.fit(X_Train,y_train)
tahmin = knn.predict(X_Test)
print(tahmin)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,tahmin)
print(cm)
















