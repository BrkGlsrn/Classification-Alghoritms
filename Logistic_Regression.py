# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 14:55:08 2018

@author: PackardBell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv("veriler.csv")

x = veriler.iloc[5:,1:4].values
y = veriler.iloc[5:,4:].values


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

print(y_test)
print("---------------")

#standartlastırma
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_Train=sc.fit_transform(x_train)
X_Test=sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
Log_R = LogisticRegression(random_state=0)
Log_R.fit(X_Train,y_train)

tahmin = Log_R.predict(X_Test)
print(tahmin)


#Confusion matrix çapraz da doğru sonuçları verir
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,tahmin)
print(cm)





















