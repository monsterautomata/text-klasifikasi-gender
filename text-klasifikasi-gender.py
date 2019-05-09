# -*- coding: utf-8 -*-
"""
Created on Fri May 10 01:28:51 2019

@author: Rozan
"""

import pandas as pd 
from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv("komentar kaskus gender.csv",encoding='latin1')

x = df.iloc[:,0]
y = df.iloc[:,1]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x)

dt = pd.read_csv("komentar kaskus gender test.csv")

x_train = X
y_train = y
x_test = dt.iloc[:,0]
y_test = dt.iloc[:,1]
x_test = vectorizer.transform(x_test)
clf = MultinomialNB()
clf.fit(x_train,y_train)
predict=clf.predict(x_test)
scoremnnb=clf.score(x_train,y_train)

from sklearn.svm import SVC
clf2 = SVC(gamma='scale',decision_function_shape='ovo')
clf2.fit(x_train,y_train) 
predict2=clf2.predict(x_test)
scoresvm=clf2.score(x_train,y_train)