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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = MultinomialNB()
clf.fit(X_train,y_train)
predict=clf.predict(X_test)
clf.score(X_train,y_train)