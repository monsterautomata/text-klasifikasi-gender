def preprocessing(data): #fungsi preprocessing
    stop_words = stopwords.words('Indonesian') #stopword bahasa indonesia
    data["Komentar"] = data["Komentar"].str.lower() #casefolding
    data['Komentar'] = data.Komentar.str.replace("[^\w\s]", "") #punctuation removal
    data.Komentar = data.Komentar.replace('\d+', '', regex = True) #number removal
    data['Komentar'] = data['Komentar'].apply(lambda x: ' '.join([item for item in x.split() if item not in stop_words]))#stopword removal
    return data

import pandas as pd 
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics	
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import BaggingClassifier

df = pd.read_csv("komentar kaskus gender.csv",encoding='latin1')#baca data training
df = preprocessing(df)#preprocessing data training
x = df.iloc[:,0]#ambil berdasarkan kolom komentar
y = df.iloc[:,1]#ambil berdasarkan kolom gender

vectorizer = CountVectorizer()#panggil fungsi countvector
X = vectorizer.fit_transform(x)#fit countvector pada kolom komentar training


dt = pd.read_csv("komentar kaskus gender test.csv")#baca data testing
#dt = preprocessing(dt)# #preprocessing data testing

x_train = X #alokasi variable xtraining (komentar training)
y_train = y #alokasi variable ytraining (gender training)
x_test = dt.iloc[:,0]#ambil berdasarkan kolom komentar
y_test = dt.iloc[:,1]#ambil berdasarkan kolom gender
x_test = vectorizer.transform(x_test)#fit countvector pada kolom komentar testing
clf = MultinomialNB()#panggil fungsi Mutinomial naive bayes
clf.fit(x_train,y_train)#fit fungsi MNNB pada x_train dan y_train
predict=clf.predict(x_test) #prediksi dengan MNNB
scoremnnb=clf.score(x_train,y_train) #skor dengan MNNB
print("Akurasi Prediksi  MNNB:",metrics.accuracy_score(y_test, predict)*100,'%') #print hasil akurasi MNNB


clf2 = BaggingClassifier(base_estimator=clf, n_estimators=100, random_state=10)#panggil fungsi bagging dengan MNNB sebagai base_estimator
clf2.fit(x_train,y_train)#fit fungsi bagging pada x_train dan y_train
predict2=clf2.predict(x_test)#prediksi MNNB dengan bagging
bagging = np.array(predict2.tolist())#kolom gender hasil prediksi
scorebagging=clf2.score(x_train,y_train)#skor MNNB dengan bagging
print("Akurasi Prediksi  MNNB dengan Bagging:",metrics.accuracy_score(y_test, predict2)*100,'%')#print hasil akurasi MNNB dengan bagging

