# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:30:43 2019

@author: Cevher
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('krediVeriseti.csv')

dataset['krediMiktari'].fillna(0, inplace=True)

dataset['yas'].fillna(dataset['yas'].mean(), inplace=True)

X = dataset.iloc[:, :5]

#Converting words to integer values (House)
def convert_to_int_House(word):
    word_dict = {'evsahibi':1,'kiraci':2}
    return word_dict[word]

X['evDurumu'] = X['evDurumu'].apply(lambda x : convert_to_int_House(x))

#Converting words to integer values (Phone)
def convert_to_int_Phone(word):
    word_dict = {'var':1,'yok':2}
    return word_dict[word]

X['telefonDurumu'] = X['telefonDurumu'].apply(lambda x : convert_to_int_Phone(x))

#Converting words to integer values (Credi)
def convert_to_int_Credi(word):
    word_dict = {'krediver':1,'verme':2}
    return word_dict[word]



y = dataset['KrediDurumu']

y = y.apply(lambda x : convert_to_int_Credi(x))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
knn=classifier.fit(X_train, y_train)

print("Training set score: {:.2f}".format(knn.score(X_train, y_train)))
print("Test set score: {:.7f}".format(knn.score(X_test, y_test)))


pickle.dump(classifier, open('modelKNN.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('modelKNN.pkl','rb'))
print(model.predict([[12500,62,2,0,2]]))



 
