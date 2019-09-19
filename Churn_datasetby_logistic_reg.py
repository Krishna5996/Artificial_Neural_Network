# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:32:13 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
os.getcwd()

os.chdir('D:\\Study\\01_2019_DS\\01_2019_DS\\Datasets')
print(os.getcwd())

dataset = pd.read_csv('Churn_Modelling.csv')

x = dataset.iloc[:, 3:13].values
x
y = dataset.iloc[:, 13].values
y


# Dummy Vars & Encoders
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
dataset2 = ['Pizza','Burger','Bread','Bread','Bread','Burger','Pizza','Burger']

values = array(dataset2)
print(values)

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values)

print(integer_encoded)


onehot=OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
print(integer_encoded)
 
onehot_encoded = onehot.fit_transform(integer_encoded)
print(onehot_encoded)

#inverted_result = label_encoder.inverse_transform([argmax(onehot_encoded[0,:])])
#print(inverted_result)

# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
                

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
'''X
print(X)
'''
x[:, 1] = labelencoder_X_1.fit_transform(x[:, 1])
x

labelencoder_X_2 = LabelEncoder()
x[:, 2] = labelencoder_X_2.fit_transform(x[:, 2])
x

x.shape

tmpDF = pd.DataFrame(x)
tmpDF

onehotencoder = OneHotEncoder(categorical_features = [1])
x= onehotencoder.fit_transform(x).toarray()
x

tmpDF = pd.DataFrame(x)
tmpDF


x.shape


x = x[:, 1:]
x

x.shape



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=1000000)
x_train_backup=X_train

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

#Importing logistic regression
from sklearn.linear_model import LogisticRegression
import numpy as np
'''
data = df_base

arr = df_base.values

X = data.ix[:,(1,10,11,13)] # ivs for train
X

y = data.ix[:,16]
y
'''
regr = LogisticRegression()

regr.fit(X_train, y_train)

pred = regr.predict(X_test)
pred

from sklearn.metrics import confusion_matrix,classification_report,mean_squared_error
cm_df = pd.DataFrame(confusion_matrix(y_test, pred).T, index=regr.classes_,
columns=regr.classes_)
'''
cm_df=pd.DataFrame(Confusion_matrix(y_train,pred)).T,index=regr.classes_,
columns=regr.classes_)'''

cm_df.index.name = 'Predicted'
cm_df.columns.name = 'True'
print(cm_df)

print(classification_report(y_test,pred))

regr.score(X_test,y_test)
'''
import pickle
saved_model=pickle.dumps(regr)
regr_from_pickle=pickle.loads(saved_model)
#regr_from_pickle.predict(X_test)

#regr_from_pickle.predict(X_test[1,:])

X_test_rowone=X_test[0:11,:].reshape(11,468)

regr_from_pickle.predict(X_test_rowone)
=================================
X_test_rowone=X_test[0:11,:].reshape(11,468)
predone=regr.predict(X_test_rowone)
predone
'''
