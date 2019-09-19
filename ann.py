# Artificial Neural Network

# Installing Theano
#!pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
#!pip install tensorflow

# Installing Keras
#!pip install --upgrade keras
#!pip install keras

#!conda upgrade pandas pytables h5py

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.getcwd()

os.chdir('D:\\Study\\01_2019_DS\\01_2019_DS\\Datasets')
print(os.getcwd())

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values

y = dataset.iloc[:, 13].values

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

onehot = OneHotEncoder(sparse=False)

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
X
print(X)

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
X

X.shape

tmpDF = pd.DataFrame(X)
tmpDF

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X 

tmpDF = pd.DataFrame(X)
tmpDF


X.shape


X = X[:, 1:]
X

X.shape

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

x_train_backup=X_train
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
#!conda install keras
#import tensorflow
import keras
from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD

# Initialising the ANN
classifier = Sequential()

#classifier.summary()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

#classifier.summary()
# Adding the second hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', 
                     activation = 'sigmoid'))

#classifier.summary()
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#classifier.compile(optimizer = SGD(), loss = 'binary_crossentropy', metrics = ['accuracy'])

#classifier.summary()
# Fitting the ANN to the Training set
#classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)

classifier.summary()

#history
# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
    
score = classifier.evaluate(X_test,y_test)
print(score)
print('loss = ', score[0])
print('acc = ', score[1])

# change the epochs to 5, 10 from 2
# got 79% acc with 2 & 5 & 20 epochs with SGD
# got 83% acc with 20 epochs with adam



# Initialising the ANN
classifier = Sequential()
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
# Adding the second hidden layer
classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the third hidden layer
classifier.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#classifier.summary()
history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 20)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

score = classifier.evaluate(X_test,y_test)
print(score)
print('loss = ', score[0])
print('acc = ', score[1])



classifier.save("bank_exited.h5")
from keras.models import load_model
mod=load_model("bank_exited.h5")
x_new=mod.predict(X_test)