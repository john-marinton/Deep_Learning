# -*- coding: utf-8 -*-

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#labelEncoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder1=LabelEncoder()
x[:,1]=encoder1.fit_transform(x[:,1])
encoder2=LabelEncoder()
x[:,2]=encoder2.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()

#dummy variable trap
x=x[:,1:]

#spliting the dataset into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#implementing Ann
#iInvoking Libraries
from keras.models import Sequential
from keras.layers import Dense,Dropout

#Intialization
classifier=Sequential()

#Adding input and hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',input_dim=11,activation='relu'))
#classifier.add(Dropout(rate=0.1)) If needed if model is overfitting

#Adding the second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
#classifier.add(Dropout(rate=0.1)) If needed if model is overfitting

#Adding the output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#Compiling the ann
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Training the model
classifier.fit(x_train,y_train,batch_size=10,epochs=100)

#Prediction
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#implementing new values to test
new_predict=classifier.predict(sc_x.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_predict=(new_predict>0.5)

#KFold Cross Validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(units=6,kernel_initializer='uniform',input_dim=11,activation='relu'))
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,epochs=100)
kfold=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10,
                      scoring='accuracy',n_jobs=1)
bias=kfold.mean()
variance=kfold.std()

#Parameter Tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(units=6,kernel_initializer='uniform',input_dim=11,activation='relu'))
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[25,32],
            'epochs':[100,500],
            'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10)
grid_search=grid_search.fit(x_train,y_train)
best_score=grid_search.best_score_
best_params=grid_search.best_params_









