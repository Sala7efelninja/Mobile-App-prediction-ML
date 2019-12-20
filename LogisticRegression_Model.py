# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 17:22:05 2019

@author: Asalla
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pre_processing as pp
import time


data = pd.read_csv('Mobile_App_Success_Milestone_2.csv')

data.dropna(axis=0,how='any',thresh=5,inplace=True)
data = data[pd.notnull(data['App_Rating'])]

data=pp.remove_symbols(data)

data=pp.Encode_AppRating(data)

columns_to_be_validated=['Price','Size','Reviews','Installs']
data=pp.to_float(data,columns_to_be_validated)

data=pp.replace_nan_values(data,columns_to_be_validated)
data.dropna(axis=0,how='any',inplace=True)
data=pp.delete_noise_data(data,columns_to_be_validated)

columns_to_be_transfomered = ['Category', 'Minimum Version', 'Content Rating']
data2=pp.label_encoder_trans(data,columns_to_be_transfomered)

#plt.subplots(figsize=(10, 8))
#corr =data2.corr()
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,annot=True)
#plt.show()

X = data.iloc[:,[1,2,3,4,5,6,8]]
Y = data['App_Rating']

columns_to_be_scaled = ['Installs', 'Reviews']
encodedFeatures =pp.one_hot_trans(X,columns_to_be_transfomered)
scaled_columns=pp.feature_scaling(X,columns_to_be_scaled)

X.drop(columns=columns_to_be_transfomered, inplace=True)
X.drop(columns=columns_to_be_scaled, inplace=True)

X = np.array(X)

features = np.concatenate((X, encodedFeatures,scaled_columns), axis=1)
features = pd.DataFrame(features)

X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.20, shuffle=True)


x_trainPca,x_testPca,ratio=pp.apply_PCA(X_train,X_test,2)

classifier = linear_model.LogisticRegression(multi_class='ovr',C=1000)

t0=time.time()
classifier.fit(X_train,y_train)
print ("training time:", round(time.time()-t0, 3), "s")
t1=time.time()

y_pred = classifier.predict(X_test)
print ("predict time:", round(time.time()-t1, 3), "s")
accuracy = np.mean(y_pred == y_test)
print(accuracy * 100)

from sklearn.metrics import confusion_matrix
labels=['Low','Intermediate','High']
classes=[1,2,3]
cm=confusion_matrix(y_test,y_pred,labels=classes)


ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,linewidths=1,fmt = 'd'); 

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);

filename = 'Logistic_Regression_Model.sav'
pickle.dump(classifier, open(filename, 'wb'))
