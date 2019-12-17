# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:49:33 2019

@author: Amr Emam
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

# Loading data
data = pd.read_csv('Mobile_App_Success_Milestone_2.csv')
#data.drop([6941, 12624, 18477], inplace=True)
#data = data[~data['Size'].isin(['Varies with device'])]
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
#data2=pp.label_encoder_trans(data,columns_to_be_transfomered)
#plt.subplots(figsize=(10, 8))

#corr =data2.corr()
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns,annot=true)
#plt.show()
X = data.iloc[:,[1,2,3,4,5,6,8]]
#X=pp.label_encoder_trans(X,['Latest Version'])

Y = data['App_Rating']
# pre-processing


columns_to_be_scaled = ['Installs', 'Reviews']
#X=pp.label_encoder_trans(X,columns_to_be_transfomered)


#dummy=pp.one_hot_trans(data,['Latest Version'])


encodedFeatures =pp.one_hot_trans(X,columns_to_be_transfomered)
scaled_columns=pp.feature_scaling(X,columns_to_be_scaled)

X.drop(columns=columns_to_be_transfomered, inplace=True)
X.drop(columns=columns_to_be_scaled, inplace=True)
X = np.array(X)
features = np.concatenate((X, encodedFeatures,scaled_columns), axis=1)


features = pd.DataFrame(features)
X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.30, shuffle=True)
#Get the correlation between the features
x_trainPca,x_testPca,ratio=pp.apply_PCA(X_train,X_test,2)



#filename = 'ModelName.sav'
#pickle.dump(ModelName, open(filename, 'wb'))


