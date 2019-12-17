1import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pre_processing as pp

# Loading data
data = pd.read_csv('Predicting_Mobile_App_Success.csv')
#data.drop([6941, 12624, 18477], inplace=True)
data = data[~data['Size'].isin(['Varies with device'])]
data.dropna(axis=0,how='any',inplace=True)
data=pp.remove_symbols(data)

columns_to_be_validated=['Rating','Price','Size','Reviews','Installs']
data=pp.delete_noise_data(data,columns_to_be_validated)
data=pp.to_float(data,columns_to_be_validated)

#X = data .loc[:, data.columns != 'Rating']

X = data.iloc[:,[1,3,4,5,6,7,9]]
X_poly= data.iloc[:,[1,3,4,5,6,7,9]]

# pre-processing
columns_to_be_transfomered = ['Category', 'Minimum Version', 'Content Rating']
columns_to_be_scaled = ['Installs', 'Reviews']
encodedFeatures =pp.one_hot_trans(X,columns_to_be_transfomered)
scaled_columns=pp.feature_scaling(X,columns_to_be_scaled)
X.drop(columns=columns_to_be_transfomered, inplace=True)
X.drop(columns=columns_to_be_scaled, inplace=True)
X = np.array(X)
features = np.concatenate((X, encodedFeatures,scaled_columns), axis=1)


features = pd.DataFrame(features)
#Get the correlation between the features
corr = features.corr()
#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr[-1]>0.1)]
print(len(top_feature.values))
#Correlation plot
plt.subplots(figsize=(10, 8))
top_corr = fifa_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()



multi_loaded_model_filename = 'multiLinearRegModel.sav'
multi_loaded_model = pickle.load(open(multi_loaded_model_filename, 'rb'))
prediction = multi_loaded_model.predict(features)
predicted_player_value = prediction[0]
print('Predicted rate for the  first application  in the test set  is : ' + str(predicted_player_value))




print ('poly model')
X_poly=pp.label_encoder_trans(X_poly,columns_to_be_transfomered)
scaled_columns=pp.feature_scaling(X_poly,columns_to_be_scaled)
X_poly.drop(columns=columns_to_be_scaled, inplace=True)
X_poly = np.array(X_poly)
features = np.concatenate((X_poly,scaled_columns), axis=1)
features = pd.DataFrame(features)
poly_features = PolynomialFeatures(degree=2)
# transforms the existing features to higher degree features.
poly_file_name = 'poly_model.sav'
poly_loaded_model = pickle.load(open(poly_file_name, 'rb'))
prediction = poly_loaded_model.predict(poly_features.fit_transform(features))
predicted_player_value = prediction[0]
print('Predicted rate for the  first application  in the test set  is : ' + str(predicted_player_value))