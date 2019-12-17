
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,Imputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

def isfloat(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def delete_noise_data(data,cols):
    for i in cols:
        for x in data[i]:
            if not isfloat(x):
                data.drop(data[data[i]==x].index[0],inplace=True)
    return data
def to_float(data,cols):
    
    for i in cols:
        sum=0
        c=0
        for j in range(len(data[i])):
            x=data[i].values[j]
            if isfloat(x):
                data[i].values[j]=float(x);
                sum+=float(x)
                c=c+1
        avg=sum/c
        for j in range(len(data[i])):
            x=data[i].values[j]
            if not isfloat(x):
                data[i].values[j]=avg;
        data[i]=data[i].astype(float)

    return data




def replace_nan_values(X,cols):
    imputer=Imputer(missing_values=np.nan,strategy='mean',axis=0)
    X[cols]=imputer.fit_transform(X[cols])
    return X



def apply_PCA(x_train,x_test,m=None):
    pca=PCA(n_components=m)
    x_train= pca.fit_transform(x_train)
    x_test= pca.transform(x_test)
    ratio=pca.explained_variance_ratio_
    return x_train,x_test,ratio
    

def Encode_AppRating(X):
     X['App_Rating'] = X['App_Rating'].str.replace('High_Rating', '3')
     X['App_Rating'] = X['App_Rating'].str.replace('Intermediate_Rating', '2')
     X['App_Rating'] = X['App_Rating'].str.replace('Low_Rating', '1')
     X['App_Rating'] = X['App_Rating'].astype(float)
     
     return X

def remove_symbols(X):
    X['Price'] = X['Price'].str.replace('$', '')
    X['Installs'] = X['Installs'].str.replace(',', '')
    X['Installs'] = X['Installs'].str.replace('+', '')
    X['Size'] = X['Size'].str.replace(',', '')
    X['Size'] = [(lambda x: float(x[0:-1]) if x[-1] == 'M' else ( float(x[0:-1]) / 1000 if x[-1] =='K' else x))(x) for x in X['Size']]
    return  X


def label_encoder_trans(X, columns_to_be_transfomered):
    labelEncoder = LabelEncoder()
    for s in columns_to_be_transfomered:
        X[s] = labelEncoder.fit_transform(X[s])
    return X


def one_hot_trans(X,columns_to_be_transfomered):

    colT = ColumnTransformer(
        [("dummy_col", OneHotEncoder(categories='auto'), columns_to_be_transfomered)])
    encodedFeatures = colT.fit_transform(X).toarray()
    return  encodedFeatures

def feature_scaling(X,columns_to_be_scaled):
    colT = ColumnTransformer([('std', MinMaxScaler(), columns_to_be_scaled)])
    scaled_feature= colT.fit_transform(X)
    return scaled_feature


