import numpy as np 
import pandas as pd
from sklearn import preprocessing


def feature_deletion(df, columns_to_drop = None): 
    """
    Delete features (i.e columns) that are not usefull for prediction. There are two types of columns that can be deleted: 
    (1) columns given as argument by the user and (2) columns that are filled with NaNs. 
    :param df: the processed dataframe 
    :param columns_to_drop: the features to be deleted from the dataframe 
    :return: the inputed dataframe without the deleted features
    """
    
    # drop the columns given as arguments 
    if columns_to_drop != None:
        if (set(columns_to_drop).issubset(df.columns)): 
            df = df.drop(columns=columns_to_drop)
        else: 
            print("Cannot delete features that do not exist. Check columns to drop input")
            
    
    # drop the columns that are all NaNs
    df = df.dropna(axis='columns', how ='all')
    
    # automatic drop where values are always the same ? 
    
    print("Features Kept:", df.columns.array.to_numpy())
    return df 


def feature_transform_notNan(df, feature_to_transform, new_feature_name): 
    """
    Some features are too complex. For instance: a phone number might be too complex information, the only informaton it may bring
    is that the customer indeed gave his phone number. These features can be be transformed to boolean features. 
    :param df: the processed dataframe 
    :param feature_to_transform: the name of the feature to transform  
    :param new_feature_name: the new feature's name
    :return: the inputed dataframe with the transformed feature 
    """
    df[new_feature_name] = df[feature_to_transform].notna().astype(int)
    df = df.drop(columns=[feature_to_transform])
    return df

def non_numerical_features(df, print_=False):
    """
    Returns the list of features that are not numerical. 
    :return: the list of features that are not numerical
    """
    special_type_columns = df.dtypes[df.dtypes == object].index.array.to_numpy()
    if print_:
        print("These features need special treatment regarding their type:",  special_type_columns)
    return special_type_columns 

def feature_ordinalEncoding(df, feature_name):
    le = preprocessing.LabelEncoder()
    df[feature_name] = le.fit_transform(df[feature_name])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    return df, mapping