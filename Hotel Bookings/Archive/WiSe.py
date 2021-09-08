import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFE
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_decomposition import PLSRegression

#########################################
# Helper methods for univariate filtering 
#########################################

def fdr_selection(p_values):
    """
    Given dictionary of [features] --> [p-values] performs FDR (False Discovery Rate) selection method. 
    :param p_values: dictionary with features associated to p-values 
    :return: features: selected using FDR method
    """
    p_values = {k: v for k, v in sorted(p_values.items(), key=lambda item: item[1], reverse=False)}

    largest_index = 0
    n_features = len(p_values)
    for idx, col in enumerate(reversed(p_values)):
        if p_values[col] <= (n_features - idx)/len(p_values)*0.2:
            largest_index = n_features - idx
            break
    features = list(p_values.keys())[:largest_index]
    return features


def p_value_computation(feature, target, coeff, measure, n_iterations=100):
    """
    Computed the p-value of the feature wrt. to the target.
    :param feature: a np.array on which that represents the feature under investigation
    :param target: the target 
    :param coeff: the reference coefficient obtained with real target column (and feature column under study)
    :n_iterations: the number of iterations used to compute the p-value of the feature 
    """
    p_value = 0
    for _ in range(0,n_iterations):
        p_rand = measure(feature, np.random.permutation(target))[0]
        p_value +=  1 if p_rand >= coeff else 0
    return p_value/n_iterations


def univariate_filtering(df, target, measure):
    """
    Given a dataframe with target datapoints, filters features according to their relevance with the target. 
    The relevance of each measure is evaluated using the `measure`. 
    :param df: the input dataframe 
    :param target: the name of the column that one wants to predict 
    :param measure: the measure (eg. Pearson correlation or Spearman Correlation) used for filtering  
    """
    p_values = dict()
    for col in df.columns: 
        if col != target:
            r, _ = measure(df[col], df[target])
            p_value = p_value_computation(df[col], df[target], r, measure)
            p_values[col] = p_value
    return fdr_selection(p_values) 

#########################################
# Univariate Filtering Methods 
#########################################
def univariate_filter_pearson(df, target):
    return univariate_filtering(df, target, pearsonr)  

def univariate_filter_spearman(df, target):
    return univariate_filtering(df, target, spearmanr)


#########################################
# Model Selection Methods 
#########################################

def model_selection_lasso(df, target, features, alphas=np.logspace(-4, -1, 30)):
    """
    Select features deemed most relevant by a Lasso model, trained using CV. 
    :param df: the complete training dataframe
    :param target: the name of the target to predict 
    :param features: the name of the features that were selected for training so far
    :param alphas: np.array containing the value of the different alpha value used to train the Lasso model 
    """
    X, y = df[features], df[target]
    lasso = LassoCV(alphas=alphas,cv=5, random_state=2, max_iter=10000).fit(X, y)
    selector = SelectFromModel(lasso, prefit=True)
    support = selector.get_support()
    return df[features].columns[support]  

def model_selection_RFE(df, target, features, n_features_to_select, estimator=SVR(kernel="linear")):
    """
    Select features using RFE techniques (recursive feature elimination). 
    :param df: the complete dataframe 
    :param target: the name of the taget to predict 
    :param featuees: the name of the features that were selected for training so far
    :param n_features_to_select: the number of features to select 
    :param estimator: the estimator on which the selection should be based on (default SVR(kernel="linear"))
    """
    X, y = df[features], df[target]
    selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)
    selector = selector.fit(X, y)
    support = selector.support_
    return df[features].columns[support]