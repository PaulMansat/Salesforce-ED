import numpy as np
import pandas as pd
import xgboost as xgb
import my_scripts.data_cleaning as dc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, auc, roc_curve, accuracy_score, matthews_corrcoef, f1_score 
 

class XGBoost_classifier_trainer:
    def __init__(self, df, target, split_test=True, learning_rate=1e-2, max_depth=5, n_estimators=10): 
        # store relevant variables for later use 
        self.target = target
        self.split_test = split_test
        
        # encode dataframe 
        self.df, self.mappings = self.data_encoding(df)
        
        # prepare the train, test and validation set 
        if split_test: 
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df.drop(columns=[target]), self.df[target], test_size=0.2)
        else: 
            self.X_train, self.y_train = self.df.drop(columns=[target]), self.df[target]
            
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, test_size=0.2)
            
        # initialy, no assumption on features on which model should be trained 
        self.features = self.X_train.columns
        
        # initialize model 
        self.model = clf_xgb = xgb.XGBClassifier(max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    verbosity=0,
                    objective='binary:logistic',
                    booster='gbtree',
                    n_jobs=-1,
                    subsample=0.7,
                    use_label_encoder=False,
                )
        
        self.best_params = None
        
    def data_encoding(self, df): 
        mappings = {}
        for col in dc.non_numerical_features(df):
            df, mapping = dc.feature_ordinalEncoding(df, col)
            mappings[col] = mapping
        return df, mappings
    
    
    def get_model(self):
        return self.model
    
    def get_training_data(self):
        return self.X_train, self.y_train
    
    def get_testing_data(self):
        if self.split_test:  
            return self.X_test, self.y_test
        else:
            print("No training set defined. Please define a training set.")            
            return
    
    def get_validation_data(self):
        return self.X_valid, self.y_valid
    
    def get_best_params(self):
        return self.best_params
    
    def print_kpi(self, features):
        if self.split_test:
            y_pred = self.model.predict(self.X_test[features])
            # plot some relevant KPI
            print('Accuracy: {:.2f}'.format(accuracy_score(y_pred, self.y_test)))
            print('MCC Score: {:.2f}'.format(matthews_corrcoef(self.y_test, y_pred)))
            print('F1 Score: {:.2f}'.format(f1_score(self.y_test, y_pred)))
            
            # plot ROC curve 
            fpr, tpr, _ = roc_curve(self.y_test, self.model.predict_proba(self.X_test[features])[:,1])
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve - Area = {:.5f}".format(auc(fpr, tpr)));
        else: 
            print("No training set defined. Please define a training set.")
        return
    
    def train_model(self, features=None, verbose=0, end_evaluation=False):
        if type(features) == type(None): 
            features = self.X_train.columns
        
        self.features = features
           
        # train the model 
        self.model.fit(self.X_train[features], self.y_train,
                        eval_set=[(self.X_valid[features], self.y_valid)],
                        early_stopping_rounds=100,
                        verbose=verbose
                      )
        
        # print some key kpi
        if end_evaluation:
            self.print_kpi(features)
                
    def find_best_params(self, max_depths, learning_rates, n_estimators, features, scoring='roc_auc'):    
        min_max_depth = float('-inf')
        min_learning_rate = float('-inf')
        min_n_estimator = float('-inf')
        best_min_score = float('-inf')

        for max_depth in max_depths:
            for learning_rate in learning_rates: 
                for n_estimator in n_estimators:                    
                    # model definition
                    clf_xgb = xgb.XGBClassifier(max_depth=max_depth,
                        learning_rate=learning_rate,
                        n_estimators=n_estimator,
                        verbosity=0,
                        objective='binary:logistic',
                        booster='gbtree',
                        n_jobs=-1,
                        subsample=0.7,
                        use_label_encoder=False,
                    )

                    # cross validation training
                    min_current_score = np.min(cross_val_score(clf_xgb, self.X_train[features], self.y_train))

                    if best_min_score < min_current_score: 
                        min_max_depth = max_depth
                        min_learning_rate = learning_rate
                        min_n_estimator = n_estimator
                        best_min_score = min_current_score

        print("Best CV min score: {:.2f}".format(best_min_score))
        self.best_params = {'max_depth': min_max_depth, 'learning_rate': min_learning_rate, 'n_estimator': min_n_estimator}
        return self.best_params
    
    def train_model_on_best_params(self, max_depths, learning_rates, n_estimators, features=None, scoring='roc_auc', end_evaluation=False):
        if type(features) == type(None): 
            features = self.X_train.columns
            
        self.features = features
        
        self.find_best_params(max_depths, learning_rates, n_estimators, features, scoring)
        
        learning_rate = self.best_params['learning_rate']
        max_depth = self.best_params['max_depth']
        n_estimators = self.best_params['n_estimator']
        
        self.model = xgb.XGBClassifier(max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        n_estimators=n_estimators,
                                        verbosity=0,
                                        objective='binary:logistic',
                                        booster='gbtree',
                                        n_jobs=-1,
                                        subsample=0.7,
                                    )
        
        # train the model 
        self.model.fit(X_train[features], y_train,
                    eval_set=[(X_valid[features], y_valid)],
                    early_stopping_rounds=100,
                    verbose=0)

        if end_evaluation: 
            self.print_kpi(features)
            
    def print_feature_importance(self, nbr_features_to_print=10):
        features=self.features
        dict_feature_importance = dict(zip(features, self.model.feature_importances_))
        dict_feature_importance = {k: v for k, v in sorted(dict_feature_importance.items(), key=lambda item: item[1], reverse=True)}

        fig, ax = plt.subplots()
        y_pos = np.arange(len(features))

        ax.barh(y_pos[:nbr_features_to_print], list(dict_feature_importance.values())[:nbr_features_to_print], align='center')
        ax.set_yticks(y_pos[:nbr_features_to_print])
        ax.set_yticklabels(list(dict_feature_importance.keys())[:nbr_features_to_print])
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Relative Importance in the Descision')
        ax.set_title('Feature Importance')
        plt.show()       
        return
    
    def return_k_most_important_features(self, k=1):
        features=self.features
        dict_feature_importance = dict(zip(features, self.model.feature_importances_))
        dict_feature_importance = {k_: v for k_, v in sorted(dict_feature_importance.items(), key=lambda item: item[1], reverse=True)}
        return np.array(list(dict_feature_importance.keys())[:k])