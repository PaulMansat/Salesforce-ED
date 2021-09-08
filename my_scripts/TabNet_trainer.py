import torch
import numpy as np
import pandas as pd
import my_scripts.data_cleaning as dc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import mean_absolute_error, auc, roc_curve, accuracy_score, matthews_corrcoef, f1_score 

# target should already be encoded with O or 1
class TabNet_binary_classifer_trainer:
    def __init__(self, df, target, n_d=15, n_a=15, lr=1e-1, lambda_sparse=1e-5, step_size=75, gamma=0.7):
        # save values params
        self.n_d = n_d
        self.n_a = n_a
        self.lr = lr
        self.lambda_sparse = lambda_sparse
        self.step_size = step_size
        self.gamma = gamma
        
        # ordinal encoding 
        df, self.mappings, self.cat_dims_full = self.data_encoding(df)
        X, y = df.drop(columns=[target]), df[target]

        # split train, validation, test set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, test_size=0.2)
        
        # original column ordering 
        self.original_ordering = np.array(self.X_train.columns)
        self.categorical_features = list(self.mappings.keys())
        
        # initialy, no assumption on features on which model should be trained 
        self.features = self.X_train.columns
        self.all_features = self.X_train.columns
        
        # convert to numpy array
        self.X_train = self.X_train.values
        self.X_test = self.X_test.values
        self.X_valid = self.X_valid.values
        self.y_train = self.y_train.values
        self.y_test = self.y_test.values
        self.y_valid = self.y_valid.values
        
        # model initialize later because requires categorical encodings  
        self.model = None
        self.cat_idxs = None
        self.cat_dims = None
        
    def print_kpi(self, selection_mask):
        y_pred = self.model.predict(self.X_test[:,selection_mask])
        # plot some relevant KPI
        print('Accuracy: {:.2f}'.format(accuracy_score(y_pred, self.y_test)))
        print('MCC Score: {:.2f}'.format(matthews_corrcoef(self.y_test, y_pred)))
        print('F1 Score: {:.2f}'.format(f1_score(self.y_test, y_pred)))

        # plot ROC curve 
        fpr, tpr, _ = roc_curve(self.y_test, self.model.predict_proba(self.X_test[:,selection_mask])[:,1])
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Area = {:.5f}".format(auc(fpr, tpr)));
    
    def get_model(self):
        return self.model
        
    def reinitialize_model(self, cat_idxs, cat_dims):
        self.model = TabNetClassifier(
                           n_d=self.n_d, n_a=self.n_a,
                           cat_idxs=cat_idxs,
                           cat_dims=cat_dims,
                           cat_emb_dim=1,
                           optimizer_fn=torch.optim.Adam,
                           optimizer_params=dict(lr=sel.lr),
                           lambda_sparse=self.lambda_sparse,
                           n_steps=7,
                           scheduler_params={"step_size":self.step_size, # how to use learning rate scheduler
                                             "gamma":self.gamma},
                           scheduler_fn=torch.optim.lr_scheduler.StepLR,
                           mask_type='sparsemax',
                           verbose=verbose)
        
        
        
    def data_encoding(self, df): 
        mappings = {}
        cat_dims_full = {}
        for col in dc.non_numerical_features(df):
            df, mapping = dc.feature_ordinalEncoding(df, col)
            mappings[col] = mapping
            cat_dims_full[col] = len(mapping)
            
        return df, mappings, cat_dims_full
    
    def get_cat_idxs(self, selected_features):  
        ordered_selected_features = [col for col in self.all_features if col in selected_features]
    
        cat_idxs = [idx for idx, col in enumerate(ordered_selected_features) if col in selected_features and col in self.categorical_features]
                
        cat_dims = [self.cat_dims_full[col] for col in ordered_selected_features if col in self.categorical_features]
        
        return cat_idxs, cat_dims
    
    def get_selection_mask(self, selected_features):
        is_selected = lambda x, list_: True if x in list_ else False
        selection_mask = [is_selected(col, selected_features) for col in self.all_features]
        return selection_mask
    
    
    def train_model(self, max_epochs=50, patience=50, reinitialize_model=False, features=None, 
                    eval_metric=['auc'], verbose=0, end_evaluation=False, weights=0):
        if type(features) == type(None):
            features = self.all_features
        
        self.features = features
            
        if self.model == None or reinitialize_model:
            cat_idxs, cat_dims = self.get_cat_idxs(features)
            self.model = TabNetClassifier(
                                   n_d=self.n_d, n_a=self.n_a,
                                   cat_idxs=cat_idxs,
                                   cat_dims=cat_dims,
                                   cat_emb_dim=1,
                                   optimizer_fn=torch.optim.Adam,
                                   optimizer_params=dict(lr=self.lr),
                                   lambda_sparse=self.lambda_sparse,
                                   n_steps=7,
                                   scheduler_params={"step_size":self.step_size, # how to use learning rate scheduler
                                                     "gamma":self.gamma},
                                   scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                   mask_type='sparsemax',
                                   verbose=verbose)
    
        selection_mask = self.get_selection_mask(features)
        
        self.model.fit(
            X_train=self.X_train[:,selection_mask], y_train=self.y_train,
            eval_set=[(self.X_train[:,selection_mask], self.y_train), (self.X_valid[:,selection_mask], self.y_valid)],
            eval_name=['train', 'valid'],
            eval_metric=eval_metric,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=1024, virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            weights=weights
        )
        
        if end_evaluation:
            self.print_kpi(selection_mask)
    
    def print_feature_importance(self, nbr_features_to_print=10):
        dict_feature_importance = dict(zip(self.features, self.model.feature_importances_))
        dict_feature_importance = {k: v for k, v in sorted(dict_feature_importance.items(), key=lambda item: item[1], reverse=True)}

        fig, ax = plt.subplots()

        y_pos = np.arange(len(self.features))

        ax.barh(y_pos[:nbr_features_to_print], list(dict_feature_importance.values())[:nbr_features_to_print], align='center')
        ax.set_yticks(y_pos[:nbr_features_to_print])
        ax.set_yticklabels(list(dict_feature_importance.keys())[:nbr_features_to_print])
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Relative Importance in the Descision')
        ax.set_title('Feature Importance')
        plt.show()
    
        return 
        
        
        