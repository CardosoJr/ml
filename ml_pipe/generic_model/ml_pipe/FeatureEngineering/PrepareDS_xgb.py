from .Encoding import kfold_target_encoder

from google.cloud import storage
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import category_encoders as ce
import pandas as pd
import numpy as np
import math
from datetime import datetime
from datetime import timedelta  
from dateutil.relativedelta import relativedelta
from google.cloud import bigquery
from google.cloud import storage

import os

class PrepareDSXgb:
    def __init__(self, small_categorical, large_categorical, variable_set, target = 'QTD_PROP', date = 'DAT_FIM_VIGENCIA'):
        self.small_categorical = small_categorical
        self.large_categorical = large_categorical
        self.variable_set = variable_set
        self.target = target
        self.date = date
        
    def encoding_small(self, X_train, X_test, X_predict):
        print('Encoding Small')
        variables = [x for x in self.small_categorical if x in self.variable_set]
        self.enc = ce.OneHotEncoder(cols = variables, drop_invariant=False, use_cat_names=True).fit(X_predict)
        X_predict = self.enc.transform(X_predict)
        X_train = self.enc.transform(X_train)
        X_test = self.enc.transform(X_test)
        
        return X_train, X_test, X_predict
        
    def __hash__(self):
        return hash(repr(self))
    
    def encoding_large(self, X_train, X_test, X_predict):
        variables = [x for x in self.large_categorical if x in self.variable_set]
        
        print('Encoding Large')
        t, e, v = kfold_target_encoder(train = X_train, 
                                       test = X_test, 
                                       valid = X_predict, 
                                       cols_encode = variables, 
                                       target = self.target, 
                                       folds = 10)
        
        X_train = X_train.drop(columns = variables).join(t)
        X_test = X_test.drop(columns = variables).join(e)
        X_predict = X_predict.drop(columns = variables).join(v)
        
        return X_train, X_test, X_predict
        
    def prepare(self, train_df, eval_df, predict_df, scoring_df):
        X_train = train_df[self.variable_set + [self.target]]
        X_test = eval_df[self.variable_set + [self.target]]
        X_predict = predict_df[self.variable_set + [self.target]]
        scoring = scoring_df
        scoring[self.date] = pd.to_datetime(scoring[self.date])

        all_categorical = self.large_categorical + self.small_categorical
        variables = [x for x in all_categorical if x in self.variable_set]

        # Label Encoding data 
        for var in variables:
            print('Encoding {0}'.format(var))
            X_predict[var] = X_predict[var].fillna('nan')
            X_train[var] = X_train[var].fillna('nan')
            X_test[var] = X_test[var].fillna('nan')

        X_train, X_test, X_predict = self.encoding_small(X_train, X_test, X_predict)
        X_train, X_test, X_predict = self.encoding_large(X_train, X_test, X_predict)
        
        return {'train_df' : X_train,
                
                'eval_df' : X_test, 
                
                'predict_df' : X_predict,
               
                'scoring_df' : scoring}    