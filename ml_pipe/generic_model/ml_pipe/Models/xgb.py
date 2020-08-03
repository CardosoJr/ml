from google.cloud import storage
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import math
from datetime import datetime
from datetime import timedelta  
from dateutil.relativedelta import relativedelta
from google.cloud import bigquery
from google.cloud import storage
import os
import dill
from generic_model.ml_pipe import utils

class XGB:
    def __init__(self, 
                 model_params = {},
                 ds_params={}):

        self.ds_params = ds_params
        self.metric = 'logloss'
        self.xgb_params =  model_params
    
    def __hash__(self):
        return hash(repr(self))
    
    def save_model(self, path):
        utils.create_folder(path)
        
        model_class = {'model' : 'xgboost'}
        with open(path + '/model_class.pkl', 'wb') as f:
            dill.dump(model_class, f)
            
        with open(path + '/model.pkl', 'wb') as f:
            dill.dump(self, f)
        
    @staticmethod
    def load_model(path):
        with open(path + '/model.pkl', 'rb') as f:
            xgb = dill.load(f)
            
        return xgb
    
    def elasticity_transform(self, df, var, factor):
        if var  == 'PREM_EMITIDO_ATU':
            df[var] = df['PREM_EMITIDO_ANUAL'] * factor
        elif var == 'TX_COMER_RENOV':
            mask = df[var] > 0.0
            df[mask][var] = np.round(df[mask]['PREM_EMITIDO_ANUAL'] * factor / self.lmr_casco[mask], 4)
        elif var == 'DIFERENCA_50C':
            base_round = 50.0
            df[var] = np.clip(base_round * np.round(df['PREM_EMITIDO_ANUAL'] * (factor - 1.0) / base_round), -5000.0, 5000.0).replace(to_replace = 0, value = 50)
        elif var == 'RAZAO_C':
            df[var] = np.clip(np.round(factor - 1, 2), -0.5, 1.0)
        elif var == 'SP_RENOV':
            df[var] = np.round(self.pr_total_ajuste_tend_env / (df['PREM_EMITIDO_ANUAL'] * factor), 4)
        return df
    
        
    def get_eval_results(self):
        eval_results = self.model.evals_result()
        keys = [int(x.split("_")[-1]) for x in eval_results.keys()]
        latest_eval = eval_results['validation_{0}'.format(np.max(keys))]
        return (self.metric, np.max(latest_eval[self.metric]))
                
        
    def build(self, datasets, params):
        '''
        description: 
            trains model
        input: 
            datasets {'train_df' : dataframe, 'eval_df' : dataframe}
        output:
        '''
        
        self.name = params['name']
        X_train = datasets['train_df']
        X_test = datasets['eval_df']

        y_train = X_train[self.ds_params['target']]
        X_train = X_train.drop(columns = [self.ds_params['target']])

        y_test = X_test[self.ds_params['target']]
        X_test = X_test.drop(columns = [self.ds_params['target']])

        self.model = XGBClassifier(**self.xgb_params)
        self.model.fit(X_train, 
                  y_train, 
                  eval_set=[(X_test, y_test)],
                  early_stopping_rounds = 500)

    def predict(self, datasets, params):
        '''
        description: 
            uses model to make predictions
        input: 
            datasets {'predict_df': dataframe, 'scoring_df' : dataframe}
        output:
        '''
        
        X_predict = datasets['predict_df']
        scoring = datasets['scoring_df']
        y_predict = X_predict[self.ds_params['target']]
        X_predict = X_predict.drop(columns = [self.ds_params['target']])
        
        X_predict_hat = X_predict.copy(deep = True)
        print(params)
        if params['elasticity_transform']:
            
            num_periods =  params['elasticity_periods']
            cut_off_date = scoring[self.ds_params['date_col']].min() + relativedelta(months = num_periods)
            mask = scoring[self.ds_params['date_col']] < cut_off_date
            
            # Filter X_Predict based on periods passed by parameters
            X_predict_hat = X_predict_hat[mask]
            
            # Filter Dates (dates need to have columns to calculate all)
            scoring = scoring[mask]
            
            # THIS IS A WORKAROUND, DATES DF SHOULD HAVE THIS VARIABLES
            scoring['PR_TOTAL_AJUST_TEND_ENV'] = X_predict_hat['SP_RENOV'] * X_predict_hat['PREM_EMITIDO_ATU']
            self.lmr_casco = X_predict_hat['PREM_EMITIDO_ATU'] / X_predict_hat['TX_COMER_RENOV']
            self.pr_total_ajuste_tend_env = X_predict_hat['SP_RENOV'] * X_predict_hat['PREM_EMITIDO_ATU']

            for var in params['elasticity_col']:
                X_predict_hat = self.elasticity_transform(X_predict_hat, var, params['elasticity_factor'])
            
            scoring['PREM_EMITIDO_ATU'] = X_predict_hat['PREM_EMITIDO_ATU']
            
        
        y_pred = self.model.predict_proba(X_predict_hat)
        scoring['PREDICTED'] = list(y_pred[:,0])

        scoring = scoring.rename(columns = {
            self.ds_params['date_col'] : 'DATE',
            self.ds_params['target'] : 'TARGET',
        })
        
        scoring['PREDICTED'] = 1 - scoring['PREDICTED']
        self.results = scoring
        return self.results