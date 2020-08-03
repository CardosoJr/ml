import pandas as pd 
import numpy as np 

class Elasticity:
    def __init__(self, variable_converters):
        self.variable_converters = variable_converters
        
    def transform_df(self, df, n_min, n_max, step):
        
        for func in variable_converters:
            df = func(df)
            
        return df
    
    def compile_report(self, df, n_min, n_max, step, output = 'output', model_name = 'test'):
        pass