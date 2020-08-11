import pandas as pd
from dateutil.relativedelta import *
from datetime import datetime
import numpy as np
from sklearn import metrics

from .auc import Auc

class BinaryAccuracy(Auc):
    def __init__(self, 
                 threshold = 0.5,
                 date_col = 'DATE',
                 model_output = 'PREDICTED',
                 target = 'TARGET', 
                 group_col = 'group',
                 freq = 'm'):
        super().__init__(date_col, model_output, target, group_col, freq)
        self.threshold = threshold
        
    def calculate_metric(self, y_test, y_pred):
        return metrics.accuracy_score(y_test, np.where(np.array(y_pred) >= self.threshold, 1, 0))
    
    def define_metric_name(self):
        return 'accuracy'