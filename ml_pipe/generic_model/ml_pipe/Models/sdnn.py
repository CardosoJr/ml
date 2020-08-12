import pandas as pd
import numpy as np
import math
import os
import tensorflow as tf
import dill
from generic_model.tool_config import config_methods
from datetime import datetime
import os

from .dnn import DNN

class SDNN(DNN):
    """
    Class DNN 
    
    this represents Stacked Deep Neural Network to run with ml-pipe in d6tflow 
    
    Known Bugs: 
    Got error when pickling this class due to weakref. Not sure which object, maybe the strategy one. 
    Using Dill (instead of pickle) to deal with this, but gotta investigate more, if this will impact further
    """

    def __init__(self, 
                 model_params = {},
                 ds_params = {}):
        super().__init__(model_params, ds_params)

    def save_model(self, path):
        self.model.save(path)
        model_input = { 
            'model_params' : self.model_params,
            'ds_params' : self.ds_params,
        }
        
        model_class = {'model' : 'sdnn'}
        
        with open(path + '/model_class.pkl', 'wb') as f:
            dill.dump(model_class, f)
        
        with open(path + '/config.pkl', 'wb') as f:
            dill.dump(model_input, f)
        
    @staticmethod
    def load_model(path):
        with open(path + '/config.pkl', 'rb') as f:
            config = dill.load(f)
            
        dnn = SDNN(**config)
        dnn.model =  tf.keras.models.load_model(path)
        return dnn

        
    def build_block(self, size,  block_input):
        r = tf.keras.layers.Dense(units = size,
                                        activation = self.model_params['activation'],
                                        kernel_regularizer = tf.keras.regularizers.l2(self.model_params['regularizer']))(block_input)
        
        if self.model_params['dropout'] > 0:
            r = tf.keras.layers.Dropout(rate = self.model_params['dropout'])(r)
        
        r = tf.keras.layers.Dense(1, activation = 'sigmoid')(r)
            
        return r
    

    def get_compiled_model(self):
        model_layers = []

        nn_input = tf.keras.Input(shape = (self.model_params['initial_size'],))
        r = self.build_block(self.model_params['layer_size'], nn_input)
        block_input = tf.identity(nn_input)
        
        for index in np.arange(1, model_params['num_blocks']):
            block_input = tf.concat([block_input, r], 0)
            r = self.build_block(self.model_params['layer_size'], block_input)
            
        
        model = tf.keras.Model(inputs = nn_input, outputs = r, name = 'sdnn')
        
        model.compile(optimizer = self.get_optimizer(),
                           loss = 'binary_crossentropy',
                           metrics = [tf.keras.metrics.BinaryAccuracy(name = 'binary_acuracy'), 
                                      tf.keras.metrics.AUC(name = 'auc'),
                                      tf.keras.metrics.AUC(name = 'auc_pr', curve = 'PR'),
                                      tf.keras.metrics.Mean(name = 'mean')])
        
        return model