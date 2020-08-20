import pandas as pd
import numpy as np
import tensorflow as tf
import dill
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

        
    def build_block(self, input_size):
        inputs = tf.keras.Input(shape = (input_size,))
        r = tf.keras.layers.Dense(units = self.model_params['layer_size'],
                                        activation = self.model_params['activation'],
                                        kernel_regularizer = tf.keras.regularizers.l2(self.model_params['regularizer']))(inputs)
        
        if self.model_params['dropout'] > 0:
            r = tf.keras.layers.Dropout(rate = self.model_params['dropout'])(r)
        
        r = tf.keras.layers.Dense(1, activation = 'sigmoid')(r)
        
        model = tf.keras.Model(inputs = inputs, outputs = r)
        
        return model
    

    def get_compiled_model(self):
        model_layers = []
        blocks = []
        for i in np.arange(self.model_params['num_blocks']):
            blocks.append(self.build_block(self.model_params['initial_size'] + i))

        inputs = tf.keras.Input(shape = (self.model_params['initial_size'],))
        outputs = blocks[0](inputs)
        block_input = tf.concat([inputs, outputs], 1)
        
        for index in np.arange(1, self.model_params['num_blocks']):
            outputs = blocks[index](block_input)
            block_input = tf.concat([block_input, outputs], 1)
        
        model = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'sdnn')
        
#         tf.keras.utils.plot_model(model = model, to_file = 'model.png', show_shapes = True, expand_nested = True)
        
        model.compile(optimizer = self.get_optimizer(),
                           loss = 'binary_crossentropy',
                           metrics = [tf.keras.metrics.BinaryAccuracy(name = 'binary_acuracy'), 
                                      tf.keras.metrics.AUC(name = 'auc'),
                                      tf.keras.metrics.AUC(name = 'auc_pr', curve = 'PR'),
                                      tf.keras.metrics.Mean(name = 'mean')])
        
        return model