import pandas as pd
import numpy as np
import math
import os
import tensorflow as tf
import dill
from generic_model.tool_config import config_methods
from datetime import datetime
import os

class DNN:
    """
    Class DNN 
    
    this represents Deep Neural Network to run with ml-pipe in d6tflow 
    
    TODO: 
    * Custom loss / metrics
    * Checkpoint
    * Tensorboard
    * Class weight
    * Learning Rate decay
    
    Known Bugs: 
    Got error when pickling this class due to weakref. Not sure which object, maybe the strategy one. 
    Using Dill (instead of pickle) to deal with this, but gotta investigate more, if this will impact further
    """

    def __init__(self, 
                 model_params = {},
                 ds_params = {}):
        
        self.use_gpu = len(tf.config.experimental.list_physical_devices('GPU')) > 0 and model_params['gpu']
        
        self.__configure_tf()     
        self.device = None
              
#         tf.debugging.set_log_device_placement(True)
        self.ds_params = ds_params
        self.model_params =  model_params
        strategy = self.configure_gpu()
        if strategy is not None: 
            with strategy.scope():
                self.model = self.get_compiled_model()
        else:
            self.model= self.get_compiled_model()    
            
            
    def __del__(self):
        self.release_devices()
    
    def __hash__(self):
        return hash(repr(self))
    
    def __configure_tf(self):
        if self.use_gpu:
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
    
    def save_model(self, path):
        self.model.save(path)
        model_input = { 
            'model_params' : self.model_params,
            'ds_params' : self.ds_params,
        }
        
        model_class = {'model' : 'dnn'}
        
        with open(path + '/model_class.pkl', 'wb') as f:
            dill.dump(model_class, f)
        
        with open(path + '/config.pkl', 'wb') as f:
            dill.dump(model_input, f)
        
    @staticmethod
    def load_model(path):
        with open(path + '/config.pkl', 'rb') as f:
            config = dill.load(f)
            
        dnn = DNN(**config)
        dnn.model =  tf.keras.models.load_model(path)
        return dnn
        
    def make_or_restore_model(self):
        # Either restore the latest model, or create a fresh one
        # if there is no checkpoint available.
        checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print("Restoring from", latest_checkpoint)
            return keras.models.load_model(latest_checkpoint)
        print("Creating a new model")
        return get_compiled_model()
    
    def configure_gpu(self):
        if self.use_gpu:
            print(tf.config.experimental.list_physical_devices())
            
            self.device = config_methods.device_manager.get_available_device()
            self.device.append("/cpu:0")
            print('{0} - Process {1} Acquiring'.format(datetime.now().strftime("%H:%M:%S-%f"), os.getpid()), self.device)

            all_devs = tf.config.list_physical_devices()
            aux_devs = ["/physical_device:" + x.upper().replace("/", "") for x in self.device]
            selected_devs = [x for x in all_devs if x.name in aux_devs]
            print("Selected Devices", self.device)
            
            tf.config.experimental.set_visible_devices(selected_devs)
            strategy = tf.distribute.MirroredStrategy(devices = self.device)
            print('Using GPU. Number of devices: {}'.format(strategy.num_replicas_in_sync))
            return strategy
        else:
            print('Not using GPU')
            self.device = None
            tf.config.experimental.set_visible_devices([], 'GPU')
            return None
        
    def release_devices(self):
        if self.device is not None and config_methods.device_manager is not None: 
            print('{0} - Process {1} Releasing'.format(datetime.now().strftime("%H:%M:%S-%f"), os.getpid()), self.device)
            config_methods.device_manager.release(self.device)
            
    def get_early_stopping_callback(self, metric, epochs):
        return tf.keras.callbacks.EarlyStopping(monitor = metric, patience = epochs, mode = 'auto', verbose = 1)
    
    def get_tensorboard_callback(self):
        return tf.keras.callbacks.TensorBoard(log_dir = self.model_params['tensorboard_path'],
                                              update_freq = 'epoch' if self.model_params['log_interval'] < 0 else self.model_params['log_interval'])
    
    def build_dataset(self, df, label, batch_size = 1, training = True):
        target = df.pop(label)
        dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
        
        if training:
            dataset = dataset.shuffle(buffer_size = 1000).repeat().batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)
            
        dataset = dataset.prefetch(buffer_size = 1)
        
        return dataset     
    
    def build_predict_dataset(self, df, batch_size = 1):
        dataset = tf.data.Dataset.from_tensor_slices(df.values).batch(batch_size)
        return dataset     
        
    def build_block(self, size, num_layers, rate, min_size):
        layers = []
        layers.append(tf.keras.layers.Dense(units = size,
                                            activation = self.model_params['activation'],
                                            kernel_regularizer = tf.keras.regularizers.l2(self.model_params['regularizer'])))
        if self.model_params['dropout'] > 0:
            layers.append(tf.keras.layers.Dropout(rate = self.model_params['dropout']))
        for nl in np.arange(num_layers - 1):
            size = size * rate
            if size < min_size: 
                size = min_size
            
            layers.append(tf.keras.layers.Dense(units = size,
                                                activation = self.model_params['activation'],
                                                kernel_regularizer = tf.keras.regularizers.l2(self.model_params['regularizer'])))
            if self.model_params['dropout'] > 0:
                layers.append(tf.keras.layers.Dropout(rate = self.model_params['dropout']))
            
        return layers
    
    def get_learning_rate(self):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.001 * self.model_params['lr_rate_mult'],
                                                                       decay_steps = 10000,
                                                                       decay_rate = 0.96,
                                                                       staircase = True)
        
        return lr_schedule
        
    def get_optimizer(self):
        if self.model_params['optimizer'].lower() == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate = self.get_learning_rate())
        elif self.model_params['optimizer'].lower() == "nadam":
            return tf.keras.optimizers.Nadam(learning_rate = self.get_learning_rate())
        else:
            return tf.keras.optimizers.Adam(learning_rate = self.get_learning_rate())
    
    def get_compiled_model(self):
        model_layers = []
        
        model_layers.extend(self.build_block(size = self.model_params['fb_initial_size'],
                                        num_layers = self.model_params['fb_layers'], 
                                        rate  = self.model_params['fb_rate'], 
                                        min_size =  self.model_params['fb_min_size']))
        
        if self.model_params['sb_layers'] > 0:
            model_layers.extend(self.build_block(size = self.model_params['sb_initial_size'],
                                            num_layers = self.model_params['sb_layers'], 
                                            rate  = self.model_params['sb_rate'], 
                                            min_size =  self.model_params['sb_min_size']))
        
        
        if self.model_params['batch_norm']:
            model_layers.append(tf.keras.layers.BatchNormalization())
        
        # last layer
        model_layers.append(tf.keras.layers.Dense(1, activation = 'sigmoid'))
        
        model = tf.keras.Sequential(model_layers)
        model.compile(optimizer = self.get_optimizer(),
                           loss = 'binary_crossentropy',
                           metrics = [tf.keras.metrics.BinaryAccuracy(name = 'binary_acuracy'), 
                                      tf.keras.metrics.AUC(name = 'auc'),
                                      tf.keras.metrics.AUC(name = 'auc_pr', curve = 'PR'),
                                      tf.keras.metrics.Mean(name = 'mean')])
        
        return model
                
        
    def build(self, datasets, params):
        '''
        description: 
            trains model
        input: 
            datasets {'train_df' : dataframe, 'eval_df' : dataframe}
        output:
        '''
        
        self.name = params['name']

        print('DROPNA MISSING', len(datasets['train_df'].dropna()) / len(datasets['train_df']))
        
        X_train = datasets['train_df'].dropna()
        X_test = datasets['eval_df'].dropna()

        steps_per_epoch = len(X_train) // (self.model_params['batch_size'] * 10) # 10 epochs for completing the dataset 

        train_dataset = self.build_dataset(X_train, 
                                               self.ds_params['target'], 
                                               batch_size = self.model_params['batch_size'], 
                                               training = True)
        
        eval_dataset = self.build_dataset(X_test, 
                                               self.ds_params['target'], 
                                               batch_size = self.model_params['batch_size'], 
                                               training = False)
        
        
        callbacks = [self.get_early_stopping_callback('auc', self.model_params['early_stopping_it'])]
        
        if self.model_params['visualize']:
            callbacks.append(self.get_tensorboard_callback())
        
        self.model.fit(
                    train_dataset,
                    epochs = self.model_params['iterations'],
                    steps_per_epoch = steps_per_epoch,
                    validation_data = eval_dataset, 
                    callbacks = callbacks)
        
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
        
        predict_dataset = self.build_predict_dataset(X_predict, self.model_params['batch_size'])
        y_pred = self.model.predict(predict_dataset)
        
        scoring['PREDICTED'] = np.squeeze(list(y_pred))

        scoring = scoring.rename(columns = {
            self.ds_params['date_col'] : 'DATE',
            self.ds_params['target'] : 'TARGET',
        })
        
        return scoring