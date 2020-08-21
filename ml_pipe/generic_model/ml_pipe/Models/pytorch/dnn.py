import pandas as pd
import numpy as np
import math
import os
import dill
from generic_model.tool_config import config_methods
from datetime import datetime
import os
from torch import nn, optim
import torch
from .utils import DFDataset, EarlyStopping, binary_acc
from generic_model.ml_pipe import utils
from torch.utils.tensorboard import SummaryWriter

import pytorch_lightning.metrics.functional as pl_metrics 
from collections import Counter

import copy


class TorchDNN(nn.Module):
    def __init__(self, model_params):
        super(TorchDNN, self).__init__()
        self.block1, output_size = self.build_block(model_params = model_params,
                                        size = model_params['fb_initial_size'],
                                        num_layers = model_params['fb_layers'], 
                                        rate  = model_params['fb_rate'], 
                                        min_size =  model_params['fb_min_size'], 
                                        input_size = model_params['initial_size'])
        
        if model_params['sb_layers'] > 0:
            self.block2, output_size =  self.build_block(model_params = model_params,
                                            size = model_params['sb_initial_size'],
                                            num_layers = model_params['sb_layers'], 
                                            rate  = model_params['sb_rate'], 
                                            min_size =  model_params['sb_min_size'],
                                            input_size = output_size)
        
        else:
            self.block2 = None
        
        self.final_layer = nn.Linear(output_size, 1)
        
        # Initializations 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    
    def build_block(self, model_params, size, num_layers, rate, min_size, input_size):
        layers = []
        layers.append(nn.Linear(input_size, size))
        layers.append(self.get_activation(model_params['activation']))
        previous_size = int(size)
        
        if model_params['dropout'] > 0:
            layers.append(nn.Dropout(model_params['dropout']))
            
        for nl in np.arange(num_layers - 1):
            size = int(size * rate)
            if size < min_size: 
                size = min_size
            
            layers.append(nn.Linear(previous_size, size))
            previous_size = size
            
            layers.append(self.get_activation(model_params['activation']))
            if model_params['dropout'] > 0:
                layers.append(nn.Dropout(model_params['dropout']))
            
        return nn.Sequential(*layers), size
    
    def get_activation(self, activation):
        if activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'elu':
            return nn.ELU()
        elif activation.lower() == 'selu':
            return nn.SELU()
        else:
            return nn.ReLU()
            
    def forward(self, x):
        out = self.block1(x)
        if self.block2 is not None:
            out = self.block2(out)
        logit = self.final_layer(out)
        return logit # nn.functional.sigmoid(logit)
        

class DNNBuilder:
    """
    Class DNN in pytorch 
    
    this represents Deep Neural Network to run with ml-pipe in d6tflow 
    
    TODO: 
    * Custom loss / metrics
    * Class weight
    * Learning Rate decay
    
    Known Bugs: 
    Got error when pickling this class due to weakref. Not sure which object, maybe the strategy one. 
    Using Dill (instead of pickle) to deal with this, but gotta investigate more, if this will impact further
    """

    def init(self, 
                 model_params = {},
                 ds_params = {}):
        
        self.writer = None
        self.device = []
        self.ds_params = ds_params
        self.model_params =  model_params
        self.kwargs = {}
        self.configure_gpu(self.model_params['gpu'])
        self.model = None
        torch.manual_seed(42)
        self.initialize_logger()
            
    def __del__(self):
        self.release_devices()
        if self.writer is not None: 
            self.writer.close()
    
    def __hash__(self):
        return hash(repr(self))
    
    def initialize_logger(self):
        if self.model_params['visualize']:
            self.writer = SummaryWriter(self.model_params['tensorboard_path'])
        
    def tb_log_metrics(self, metrics, eval_metrics, iteration):
        if self.model_params['visualize']:
            for key in metrics.keys():
                values = {'Train' : metrics[key], 'Eval' : eval_metrics[key]}
                self.writer.add_scalars(key, values, iteration)
            
#             for name, value in metrics.items():
#                 self.writer.add_scalar(name + "/" + "Train" if training else "Eval", value, epoch)
        
    def tb_log_graph(self):
        if self.model_params['visualize']:
            self.writer.add_graph(self.model, torch.zeros(self.model_params['initial_size']))
    
    def save_model(self, path):
        utils.create_folder(path)
        torch.save(self.model.state_dict(), path + "/model.pt")
        model_input = { 
            'model_params' : self.model_params,
            'ds_params' : self.ds_params,
        }
        
        model_class = {'model' : 'dnn_pt'}
        
        with open(path + '/model_class.pkl', 'wb') as f:
            dill.dump(model_class, f)
        
        with open(path + '/config.pkl', 'wb') as f:
            dill.dump(model_input, f)
        
    @staticmethod
    def load_model(path):
        with open(path + '/config.pkl', 'rb') as f:
            config = dill.load(f)
            
        dnn = DNNBuilder(**config)
        dnn.build_model()
        dnn.model.load_state_dict(torch.load(path + "/model.pt"))
        return dnn
        
    def configure_gpu(self, use_gpu = False):
        self.device = torch.device("cuda" if use_gpu else "cpu")

    def release_devices(self):
        pass
#         if self.device is not None and config_methods.device_manager is not None: 
#             print('{0} - Process {1} Releasing'.format(datetime.now().strftime("%H:%M:%S-%f"), os.getpid()), self.device)
#             config_methods.device_manager.release(self.device)
            
    def build_dataset(self, df, label, batch_size = 32, use_gpu = False, training = True):
        loader = torch.utils.data.DataLoader(DFDataset(df, label), 
                                             batch_size = batch_size,
                                             shuffle = training, 
                                             pin_memory = use_gpu,
                                             num_workers = 4 if training else 1)
        return loader   
    
    def get_optimizer(self):
        if self.model is None: 
            raise Exception("Model should already be initialized")
        
        if self.model_params['optimizer'].lower() == "rmsprop":
            return optim.RMSprop(self.model.parameters(), lr = 0.001 * self.model_params['lr_rate_mult'], weight_decay = self.model_params['regularizer'])
        else:
            return optim.Adam(self.model.parameters(), lr = 0.001 * self.model_params['lr_rate_mult'], weight_decay = self.model_params['regularizer'])
    
    def build_model(self):
        self.model = TorchDNN(self.model_params)
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.optimizer = self.get_optimizer()
        self.loss_fn = nn.BCEWithLogitsLoss() # Sigmoid and BCE loss  ## nn.CrossEntropyLoss()
        if self.model_params['early_stopping_it'] < 0:
            patience = 1e9
        else:
            patience = self.model_params['early_stopping_it']
            
        self.early_stopping = EarlyStopping(patience = patience)
        
    def build(self, datasets, params):
        '''
        description: 
            trains model
        input: 
            datasets {'train_df' : dataframe, 'eval_df' : dataframe}
        output:
        '''
        
        self.name = params['name']
        
        X_train = datasets['train_df'].dropna()
        X_test = datasets['eval_df'].dropna()

        steps_per_epoch = len(X_train) // (self.model_params['batch_size'] * 10) # 10 epochs for completing the dataset 

        train_dataset = self.build_dataset(X_train, 
                                               self.ds_params['target'], 
                                               batch_size = self.model_params['batch_size'], 
                                               training = True,
                                               use_gpu = self.model_params['gpu'])
        
        eval_dataset = self.build_dataset(X_test, 
                                               self.ds_params['target'], 
                                               batch_size = self.model_params['batch_size'], 
                                               training = False,
                                               use_gpu = self.model_params['gpu'])
        
        input_size = len(X_train.columns) - 1 # - 1 due to TARGET column 
        
        if 'initial_size' in self.model_params.keys():
            if input_size != self.model_params['initial_size']:
                raise Exception("Model input with different dimension")
        else:
            self.model_params['initial_size'] = input_size
        
        self.build_model()
        self.tb_log_graph()
        
        self.early_stopping.on_train_begin()
        for epoch in np.arange(self.model_params['iterations']):
            tr_loss, tr_metrics = self.train(train_loader = train_dataset, test_loader = eval_dataset, epoch = epoch)
            val_loss, val_metrics = self.test(eval_dataset)
            
            if self.model_params['log_interval'] < 0 and self.model_params['visualize']:
                self.tb_log_metrics({"loss" : tr_loss, **tr_metrics}, 
                                      {"loss" : val_loss, **val_metrics}, 
                                      (epoch + 1) * len(train_dataset))
            
            if self.early_stopping.on_epoch_end(epoch, val_loss):
                break
        
        self.early_stopping.on_train_end()
        
    def calculate_metrics(self, pred, target, prvs_metrics = None):
#          output.cpu().detach().numpy()
        metrics = {'accuracy' : pl_metrics.accuracy(torch.round(pred), torch.round(target)),
                       'aucroc'   : pl_metrics.auroc(pred, target),
                       'f1_score' : pl_metrics.f1_score(torch.round(pred), torch.round(target))} 
        
        if prvs_metrics is not None: 
            for key in metrics.keys():
                metrics[key] += prvs_metrics[key]
            metrics['counter'] = prvs_metrics['counter'] + 1
        else:
            metrics['counter'] = 1

        return metrics
        
    def train(self, train_loader, test_loader, epoch):
        running_loss = 0
        epoch_loss = 0
        metrics = None
        epoch_metrics = None
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target.float().unsqueeze(1))
            loss.backward()
            self.optimizer.step()            
            running_loss += loss.item()
            epoch_loss += loss.item()
            
            if self.model_params['visualize']:
                metrics = self.calculate_metrics(torch.sigmoid(output).squeeze(1), target.float(), metrics)
                epoch_metrics = copy.deepcopy(metrics)
            
            if batch_idx > 0 and self.model_params['log_interval'] > 0 and batch_idx % self.model_params['log_interval'] == 0:
                if self.model_params['visualize']:
                    val_loss, val_metrics = self.test(test_loader)
                    metrics = self.normalize_metrics(metrics)
                    
                    self.tb_log_metrics({"loss" : running_loss / metrics['counter'], **metrics},
                                          {"loss" : val_loss, **val_metrics},
                                          epoch * len(train_loader) + batch_idx + 1)
                
                    metrics = None # reseting metrics calculations
                    running_loss = 0 # reseting loss
                    self.model.train() ## get back to training mode
                
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                
                
        
        return epoch_loss / len(train_loader), self.normalize_metrics(epoch_metrics)
             
    
    def normalize_metrics(self, metrics):
        if metrics is not None:
            for k in metrics.keys():
                if k == "counter":
                    continue
                metrics[k] /= metrics['counter']         
        
        return metrics

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0
        metrics = None
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_fn(output, target.float().unsqueeze(1)).item()  # sum up batch loss
                
                if self.model_params['visualize']:
                    metrics = self.calculate_metrics(torch.sigmoid(output).squeeze(1), target.float(), metrics)

        test_loss /= len(test_loader)
        print('\nTest set: Avg loss: {:.4f}\n'.format(test_loss))
        
        metrics = self.normalize_metrics(metrics)
        
        return test_loss, metrics
        
    def get_output(self, predict_loader):
        y_pred_list = np.array([])
        self.model.eval()
        metrics = None
        with torch.no_grad():
            for data in predict_loader:
                data = data.to(self.device)
                output = torch.sigmoid(self.model(data))
                y_pred_list = np.append(y_pred_list, output.cpu().detach().numpy().squeeze())

        return y_pred_list
        
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
        
        predict_dataset = self.build_dataset(X_predict, 
                                               None, 
                                               batch_size = self.model_params['batch_size'], 
                                               training = False,
                                               use_gpu = self.model_params['gpu'])
        
        
        y_pred = self.get_output(predict_dataset)
        
        scoring['PREDICTED'] = y_pred

        scoring = scoring.rename(columns = {
            self.ds_params['date_col'] : 'DATE',
            self.ds_params['target'] : 'TARGET',
        })
        
        return scoring