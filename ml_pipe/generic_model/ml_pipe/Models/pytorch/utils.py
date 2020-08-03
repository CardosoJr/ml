from torch.utils.data import Dataset
from torch import nn, optim
import torch

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


class DFDataset(Dataset):
    def __init__(self, df, target_field):
        self.has_target  = False
        if target_field is not None:
            self.has_target = True
            self.target = df[target_field].values # astype("float32")
        self.data = df.loc[:, df.columns != target_field].astype("float32").values
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.has_target:
            return [self.data[idx], self.target[idx]]
        else:
            return self.data[idx]


class EarlyStopping():
    def __init__(self, 
                 min_delta=0,
                 patience=5):
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e-15
        self.stopped_epoch = 0

    def on_train_begin(self):
        self.wait = 0
        self.best_loss = 1e15

    def on_epoch_end(self, epoch, current_loss):
        stop_training = False                                   
        if current_loss is not None:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    stop_training = True
                self.wait += 1
                                           
        return stop_training

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print('\nTerminated Training for Early Stopping at Epoch %04i' % 
                (self.stopped_epoch))