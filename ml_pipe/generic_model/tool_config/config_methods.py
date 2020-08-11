# DS CREATORS

from generic_model.ml_pipe.DSGenerators import CreateDS

# FE ENGINEERING
from generic_model.ml_pipe.FeatureEngineering.PrepareDS_xgb import PrepareDSXgb

# METRICS
from generic_model.ml_pipe.Metrics import abs_perc_err, auc, elasticity_err, accuracy, f1_score

# MODELS
from generic_model.ml_pipe.Models import dnn, xgb
from generic_model.ml_pipe.Models import pytorch as pt
import generic_model.ml_pipe.Models.keras.device_manager as dev

# Utils
from generic_model.ml_pipe.Utils import price_elasticity

# OTHER LIBS
from multiprocessing import Process, Manager, Lock
from multiprocessing.managers import BaseManager

train_eval_creator = {
    'gcp_bq' : lambda params: CreateDS.CreateDataSet(**params)}


predict_score_creator = {
     'gcp_bq' : lambda params: CreateDS.CreateDataSet(**params)}

engineer_creator = {
    'one_hot_mean_enc' : lambda params: PrepareDSXgb(**params)
}

model_creator = {
    'xgboost' : lambda params: xgb.XGB(**params),
    'dnn'     : lambda params: dnn.DNN(**params),
    'dnn_pt'  : lambda params: pt.dnn.DNNBuilder(**params),
    'sdnn_pt' : lambda params: pt.sdnn.SDNBuilder(**params)}

elasticity = price_elasticity.Elasticity([])

metrics_creator = {
    'abs_perc_err'   : lambda params: abs_perc_err.AbsPercErr(**params),
    'auc'            : lambda params: auc.Auc(**params),
    'elasticity_err' : lambda params: elasticity_err.ElasticityErr(**params),
    'accuracy'       : lambda params: accuracy.BinaryAccuracy(**params),
    'f1_score'       : lambda params: f1_score.F1Score(**params)
}

optimization_lock = Lock()

## Using Manager / Proxy objects because DeviceManager will be shared in different processes
## This approach is not recommended if DeviceManager is more complex (like a nested class). It would be better to use low-level methods
# device_manager = dev.DeviceManager()
BaseManager.register('DeviceManager', dev.DeviceManager)
manager = BaseManager()
manager.start()
device_manager = manager.DeviceManager()