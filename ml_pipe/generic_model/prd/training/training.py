import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
from   google.cloud import storage
import os
import d6tflow
import luigi
import gc
import yaml
import datetime
import modelos.xgboost.xgboost_tasks as tasks
import pickle
from   pathlib import Path
import lib.Utils as Utils

# Load do arquivo de configuração geral
with open('config_geral.yaml', 'r') as stream:
    config_geral = yaml.safe_load(stream) 

    
# PROJECT       = config_geral['PROJECT']
# BUCKET_NAME   = config_geral['BUCKET_NAME']
VERSAO_CODIGO = config_geral['VERSAO_CODIGO']
GCP_BASE_PATH = config_geral['GCP_BASE_PATH']

def config_pre_tratamento(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    
            
    config['table_dataset'] = "`" + config['table_dataset'] + "`"
    
    if isinstance(config['train_begin'], str):
        config['train_begin'] = datetime.datetime.strptime(config['train_begin'],'%Y-%m-%d')
        config['train_end']   = datetime.datetime.strptime(config['train_end'],  '%Y-%m-%d')
#     config['train_begin'] = datetime.datetime(config['train_begin'].year, config['train_begin'].month, config['train_begin'].day)
#     config['train_end']   = datetime.datetime(config['train_end'].year, config['train_end'].month, config['train_end'].day)
    return config

#Pegando os parâmetros necessários para rodar o modelo
def get_tasks(config):
    
    task_te_params = {
        
        'project': config['project'],
        'table_dataset' : config['table_dataset'],
        'period_begin' : config['train_begin'],
        'period_end' : config['train_end'],
        'cols' : config['all_features'],
        'date_col':config['original_params']['date_col'],
        'key_col':config['original_params']['hash_col'],
        'target':config['original_params']['target'],
        'dataset_filter':config['dataset_filter']
    }
    task_ps_params = {
        'escoragem': False,
        'project': config['project'],
        'table_dataset' : config['table_dataset'],
        'period_begin' : config['train_begin'],
        'period_end' : config['train_end'],
        'cols' : config['all_features'],
        'date_col':config['original_params']['date_col'],
        'key_col':config['original_params']['hash_col'],
        'target':config['original_params']['target'],
        'analysis_variables' : config['analysis_variables'],
        'dataset_filter':config['dataset_filter']
        
    }
    
    task_engineer_params = {
        'small_categorical' : config['small_categorical'],
        'large_categorical' : config['large_categorical'],
        'variable_set'      : config['all_features'],
        'date'              : config['original_params']['date_col']
    }
    
    task_model_params =  {
    
        'model_params':config['model_params']
    }
    
    task_predict_params = {}
    
    task_prd_report_params = {
        'target' : config['params']['target'],
        'model_output' : config['params']['model_output'],
        'key' : config['params']['key'],
        }
    
    params = {'task_te_params': task_te_params,
              'task_ps_params': task_ps_params,
              'task_engineer_params': task_engineer_params,
              'task_model_params': task_model_params,
              'task_predict_params': task_predict_params,
              'task_prd_report_params': task_prd_report_params}

    
    return params

#Nessa função estamos pegando da pasta local os arquivos de config do treino/o pickle do modelo e o report para o frontend;
#e gravando no caminho do Storage;

def save_artifact(config, pickle_local_path, bucket, caminho):
    import json
    
    html_path = save_report_html(pickle_local_path, bucket, caminho)
    
    metadata = {
        'outputs' : [
        # Markdown that is hardcoded inline
        {
          'type': 'web-app',
          'storage': 'gcs',
          'source': html_path,
        }, 
        {
              'storage': 'inline',
              'source': '# Modelo Final\n**Versão Código:** {0}\n**Versão Modelo:** {1}'.format(VERSAO_CODIGO, config['versao_modelo']),
              'type': 'markdown',
        }
        ]
    }

    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)
    

def save_report_html(pickle_local_path, artifact_bucket, artifact_folder):
    import visualizacao.frontend as frontend
    import matplotlib.pyplot as plt 
    import base64
    from io import BytesIO
    
    t4 = frontend.Production(pickle_local_path)
    t4.load_data(None)

    t4.train_eval(6, 'elasticidade')
    fig = plt.gcf()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    fig_1 = base64.b64encode(tmpfile.getvalue()).decode('utf-8')


    t4.train_eval(6, 'AUC')
    fig = plt.gcf()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    fig_2 = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    t4.real_predict(6)
    fig = plt.gcf()
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    fig_3 = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = """
    <img src=\'data:image/png;base64,{0}\'>
    <img src=\'data:image/png;base64,{1}\'>
    <img src=\'data:image/png;base64,{2}\'>
    """.format(fig_1, fig_2, fig_3)

    with open('report.html','w') as f:
        f.write(html)
        
    file_path = 'gs://{0}/{1}/modelagem/prd_report_{2}.html'.format(artifact_bucket, artifact_folder, datetime.datetime.now().strftime("%Y%m%d%H%M%f"))
    Utils.upload_file_to_gcp(local_path = './', local_filename = 'report.html', bucket_name = artifact_bucket, gcs_path = file_path)
    
    return file_path


def salvar_modelo(t,model,config,local_path='./experimentos/producao/'):
    output = { 
        'prod_metric' : t.output()['prod_metric'].load(),
        'prod_real_predict': t.output()['prod_real_predict'].load()
    }
     
    caminho = 'gs://{}/{}{}/{}/'.format(config['bucket_name'],GCP_BASE_PATH,VERSAO_CODIGO,config['versao_modelo'])
    with open(local_path+"modelo.pickle", 'wb') as f:
        pickle.dump(model, f)
    with open(local_path+'config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
        
    Path(local_path + 'arquivos_prod').mkdir(parents=True, exist_ok=True)    
    with open(local_path + 'arquivos_prod' + '/report.pickle'.format(config['versao_modelo']), 'wb') as f:
        pickle.dump(output, f)
        
    Utils.upload_file_to_gcp(local_path = local_path, local_filename='modelo.pickle', bucket_name = config['bucket_name'], gcs_path = caminho+'modelo.pickle' )
    Utils.upload_file_to_gcp(local_path = local_path, local_filename='config.yaml',   bucket_name = config['bucket_name'], gcs_path = caminho+'config.yaml' )
    Utils.upload_file_to_gcp(local_path = local_path + 'arquivos_prod/', local_filename = 'report.pickle', bucket_name = config['bucket_name'], gcs_path = caminho+'report.pickle' )

    # salvando a versao do modelo no txt
    with open(local_path+'output_versao_codigo.txt', 'w') as text_file:
        text_file.write(VERSAO_CODIGO)
        
   # salvando o caminho no txtx que salva os arquivos do treino     
    with open (local_path+'output_caminho_treino.txt', 'w') as text_file:
        text_file.write(caminho)
    
    print('arquivos gravados no GCP {}'.format(caminho + 'modelo.pickle'))
    
    print('Consegui atualizar a imagem por cima')
    
    save_artifact(config, local_path, config['bucket_name'], '{0}{1}/{2}'.format(GCP_BASE_PATH, VERSAO_CODIGO, config['versao_modelo']))

# config path --> caminho do storage
def run(config_path):
    
    local_path = './experimentos/producao/'
    filename   = 'config.yaml'
    
    
    #caminho do bucket para fzer o download do arquivo para a maquina local
    bucket_name_download= config_path[5:].split('/')[0]
    
    Utils.download_file_from_gcp(config_path, local_path = local_path, filename=filename, bucket_name=bucket_name_download)
    
    #variáveis de ambiente de acordo com desenvolvimento ou produção
    config = config_pre_tratamento(local_path+filename)
    project= config['project']
    config['caminho_saida_dados'] = local_path
    d6tflow.set_dir(config['caminho_saida_dados'])
    params = get_tasks(config)
    t = tasks.TaskPrdReport(**params)
    
    d6tflow.preview(t)
    d6tflow.run(t, workers = config['workers'])
    
    model = tasks.TaskTrainModel( task_engineer_params = params['task_engineer_params'],
                                  task_te_params       = params['task_te_params'],
                                  task_ps_params       = params['task_ps_params'], 
                                  task_model_params    = params['task_model_params']).output().load()
    salvar_modelo(t,model,config)
    return True

