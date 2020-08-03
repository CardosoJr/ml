import pandas as pd 
import numpy as np 
from google.cloud import bigquery
from google.cloud import storage
from datetime import datetime
from pathlib import Path

def download_file_from_gcp(file_path, local_path = './', project = None):
    '''
    Download de arquivo do GCP no caminho file_path para o local_path. O nome do arquivo é mantido o mesmo. 
    
    Args:
        file_path: caminho do arquivo no GCP para download
        local_path: pasta em que o arquivo será salvo 
        project: nome do projeto no GCP, se None, projeto será o default
    Outputs:
        full path do arquivo baixado 
    '''
    
    bucket_name = file_path.split('/')[2]
    filename = file_path.split('/')[-1]
    if project is None:
        storage_client = storage.Client()
    else:
        storage_client = storage.Client(project = project)
    bucket = storage_client.get_bucket(bucket_name)
    p = file_path.split(bucket_name + '/')[1]
    blob =  bucket.blob(p)
    blob.download_to_filename(local_path + filename)
    
    return local_path + filename
    
def upload_file_to_gcp(local_filename, gcs_path = '', project = None):
    '''
    Upload de arquivo de arquivo local para GCP
    
    Args:
        local_filename: caminho do arquivo local para upload
        gcs_path: caminho completo no GCP para upload
        project: nome do projeto no GCP, se None, projeto será o default
    '''
    
    bucket_name = gcs_path.split('/')[2]
    if project is None:
        storage_client = storage.Client()
    else:
        storage_client = storage.Client(project = project)
    bucket = storage_client.get_bucket(bucket_name)
    p = gcs_path.split(bucket_name + '/')[1]
    blob =  bucket.blob(p)
    blob.upload_from_filename(local_filename)
    
    
def delete_file_in_gcs(gcs_path, project = None):
    """
    Deleta arquivo no GCP
    
    Args:
        gcs_path: caminho completo no GCP para deleção
        project: nome do projeto no GCP, se None, projeto será o default
    """
    bucket_name = gcs_path.split('/')[2]
    if project is None:
        storage_client = storage.Client()
    else:
        storage_client = storage.Client(project = project)
    bucket = storage_client.get_bucket(bucket_name)
    p = file_path.split(bucket_name + '/')[1]
    blob =  bucket.blob(p)
    blob.delete()
    
    
def read_multiple_csv(bucket_name, filename_pattern, project = None):
    """
    Lê multiplos arquivos no GCS com mesmo filename pattern
    
    Args:
        bucket_name: bucket do arquivo 
        filename_pattern: pattern do arquivo a ser lido
        project: nome do projeto no GCP, se None, projeto será o default
    Output:
        Dataframe concatenado de todos os arquivos csv encontrados
    """
    
    
    lookup = filename_pattern[filename_pattern.find(bucket_name) + len(bucket_name) + 1:]
    print('looking for {0} in {1} bucket'.format(lookup, bucket_name))

    if project is None:
        client = storage.Client()
    else:
        client = storage.Client(project = project)
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix= pattern))
    file_list = ["gs://" + str(bucket_name) + "/" + b.name for b in blobs]
    file = pd.concat([pd.read_csv(f) for f in file_list], ignore_index = True)
    return file