import pandas as pd 
import numpy as np 
from google.cloud import bigquery
from google.cloud import storage
from datetime import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta

def create_table_from_csv(csv_file_name, project_name, dataset_name, table_name, append = False, schema = None):
    """
    Cria uma tabela no BQ dado um CSV
    
    Args:
        csv_file_name: nome do arquivo CSV para construção da tabela
        project_name: nome do projeto a ser salva a tabela
        dataset_name: nome do dataset a ser salva a tabela
        table_name: nome da tabela
        append = False: False se a tabela vai ser reescrita. True se for feito append
        schema = None: Caso None, o schema será autodetect
    """
    
    client = bigquery.Client(project = project_name)
    dataset_ref = client.dataset(dataset_name)
    job_config = bigquery.LoadJobConfig()
    if schema is None:
        job_config.autodetect = True
    else:
        job_config.schema = schema 
        job_config.skip_leading_rows = 1
        
    job_config.source_format = bigquery.SourceFormat.CSV
    job_config.field_delimiter = ","
    
    if append:
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    else:
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        
    load_job = client.load_table_from_uri(csv_file_name, dataset_ref.table(table_name), job_config = job_config)  

    print("Starting job {}".format(load_job.job_id))

    load_job.result()  
    print("Job finished.")

    destination_table = client.get_table(dataset_ref.table(table_name))
    print("Loaded {} rows.".format(destination_table.num_rows))

def check_num_rows(project_name, dataset_name, table_name):
    """
    Retorna número de linhas da tabela
    
    Args:
        project_name: nome do projeto da tabela
        dataset_name: nome do dataset da tabela
        table_name: nome da tabela
    Output:
        Quantidade de linhas da tabela
    """
    client = bigquery.Client(project = project_name)
    dataset_ref = client.dataset(dataset_name)
    destination_table = client.get_table(dataset_ref.table(table_name))
    return destination_table.num_rows

def create_table_from_query(query, project_name, dataset_name, table_name, expiration_days = None, append = False):
    """
    Cria uma tabela no BQ dada um query
    
    Args:
        query: query para construção da tabela
        project_name: nome do projeto a ser salva a tabela
        dataset_name: nome do dataset a ser salva a tabela
        table_name: nome da tabela
        append = False: False se a tabela vai ser reescrita. True se for feito append
        expiration_days = None: Quantidade de dias que a tabela estará viva.
    """
    bq = bigquery.Client(project = project_name)
    job_config = bigquery.QueryJobConfig()
    table_ref = bq.dataset(dataset_name).table(table_name)
    
    if expiration_days is not None:
        table = bigquery.Table(table_ref)
        table.expires = datetime.now() + relativedelta(days = expiration_days)
        bq.create_table(table)

    job_config.destination = table_ref
    
    if append:
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    else:
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        
    query_job = bq.query(query, job_config = job_config)
    query_job.result()
    print("Query result loaded to: {}".format(table_ref.path))
    
def delete_table(project_name, dataset_name, table_name):
    """
    Deleta uma tabela do BQ
    
    Args:
        project_name: nome do projeto da tabela
        dataset_name: nome do dataset da tabela
        table_name: nome da tabela
    """
    
    bq = bigquery.Client(project = project_name)
    job_config = bigquery.QueryJobConfig()
    table_ref = bq.dataset(dataset_name).table(table_name)
    bq.delete_table(table_ref)
    
def create_empty_table(project_name, table_full_name, expiration_days = None):
    """
    Cria uma tabela no BQ vazia
    
    Args:
        project_name: nome do projeto a ser salva a tabela
        dataset_name: nome do dataset a ser salva a tabela
        table_name: nome da tabela
        expiration_days = None: Quantidade de dias que a tabela estará viva.
    """
    
    client = bigquery.Client(project = project_name)
    table = bigquery.Table(table_full_name)
    if expiration_days is not None:
        table.expires = datetime.now() + relativedelta(days = expiration_days)
    table = client.create_table(table)
    
def extract_csv_from_bq(sql, file_names, project, dataset, output_table):
    """
    Exporta uma query para CSV no GCS
    
    Args:
        query: query para export
        file_names: nome do arquivo que será exportado. Caso a query retorne uma tabela grande, será necessário que o arquivo seja um pattern, exemplo: 
                    padrao*.csv. Assim, o resultado será particionado em N arquvios csv.
        project: nome do projeto a ser salva a tabela
        dataset: nome do dataset a ser salva a tabela
        output_table: nome da tabela temporária que será salva a query antes de ser exportada
    """
    
    bq = bigquery.Client(project = project)
    # Configures the job that will export the query to a temporary table. This table will be exported to csv
    job_config = bigquery.QueryJobConfig()
    table_ref = bq.dataset(dataset).table(output_table)
    job_config.destination = table_ref
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    query_job = bq.query(sql, job_config=job_config)
    query_job.result() 
    print("Query result loaded to: {}".format(table_ref.path))
    
    # Configure the job that will export the query result to csv files in the bucket
    dataset_ref = bq.dataset(dataset, project = project)
    table_ref = dataset_ref.table(output_table)
    
    extract_job = bq.extract_table(table_ref, file_names) 
    extract_job.result()

    # Deletes the temporary table
    print('Query results extracted to GCS: {}'.format(file_names))
    bq.delete_table(table_ref) #Deletes table in BQ
    print('Table {} deleted'.format(table_ref))
    
    