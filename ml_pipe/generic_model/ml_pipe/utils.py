from datetime import datetime
from google.cloud import bigquery
from google.cloud import storage
import pandas as pd
import numpy as np


import os  

def extract_table_bq(ds_params, output_table, sql, file_names):
    bq = bigquery.Client(project=ds_params['PROJECT'])
    # Configures the job that will export the query to a temporary table. This table will be exported to csv
    job_config = bigquery.QueryJobConfig()
    table_ref = bq.dataset(ds_params['DATASET']).table(output_table)
    job_config.destination = table_ref
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    query_job = bq.query(sql, job_config=job_config)
    query_job.result() 
    print("Query result loaded to: {}".format(table_ref.path))
    
    # Configure the job that will export the query result to csv files in the bucket
    dataset_ref = bq.dataset(ds_params['DATASET'], project=ds_params['PROJECT'])
    table_ref = dataset_ref.table(output_table)
    
    extract_job = bq.extract_table(table_ref, file_names) 
    extract_job.result()

    # Deletes the temporary table
    print('Query results extracted to GCS: {}'.format(file_names))
    bq.delete_table(table_ref) #Deletes table in BQ
    print('Table {} deleted'.format(table_ref))


def get_ds_paths(ds_name, periods,base_dir = "gs://marketprice/renov/ds/"):
    datasets = {}
    for k, dates in periods.items():
        folder_format = "{0}_{1}".format(dates[0].strftime("%Y%m"), dates[1].strftime("%Y%m"))
        file_format = "{0}_{1}".format(dates[0].strftime("%Y-%m-%d"), dates[1].strftime("%Y-%m-%d"))

        datasets[k] = {
            'train_data_path' : base_dir + '{0}/{2}/train_{1}.csv'.format(folder_format, file_format, ds_name),
            'eval_data_path' : base_dir + '{0}/{2}/eval_{1}.csv'.format(folder_format, file_format, ds_name),
            'predict_data_path' : base_dir + '{0}/{2}/predict_{1}.csv'.format(folder_format, file_format, ds_name),
            'dates_data_path' : base_dir + '{0}/{2}/dates_{1}.csv'.format(folder_format, file_format, ds_name),
        }
    return datasets

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_blob(bucket_name, pattern):
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix= pattern))
    file_list = ["gs://" + str(bucket_name) + "/" + b.name for b in blobs]
    return file_list


def read_multiple_csv(bucket_name, filename_pattern):
    lookup = filename_pattern[filename_pattern.find(bucket_name) + len(bucket_name) + 1:]
    print('looking for {0} in {1} bucket'.format(lookup, bucket_name))

    files = get_blob(bucket_name, lookup)
    print(files)
    file = pd.concat([pd.read_csv(f) for f in files], ignore_index = True)
    return file
