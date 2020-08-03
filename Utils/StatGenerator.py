import pandas as pd 
import numpy as np 
from google.cloud import bigquery
from google.cloud import storage
from datetime import datetime
from pathlib import Path
from .BigqueryUtils import extract_csv_from_bq
from .StorageUtils import download_file_from_gcp, upload_file_to_gcp

def generate_stat_table(table_name,
                        date_col = 'DATE', 
                        data_file_name = 'stat.csv',
                        stats = ['AVG', 'SUM'],
                        stat_cols = [],
                        grouping_cols = []):
    '''
    Constrói Estatísticas de uma tabela em BQ em agrupamentos fornecidos por argumentos
    
    Args:
        table_name: nome da tabela a ser sumarizada
        date_col: coluna de datetime 
        data_file_name: nome do arquivo csv a ser salvo
        stats: estatísticas básicas (funções BQ) 
        stat_cols: colunas a serem calculadas
        grouping_cols: colunas de agrupamento
    Outputs:
        Dataframe com dados da tabela sumarizada
    '''
    
    all_features = []
    for f in stat_cols: 
        all_features.extend(["{0}({1}) AS {1}_{0}".format(x, f) for x in stats])
    
    
    query = """
		SELECT 
        AUX_ANOMES,
        {2},        
        COUNT(*) AS COUNT,
        {3} 
		
		FROM(
			SELECT 
				FORMAT_DATE("%Y%m", CAST({1} AS DATE)) as AUX_ANOMES,
				{2},
                {4}
			
			FROM `{0}`
			)
        
        GROUP BY AUX_ANOMES, {2} 
		
		ORDER BY AUX_ANOMES
    
    """.format(table_name, 
               date_col,
              ','.join(grouping_cols),
              ',\n'.join(all_features),
              ','.join(stat_cols))
    
    bq = bigquery.Client(project = table_name.split('.')[0])
    df = bq.query(query).to_dataframe()
    df.to_csv(data_file_name, index = False)
    return df
    
def __get_features(table_name):
    query = """
    SELECT * FROM `{0}` LIMIT 1
    """.format(table_name)
    bq = bigquery.Client(project = table_name.split('.')[0])
    df = bq.query(query).to_dataframe()
    
    return df.columns.tolist()    

def get_min_sample_size(sampling_error, dataset_size, target_proportion = None):
    """
    Retorna tamanho de amostra mínimo
    
    Args: 
        sampling_error: erro de amostragem 
        dataset_size: tamanho total do dataset a ser amostrado 
        target_proportion: proporção da classe target dentro do dataset
    Output:
        sample_size: tamanho da amostra mínimo
    """
    
    import scipy.stats as st
    
    z_alfa_2 =st.norm.ppf(1 - sampling_error)
    
    if target_proportion is None:
        # default values
        p = 0.5 
    else:
        p = round(target_proportion, 4)
        
        
    q_default = round(1 - p, 4)
    
    #Calculo da amostra mínima - default
    sample_size = (dataset_size * p * q * (z_alfa_2**2))/((p * q *(z_alfa_2**2)) + ((dataset_size - 1) * (sampling_error**2)))
    sample_perc = round(sample_size / dataset_size, 2)
        
    return sample_size

def get_stratified_sample_by_df(df, sample_perc, strat_cols):
    from sklearn.model_selection import train_test_split

    """
    Dado um DF, retorna uma amostra estratificada
    
    Args: 
        df: 
        sample_perc:
        strat_cols:
    Outputs:
        DF de amostra
    """
    _, sample = train_test_split(df,
                                 test_size = sample_perc,
                                 random_state = 0, 
                                 stratify = df[strat_cols])
    
    return sample

def get_stratified_sample_by_query(table_name, key_col, sample_perc, strat_cols):
    """
    Retorna query com amostragem estratificada exata no BQ. A query retorna uma lista de keys
    
    Args:
        table_name: nome completo da tabela a ser amostrada
        key_col: coluna de chave primária da tabela
        sample_perc: percentual de amostragem a ser feito
        strat_cols: colunas para estratificação
    Outputs:
        string da query para amostragem
    References:
        https://stackoverflow.com/questions/52901451/stratified-random-sampling-with-bigquery
    """
    group_col = "CONCAT({0})".format(',"_",'.join(strat_cols))
    
    query = """
    WITH 
        table AS (
        SELECT *, {1} AS GROUP_COL
        FROM  `{0}` a),
    table_stats AS (
        SELECT *, SUM(c) OVER() total 
        FROM (
        SELECT {1} AS GROUP_COL, COUNT(*) c 
        FROM table
        GROUP BY 1))

    SELECT sample.{3}
    FROM (
        SELECT ARRAY_AGG(a ORDER BY RAND()) cat_samples, GROUP_COL, ANY_VALUE(c) c
        FROM table a
        JOIN table_stats b
        USING(GROUP_COL)
        GROUP BY GROUP_COL), UNNEST(cat_samples) sample WITH OFFSET off
    WHERE off < {2} * c
    """.format(table_name, group_col, sample_perc, key_col)
    
    return query

def generate_stat_html(table_name,
                       file_name_pattern, 
                       query_filter = "", 
                       table_name_default = None, 
                       html_file_name = 'stats.html',
                       sample_size = 300000, 
                       stratified = None):
    '''
    Constrói report tensorflow-data-validation para comparação de uma tabela atual (current) com uma tabela default
    
    Args:
        table_name: nome da tabela current para ser feita report
        file_name_pattern: pattern de nome de arquivo para ser exportado do BQ 
        query_filter: filtros para serem feitos na query
        table_name_default: nome completo da tabela BQ default para comparação. Em caso de None, report será feito apenas para table_name
        html_file_name: nome do arquivo html para ser salvo report
        sample_size: tamanho da amostra máxima para ser analisada
        stratified: lista de colunas para fazer estratificação. Se as colunas não existirem nas tabelas (current e default) serão desconsideradas.
                    Em caso de None, não será feita estratificação
    Outputs:
        html do report
    '''
    
    import pyarrow
    import tensorflow_data_validation as tfdv
    
    feat = __get_features(table_name)
    
    if table_name_default is not None: 
        feat_default = __get_features(table_name_default)
        feat = list(set(feat) & set(feat_default))
    
    feat_query = "{0}".format(','.join(feat))
    
    query_pattern = """        
        WITH TOTAL AS (
        SELECT COUNT(*) AS TOTAL_COUNT 
        FROM `{0}`) 

        SELECT {2}
        FROM 
        (SELECT 
        *,
        FROM `{0}`
        {3}
        )
        CROSS JOIN TOTAL

        WHERE RAND() <= {1} / TOTAL_COUNT   
    """ 
    
    extract_csv_from_bq(query_pattern.format(table_name, sample_size, feat_query, query_filter),
                        file_name_pattern + "_current.csv", 
                        table_name.split('.')[0], 
                        table_name.split('.')[1], 
                        'tmp_{0}'.format(datetime.now().strftime("%Y%m%d%H%M%f")))
    
#     current_file_name = download_file_from_gcp(file_name_pattern + "_current.csv")
    current_file_name = file_name_pattern + "_current.csv"
    stats  = tfdv.generate_statistics_from_csv(current_file_name)

    
    if table_name_default is not None:
        extract_csv_from_bq(query_pattern.format(table_name_default, sample_size, feat_query, query_filter),
                            file_name_pattern + "_default.csv", 
                            table_name_default.split('.')[0], 
                            table_name_default.split('.')[1], 
                            'tmp_{0}'.format(datetime.now().strftime("%Y%m%d%H%M%f")))
        
#         default_file_name = download_file_from_gcp(file_name_pattern + "_default.csv")
        default_file_name = file_name_pattern + "_default.csv"
        stats_default = tfdv.generate_statistics_from_csv(default_file_name)
        html = tfdv.utils.display_util.get_statistics_html(lhs_statistics = stats, lhs_name = "Current", rhs_statistics = stats_default, rhs_name = "Default")
    else:
        html = tfdv.utils.display_util.get_statistics_html(lhs_statistics=stats)
    
    with open('./stat.html', 'w') as f:
        f.write(html)
        
    upload_file_to_gcp(local_filename = './stat.html', gcs_path = html_file_name)
    
    return html