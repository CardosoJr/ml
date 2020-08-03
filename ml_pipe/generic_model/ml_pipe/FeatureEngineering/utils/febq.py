# Feature Engineering functions for bq

def bucketize(table_name, feature, buckets):
    # USE ML.BUCKETIZE OR ML.QUANTILE_BUCKETIZE
    pass


def fill_most_frequent(table_name, features):
    med_feat = ', \n'.join(["APPROX_TOP_COUNT({0}, 1)[offset(0)].value AS {0}_MED".format(x) for x in features])
    
    new_features = ', \n'.join(["CASE WHEN {0} IS NULL THEN {0}_MED ELSE {0} END AS {0}".format(x) for x in features])
    
    sql = """ WITH PERC_VALUES AS (
    SELECT 
        {0}
    FROM {1})
    
    SELECT table.* EXCEPT({2}),
    {3}
    
    FROM {1} as table   
    CROSS JOIN PERC_VALUES
    
    """.format(med_feat,
               table_name,
               ','.join(features),
               new_features)
    
    return sql

def fill_missing_median(table_name, features):
    med_feat = ', \n'.join(["APPROX_QUANTILES({0}, 100)[offset(50)] AS {0}_MED".format(x) for x in features])
    
    new_features = ', \n'.join(["CASE WHEN {0} IS NULL THEN {0}_MED ELSE {0} END AS {0}".format(x) for x in features])
    
    sql = """ WITH PERC_VALUES AS (
    SELECT 
        {0}
    FROM {1})
    
    SELECT table.* EXCEPT({2}),
    {3}
    
    FROM {1} as table   
    CROSS JOIN PERC_VALUES
    
    """.format(med_feat,
               table_name,
               ','.join(features),
               new_features)
    
    return sql
    
def scaler_min_max(table_name, features, lower_bound_percent = 0, upper_bound_percent = 100): 
    ## APPROX_QUANTILES(TX_COMER_RENOV, 100)[offset(30)]
    
    min_feat = ', \n'.join(["APPROX_QUANTILES({0}, 100)[offset({1})] AS {0}_MINPERC".format(x, lower_bound_percent) for x in features])
    max_feat  = ', \n'.join(["APPROX_QUANTILES({0}, 100)[offset({1})] AS {0}_MAXPERC".format(x, upper_bound_percent)  for x in features])
    
    new_features = ', \n'.join(["CASE WHEN {0}_MAXPERC = {0}_MINPERC THEN {0} ELSE ({0} - {0}_MINPERC) / ({0}_MAXPERC - {0}_MINPERC) END AS {0}".format(x) for x in features])
    
    sql = """ WITH PERC_VALUES AS (
    SELECT 
        {0},
        {1}
    FROM {2})
    
    SELECT table.* EXCEPT({3}),
    {4}
    
    FROM {2} as table   
    CROSS JOIN PERC_VALUES
    
    """.format(min_feat, 
               max_feat,
               table_name,
               ','.join(features),
               new_features)
    
    
    return sql

def filter_table(table_name, where):
    sql = """
    SELECT * FROM {0} {1}
    
    """.format(table_name, where)
    
    return sql
    

def normalize(features, table_name):
    new_features = ', \n'.join(["({0} - MEAN({0}) OVER ()) / (STDDEV_POP({0}) OVER ()) AS {0}".format(x) for x in features])
    
    sql = """
    SELECT * EXCEPT({0})
    {1}
    FROM {2}    
    """.format(','.join(features),
              new_features,
              table_name)
    
    return sql


methods_dict = {
    'bucketize' : bucketize,
    'scaler_min_max' : scaler_min_max,
    'filter_table' : filter_table,
    'normalize' : normalize,
    'fill_missing_median' : fill_missing_median,
    'fill_most_frequent' : fill_most_frequent
}