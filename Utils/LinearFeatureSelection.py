import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import scipy.stats as stat

class LinearFeatureSelection: 
    def fit_transform(self, df, small_categorical, large_categorical, target):
        df_aux = df.copy(deep = True)
        df_final, var_buckets, clf, global_pvalues = optimize_category_levels(df_aux, small_categorical, large_categorical, target)
        self.small_categorical = small_categorical
        self.large_categorical = large_categorical
        self.var_buckets = var_buckets
        self.clf = clf
        self.global_pvalues = global_pvalues
        self.columns = df_final.columns
        return df_final
        
    def transform(self, df):
        df_aux = df.copy(deep = True)
        df_finalX = transform(df_aux, self.var_buckets, self.large_categorical, self.columns)   
        return df_finalX

    def transform_categorical(self, df):
        df_final = self.transform(df)
        
        all_categorical = self.small_categorical + self.large_categorical
        all_columns = list(df_final.columns)
        
        for var in all_categorical:
            var_columns = [x for x in all_columns if var in x]
            if len(var_columns) > 0:
                df_final[var + "__BASE"] = 1
                for col in var_columns:
                    df_final[var + "__BASE"] = df_final[var + "__BASE"] - df_final[col]
                var_columns.append(var + "__BASE")
                dummies = df_final[var_columns]
                reverse = dummies.idxmax(axis=1)
                df_final = df_final.drop(columns = var_columns)
                df_final[var] = reverse
        return df_final
    
def likelihood_ratio_test(features_alternate, labels, lr_model, features_null=pd.DataFrame([])):
    """
    Compute the likelihood ratio test for a model trained on the set of features in
    `features_alternate` vs a null model.  If `features_null` is not defined, then
    the null model simply uses the intercept (class probabilities).  Note that
    `features_null` must be a subset of `features_alternative` -- it can not contain
    features that are not in `features_alternate`.

    Returns the p-value, which can be used to accept or reject the null hypothesis.
    """
    labels = np.array(labels)
    features_alternate = np.array(features_alternate)

    if not features_null.empty:
        features_null = np.array(features_null)

        if features_null.shape[1] >= features_alternate.shape[1]:
            raise ValueError

        lr_model.fit(features_null, labels)
        null_prob = lr_model.predict_proba(features_null)[:, 1]
        df = features_alternate.shape[1] - features_null.shape[1]
    else:
        null_prob = sum(labels) / float(labels.shape[0]) * \
                    np.ones(labels.shape)
        df = features_alternate.shape[1]

    lr_model.fit(features_alternate, labels)
    alt_prob = lr_model.predict_proba(features_alternate)

    alt_log_likelihood = -log_loss(labels,
                                   alt_prob,
                                   normalize=False)
    null_log_likelihood = -log_loss(labels,
                                    null_prob,
                                    normalize=False)

    G = 2 * (alt_log_likelihood - null_log_likelihood)
    p_value = stat.chi2.sf(G, df)

    return G,p_value

def order_variable_importance(df, features, target):
    """
    Recebe um dataframe e variáveis alvo. Calcula o likelihood-test-ratio individual para todos
    usando um classificador logistico do scikit-learn. Retorna uma lista de variáveis ordenadas
    por deviance    
    """
    num_buckets = 400
    res = []
    for feat in features:
        X = pd.get_dummies(df[feat],prefix='feat')
        y = df[target]
        clf = LogisticRegression(random_state=0, solver='liblinear',max_iter =10000, warm_start =True)
        G, p_value = likelihood_ratio_test(X, y, clf)
        res.append([feat,G])


    return sorted(res, key=lambda x: -x[1])


def calculate_grouping(df,clf,X,index_drop,top_volume_count):
    """
    Função usada para calcular o agrupamento inicial de variáveis categóricas com muitos níveis.
    Os níveis são agrupados por proximidade dos betas, limitados por um volume máximo pré-determinado
    em cada nível.
    Retorna um dicionário: {nivel_original: {'agrupamento': novo_agrupamento}} 
    """
    
    arr = list(zip(X.columns,clf.coef_[0],X.sum().values))
    ldf = pd.DataFrame(arr,columns=['nivel','betas','peso'])
    ldf = ldf.append({'nivel':index_drop,'betas':0.0,'peso':top_volume_count}, ignore_index=True)
    ldf['exp'] = np.exp(ldf['betas'])
    ldf = ldf.sort_values(by='exp')
    ldf['lift'] = ldf['exp'].pct_change()
    ldf.fillna(0,inplace=True)
    agrupamento = []
    peso_acumulado = []
    current = 1
    for i, index_row in enumerate(ldf.iterrows()):
        if i == 0:
            agrupamento.append(current)
            peso_acumulado.append(index_row[1]['peso'])
            continue
        perc_acumulado = (peso_acumulado[-1]+ index_row[1]['peso'])/len(df.index)
        if (index_row[1]['lift'] < 0.03) and (perc_acumulado < 0.005):
            agrupamento.append(current)
            peso_acumulado.append(peso_acumulado[-1] + index_row[1]['peso'])
        else:
            current += 1
            agrupamento.append(current)
            peso_acumulado.append(index_row[1]['peso'])

    ldf['peso_acumulado'] = peso_acumulado
    ldf['agrupamento'] = agrupamento
    ldf = ldf.set_index('nivel')
    translation_dict = ldf[['agrupamento']].to_dict('index')

    return translation_dict
def fix_other_variables(df,df_final,var,target, var_buckets,threshold=0.05):
    print('Fix global')
    flag = True
    while flag:
        flag = False
        clf = LogisticRegression(random_state=0, solver='liblinear',max_iter =10000, warm_start =True, C=100000)
        clf.fit(df_final, target)
        pvalues = calculate_pvalues(clf, df_final)
        failure_count_global = len([i for i in pvalues if i > threshold])
        print('Total niveis local:{}   falhas_global:{}'.format(len(df_final.columns), failure_count_global))
        plist = list(zip(df_final.columns,pvalues))
        for col,pvalue in plist:
            if pvalue > threshold:
                var = col.split('___')[0]
                flag = True
                ex_var = col.split('___')[-1]
                df_final = df_final.drop(columns=[col])
                var_buckets[var][1].append(ex_var)

    return df_final,var_buckets
def test_global_pvalue(df, df_final, var,target,threshold = 0.05):
    buckets = []
    df['teste'] = df[var]
    X = pd.get_dummies(df['teste'],prefix=var)

    num_dummies = len(list(X.columns))
    df_final_teste = df_final.copy(deep=True)
    for col in list(X.columns):
        df_final_teste[col] = X[col].values

    clf = LogisticRegression(random_state=0, solver='liblinear',max_iter =10000, warm_start =True)
    G,pvalue = likelihood_ratio_test(df_final_teste, target, clf, features_null=df_final)    
    if pvalue > threshold:
        return False, pvalue, G
    else:
        return True, pvalue, G
    
def calculate_pvalues(clf,X):
    """
    Função para calcular os p_valores de todos os níveis de uma variável categórica.
    
    """
    denom = (2.0*(1.0+np.cosh(clf.decision_function(X))))
    denom = np.tile(denom,(X.shape[1],1)).T
    F_ij = np.dot((X/denom).T,X) ## Fisher Information Matrix
    Cramer_Rao = np.linalg.pinv(F_ij) ## Inverse Information Matrix
    sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
    z_scores = clf.coef_[0]/sigma_estimates # z-score for eaach model coefficient
    p_values = [stat.norm.sf(abs(x))*2 for x in z_scores] ### two tailed test for p-values
    return p_values

def find_relevant_buckets_large_cat(df, df_final,var,target, var_buckets, threshold = 0.05):
    def translation(el,translation_dict):
        v = var + '___' + str(el)
        return translation_dict[v]['agrupamento']
    X = pd.get_dummies(df[var],prefix=var, prefix_sep='___')
    counts = df[var].value_counts()
    top_volume_var = counts.index[0]
    top_volume_count = counts.iloc[0]
    index_drop =  var + '___' + str(top_volume_var)
    X.drop(columns=index_drop, inplace=True)
    df_final_teste = df_final.copy(deep=True)
    for col in list(X.columns):
        df_final_teste[col] = X[col].values
    clf = LogisticRegression(random_state=0, solver='liblinear',max_iter =10000, warm_start =True, C=100000)
    clf.fit(df_final_teste, target)
    translation_dict = calculate_grouping(df,clf,X,index_drop,top_volume_count)


    df[var] = df[var].apply(translation, args=[translation_dict])  
    df_final_teste,var_buckets = find_relevant_buckets_small_cat(df, df_final,var,target, var_buckets)
    #readicionar o topvalue no agrupamento top
    translation_dict[index_drop] = {'agrupamento': var_buckets[var][0]}
    var_buckets[var].append(translation_dict)

    return df_final_teste,var_buckets

def find_relevant_buckets_small_cat(df, df_final, var, target, var_buckets={}, threshold=0.05):
    
    df['teste'] = df[var]
    df['teste'] = df['teste'].astype(str)
    flag = True
    fix_list = []
    print('final:{}'.format(len(df_final.columns)))
    while flag:
        X = pd.get_dummies(df['teste'],prefix=var, prefix_sep='___')
        top_volume_var = df['teste'].value_counts().index[0]
        index_drop =  var + '___' + str(top_volume_var)
        X.drop(columns=index_drop, inplace=True)
        num_dummies = len(X.columns)
        df_final_teste = df_final.copy(deep=True)
        for col in list(X.columns):
            df_final_teste[col] = X[col].values
        clf = LogisticRegression(random_state=0, solver='liblinear',max_iter =10000, warm_start =True, C=100000)
        clf.fit(df_final_teste, target)
        pvalues = calculate_pvalues(clf, df_final_teste)
        failure_count_local = len([i for i in pvalues[-num_dummies :] if i > threshold])
        failure_count_global = len([i for i in pvalues if i > threshold])
        print('Total niveis local:{}  falhas_local:{}  falhas_global:{}'.format(len(X.columns),failure_count_local, failure_count_global))
        flag = False
        for p in list(zip(pvalues[-num_dummies :],X.columns)):
            if p[0] > 0.05:
                flag=True
                ex_var = p[1].split('___')[-1]
                df['teste'] = np.where(df['teste']==ex_var,top_volume_var,df['teste'])
                fix_list.append(ex_var)

    var_buckets[var] = [top_volume_var,fix_list]
    return df_final_teste,var_buckets

def optimize_category_levels(df, small_categorical, large_categorical, target):
    features = small_categorical + large_categorical
    ordered_variables = [y[0] for y in order_variable_importance(df, features, target)]
    print(ordered_variables)
    var_buckets = {}
    df_final = pd.DataFrame([])
    target = df[target]
    global_pvalues = []
    for var in ordered_variables:
        #testar p-valor global
        test, global_pvalue, G = test_global_pvalue(df, df_final, var, target)
        global_pvalues.append(global_pvalue)
        if not test:
            print('Variável {} não adicionada. pvalor: {}, deviance: {}'.format(var,global_pvalue,G))
            continue
        print('Variável {} adicionada. pvalor: {}, deviance: {}'.format(var,global_pvalue,G))
        if var in small_categorical:
            df_final,var_buckets = find_relevant_buckets_small_cat(df, df_final,var,target, var_buckets)
        elif var in large_categorical:
            df_final,var_buckets = find_relevant_buckets_large_cat(df, df_final,var,target, var_buckets)


        df_final,var_buckets = fix_other_variables(df, df_final,var,target, var_buckets)

    df_final = df_final.reindex(sorted(df_final.columns), axis=1)
    clf = LogisticRegression(random_state=0, solver='liblinear',max_iter =10000, warm_start =True, C=100000)
    clf.fit(df_final, target)
    return df_final, var_buckets, clf, global_pvalues

def transform(df,var_buckets,large_cat, model_columns):
    def translation(el,translation_dict,main):
        v = var + '___' + str(el)
        if v in translation_dict.keys():
            return translation_dict[v]['agrupamento']
        else:
            return main
    
    df_final = pd.DataFrame([])
    for var,el in var_buckets.items():

        if var in large_cat:
            translation_dict = el[2]
            main = el[0]
            df[var] = df[var].apply(translation, args=[translation_dict,main])
            null_list = el[1]
            df['teste'] = np.where(df[var].isin(null_list), el[0], df[var])
            X = pd.get_dummies(df['teste'],prefix=var, prefix_sep='___')
            index_drop =  var + '___' + str(main)

        else:
            main = el[0]
            null_list = el[1]
            df['teste'] = np.where(df[var].isin(null_list), el[0], df[var])
            X = pd.get_dummies(df['teste'],prefix=var, prefix_sep='___')
            index_drop =  var + '___' + str(main)
            
        for col in list(X.columns):
            if col in model_columns:
                df_final[col] = X[col].values
            
        #A ordem das colunas importa
        df_final = df_final.reindex(sorted(df_final.columns), axis=1)
    return df_final
