from os import listdir
from os.path import isfile, join
import time
import os
import json
import logging
import datetime
import csv
import pandas as pd
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import random

import traceback

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from joblib import dump,load

import category_encoders as ce
import random
from deap import creator, base, tools, algorithms
import numpy as np
from kmodes.kprototypes import KPrototypes
import hdbscan as hdb
import multiprocessing
from sklearn.cluster import KMeans

import time


class GeneticClusterOptimization:
    
    def __init__(self,df, initialization=False ,target_column='', periodo_features=["",""],
        add_target=False ,encode=False, binary=False, 
        cluster_method='',periodo_cluster=['',''], max_onehot = 10,dim_reduc=False, 
        num_cluster_runs = 1, cluster_param=15, 
        initializing_mode = 'ratio',cluster_columns=''):
        print('Building GCO...')
        self.df = df
        df = df.sort_values(by =[initializing_mode])
        self.start, self.end = datetime.datetime.strptime(periodo_cluster[0], '%Y-%m-%d'), datetime.datetime.strptime(periodo_cluster[1], '%Y-%m-%d')
        df = df[(df['date']>self.start)&(df['date']<self.end)]
        if True:
            self.df = self.df[(self.df['date']>self.start)&(self.df['date']<self.end)]

        self.X = df[cluster_columns]
        self.y = df[target_column]
      
        self.AsBinary = binary
        self.Method = cluster_method   
        self.cluster_param = cluster_param
        self.cluster_columns = cluster_columns 
        self.encode = encode
        self.periodo_cluster = periodo_cluster
        self.periodo_features = periodo_features

        self.models = []
        self.num_cluster_runs = num_cluster_runs
        self.initializing_mode = initializing_mode
        self.seed = 20      
        self.add_target = add_target
        self.initialization = initialization
        
                  
        #find ttypes categorical numerical. Standalize numerical
        if cluster_method == 'kprototypes':
            categorical = []
            numerical = []
            date_type = []
            for col in self.X.columns:
                if ( self.X[col].dtypes == np.int64) or ( self.X[col].dtypes == np.float64):
                    numerical.append(col)
                elif ( self.X[col].dtypes == '<M8[ns]'):
                    date_type.append(col)
                else:
                    categorical.append(col)
            if date_type:
                raise Exception('datetime field not allowed')
            
            
            self.X = self.X[numerical+categorical]
            if encode:
                self.encode_variables(standard_scaler = numerical, boolean_list = categorical)

            if self.add_target:
                #adicionar  target ao cluster              
                self.X['target'] = self.y
                cols = list(self.X.columns)
                
                cols.remove('target')
                cols = ['target'] + cols
                self.X = self.X[cols]
                numerical = ['target'] + numerical
            
            self.categorical_index = [self.X.columns.get_loc(x) for x in categorical]
            self.numerical_index = [self.X.columns.get_loc(x) for x in numerical]
            
        if cluster_method == 'hdbscan':
            if encode:
                bool_list, one_hot_list, many_list, standard_scaler = self.generate_encoding_lists(many=max_onehot)
                self.encode_variables(one_hot_list, bool_list, many_list, standard_scaler,dim_reduc = dim_reduc)      

        self.IndividualSize = len(self.X.columns)

        if self.add_target:
            self.IndividualSize -= 1
        
        # Creating Cluster Initializations 
        if initialization:
            random.seed(self.seed)
            self.cluster_init = []
            for test in range(self.num_cluster_runs):
                self.cluster_init.append(self.get_initial_state(self.cluster_param)) 

        print('Finished building GCO')        #open result file
       
    def generate_encoding_lists(self, many=10): 
        '''
        Retira datetime, int e float. 
        Se nao forem esses tipos, conta quantos valores unicos existem. para 2 faz ordinal encoding, ate 4 one hot e mais de 4 
        faz hash trick
        '''
        bool_list = []
        one_hot_list = []
        many_list = []
        standard_scaler = []
        for col in self.X.columns:
            number_type_flag = (self.X[col].dtypes == np.int64) or (self.X[col].dtypes == np.float64) 
            date_type_flag = (self.X[col].dtypes == '<M8[ns]')
            if number_type_flag:
                standard_scaler.append(col)
            elif not date_type_flag:
                num_values = len(self.X[col].unique())
                if num_values <= 2:
                    bool_list.append(col)
                elif (num_values > 2) and (num_values < many):
                    one_hot_list.append(col)
                elif num_values >= many:
                    many_list.append(col)
                    
        return bool_list, one_hot_list, many_list, standard_scaler
        
    def encode_variables(self, one_hot_list=[], boolean_list=[], many_list=[], standard_scaler=[],dim_reduc = False):
       
        ###Boolean
        if boolean_list:
            input_cols = list(self.X.columns)
            bool_enc = ce.OrdinalEncoder(cols=boolean_list,drop_invariant=True).fit(self.X)
            self.X = bool_enc.transform(self.X)
            self.models.append({'model_type':'boolean', 'model_file':bool_enc,'input_cols':input_cols, 'model_variables':boolean_list})

        ####  ONE HOT
        if one_hot_list:
            input_cols = list(self.X.columns)
            onehot_enc = ce.OneHotEncoder(cols=one_hot_list,drop_invariant=True,use_cat_names=True).fit(self.X)
            self.X = onehot_enc.transform(self.X)        
            self.models.append({'model_type':'one_hot', 'model_file':onehot_enc,'input_cols':input_cols, 'model_variables':one_hot_list})
        
        ##MANY VALUES
        if many_list:
            input_cols = list(self.X.columns)            
            hash_enc = ce.HashingEncoder(cols=many_list,n_components=15, drop_invariant=True).fit(self.X)
            self.X = hash_enc.transform(self.X)
            self.models.append({'model_type':'hash_trick', 'model_file':hash_enc, 'input_cols':input_cols,'model_variables':many_list})
        
        ##Scalling
        if standard_scaler:
            input_cols = list(self.X.columns)        
            df_standard = self.X[standard_scaler]
            scaler = StandardScaler()
            self.X[standard_scaler] = scaler.fit_transform(df_standard)
            self.models.append({'model_type':'standard_scaler', 'model_file':scaler, 'model_variables':standard_scaler})
        #SVD
        if dim_reduc == 'SVD':
            self.dimensional_reduction(method='SVD',n_components=10)
            
    def encoding_df(self,df,columns):


        with open(self.folder + 'model_docs.json') as f:
            model_docs = json.load(f)

        for mc in model_docs:

            if mc['model_type'] == 'boolean':
                m = load(self.folder + 'boolean.joblib')
                df = m.transform(df[mc['input_cols']])


            elif mc['model_type'] == 'one_hot':
                m = load(self.folder + 'one_hot.joblib')
                df = m.transform(df[mc['input_cols']])


            elif mc['model_type'] == 'standard_scaler':
                m = load(self.folder + 'standard_scaler.joblib')
                df[mc['model_variables']] = m.transform(df[mc['model_variables']])

            elif mc['model_type'] == 'svd':
                m = load(self.folder + 'svd.joblib')            
                t = m.transform(df[mc['input_cols']])
                columns = ['svd%i' % i for i in range(t.shape[1])]
                df = pd.DataFrame(t, columns=columns, index=df.index)

        df = df[columns]
        return df
    

    def dimensional_reduction(self,method='SVD',n_components=10,replace_df=True):
        svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
        input_cols = list(self.X.columns)
        svd_model = svd.fit(self.X)  
        columns = ['svd%i' % i for i in range(n_components)]
        self.X = pd.DataFrame(svd.transform(self.X), columns=columns, index=self.X.index)
        self.models.append({'model_type':'svd', 'model_file':svd_model, 'input_cols':input_cols,'output_cols':list(self.X.columns)})
        return svd.explained_variance_ratio_.sum()
    
    def get_initial_state(self, num_clusters):   
             
        num_elements = int(len(self.X) / num_clusters)        
        indexes = []
        for i in range(num_clusters):
            indexes.append(random.randint(i * num_elements, (i + 1) * num_elements))
        v1,v2 = self.X.iloc[indexes,:len(self.numerical_index)].as_matrix(),self.X.iloc[indexes,len(self.numerical_index):].as_matrix()
        return [v1,v2]
    
    def generate_features(self,df_feat,df_model, lag = 0):
        pass
        
    
    def calculate_fitness(self, labels,model):
        if not self.iv_validation:
            df_cluster = self.X.copy()
            df_cluster['cluster'] = labels       
            df_cluster['target'] = np.log(self.y)

            df_grouped = df_cluster.groupby(['cluster'])['target'].std()
            std_sum = (df_grouped.values).sum()
            if np.isnan(std_sum):
                raise Exception('std is Nan')   
            return std_sum/len(set(labels))
        else:        
            max_bin = 20
            force_bin = 3
            
            # define a binning function
            def mono_bin(Y, X, n = max_bin):

                df1 = pd.DataFrame({"X": X, "Y": Y})
                justmiss = df1[['X','Y']][df1.X.isnull()]
                notmiss = df1[['X','Y']][df1.X.notnull()]
                r = 0
                while np.abs(r) < 1:
                    try:
                        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
                        d2 = d1.groupby('Bucket', as_index=True)
                        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
                        n = n - 1 
                    except Exception as e:
                        n = n - 1

                if len(d2) == 1:
                    n = force_bin         
                    bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
                    if len(np.unique(bins)) == 2:
                        bins = np.insert(bins, 0, 1)
                        bins[1] = bins[1]-(bins[1]/2)
                    d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
                    d2 = d1.groupby('Bucket', as_index=True)

                d3 = pd.DataFrame({},index=[])
                d3["MIN_VALUE"] = d2.min().X
                d3["MAX_VALUE"] = d2.max().X
                d3["COUNT"] = d2.count().Y
                d3["EVENT"] = d2.sum().Y
                d3["NONEVENT"] = d2.count().Y - d2.sum().Y
                d3=d3.reset_index(drop=True)

                if len(justmiss.index) > 0:
                    d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
                    d4["MAX_VALUE"] = np.nan
                    d4["COUNT"] = justmiss.count().Y
                    d4["EVENT"] = justmiss.sum().Y
                    d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
                    d3 = d3.append(d4,ignore_index=True)

                d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
                d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
                d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
                d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
                d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
                d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
                d3["VAR_NAME"] = "VAR"
                d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
                d3 = d3.replace([np.inf, -np.inf], 0)
                d3.IV = d3.IV.sum()
                return(d3)

            def char_bin(Y, X):

                df1 = pd.DataFrame({"X": X, "Y": Y})
                justmiss = df1[['X','Y']][df1.X.isnull()]
                notmiss = df1[['X','Y']][df1.X.notnull()]    
                df2 = notmiss.groupby('X',as_index=True)

                d3 = pd.DataFrame({},index=[])
                d3["COUNT"] = df2.count().Y
                d3["MIN_VALUE"] = df2.sum().Y.index
                d3["MAX_VALUE"] = d3["MIN_VALUE"]
                d3["EVENT"] = df2.sum().Y
                d3["NONEVENT"] = df2.count().Y - df2.sum().Y

                if len(justmiss.index) > 0:
                    d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
                    d4["MAX_VALUE"] = np.nan
                    d4["COUNT"] = justmiss.count().Y
                    d4["EVENT"] = justmiss.sum().Y
                    d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
                    d3 = d3.append(d4,ignore_index=True)

                d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
                d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
                d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
                d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
                d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
                d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
                d3["VAR_NAME"] = "VAR"
                d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
                d3 = d3.replace([np.inf, -np.inf], 0)
                d3.IV = d3.IV.sum()
                d3 = d3.reset_index(drop=True)

                return(d3)

            def data_vars(df1, target):

                stack = traceback.extract_stack()
                filename, lineno, function_name, code = stack[-2]
                vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
                final = (re.findall(r"[\w']+", vars_name))[-1]

                x = df1.dtypes.index
                count = -1

                for i in x:
                    if i.upper() not in (final.upper()):
                        if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                            conv = mono_bin(target, df1[i])
                            conv["VAR_NAME"] = i
                            count = count + 1
                        else:
                            conv = char_bin(target, df1[i])
                            conv["VAR_NAME"] = i            
                            count = count + 1

                        if count == 0:
                            iv_df = conv
                        else:
                            iv_df = iv_df.append(conv,ignore_index=True)

                iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
                iv = iv.reset_index()
                return(iv_df,iv)
            df_model_cluster = self.df[self.cluster_columns]
            df_model_cluster = self.encoding_df(df_model_cluster,self.cluster_columns)
            df_model_cluster['cluster'] = model.predict(df_model_cluster.values, categorical=self.categorical_index)
            df_model_full = self.df.merge(df_model_cluster['cluster'], left_index=True, right_index=True)
            df_model_full['indice'] = df_model_full.index
            df_feature_cluster = self.df_feature_cluster.copy()
            df_feature_cluster['cluster'] = model.predict(df_feature_cluster.values, categorical=self.categorical_index)
            df_feature_full = self.df_feature.merge(df_feature_cluster['cluster'], left_index=True, right_index=True)
            result_features = self.generate_features(df_feature_full, df_model_full)
            
            x= result_features.merge(self.y, left_on='indice', right_index=True)
            x = x.set_index('indice')
            df,IV = data_vars(x, x['qtd_prop'])
            print(IV)
            return IV.max().loc['IV']
            
        
    def save_scoring(self,ind,fit,model):
        row = ind + [fit]
        with open(self.folder+'data.csv','a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

        #save bestmodel

        best = float(fit) < self.best_ftnss[0]
        
        if best:
            with open(self.folder+'best_individual','w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
           
            self.best_ftnss[0] = fit
        return best
                   
    def kmeans_fitness(self, individual):        
        df_cluster = self.X.copy()
               
        remove = []
        for i in range(len(df_cluster.columns)):
            if individual[i] < 0.01:
                remove.append(df_cluster.columns[i])
                
            else:
                df_cluster.iloc[:,i] = individual[i] * df_cluster.iloc[:,i] 
        if len(individual) == len(remove):
            return 100000,
        
        if remove:
            df_cluster = df_cluster.drop(remove,axis=1)
        
        kmeans = KMeans(n_clusters=self.cluster_param ,random_state = 10)
        cluster_labels = kmeans.fit_predict(df_cluster)      
        return self.calculate_fitness(cluster_labels),   
    
    def matching_dissim_weighted(self,a, b, **_):
            return np.sum((a != b) * self.individual[:len(self.categorical_index)], axis=1)
    
    def k_prototypes_fitness(self,individual):
    
    
        self.individual = individual
        df_cluster=self.X.copy()   
        if self.add_target:
            self.individual = [1] + self.individual
        
        
        #check if calculation was already made upt to 2nd decimal
        inf_curr = [round(float(y),2) for y in individual]
        for x in self.results:
            ind_test_norm = [round(float(y),2) for y in x[:-1]]
            if ind_test_norm == inf_curr:
                print('já calculado')
                return float(x[-1]),
        
        #weights on kmeans
        for i in self.numerical_index:
            df_cluster.iloc[:,i] = self.individual[i] * df_cluster.iloc[:,i] 
        random.seed(10)
        kproto = KPrototypes(n_clusters=self.cluster_param, cat_dissim=self.matching_dissim_weighted, #num_dissim=euclidean_dissim_weighted, 
                        max_iter=5, verbose=0, gamma=1,n_init=1, init = 'random', random_state=10)
        kproto.fit(df_cluster.values,categorical = self.categorical_index)            
        ftnss = self.calculate_fitness(kproto.labels_,kproto)                
        
        self.save_scoring(self.individual,ftnss,kproto)
        self.results.append(self.individual + [ftnss])
        
        return ftnss,
        
    def hdbscan_fitness(self, individual):
        #check if calculation was already made upt to 2nd decimal
        inf_curr = [round(float(y),2) for y in individual]
        for x in self.results:
            ind_test_norm = [round(float(y),2) for y in x[:-1]]
            if ind_test_norm == inf_curr:
                print('já calculado')
                return float(x[-1]),

        df_cluster = self.X.copy()
        remove = []
        for i in range(len(df_cluster.columns)):
            if individual[i] < 0.01:
                remove.append(df_cluster.columns[i])                
            else:
                df_cluster.iloc[:,i] = individual[i] * df_cluster.iloc[:,i] 
        if len(individual) == len(remove):
            return 100000,
        
        if remove:
            df_cluster = df_cluster.drop(remove,axis=1)
        
        
        clusterer = hdb.HDBSCAN(min_cluster_size=self.cluster_param, prediction_data=True)
        clusterer.fit(df_cluster)     
        ftnss = self.calculate_fitness(clusterer.labels_)   
        self.save_scoring(individual,ftnss,clusterer)
        self.results.append(individual + [ftnss])
        return ftnss,   

    def check_tested(self,row):
        results = []
        chunks = [x for x in os.walk(os.getcwd()+'/log_dados')]
        for c in chunks:
            for file in c[2]:
                if file == 'model_config':
                    with open(c[0] + '/' +file) as csvfile:
                        csv_reader = csv.reader(csvfile,delimiter=',')
                        a = next(csv_reader)[:6]
                        b = [str(x) for x in row][:6]
                        if a==b: # only 6 first parameters
                            with open(c[0] + '/data.csv') as csvfile2:
                                csv_reader2 = csv.reader(csvfile2,delimiter=',')
                                best = 100000
                                for r in csv_reader2:
                                    results.append(r)
                                    if float(r[-1]) < best:
                                        best = float(r[-1])
                    if results:
                        return {'check':True,'folder':c[0].split('/')[-1],'results':results, 'best':best}

                
        
        return {'check':False}
    
    def save_models(self):
        models_docs = []
        for m in self.models:
            #cria um arquivo
            filename = m['model_type'] + '.joblib'
            dump(m['model_file'], self.folder + filename)
            x = m.copy()
            x.pop('model_file')
            models_docs.append(x)
        #write file with model docs
        
        if self.initialization:

            cluster_list = []
            for el in self.cluster_init:
                cluster_list.append([el[0].tolist(),el[1].tolist()])

            with open(self.folder +'cluster_init.json','w') as f:
                json.dump(cluster_list,f)
                
        with open(self.folder +'model_docs.json','w') as f:
            json.dump(models_docs,f)
                   
    def genetic_cluster_optimizaton(self,ngen=2,pop=100,cxpb=0.8,mutpb=0.1,indpb=0.05,processes=1,iv_validation=False,df_feature=''):

        manager = multiprocessing.Manager()
        self.results = manager.list()
        self.best_ftnss = manager.list()

        self.results = []
        self.best_ftnss = []
        self.iv_validation = iv_validation

        
        tournsize = int(np.max([3, pop / 100]))
        row = [str(list(self.X.columns)),self.periodo_cluster,self.periodo_features,self.encode,self.AsBinary,self.Method,
               self.cluster_param,ngen,pop,cxpb,mutpb,tournsize,indpb,datetime.datetime.now().strftime("%Y%m%dT%H%M%S")]
        tested = self.check_tested(row)
        
        if tested['check']:
            self.folder = 'log_dados/' + tested['folder'] + '/'
            self.results.extend(tested['results'])
            self.best_ftnss.append(tested['best'])
            
            
        else:
            #new_model
            
            #check number of last model
            subdirs = [x[0] for x in os.walk(os.getcwd()+'/log_dados')]
            n = []
            for s in subdirs:
                if 'model' in s:
                    n.append(int(s.split('_')[-1]))
            folder_name = 'model_' + str(max(n)+1).zfill(3) if n else 'model_001'
            os.makedirs('log_dados/'+folder_name)
            self.folder = 'log_dados/'+ folder_name + '/'
            with open(self.folder + 'model_config','w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
            self.best_ftnss.append(float(10000))
            with open(self.folder+'best_individual','w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([0,0])
                
        self.save_models()
        
        
        #base de mercado para geracao de variaveis
        if iv_validation:   
            self.df_feature = df_feature
            df_feature = df_feature[(df_feature['dat_calculo'] > self.start) & (df_feature['dat_calculo'] < self.end)] 
            self.df_feature_cluster = self.encoding_df(df_feature,self.cluster_columns)           
            
                
        # Defining Statistics
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)        
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        
        # Parallel
        pool = multiprocessing.Pool(processes=processes)
        toolbox.register("map", pool.map)
        
        # Binary Individual
        if self.AsBinary:
            toolbox.register("attr_bool", random.randint, 0, 1)
            toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
        else:
            toolbox.register("attr_bool", random.uniform, 0, 1)
            toolbox.register("mutate", tools.mutPolynomialBounded, indpb=indpb, low=0, up=1, eta=0.5)
        

        if len(self.results) < pop:
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=self.IndividualSize)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            population = toolbox.population(n=pop)
        else:
        
            def initIndividual(icls, content):
                return icls(content)

            def initPopulation(pcls, ind_init):
                contents = []
                for x in self.results[-pop:]:
                    new_ind = [float(y) for y in x[:-1]]
                    contents.append(new_ind)
                
                return pcls(ind_init(c) for c in contents)

            toolbox.register("individual_guess", initIndividual, creator.Individual)
            toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess)

            population = toolbox.population_guess()

        
        if self.Method == 'kprototypes':
            toolbox.register("evaluate", self.k_prototypes_fitness)
        elif self.Method == 'hdbscan':
            toolbox.register("evaluate", self.hdbscan_fitness)
        elif self.Method == 'kmeans':
            toolbox.register("evaluate", self.kmeans_fitness)
        else:
            raise Exception('cluster_method not defined')

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("select", tools.selTournament, tournsize=tournsize)     

        
        for gen in range(ngen):
            print('Generation: %s' % gen)
            offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))
            record = stats.compile(population)
            print(record)

        top10 = tools.selBest(population, k=10)

        with open(self.folder+'result','w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(top10)


        return top10 

import ast
from lightgbm import LGBMClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

class GenerateOutputs:

    def __init__(self,cluster_method='', model_folder='',add_target=True):
        'dfs devem ter as mesmas variaveis utilizadas para o clustering original (e outras)'
        self.folder = model_folder
        self.cluster_method = cluster_method
        self.add_target = add_target

    def encoding_df(self,df,columns):


        with open(self.folder + 'model_docs.json') as f:
            model_docs = json.load(f)

        for mc in model_docs:

            if mc['model_type'] == 'boolean':
                m = load(self.folder + 'boolean.joblib')
                df = m.transform(df[mc['input_cols']])

            elif mc['model_type'] == 'one_hot':
                m = load(self.folder + 'one_hot.joblib')
                df = m.transform(df[mc['input_cols']])

            elif mc['model_type'] == 'standard_scaler':
                m = load(self.folder + 'standard_scaler.joblib')
                df[mc['model_variables']] = m.transform(df[mc['model_variables']])

            elif mc['model_type'] == 'svd':
                m = load(self.folder + 'svd.joblib')            
                t = m.transform(df[mc['input_cols']])
                columns = ['svd%i' % i for i in range(t.shape[1])]
                df = pd.DataFrame(t, columns=columns, index=df.index)

        df = df[columns]
        return df

    def matching_dissim_weighted(self,a, b, **_):
        return np.sum((a != b) * self.individual[:len(self.categorical_index)], axis=1)

    def return_best_cluster(self,df_cluster,cluster_param):
        if self.cluster_method == 'kprototypes':
            
            #weights on kmeans
            
          
            for i in self.numerical_index:
                df_cluster.iloc[:,i] = self.individual[i] * df_cluster.iloc[:,i] 

            if os.path.exists(self.folder + 'cluster_init.json'):
                with open(self.folder + 'cluster_init.json') as f:
                    cluster_init = json.load(f)
                ftnss = 100000
                for init in cluster_init:    
                    init = [ np.array(init[0]),np.array(init[1])] 
                    kproto = KPrototypes(n_clusters=cluster_param, cat_dissim=self.matching_dissim_weighted, #num_dissim=euclidean_dissim_weighted, 
                                    max_iter=5, verbose=1, gamma=1,n_init=1, init = init)
                    kproto.fit(df_cluster.values,categorical = self.categorical_index)    

                    x = pd.DataFrame([])
                    x['cluster'] = kproto.labels_
                    x['target'] = self.target
                    df_grouped = x.groupby(['cluster'])['target'].max() - x.groupby(['cluster'])['target'].min()
                    curr_ftnss = (df_grouped.values).sum()   
                    print(ftnss) 

                    print(curr_ftnss < ftnss)       
                    winner_model = kproto          
                           
                    if curr_ftnss < ftnss:
                        ftnss = curr_ftnss     
                        winner_model = kproto  
            else:
                kproto = KPrototypes(n_clusters=self.cluster_param, cat_dissim=self.matching_dissim_weighted, #num_dissim=euclidean_dissim_weighted, 
                                max_iter=5, verbose=1, gamma=1,n_init=1, init = 'Cao')
                kproto.fit(df_cluster.values,categorical = self.categorical_index)            
                curr_ftnss = self.calculate_fitness(kproto.labels_)                
                winner_model = kproto                     
            
            dump(winner_model,self.folder+'best_model.joblib')
            self.df['cluster'] = winner_model.labels_
            return winner_model
        
        elif self.cluster_method == 'hdbscan':
            clusterer = hdb.HDBSCAN(min_cluster_size=cluster_param, prediction_data=True)
            clusterer.fit(df_cluster)    
            dump(clusterer,self.folder+'best_model.joblib')
            self.df['cluster'] = clusterer.labels_
            return clusterer   

    def generate_best(self,df='',df_extra='',runs=1,iter=5,verbose=1):
        
        with open(self.folder + 'model_config') as csvfile:
            reader = csv.reader(csvfile)
            content = next(reader)
        
        columns, dt, cluster_param = ast.literal_eval(content[0]), ast.literal_eval(content[1]),int(content[5])
        start, end = datetime.datetime.strptime(dt[0],'%Y-%m-%d'), datetime.datetime.strptime(dt[1],'%Y-%m-%d')

        with open(self.folder + 'model_docs.json') as f:
            model_docs = json.load(f)

        if self.cluster_method == 'kprototypes':
            
            cat_len = len(model_docs[0]['model_variables'])
            num_len = len(columns) - cat_len
            self.categorical_index = list(range(num_len,num_len+cat_len))
            self.numerical_index = list(range(num_len))

        
        df = df[(df['date'] > start) & (df['date'] < end)]
        df = df.sort_values('date')
        df = df.set_index('indice')

        self.df =df
        self.target = df.iloc[:,-1]

        if self.add_target:
            #adicionar  target ao cluster              
            self.df['target'] = self.target
            cols = list(self.df.columns)
            cols.remove('target')
            cols = ['target'] + cols
            self.df = self.df[cols]
        

        df = df.drop('date',axis=1)
        
        df_cluster = self.encoding_df(df,columns)


        with open(self.folder +'best_individual','r') as f:
            reader = csv.reader(f)
            individual = next(reader)[:-1]

        self.individual = [float(x) for x in individual]
       
        model = self.return_best_cluster(df_cluster,cluster_param)
        self.df.to_csv(self.folder + 'renovacao_tag.csv')

        
        df_extra = df_extra[(df_extra['dat_calculo'] > start) & (df_extra['dat_calculo'] < end)] 
        self.df_extra = df_extra   
        df_extra = df_extra.drop('dat_calculo',axis=1)

        #ultima entrada e o target
        if self.add_target:
            cols = list(df_extra.columns)
            del cols[-1]
            cols.append('target')
            df_extra.columns = cols
       
        df_extra_cluster = self.encoding_df(df_extra,columns)

        if self.cluster_method == 'hdbscan':
            test_labels, strengths = hdb.approximate_predict(model, df_extra_cluster.values)
            self.df_extra['cluster'] = test_labels
        
        elif self.cluster_method == 'kprototypes':
            self.df_extra['cluster'] = model.predict(df_extra_cluster.values, categorical=self.categorical_index)

        print(self.df_extra['cluster'].value_counts())
        self.df_extra.to_csv(self.folder + 'mercado_tag.csv')

    def generate_features(self, lag = 0):
        df_featuretag = pd.read_csv(self.folder + 'mercado_tag.csv', parse_dates = ["dat_calculo"])
        df_modeltag = pd.read_csv(self.folder + 'renovacao_tag.csv', parse_dates = ["date"])
        pass
            
    def test_model(self,df): 
        cat_index = []
        for col in df.columns:
            number_type_flag = (df[col].dtypes == np.int64) or (df[col].dtypes == np.float64) 
            if not number_type_flag:
                cat_index.append(df.columns.get_loc(col))
        X,y = df.iloc[:,:-1],df.iloc[:,-1]
        
        for i in cat_index:
            X.iloc[:,i] = X.iloc[:,i].astype('category')
       
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(X, y, train_size=0.7, stratify=y)
        
        parameters = {
            'n_estimators':1000,
            'application': 'binary',
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': 'true',
            'class_weight':{0:1.05,1:1},
            'boosting': 'gbdt',
            'num_leaves': 31,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 20,
            'learning_rate': 0.05,
            'verbose': 0,
            'early_stopping_rounds':20,
            'random_state': 10
            
        }

        model = LGBMClassifier(**parameters)


        model.fit(self.X_train, self.y_train,eval_set=(self.X_validation, self.y_validation))
        self.model = model

    def plot_confusion_matrix(self,cut=0.5):
        y_pred_prob = self.model.predict_proba(self.X_validation)
        y_pred = []
        for x in y_pred_prob:
            if x[0] > cut:
                y_pred.append(0)
            else:
                y_pred.append(1)
        cm = confusion_matrix(self.y_validation, y_pred)
        print(accuracy_score(self.y_validation, y_pred))
        l = []
        for x in cm:
            k=[]
            s = sum(x)
            for y in x:
                k.append(round(float(y/s),2))
            l.append(k)
        class_name = ['Demais','Sulamerica']
        df_cm = pd.DataFrame(l, index = class_name,
                        columns = class_name)
        plt.figure(figsize = (10,7))
        sns.heatmap(df_cm, annot=True)
        
    def plot_performance(self,cut = 0.5,top=None,model_choice='lgbm'):
        y_pred_prob = self.model.predict_proba(self.X_validation)
        y_pred = []
        for x in y_pred_prob:
            if x[0] > cut:
                y_pred.append(0)
            else:
                y_pred.append(1)
        print(classification_report(self.y_validation, y_pred))
        if model_choice == 'catboost':
            imp = self.model.get_feature_importance(prettified=True)
            if top:
                imp = imp[:top]
                
        elif model_choice == 'lgbm':
            imp = sorted(zip(self.X_train.columns,self.model.feature_importances_), key = lambda x: -x[1])
            if top:
                imp = imp[:top]
            
            
        fig,ax = plt.subplots(figsize=(10,10))

        score = [i[0]for i in imp ]
        y_pos = np.arange(len(score))
        performance = [i[1]for i in imp ]

        ax.barh(y_pos, performance, align='center',
                color='green', ecolor='black')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(score)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Performance')
        return imp