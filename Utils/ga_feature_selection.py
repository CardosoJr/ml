############################################################################################################################################################
### TODO: 
### 1. Add other clustering metrics 
### 2. Improve GA performance. GA have been losing diversity very quickly. 
###     2.1 Remove duplicate offspring (use some kind of hash function)
###     2.2 Optimize parameters (increase mutation probabilities), I think tournament selection is giving poor results
### 3. Implement MOO Version 
### 4. Scale datasets before performing optimization 
############################################################################################################################################################


from re import S
import numpy as np 
import pandas as pd 
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import multiprocessing
import category_encoders as ce
import random
import csv
from  datetime import datetime
import os
from deap import creator, base, tools, algorithms
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
import pickle
from c_index import calc_c_index

def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)
    
def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    #values = values[np.nonzero(values)]
    return np.max(values)

def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    return di

WORST_OBJ_FUNC_SOLUTION = -9999999999 # all objectives are transformed into maximization problems

metrics_sense = {
    "Wemmert-Gancarski": "maximize",
    "Silhouette": "maximize",
    "Calinski-Harabasz": "maximize",
    "Davies-Bouldin": "minimize",
    "Index-C": "minimize",
    "Dunn" : "maximize",
    "Banfeld-Raftery": "minimize",
    "Xie-Beni" : "minimize",
    "Ratkowsky-Lance" : "maximize",
    "KNN" : "maximize"  
}

def calculate_metrics_new(df, clusters, metrics):
    result = {}
    for metric in metrics:
        if metric == "Calinski-Harabasz":
            result['Calinski-Harabasz'] = calinski_harabasz_score(df, clusters)
        elif metric == "Davies-Bouldin":
            result['Davies-Bouldin'] = davies_bouldin_score(df, clusters)
        elif metric == "Silhouette":
            result['Silhouette'] = silhouette_score(df, clusters)
        elif metric == "Index-C":
            result['Index-C'] = calc_c_index(df, clusters)
        elif metric == "Dunn":
            result['Dunn'] = dunn_fast(df, clusters)
        elif metric == "KNN":
            X_train, X_test, y_train, y_test = train_test_split(df, clusters, test_size=0.33, random_state=42)
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train, y_train)
            result['KNN'] = accuracy_score(y_test, knn.predict(X_test))
    return result

class GeneticFeatureSelection:
    """
        Performs Feature Selection Optimization using Genetic Algorithm and Cluster quality indices

        Input:
            - df: Dataframe with features and target
            - target_feature: Target feature
            - experiment_name: Name of the experiment
            - metrics: List of metrics to optimize, can be:
                - "Wemmert-Gancarski"
                - "Silhouette"
                - "Calinski-Harabasz"
                - "Davies-Bouldin"
                - "Index-C"
                - "Dunn"
                - "Banfeld-Raftery"
                - "Xie-Beni"
                - "Ratkowsky-Lance"
                - "KNN"
            - categorical_features: List of categorical features
            - binary: If True, optimization is performed using binary features
            - seed: Seed for random number generator
        Output:
            - best_individual: Best individual found
            - best_ftnss: Best fitness found
            - best_model: Best model found
            - best_results: Results of best model
            - best_folder: Folder where best model was found
    """

    def __init__(self, df, target_feature, experiment_name, metrics, categorical_features = None, binary = True, seed = 42, max_it_without_improvement = 40, dir = ""):
        self.experiment_name = experiment_name
        self.dir = Path(dir) / "opt_log"
        self.df = df.copy()
        self.target_feature = target_feature
        self.binary_optimization = binary 
        self.metrics = metrics
        self.max_it_without_improvement = max_it_without_improvement

        if len(metrics) > 1:
            self.moop = True
        else:
            self.moop = False

        self.aux_result = []
        self.freq_save = 10
        random.seed(seed)

        if categorical_features is not None: 
            cat_encoder = ce.CatBoostEncoder(cols = categorical_features, drop_invariant = True).fit(self.df, self.df[target_feature])
            self.df = cat_encoder.transform(self.df)

        self.target = self.df.pop(target_feature)
        self.ind_size = len(self.df.columns)
        
    def save_scoring(self, ind, fit):
        row = ind + [fit]
        self.aux_result.append(row)

        with open(self.folder / 'data.csv','a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

        #save bestmodel
        best = float(fit) < self.best_ftnss[0]
        if best:
            with open(self.folder / 'best_individual', 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)
            self.best_ftnss[0] = fit
        return best

    def check_tested(self, row):
        files = [x for x in self.dir.rglob("*") if x.is_file()]
        for file in files:
            if file.stem == 'model_config':
                with open(file) as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter = ',')
                    a = next(csv_reader)[:3]
                    b = [str(x) for x in row][:3]
                    if a == b: 
                        return {'check': True, 'folder': file.parents[0], 'checkpoint': file.parents[0] / "checkpoint.pkl"}
        return {'check': False}

    def calculate_fitness(self, individual):
        #check if calculation was already made upt to 2nd decimal
        inf_curr = [round(float(y),2) for y in individual]
        for x in self.results:
            ind_test_norm = [round(float(y),2) for y in x[:-1]]
            if ind_test_norm == inf_curr:
                print('Already calculated')
                return float(x[-1]),

        df_cluster = self.df.copy()
        remove = []
        for i in range(len(df_cluster.columns)):
            if individual[i] < 0.01:
                remove.append(df_cluster.columns[i])                
            else:
                df_cluster.iloc[:,i] = individual[i] * df_cluster.iloc[:,i] 

        if len(individual) == len(remove):
            return WORST_OBJ_FUNC_SOLUTION,
        if remove:
            df_cluster = df_cluster.drop(remove, axis = 1)
        metrics = calculate_metrics_new(df_cluster, self.target, self.metrics)
        results = [metrics[x] * (1.0 if metrics_sense[x] == "maximize" else -1.0) for x in self.metrics]
        return tuple(results)

    def fit(self, ngen = 2, pop = 100, cxpb = 0.8, mutpb = 0.1, indpb = 0.05, processes = 1):
        manager = multiprocessing.Manager()
        self.results = manager.list()
        self.best_ftnss = manager.list()

        self.results = []
        self.best_ftnss = []
        
        tournsize = int(np.max([3, pop / 100]))
        row = [str(list(self.df.columns)), str(list(self.metrics)), self.experiment_name, ngen, pop, cxpb, mutpb, indpb, datetime.now().strftime("%Y%m%dT%H%M%S")]
        
        # Defining Statistics
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)   
        stats.register("std", np.std)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        # Parallel
        pool = multiprocessing.Pool(processes = processes)
        toolbox.register("map", pool.map)

        # Binary Individual
        if self.binary_optimization:
            toolbox.register("attr_bool", random.randint, 0, 1)
            toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
        else:
            toolbox.register("attr_bool", random.uniform, 0, 1)
            toolbox.register("mutate", tools.mutPolynomialBounded, indpb =indpb, low=0, up=1, eta=0.5)

        toolbox.register("evaluate", self.calculate_fitness)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("select", tools.selTournament, tournsize=tournsize)     

        tested = self.check_tested(row)
        if tested['check'] and tested['checkpoint'].exists():
            self.folder = tested['folder']
            with open(tested['checkpoint'], "rb") as f:
                chkp = pickle.load(f)
            population = chkp["population"]
            start_gen = chkp["generation"]
            halloffame = chkp["halloffame"]
            logbook = chkp["logbook"]
            random.setstate(chkp["rndstate"])
        else:
            #check number of last model
            built_models = [int(x.parts[-1].split("_")[-1]) for x in self.dir.rglob("*") if x.is_dir() if "model_" in x.parts[-1]]
            folder_name = 'model_' + str(max(built_models) + 1).zfill(3) if built_models else 'model_001'
            self.folder = self.dir / folder_name
            os.makedirs(self.folder)
            with open(self.folder / 'model_config','w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)

            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n = self.ind_size)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            population = toolbox.population(n=pop)
            halloffame = tools.HallOfFame(maxsize = 10)
            logbook = tools.Logbook()
            start_gen = 0

        last_improvement = start_gen
        best_fo = WORST_OBJ_FUNC_SOLUTION

        # Evaluate the entire population
        fits = toolbox.map(toolbox.evaluate, population)
        for fit, ind in zip(fits, population):
            ind.fitness.values = fit

        for gen in range(start_gen, ngen):
            offspring = algorithms.varOr(population, toolbox, cxpb = cxpb, mutpb = mutpb, lambda_ = 2 * pop)
            # offspring = algorithms.varAnd(population, toolbox, cxpb = cxpb, mutpb = mutpb)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            population = toolbox.select(population + offspring, k = pop)
            record = stats.compile(population)
            halloffame.update(population)
            logbook.record(gen=gen, evals=len(fits), **record)

            if record['max'] > best_fo:
                best_fo = record['max']
                last_improvement = gen

            print(f"Gen {gen + 1}/{ngen}", record, len(fits))

            if gen - last_improvement > self.max_it_without_improvement:
                print('No improvement for a long time')
                break

            if (gen + 1) % self.freq_save == 0:
                cp = dict(population=population, generation=gen, halloffame=halloffame,
                      logbook=logbook, rndstate=random.getstate())
                with open (self.folder / "checkpoint.pkl", 'wb') as cp_file: 
                    pickle.dump(cp, cp_file)

        top10 = tools.selBest(population, k = 10)
        df = pd.DataFrame(top10, index = [1] * np.shape(top10)[0])
        df.to_csv(self.folder / 'result.csv', index = False)
        return top10 