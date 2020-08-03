"""
Helper functions for categorical encodings
"""
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import KFold
from sklearn import base
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, colnames,targetName,n_fold=5,verbosity=True,discardOriginal_col=False):

        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col

    def fit(self, X, y=None):
        return self


    def transform(self,X):

        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold, shuffle = False, random_state=2019)



        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
#             print(tr_ind,val_ind)
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())

        X[col_mean_name].fillna(mean_of_target, inplace = True)

        if self.verbosity:

            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                                                                      self.targetName,
                                                                                      np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
            

        return X

class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self,train,colNames,encodedName):
        
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
        
        
    def fit(self, X, y=None):
        return self

    def transform(self,X):


        mean = self.train[[self.colNames,self.encodedName]].groupby(self.colNames).mean().reset_index() 
        
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]

        
        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd})

        return X
    
    
def kfold_target_encoder(train, test, valid, cols_encode, target, folds=10):
    """
    Mean regularized target encoding based on kfold
    """
    train_new = train.copy()
    test_new = test.copy()
    valid_new = valid.copy()
    kf = KFold(n_splits=folds, shuffle = False, random_state=1)
    
    for col in cols_encode:
        global_mean = train_new[target].mean()
        train_new[col + "_mean_enc"] = [global_mean] * len(train_new)
        for train_index, test_index in kf.split(train_new):
            mean_target = train_new.iloc[train_index].groupby(col)[target].mean()
            indexes = train_new.iloc[test_index].index
            train_new.loc[indexes, col + "_mean_enc"] = train_new.iloc[test_index][col].map(mean_target)
        
        # making test encoding using full training data
        col_mean = train_new.groupby(col)[target].mean()
        test_new[col + "_mean_enc"] = test_new[col].map(col_mean)
        test_new[col + "_mean_enc"].fillna(global_mean, inplace=True)
        valid_new[col + "_mean_enc"] = valid_new[col].map(col_mean)
        valid_new[col + "_mean_enc"].fillna(global_mean, inplace=True)
    
    # filtering only mean enc cols
    train_new = train_new.filter(like="mean_enc", axis=1)
    test_new = test_new.filter(like="mean_enc", axis=1)
    valid_new = valid_new.filter(like="mean_enc", axis=1)

    return train_new, test_new, valid_new
        
def catboost_target_encoder(train, test, cols_encode, target):
    """
    Encoding based on ordering principle
    """
    train_new = train.copy()
    test_new = test.copy()
    for column in cols_encode:
        global_mean = train[target].mean()
        cumulative_sum = train.groupby(column)[target].cumsum() - train[target]
        cumulative_count = train.groupby(column).cumcount()
        train_new[column + "_cat_mean_enc"] = cumulative_sum/cumulative_count
        train_new[column + "_cat_mean_enc"].fillna(global_mean, inplace=True)
        # making test encoding using full training data
        col_mean = train_new.groupby(column).mean()[column + "_cat_mean_enc"]  #
        test_new[column + "_cat_mean_enc"] = test[column].map(col_mean)
        test_new[column + "_cat_mean_enc"].fillna(global_mean, inplace=True)
    # filtering only mean enc cols
    train_new = train_new.filter(like="cat_mean_enc", axis=1)
    test_new = test_new.filter(like="cat_mean_enc", axis=1)
    return train_new, test_new

def one_hot_encoder(train, test, cols_encode, target=None):
    """ one hot encoding"""
    ohc_enc = OneHotEncoder(handle_unknown='ignore')
    ohc_enc.fit(train[cols_encode])
    train_ohc = ohc_enc.transform(train[cols_encode])
    test_ohc = ohc_enc.transform(test[cols_encode])
    return train_ohc, test_ohc
    
def label_encoder(train, test, cols_encode=None, target=None):
    """
    Code borrowed from fast.ai and is tweaked a little.
    Convert columns in a training and test dataframe into numeric labels 
    """
    train_new = train.drop(target, axis=1).copy()
    test_new = test.drop(target, axis=1).copy()
    
    for n,c in train_new.items():
        if is_string_dtype(c) or n in cols_encode : train_new[n] = c.astype('category').cat.as_ordered()
    
    if test_new is not None:
        for n,c in test_new.items():
            if (n in train_new.columns) and (train_new[n].dtype.name=='category'):
                test_new[n] = pd.Categorical(c, categories=train_new[n].cat.categories, ordered=True)
            
    cols = list(train_new.columns[train_new.dtypes == 'category'])
    for c in cols:
        train_new[c] = train_new[c].astype('category').cat.codes
        if test_new is not None: test_new[c] = test_new[c].astype('category').cat.codes
    return train_new, test_new

def hash_encoder(train, test, cols_encode, target=None, n_features=10):
    """hash encoder"""
    h = FeatureHasher(n_features=n_features, input_type="string")
    for col_encode in cols_encode:
        h.fit(train[col_encode])
        train_hash = h.transform(train[col_encode])
        test_hash = h.transform(test[col_encode])
    return train_hash, test_hash


# def add_target_cluster(daset, c,  target_c):
#     # prepare w2v sentences with target histories
#     gp = daset.loc[tr_index, [c,target_c]].groupby(c)[target_c]
#     hist = gp.agg(lambda x: ' '.join(x).astype(str)))
#     gp_index = hist.index
#     sentences = [x.split(' ') for x in hist.values]
#     n_features = 500
#     w2v = Word2Vec(sentences=sentences, min_count=1, size=n_features)
#     w2v_feature = transeform_to_matrix(w2v, sentences)
#     # clustering
#     cluster_labels = any cluster method e.g K-Means
#     cluster_labels = pd.Series(cluster_labels, name=c + '_cluster', index=gp_index)
#     daset[c + '_cluster'] = daset[c].map(cluster_labels).fillna(-1).astype(int)

# def apply_w2v(sentences, model, num_features): 
#     def _average_word_vectors(words, model, vocabulary, num_features): 
#         feature_vector = np.zeros((num_features,), dtype="float64") 
#         n_words = 0. 
#         for word in words: if word in vocabulary: 
#                 n_words = n_words + 1. 
#                 feature_vector = np.add(feature_vector, model[word]) 
#                 if n_words: 
#                     feature_vector = np.divide(feature_vector, n_words) 
#         return feature_vector 
#     vocab = set(model.wv.index2word) 
#     feats = [_average_word_vectors(s, model, vocab, num_features) for s in sentences] 
#     return csr_matrix(np.array(feats))
                
def factorize_encoding():
    for c in categorical_columns:
        daset[c+'_freq'] = daset[c].map(daset.groupby(c).size() / daset.shape[0])
        indexer = pd.factorize(daset[c], sort=True)[1]
        daset[c] = indexer.get_indexer(daset[c])

