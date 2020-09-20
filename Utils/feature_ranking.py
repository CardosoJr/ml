import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from dateutil.relativedelta import *
from sklearn import metrics
from sklearn.model_selection import KFold


class FeatureRanking: 
    def __init__(self, df, target, categorical_feat = None):
        self.df = df
        self.categorical_feat = categorical_feat
        self.target = target
        if categorical_feat is not None:
            self.__handle_cat_features()
            
    def get_rank(self,
                 num_top_features = 20,
                 rank_plot = False,
                 correlation_threshold = None, 
                 missing_threshold = 0.6,
                 sample_percent = 0.2):

        ranking = {}
        
        if correlation_threshold is None: 
            correlation_threshold = 1.0
        
        
        iv_rank = self.iv(rank_plot, num_top_features)
        self.__update_ranking(ranking, iv_rank[::-1])
        
        lr_rank = self.lr(rank_plot)
        self.__update_ranking(ranking, lr_rank)
        
        gb_rank = self.gb(rank_plot, sample_percent, correlation_threshold, missing_threshold)
        self.__update_ranking(ranking, gb_rank)
        
        final_features = self.__handle_correlated(ranking, correlation_threshold)

        df_dict = {}
        rk_vars =  []
        rks = []

        for var, rankings in ranking.items():
             for rk in rankings:
                    rk_vars.append(var)
                    rks.append(rk)

        df_dict['Feature'] = rk_vars
        df_dict['Ranking'] = rks

        ranking_df = pd.DataFrame(df_dict)
        med_ranking_df = ranking_df.groupby('Feature')['Ranking'].median().reset_index()
        
        if rank_plot:
            f, ax = plt.subplots(figsize=(20, 6))
            med = pd.DataFrame([])
            g = ranking_df.groupby('Feature')
            med['median_rank'] = g['Ranking'].mean()
            med = med.reset_index()
            med = med.sort_values(by = 'median_rank')

            g_ranking = pd.merge(left = ranking_df, right = med.head(60), on = 'Feature', how = 'right')
            g_ranking = g_ranking.sort_values(by = 'median_rank')

            sns.barplot('Feature', 'Ranking', data = g_ranking, ax = ax)
            _ = plt.xticks(size = 12, rotation = 90)
            plt.show()
        
        return med_ranking_df, final_features
    
    def __update_ranking(self, ranking, test_rank):
        i = 1 
        for var in test_rank: 
            if var not in ranking.keys():
                ranking[var] = []
            ranking[var].append(i)
            i = i + 1
    
    def __handle_correlated(self, ranking, correlation_threshold):
        if correlation_threshold is None or correlation_threshold == 1.0:
            pass
        
        df_final = self.df.copy(deep = True)
        df_final.pop(self.target)
        
        au_corr = df_final.corr().abs().unstack()

        labels_to_drop = set()
        cols = df_final.columns
        for i in range(0, df_final.shape[1]):
            for j in range(0, i+1):
                labels_to_drop.add((cols[i], cols[j]))

        au_corr = au_corr.drop(labels=labels_to_drop, errors  = 'ignore' ).sort_values(ascending=False)
        au_corr = au_corr[au_corr > correlation_threshold]

        features = list(df_final.columns)

        for feat_a, feat_b in au_corr.index:
            print(feat_a, feat_b)
            if feat_a in features and feat_b in features:
                if np.mean(ranking[feat_a]) > np.mean(ranking[feat_b]):
                    print('removing', feat_b)
                    features.remove(feat_b)
                else:
                    print('removing', feat_a)
                    features.remove(feat_a)
                    
        return features

    
    def __handle_cat_features(self):
        self.df[self.categorical_feat] = self.df[self.categorical_feat].fillna('MISSING')
        
        t, _, _ = kfold_target_encoder(train = self.df, 
                                       test = self.df.sample(1000), 
                                       valid = self.df.sample(1000), 
                                       cols_encode = self.categorical_feat, 
                                       target = self.target, 
                                       folds = 10)
        
        self.df = self.df.drop(columns = self.categorical_feat).join(t)

    def iv(self, rank_plot = False, num_top_features = 20):
        def calc_iv(df, feature, target, pr=False):
            d1 = df.groupby(by=feature, as_index=True)
            data = pd.DataFrame()

            data['all'] = d1[target].count()
            data['bad'] = d1[target].sum()
            data['share'] = data['all'] / data['all'].sum()
            data['bad_rate'] = d1[target].mean()
            data['d_g'] = (data['all'] - data['bad']) / (data['all'] - data['bad']).sum()
            data['d_b'] = data['bad'] / data['bad'].sum()
            data['woe'] = np.log(data['d_g'] / data['d_b'])
            data = data.replace({'woe': {np.inf: 0, -np.inf: 0}})
            data['iv'] = data['woe'] * (data['d_g'] - data['d_b'])

            data.insert(0, 'variable', feature)
            data.insert(1, 'value', data.index)
            data.index = range(len(data))

            iv = data['iv'].sum()

            if pr:
                print(data)
                print('IV = %s' % iv)

            return iv, data
        
        ivs = []
        for var in self.df.columns:
            if var == self.target:
                continue
            iv, _ = calc_iv(self.df, var, self.target)
            ivs.append(iv)

        def get_key(item):
            return item[0]
        save = sorted(zip(ivs, self.df.columns), key = get_key)
        
        if rank_plot:
            fig, ax = plt.subplots(figsize=(15,8))
            pos = np.arange(len( save[-num_top_features:])) + 0.5  
            plt.bar(pos, [x[0] for x in save[-num_top_features:]], align='center', color = '#416986')
            plt.xticks(pos, [x[1] for x in save[-num_top_features:]], rotation = 90)
            plt.ylabel('IV', size = 15)
            plt.xticks(size = 12)
            plt.yticks(size = 12)
            plt.show()
        
        return [x[1] for x in save]
    
    def lr(self, rank_plot = False):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss
        import scipy.stats as stat

        def likelihood_ratio_test(features_alternate, labels, lr_model, features_null=pd.DataFrame([])):
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
            num_buckets = 400
            res = []
            for feat in features:
                X = pd.get_dummies(df[feat],prefix='feat')
                y = df[target]
                clf = LogisticRegression(random_state=0, solver='liblinear',max_iter = 10000, warm_start = True)
                G, p_value = likelihood_ratio_test(X, y, clf)
                res.append([feat,G])


            return sorted(res, key=lambda x: -x[1])
        
        all_features = list(self.df.columns.ravel())
        all_features.remove(self.target)
        
        var_importance = order_variable_importance(self.df.sample(frac = 0.01), all_features, self.target)
        ordered_vars = [x[0] for x in var_importance]
        importances = [x[1] for x in var_importance]
        importances = (importances - np.min(importances)) / (np.max(importances) - np.min(importances))
        
        if rank_plot:
            fig, ax = plt.subplots(figsize=(15,8))
            pos = np.arange(len(importances[:20])) + 0.5  

            plt.bar(pos, importances[:20], align = 'center', color = '#416986')
            a = plt.xticks(pos, ordered_vars[:20], rotation = 90)
            plt.ylabel('Standardized Likelihood Ratio Test', size = 15)
            plt.xticks(size = 12)
            plt.yticks(size = 12)
            plt.show()
            
        return ordered_vars
    
    def gb(self, rank_plot, sample_percent = 0.2, correlation_threshold = 0.8, missing_threshold = 0.6):
        from feature_selector import FeatureSelector
        x = self.df.sample(frac = sample_percent)
        y = x.pop(self.target)

        fs = FeatureSelector(data = x, labels = y)

        fs.identify_all(selection_params = {'missing_threshold': missing_threshold, 'correlation_threshold': correlation_threshold, 
                                                'task': 'classification', 'eval_metric': 'auc', 
                                                'cumulative_importance': 0.99}) 
        
        train_removed_all_once = fs.remove(methods = 'all', keep_one_hot = True)    
        kept_feat = list(train_removed_all_once.columns)
        
        kept_feat_importance = fs.feature_importances[fs.feature_importances['feature'].isin(kept_feat)]
        
        if rank_plot:
            fig, ax = plt.subplots(figsize=(15,8))
            top20 = kept_feat_importance.head(20)
            plt.bar(top20['feature'], top20['normalized_importance'], color = '#416986')
            a = plt.xticks(rotation = 90)
            plt.ylabel('RF Feature Importance', size = 15)
            _ = plt.xticks(size = 12)
            _ = plt.yticks(size = 12)
            plt.title('Top 20 features (removing highly correlated)')
            plt.show()
            
        return fs.feature_importances['feature'].values.ravel()
        
    def pca_plots(self, num_components):
        from sklearn.decomposition import PCA
        pca = PCA(num_components)
        pca.fit(self.df.fillna(0.0))
        
        explained_variance = pca.explained_variance_ratio_

        plt.figure(figsize=(10, 4))
        plt.bar(range(len(explained_variance)), explained_variance, alpha=0.5, align='center', color = '#416986',
        label='individual explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.tight_layout()

        features = pd.DataFrame(pca.components_, columns = self.df.columns, index = np.arange(num_components)).transpose().abs()
        features['FEATURES'] = features.index
        features = features.reset_index()

        for c in np.arange(num_components):
            features = features.sort_values(by = c, ascending = False)
            top = features.head(30)
            plt.figure(figsize=(10, 4))
            pos = np.arange(len(top)) + 0.5  

            plt.bar(pos, top[c], align='center', color = '#416986')
            plt.xticks(pos,  top['FEATURES'], rotation = 90)

            plt.ylabel('Weights'.format(c), size = 12)
            plt.title('PCA component {0}'.format(c), size = 15)
            plt.xticks(size = 12)
            plt.yticks(size = 12)
            plt.show()
        
    def corr_plot(self, k):
        corr = self.df.corr()
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(21, 17))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.9, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.xticks(size = 10)
        plt.yticks(size = 10)
        plt.show()
        
        top = self.__get_top_abs_correlations(self.df.copy(deep = True), k)
        pairs = ["{0} / {1}".format(x[0], x[1]) for x in top.index]
        values = [float(x) for x in top.values]

        fig, ax = plt.subplots(figsize=(15,8))

        def get_key(item):
            return item[0]

        sorted_pair_corr = sorted(zip(values, pairs), key = get_key)

        pos = np.arange(len(sorted_pair_corr)) + 0.5  
        plt.bar(pos,  [x[0] for x in sorted_pair_corr], align='center', color = '#416986')
        plt.ylim(np.min(top.values)*0.999, np.max(top.values))
        plt.xticks(pos,  [x[1] for x in sorted_pair_corr], rotation = 90)
        plt.ylabel('Pearson Correlation', size = 15)
        plt.xticks(size = 9)
        plt.yticks(size = 12)
        plt.title('Top {0} Correlated Features'.format(k))
        plt.show()
        
    def __get_redundant_pairs(self, df):
        pairs_to_drop = set()
        cols = df.columns
        for i in range(0, df.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def __get_top_abs_correlations(self, df, n = 5):
        au_corr = df.corr().abs().unstack()
        labels_to_drop = self.__get_redundant_pairs(df)
        au_corr = au_corr.drop(labels=labels_to_drop, errors  = 'ignore' ).sort_values(ascending=False)
        return au_corr[0:n]

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