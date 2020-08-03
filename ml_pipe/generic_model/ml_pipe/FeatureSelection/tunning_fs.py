import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set(font_scale=1.5)

def get_translation_dict(tunning_trials, identity_func, group_func, group_tag = 'group'):
    translation = {}
    
    ## Saving Search Results        
    grps = [x for x in tunning_trials.trials[0]['misc']['vals'].keys() if group_tag in x]
    others = [x for x in tunning_trials.trials[0]['misc']['vals'].keys() if group_tag not in x]
    
    for grp in grps:
        translation[grp] = group_func

    for other in others:
        translation[other] = identity_func
        
    return translation
    

def parse_hyperopt_pickle(tunning_trials, translation): 
    loss = [x['result']['loss'] for x in tunning_trials.trials]
    pars = [x['misc']['vals'] for x in tunning_trials.trials]

    data = {'loss' : loss}
    for col in translation.keys():
        data[col] = []

    for par in pars:
        _ = [data[k].append(translation[k](v)) for k, v in par.items()]

    results = pd.DataFrame(data)
    
    return results


def select_groups(tunning_results, num_solutions, cut_off, max_groups = None):
    group_ranks = pd.DataFrame([])

    for k in num_solutions:
        group_results = pd.DataFrame([])
        used_results = tunning_results.drop_duplicates()
        used_results = used_results.sort_values(by = 'loss')
        used_results = used_results.iloc[:k,]

        for col in [x for x in list(used_results.columns) if 'group' in x]:
            aux = pd.DataFrame([])
            aux['loss'] = used_results['loss']
            aux['used'] = used_results[col]
            aux['group'] = [col] * len(aux)
            group_results = group_results.append(aux, ignore_index = True)

        rel_counting = group_results[group_results['used'] == True].groupby('group')['loss'].mean().reset_index()
        rel_counting['relative_count'] = (rel_counting['loss'] - used_results['loss'].min()) / used_results['loss'].min()
        rel_counting = rel_counting.sort_values(by ='relative_count', ascending = True)
        rel_counting['rank'] =  rel_counting['relative_count'].rank(method = 'dense') # np.arange(1, len(rel_counting) + 1)
        group_ranks = group_ranks.append(rel_counting[['group', 'rank']])    

    f, ax = plt.subplots(figsize=(14, 6))
    med = pd.DataFrame([])
    g = group_ranks.groupby('group')
    med['median_rank'] = g['rank'].mean()
    med = med.reset_index()
    
    to_select = round(cut_off * med['group'].nunique())
    
    if max_groups is not None:
        if to_select > max_groups:
            to_select = max_groups
    
    group_ranks = pd.merge(left = group_ranks, right = med, on = 'group', how = 'left')
    group_ranks = group_ranks.sort_values(by = 'median_rank')

    sns.barplot('group', 'rank', data = group_ranks, ax = ax)
    ax.axvline(to_select, color = 'black', linestyle = '--')

    _ = plt.xticks(rotation = 45)
    
    return list(med.sort_values(by = 'median_rank').iloc[:to_select,]['group'].values)
    
     
