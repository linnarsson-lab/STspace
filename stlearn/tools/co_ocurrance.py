import scanpy as sc
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm
import logging
from itertools import product
logging.basicConfig(level=logging.INFO)

def co_ocurrance_z(
		adata,
		key,
        clusters=None,
		max_distance=500,
        starting_distance=5,
        steps:int=50,
        n_perms:int = 500,
	) -> None:

    import math
    import time
    if type(max_distance) is int:
        interval = np.linspace(starting_distance, max_distance, num=steps)

    if clusters is None:
        clusters = np.unique(adata.obs[key])

    area_correction = np.array(
         [1/(math.pi * i**2 - math.pi * (interval[n-1])**2) if n > 0 else 1/(math.pi * starting_distance**2) for n,i in enumerate(interval)], 
         dtype=np.float64)
    
    df = pd.DataFrame({'X':adata.obs.X,'Y':adata.obs.Y, key:adata.obs[key]})
    centroids = np.array([df.X, df.Y]).T
    tree = KDTree(centroids)
    dfkey = np.array(df[key].values)
    results = {}
    results_permuted = {'mean':{}, 'q05':{}, 'q95':{}, 'std':{}}
    for c in tqdm(clusters, position=0, leave=True):
        points = centroids[df[key] == c]
        p2 = np.array([points for i in interval])
        idx_intervals = tree.query_ball_point(p2, r=np.array([interval for i in range(len(points))]).T, workers=-1)
        logging.info('Done finding neighbors of cluster {}'.format(c))
        result, result_boots_mean, result_boots_5, result_boots_95, results_boots_std = _export_counts(idx_intervals, clusters, dfkey, n_perms, interval)

        result = _decumulate(result).astype('float64')
        results_boots_mean = _decumulate(result_boots_mean).astype('float64')
        results_boots_5 = _decumulate(result_boots_5).astype('float64')
        results_boots_95 = _decumulate(result_boots_95).astype('float64')
        results_boots_std = _decumulate(results_boots_std).astype('float64')

        result *= area_correction
        results_boots_mean *= area_correction
        results_boots_5 *= area_correction
        results_boots_95 *= area_correction
        results_boots_std *= area_correction

        results[c] = pd.DataFrame(data=result, index=clusters, columns=[i for i in range(interval.shape[0])])
        results_permuted['mean'][c] = results_boots_mean
        results_permuted['q05'][c] = results_boots_5
        results_permuted['q95'][c] = results_boots_95
        results_permuted['std'][c] = results_boots_std

    adata.uns['co_ocurrance'] = (results, results_permuted)


def _decumulate(result):
    result_noncumulative = []
    for i in range(result.shape[1]):
        if i == 0:
            result_noncumulative.append(result[:,i])
        else:
            sum_b = result[:,0:i].sum(axis=1).flatten()
            result_noncumulative.append(result[:,i]-sum_b)
    result_noncumulative = np.array(result_noncumulative).T
    return result_noncumulative

def _export_counts(idx_intervals, clusters, dfkey, n_perms, interval):
    results_dists = []
    results_boots_mean = []
    results_boots_5 = []
    results_boots_95 = []
    results_boots_std = []

    for n,_ in enumerate(interval):
        idx = idx_intervals[n,:]
        if len(idx) > 0:
            idx = np.concatenate(idx)
            idx = np.unique(idx)
            result = dfkey[idx]
            result_boot = [np.random.permutation(dfkey)[idx] for _ in range(n_perms)]     
        else:
            result = []
            result_boot = [[] for _ in range(n_perms)]

        results_perms = []
        for i in range(n_perms):
            results_perms.append([(result_boot[i] == s).sum() if len(result_boot[i]) >0 else 0 for s in clusters])
        
        result = [(result == s).sum() if len(result) > 0 else 0 for s in clusters]
        results_dists.append(result)
        results_perms = np.array(results_perms)
        summary_mean = np.mean(results_perms, axis=0)
        summary_5 = np.quantile(results_perms, q=0.05, axis=0)
        summary_95 = np.quantile(results_perms, q=0.95, axis=0)
        summary_std = np.std(results_perms, axis=0)

        results_boots_mean.append(summary_mean)
        results_boots_5.append(summary_5)
        results_boots_95.append(summary_95)
        results_boots_std.append(summary_std)

    result = np.array(results_dists).T
    result_boots_mean = np.array(results_boots_mean).T
    result_boots_5 = np.array(results_boots_5).T
    result_boots_95 = np.array(results_boots_95).T
    result_boots_95 = np.array(results_boots_95).T
    results_boots_std = np.array(results_boots_std).T
        
    return result, result_boots_mean, result_boots_5, result_boots_95, results_boots_std


def co_ocurrance_faster(
		adata,
		key,
        clusters=None,
		max_distance=500,
        starting_distance=2.5,
        steps:int=50,
        bootstrap:bool = False,
        n_perms:int = 100,
	) -> None:
    from joblib import Parallel, delayed
    import multiprocessing
    import math

    adata_filter = adata[adata.X.sum(axis=1) > 20]
    adata_filter = adata_filter[adata_filter.obs[key].isin(clusters)]
    if adata_filter.shape[0] > 250000:
        logging.info('Subsmpling to 250000 cells')
        adata_filter = sc.pp.subsample(adata_filter, n_obs=250000, copy=True)
    ncores = multiprocessing.cpu_count()
    if type(max_distance) is int:
        interval = np.linspace(starting_distance, max_distance, num=steps)

    if clusters is None:
        clusters = np.unique(adata_filter.obs[key])
    
    area_correction = np.array(
         [1/(math.pi * i**2 - math.pi * (interval[n-1])**2) if n > 0 else 1/(math.pi * starting_distance**2) for n,i in enumerate(interval)], 
         dtype=np.float64)
    
    df = pd.DataFrame({'X':adata_filter.obs.X,'Y':adata_filter.obs.Y, key:adata_filter.obs[key]})
    df_permuted = df.copy()
    tree_dic = {}
    tree_dic_permuted = {}
    for c in clusters:
        df_c = df[df[key] == c]
        centroids = np.array([df_c.X, df_c.Y]).T
        tree_dic[c] = KDTree(centroids)

        if bootstrap:
            permuted_trees = []
            for i in tqdm(range(n_perms), position=0, leave=True):
                #df_permuted['key_perm'] = np.random.permutation(df[key])
                df_permuted['key_perm'] = np.random.choice(df[key], size = len(df[key]) )
                df_c = df_permuted[df_permuted['key_perm'] == c]
                centroids_p = np.array([df_c.X, df_c.Y]).T
                permuted_trees.append(KDTree(centroids_p))
            tree_dic_permuted[c] = permuted_trees

    d_empty = pd.DataFrame(
        np.zeros([len(clusters), len(interval)]), 
        index=clusters, 
        columns=[i for i in range(len(interval))
        ]
    )
    results = {c:d_empty.copy() for c in clusters}
    results_permuted = {c:d_empty.copy() for c in clusters}
    #@numba.jit(nopython=True)
    for c in tqdm(product(clusters, repeat=2),position=0, leave=True):
        tree1 = tree_dic[c[0]]
        tree2 = tree_dic[c[1]]
        res = tree1.count_neighbors(tree2, r=interval, cumulative=False).astype('float64')
        if c[0] == c[1]:
            #print(res)
            itself = np.zeros_like(res)
            #itself = np.ones_like(res)
            itself[0] = tree1.count_neighbors(tree2, r=1, cumulative=False)
            #itself *= tree1.count_neighbors(tree, r=1, cumulative=False)
            res -= itself
        res *= area_correction

        results[c[0]].loc[ c[1],:] =  res

        if bootstrap:
            results_bootstrap = Parallel(n_jobs=ncores)(delayed(compute_n)(tree1, tree_dic_permuted[c[1]][i], area_correction, interval) for i in range(n_perms))

            results_bootstrap = np.stack(results_bootstrap, axis=0)
            zscore = (results[c[0]].loc[c[1],:] - np.mean(results_bootstrap, axis=0)) / (np.std(results_bootstrap, axis=0) + 1)
            results_permuted[c[0]].loc[ c[1],:] = zscore.values
        
        #print(results_permuted[c[0]])

    adata.uns['co_ocurrance'] = {'count':results, 'zscore':results_permuted}

def _compute_n(tree1, tree2, area_correction, interval):
    res = tree1.count_neighbors(tree2, r=interval, cumulative=False).astype('float64')
    res *= area_correction
    return res

