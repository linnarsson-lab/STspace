import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib import colors as mcolors
import random

logging.basicConfig(level=logging.INFO)
rcParams['font.size'] = 7

cluster_colors_GBM = {
    'AC-like':'#2ecc71',#inchworm B4FF9F
    'GBL-like':'#c2f970',#'#c2f970'
    'preOPC-like':'#7befb2',#'#c2f970'
    'AC-like Prolif':'#c2f970',
    'MES-like hypoxia independent':'#e76d89',# Deep cerise
    'MES-like hypoxia/MHC':'#e76d89',# Deep cerise
    'MES-like':'#e76d89',# Deep cerise
    'NPC-like':'#ff9470', #atomic tangerine
    'RG':'#f62459',  #radical red
    'OPC-like':'#89c4f4', #bright turquoise
    
    'Astrocyte':'#26c281', #jungles greeen
    'OPC':'#bfbfbf', #silver,#mystic
    'Neuron':'#ffff9f',# canary
    'Oligodendrocyte':'#392e4a',#martynique
    
    'B cell':'#eefcf5',#white 
    'Plasma B':'#4871f7', #cornflower blue
    'CD4/CD8':'#a2ded0', #aqua island
    'DC':'#848ccf', #atomic tangerine 
    'Mast':'#825e5c', #ligh wood
    'Mono':'#f4ede4', #alabaster
    'TAM-BDM':'#e3ba8f', #wood
    'TAM-MG':'#a6915c',#red orange
    'NK':'#bedce3', #ziggurat
    
    'Endothelial':'#d5b8ff', #mauve
    'Mural cell': '#8c14fc',  #electric indigo
    'Fibroblast': '#8c14fc',
    
	'Erythrocyte': '#e33d94'
}
def random_color():
	return "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])



palette= [
    mcolors.CSS4_COLORS['hotpink'],
    mcolors.CSS4_COLORS['cyan'],
    mcolors.CSS4_COLORS['deepskyblue'],
    
    #mcolors.CSS4_COLORS['sandybrown'],
    mcolors.CSS4_COLORS['gold'],
    mcolors.CSS4_COLORS['chartreuse'],
    mcolors.CSS4_COLORS['dodgerblue'],
    mcolors.CSS4_COLORS['mediumpurple'],
    mcolors.CSS4_COLORS['magenta'],
    mcolors.CSS4_COLORS['mediumspringgreen'],
    mcolors.CSS4_COLORS['tomato'], 
]


def dotplot(
    adata,
    markers,
    cluster_key='CombinedNameMerge',
    cmap='Blues',
    threshold = 0,
    size_factor=10,
    metric='correlation',
    method='ward',
    figsize=(8,8),
    row_order=None,
    col_order=None,
    save=False,
    savepath=None,
    xtick_rotation=90,
    ytick_rotation=0,
    reverse_rows=False,
	reverse_cols=False,
	normalize_totals=True,
	totals=False,
	minmax=True,
    ):
    from sklearn.preprocessing import Normalizer, MinMaxScaler
    import scipy.cluster.hierarchy as hc
    from scipy.spatial.distance import pdist
    import fastcluster
    
    if normalize_totals:
        sc.pp.normalize_total(adata, target_sum=1e3)
    adata = adata[:,markers]
    
    if minmax:
        adatas = []
        for s in np.unique(adata.obs.Sample):
            ad = adata[adata.obs.Sample == s,]#.X.mean(axis=0)
            ad.X = MinMaxScaler().fit_transform(ad.X)
            adatas.append(ad)
        adata = adata[0].concatenate(*adatas[1:])
    
    Xs = []
    Sizes = []
    total_Cells = []
    
    for ad in np.unique(adata.obs[cluster_key]):
        
        #X = adata[adata.obs[cluster_key] == ad,markers].X.sum(axis=0)
        X = adata[adata.obs[cluster_key] == ad,markers].X.mean(axis=0)
        total_Cells.append(adata[adata.obs[cluster_key] == ad,].shape[0])
        S = (adata[adata.obs[cluster_key] == ad,markers].X > threshold).sum(axis=0)
        if totals:
            X = X.sum()
            S = S.sum()
        Xs.append(X)
        Sizes.append(S)
    Xs = np.array(Xs)
    print(Xs.shape)
    total_Cells = np.array(total_Cells)
    Sizes = np.array(Sizes)
    if totals:
        markers = ['score']
        ordering_samples_str_rows = np.unique(adata.obs[cluster_key])
        ordering_samples_str_columns = markers	
    df = pd.DataFrame(data=Xs, index=np.unique(adata.obs[cluster_key]), columns=markers)
    df_sizes = pd.DataFrame(data=Sizes, index=np.unique(adata.obs[cluster_key]), columns=markers)
    X = df.values/df.values.max(axis=0)[None,:]
    #X = df.values
    S = df_sizes.values/df_sizes.sum(axis=1)[:,None] #/df_sizes.values.sum(axis=0)[None,:]
    df =  pd.DataFrame(data = X, index=np.unique(adata.obs[cluster_key]), columns=markers)
    df_sizes = pd.DataFrame(data=S*size_factor, index=np.unique(adata.obs[cluster_key]), columns=markers)
    
    if totals == False:
        D = pdist(df.values, metric=metric)
        Z = fastcluster.linkage(D, method=method,metric=metric, preserve_input=True)
        Z = hc.optimal_leaf_ordering(Z, D, metric=metric)
        ordering_samples = hc.leaves_list(Z)
        ordering_samples_str_rows = df.index[ordering_samples]

        D = pdist(df.values.T, metric=metric)
        Z = fastcluster.linkage(D, method=method,metric=metric, preserve_input=True)
        Z = hc.optimal_leaf_ordering(Z, D, metric=metric)
        ordering_samples = hc.leaves_list(Z)
        ordering_samples_str_columns = df.columns[ordering_samples]
	
    
    if reverse_cols:
        ordering_samples_str_columns = ordering_samples_str_columns[::-1]
    if reverse_rows:
        ordering_samples_str_rows = ordering_samples_str_rows[::-1]
    
    if row_order is not None:
        ordering_samples_str_rows = row_order
    if col_order is not None:
        ordering_samples_str_columns = col_order

    df = df.loc[ordering_samples_str_rows].loc[:,ordering_samples_str_columns]
    df_melt = df.melt(ignore_index=False)

    xs, ys, values, s = [], [], [], []
    for x in df_melt.iterrows():
        xs.append(x[0])
        ys.append(x[1].variable)
        values.append(x[1].value)
        s.append(df_sizes.loc[x[0]][x[1].variable]+.1)
        
    # convert lists to matrices 
    x = np.array(xs)
    y = np.array(ys)
    sizes = np.array(s)
    values = np.array(values)

    # scatter plot
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=figsize)
    ax.scatter(y, x, s=sizes, c=values,cmap=cmap , rasterized=False, lw=0)

    ax.grid(axis='x')
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    
    ax.set_xlim(-.05*figsize[0], len(df.columns))
    ax.set_ylim(-.05*figsize[0], len(df.index))
		
    plt.xticks(range(df.shape[1]), df.columns , rotation=xtick_rotation, fontsize=10)
    plt.yticks(range(df.shape[0]), df.index, fontsize=10, rotation=ytick_rotation)
    
    plt.margins(0.01, 0.05)
    if save:
        plt.savefig(savepath, dpi=300,transparent=True,bbox_inches='tight')
    plt.show()
      
    return df 
    
