import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
import geopandas as gpd
import logging
import matplotlib.colors as mcolors
from matplotlib import rcParams
import seaborn as sns
from matplotlib import colors as mcolors
import scipy.cluster.hierarchy as hc
from scipy.spatial.distance import pdist
import fastcluster
import random

logging.basicConfig(level=logging.INFO)

#rcParams['mathtext.rm'] = 'Arial'
#rcParams['font.family'] = 'Arial'
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


def neighborhood_enrichment(
	adata,
	key='CombinedNameMerge',
	mode='zscore',
	figsize=(5,5),
	palette=None,
	save=False,
	savepath=None,

	):

	fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
	df = pd.DataFrame(
		data=adata.uns[key+'_nhood_enrichment'][mode],
		index = adata.obs[key].cat.categories,
		columns= adata.obs[key].cat.categories,
	)

	metric = 'euclidean'
	method = 'ward'
	D = pdist(df.values.T, metric=metric)
	
	# replace nans with 0 
	D[np.isnan(D)] = 0
	Z = fastcluster.linkage(D, method=method,metric=metric, preserve_input=True)
	Z = hc.optimal_leaf_ordering(Z, D, metric=metric)
	ordering_a = df.index[hc.leaves_list(Z)]
	df = df.loc[ordering_a][ordering_a]

	# Getting the Upper Triangle of the co-relation matrix
	matrix = np.triu(df,k=1)
	# using the upper triangle matrix as mask 
	sns.heatmap(df, mask=matrix, cmap='coolwarm', center=0, ax=ax)
	ax.tick_params(axis='both', which='major', pad=10, length=0) 
	wh = 0.03/10* figsize[0]
	for i, color in enumerate([mcolors.to_rgb(palette[i]) for i in ordering_a]):
		ax.add_patch(plt.Rectangle(xy=(-wh, i), width=wh, height=1, color=color, lw=0,
								transform=ax.get_yaxis_transform(), clip_on=False))

	for i, color in enumerate([mcolors.to_rgb(palette[i]) for i in ordering_a]):
		ax.add_patch(plt.Rectangle(xy=(i, -wh), height=wh, width=1, color=color, lw=0,
								transform=ax.get_xaxis_transform(), clip_on=False))

	plt.tight_layout()
	if save:
		plt.savefig(savepath, dpi=300, transparent=True, bbox_inches='tight')
	plt.show()
	
def neighborhood_enrichment_from_pandas(
	df,
	figsize=(5,5),
	palette=None,
	save=False,
	savepath=None,
	mask_up=2.5,
	mask_down=-2.5,

	):

	fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize)
	N = df.shape[0]
	metric = 'euclidean'
	method = 'ward'
	D = pdist(df.values.T, metric=metric)

	# replace nans with 0 
	D[np.isnan(D)] = 0
	Z = fastcluster.linkage(D, method=method,metric=metric, preserve_input=True)
	Z = hc.optimal_leaf_ordering(Z, D, metric=metric)
	ordering_a = df.index[hc.leaves_list(Z)]
	df = df.loc[ordering_a][ordering_a]

	# Getting the Upper Triangle of the co-relation matrix
	upper_triangle = np.triu(np.ones_like(df),k=1)
	mask =  ((df <= mask_up) & (df >= mask_down))

	# using the upper triangle matrix as mask 
	sns.heatmap(
		df, 
		mask= upper_triangle | mask, 
		cmap='coolwarm', 
		center=0, 
		ax=ax,
		linecolor='white',
		linewidth=0.4,
	)
	ax.tick_params(axis='both', which='major', pad=10, length=0) 
	wh = 0.03/10* figsize[0]
	for i, color in enumerate([mcolors.to_rgb(palette[i]) for i in ordering_a]):
		ax.add_patch(plt.Rectangle(xy=(-wh, i), width=wh, height=1, color=color, lw=0,
								transform=ax.get_yaxis_transform(), clip_on=False))

	for i, color in enumerate([mcolors.to_rgb(palette[i]) for i in ordering_a]):
		ax.add_patch(plt.Rectangle(xy=(i, -wh), height=wh, width=1, color=color, lw=0,
								transform=ax.get_xaxis_transform(), clip_on=False))
		
	ax.plot([1, N+1, 0, 0], [0, N, N, 0], clip_on=False, color='black', lw=2)
	plt.subplots_adjust(top=.88)
	plt.tight_layout()
	if save:
		plt.savefig(savepath, dpi=300, transparent=True,)
	plt.show()


def co_ocurrance(
	adata,
	cluster
    ):
    #fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    df = adata.uns['co_ocurrance']['count'][cluster]#.iloc[:,1:]
    X = df.values
    X = Normalizer().fit_transform(X)
    X = StandardScaler().fit_transform(X)
    df = pd.DataFrame(X, columns=df.columns, index=df.index)
    df = df.melt(ignore_index=False)
    df['cell_type']= df.index
    import seaborn as sns
    sns.lmplot(
        data=df, 
        x="variable", 
        y="value", 
        hue="cell_type",
        order=4,
        scatter_kws={'s':5},
        palette=palette,
        #ax=ax,
        #sizes=(.25, 2.5)
    )
    sns.despine(top=True,right=True, bottom=True, left=True)
    plt.savefig('figures/MES-like7.svg',dpi=600,transparent=True)
    #for s in spines:
    #    ax.spines[s].set_visible(False)
