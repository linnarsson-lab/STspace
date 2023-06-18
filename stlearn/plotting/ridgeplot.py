import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import matplotlib.colors as mcolors
from tqdm import tqdm, trange
from matplotlib import rcParams
from matplotlib import colors as mcolors
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


def ridge(
		adata, 
		genes:list, 
		clusters:list, 
		cluster_key:str,
		palette:dict, # dictionary of cluster colors in hex,
		norm_scale:bool=True,
		batch_correction:bool=False,
		figsize=(8,16),
		save:bool=False,
		savepath=None,
	):
	from sklearn.neighbors import KernelDensity
	import matplotlib.gridspec as grid_spec
	from sklearn.preprocessing import MinMaxScaler, Normalizer

	if batch_correction:
		sc.pp.normalize_total(adata, target_sum=1e3)
		adatas = []
		for s in np.unique(adata.obs.Sample):
			ad = adata[adata.obs.Sample == s,]#.X.mean(axis=0)
			ad.X = MinMaxScaler().fit_transform(ad.X)
			adatas.append(ad)
		adata = adata[0].concatenate(*adatas[1:])
	adata = adata[adata.obs[cluster_key].isin(clusters), :].copy()

	if norm_scale and batch_correction == False:
		adata.X = Normalizer().fit_transform(adata.X)
		sc.pp.log1p(adata)
		adata.X = MinMaxScaler().fit_transform(adata.X)


	gs = grid_spec.GridSpec(len(clusters), len(genes))
	fig = plt.figure(figsize=figsize)

	for i in trange(len(genes)):
		g = genes[i]
		ax_objs = []
		for j in range(len(clusters)):
			c = clusters[j]
			if type(adata.X) != type(np.array([])):
				x = adata[adata.obs[cluster_key] == c, g].X.toarray().flatten()
			elif type(adata.X) == type(np.array([])):
				x = adata[adata.obs[cluster_key] == c, g].X.flatten()
			else: 
				raise ValueError('adata.X must be a numpy array or scipy sparse matrix')
			#x_d = np.linspace(0,1, 1000)
			xmax = x.max()
			x_d = np.linspace(0,xmax, 1000)

			kde = KernelDensity(bandwidth=0.01, kernel='exponential',)
			kde.fit(x[:, None])

			logprob = kde.score_samples(x_d[:, None])

			# creating new axes object
			ax_objs.append(fig.add_subplot(gs[j:j+1, i:i+1]))

			# plotting the distribution
			ax_objs[-1].plot(x_d, np.exp(logprob),color="#f0f0f0",lw=1)
			ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1,color=palette[c])

			# setting uniform x and y lims
			ax_objs[-1].set_xlim(0,xmax)
			ax_objs[-1].set_ylim(0,2.5)

			# make background transparent
			rect = ax_objs[-1].patch
			rect.set_alpha(0)

			# remove borders, axis ticks, and labels
			ax_objs[-1].set_yticklabels([])

			if i == len(clusters)-1:
				ax_objs[-1].set_xlabel(g, fontweight="bold")
			else:
				ax_objs[-1].set_xticklabels([])

			spines = ["top","right","left","bottom"]
			for s in spines:
				ax_objs[-1].spines[s].set_visible(False)

			adj_cluster = str(c)#c.replace(" ","\n")
			ax_objs[-1].text(-0.02,0, adj_cluster,fontweight="bold",ha="right")

	gs.update(hspace=-.6, wspace=.1)
	plt.tight_layout()
	if save:
		plt.savefig(savepath, transparent=True,bbox_inches='tight')
	plt.show()
