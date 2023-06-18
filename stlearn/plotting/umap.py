import numpy as np
import matplotlib.pyplot as plt
import logging
import matplotlib.colors as mcolors
#from stlearn.plotting.color_utils import scattern
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

'''def umap(
		adata,
		color: list=[],
		palette=None,
		cmap='magma',
		figsize=(5,5),
		ncols=1,
		bgval=1,
		max_percentile=98,
		log=False,
		g=None,
		use_layer=None,
		show_axis:bool=False,
		save:bool=False,
		savepath=None,
		dpi=150,
		**kwargs
	):
	nrows = int(np.ceil(len(color)/ncols))
	xy = adata.obsm['X_umap']
	
	fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
	if nrows == 1 and ncols == 1:
		axs = [axs]
	else:
		axs = axs.ravel()
	grey = '#ececec'
	for c, ax in zip(color, axs):
		# remove ax ticks
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title(c)
		if type(c) == str:
			c = [c]
		exp = adata[:,c].X.sum(axis=1)
		if log:
			exp = np.log(exp+1)
			bgval = np.log(bgval+1)

		scattern(xy, ax=ax, c=exp.flatten(), bgval=bgval, max_percentile=max_percentile, g=g, cmap=cmap)
		if show_axis==False:
			ax.axis('off')


	plt.tight_layout()
	if save:
		plt.savefig(savepath,dpi=dpi, transparent=True,bbox_inches='tight')
	plt.show()'''


def gene_umap(
		adata,
		color: list=[],
		cmap='magma',
		figsize=(5,5),
		ncols=1,
		bgval=0,
		s=1,
		log=False,
		show_axis:bool=False,
		save:bool=False,
		savepath=None,
		alpha=1,
		alpha_grey=0.5,
		dpi=150,
		**kwargs
	):

	nrows = int(np.ceil(len(color)/ncols))
	xy = adata.obsm['X_umap']
	
	fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
	if nrows == 1 and ncols == 1:
		axs = [axs]
	else:
		axs = axs.ravel()

	for c, ax in zip(color, axs):
		# remove ax ticks
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title(c)
		if type(c) == str:
			c = [c]
		exp = adata[:,c].X.sum(axis=1)
		if log:
			exp = np.log(exp+1)
			bgval = np.log(bgval+1)

		ordering = np.random.permutation(xy.shape[0])
		color = exp[ordering]
		xy_ = xy[ordering, :]
		if bgval is not None:
			cells = color > bgval
			ax.scatter(xy_[:, 0], xy_[:, 1], c="lightgrey", s=s, cmap=cmap, alpha=alpha_grey, **kwargs)
			ax.scatter(xy_[cells, 0], xy_[cells, 1], c=color[cells], s=s, cmap=cmap, alpha=alpha,**kwargs)
		else:
			ax.scatter(xy_[:, 0], xy_[:, 1], c=color, s=s, cmap=cmap, **kwargs)

		if show_axis==False:
			ax.axis('off')

	plt.tight_layout()
	if save:
		plt.savefig(savepath,dpi=dpi, transparent=True,bbox_inches='tight')
	plt.show()