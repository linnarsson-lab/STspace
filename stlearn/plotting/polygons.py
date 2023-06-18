import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
import logging
import os
import matplotlib.colors as mcolors
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

def inside(p, xlim, ylim):
    if p.centroid.coords[0][0] >= xlim[0] and p.centroid.coords[0][0] < xlim[1] and p.centroid.coords[0][1] >= ylim[0] and p.centroid.coords[0][1] < ylim[1]:
        return True
    else:
        return False
    
class PlotPolygons:
	def __init__(
		    self, 
	        adata,
			color_dic= cluster_colors_GBM,
			color_palette=palette,
	        ) -> None:
		super().__init__()
		self.adata = adata
		self.color_dic = color_dic
		self.color_palette = color_palette

	def plot_polygons(
			self, 
			sample,
			colorby='CellularNgh',
			clusters = None, # List of clusters to plot
			greyclusters = [], # List of clusters to plot in grey
			figsize=(5,5),
			color_dic=None,
			show_axis:bool=False,
			save:bool=False,
			savepath=None,
			dpi=600,
			):
		
		gray_color = '#ececec'
		adata = self.adata[self.adata.obs['Sample'] == sample]
		if clusters is not None:
			adata = adata[adata.obs[colorby].isin(clusters+greyclusters)]
		polygons = adata.obs['Polygons']
		CN = adata.obs[colorby]
		
		if color_dic is None:
			#print('color by palette')
			colors_cn = np.array([self.color_palette[cn] for cn in CN])
		else:
			#print('color by dic')
			colors_cn = np.array([color_dic[cn] for cn in CN])

		gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(polygons),data={colorby:CN, 'color':colors_cn})

		fig, ax1 = plt.subplots(figsize=figsize)
		ax1.set_facecolor((1.,1.,1.))

		if len(greyclusters) > 0:
			gdf_gray = gdf[gdf[colorby].isin(greyclusters)].plot(color=gray_color,edgecolor='black',linewidth=0.05,ax=ax1,rasterized=True,facecolor=(0,0,0), alpha=0.25)
		if clusters is not None:
			gdf_col = gdf[gdf[colorby].isin(clusters)]
		
		else:
			gdf_col = gdf
		im = gdf_col.plot(color= gdf_col['color'], edgecolor='black',linewidth=0.05,ax=ax1,rasterized=True,facecolor=(0,0,0),alpha=0.75)
		if len(greyclusters) > 0:
			im_gray = gdf_gray.plot(color=gray_color,edgecolor='black',linewidth=0.05,ax=ax1,rasterized=True)
		ax1.set_rasterization_zorder(0)

		scalebar = ScaleBar(
			1,
			units='um',
			length_fraction=.1,
			location='lower right'
		) # 1 pixel = 0.2 meter
		plt.gca().add_artist(scalebar)

		#save_path = os.path.join('figures',sample+'_CN.svg')
		plt.tight_layout()
		if not show_axis:
			plt.axis('off')
			ax1.xaxis.set_visible(False) 
			ax1.yaxis.set_visible(False) 
		if save:
			plt.savefig(savepath,dpi=dpi, transparent=True,bbox_inches='tight')
		plt.show()

	def plot_polygons_obsm(
			self, 
			sample=None,
			colorby='proportions',
			clusters = [], # List of clusters to plot
			greyclusters = [], # List of clusters to plot in grey
			figsize=(20,20),
			cmap='magma',
			show_axis:bool=False,
			save:bool=False,
			min_quantile=0.5,
			max_quantile=0.99,
			dpi=600,
			):
		
		gray_color = '#ececec'
		if sample is not None:
			adata = self.adata[self.adata.obs['Sample'] == sample]
		else:
			adata = self.adata

		if len(clusters) == 0:
			clusters = adata.obsm[colorby].columns

		import os
		path = "proportions"
		# Check whether the specified path exists or not
		isExist = os.path.exists(path)
		if not isExist:
			os.makedirs(path)
			
		polygons = adata.obs['Polygons']
		self.proportions = {}
		for ct in clusters:
			data = adata.obsm["proportions"][ct].values
			data = np.clip(data, np.quantile(data, min_quantile), np.quantile(data, max_quantile))
			self.proportions[ct] = data

		gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(polygons))

		for ct in clusters:
			fig, ax = plt.subplots(figsize=figsize)
			ax.set_facecolor((1.,1.,1.))
			ax.set_title(ct)
			zero_mask = self.proportions[ct] == self.proportions[ct].min()
			
			gdf1 = gdf[~zero_mask]
			gdf1[ct] = self.proportions[ct][~zero_mask]
			
			gdf[zero_mask].plot(
				color=gray_color,
				edgecolor=gray_color,
				linewidth=0.05,
				ax=ax,
				rasterized=True,
				facecolor=(0,0,0), 
				alpha=0.5
			)

			gdf1.plot(
				column=ct, 
				cmap=cmap,
				edgecolor='black',
				linewidth=0.05,
				ax=ax,
				rasterized=True,
				facecolor=(0,0,0),
				alpha=0.75
			)

			ax.set_rasterization_zorder(0)
			if not show_axis:
				plt.axis('off')
				ax.xaxis.set_visible(False) 
				ax.yaxis.set_visible(False) 

			scalebar = ScaleBar(
				1,
				units='um',
				length_fraction=.1,
				location='lower right'
			) # 1 pixel = 0.2 meter
			
			ax.add_artist(scalebar)
			
		
			if save:
				plt.savefig('proportions/{}.png'.format(ct),dpi=dpi, transparent=True,bbox_inches='tight')
			plt.show()

	def plot_polygons_clusters(
			self, 
			sample:str,
			clusters:list,
			cluster_key='PredictedClass',
			grey_clusters=[],
			palette=None,
			facecolor=(0,0,0), #white
			figsize=(5,5),
			area_min_size=25,
			alpha=1,
			show_axis:bool=False,
			save:bool=False,
			return_ax:bool=False,
			image:np.array=None,
			flipy:bool=False,
			flipx:bool=False,
			image_downscale:int=5,
			savepath=None,
			ax=None,
			):
		
		scale_factor = 1
		gray_color = '#ececec'
		
		adata = self.adata[self.adata.obs['Sample'] == sample]
		adata = adata[adata.obs['Area'] > area_min_size]
		adata = adata[adata.obs[cluster_key].isin(clusters+grey_clusters)]
		logging.info('Plotting {} cells'.format(adata.shape[0]))
		PClass = adata.obs[cluster_key]
		polygons = adata.obs['Polygons']

		if palette is None:
			colors = np.array([palette[c] for c in PClass])
		else:
			# function to generate a random hex color
			for c in np.unique(PClass):
				if c not in palette:
					palette[c] = '#%02X%02X%02X' % (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))

			colors = np.array([palette[c] for c in PClass])

		polygons = gpd.GeoSeries.from_wkt(polygons)
		if image is not None:
			scale_factor = 0.27*image_downscale
			polygons = polygons.affine_transform([1/scale_factor, 0, 0, 1/scale_factor, 0, 0])
			#polygons = polygons.scale(xfact=1/(0.27*5*10), yfact = 1/(scale_factor))
		gdf = gpd.GeoDataFrame(geometry=polygons,data={'Class':PClass, 'color':colors})

		if ax is None:
			fig, ax1 = plt.subplots(figsize=figsize)
		else:
			ax1 = ax
		ax1.set_facecolor((1.,1.,1.))

		if image is not None:
			if flipy:
				image = np.flipud(image)
			if flipx:
				image = np.fliplr(image)
			ax1.imshow(image)

		im_gray = gdf[gdf['Class'].isin(grey_clusters)].plot(color=gray_color,edgecolor='black',linewidth=0.05,ax=ax1,rasterized=True,facecolor=(0,0,0), alpha=0.25)
		for x in clusters:
			gdf_plot = gdf[gdf['Class'] == x]
			im = gdf_plot.plot(color=gdf_plot.color, edgecolor='black',linewidth=0.05,ax=ax1,rasterized=True,facecolor=facecolor, alpha=alpha)
		ax1.set_rasterization_zorder(0)
		scalebar = ScaleBar(
			scale_factor,
			units='um',
			length_fraction=.1,
			location='lower right'
		) # 1 pixel = 0.2 meter
		plt.gca().add_artist(scalebar)
	
		plt.tight_layout()
		
		if return_ax:
			return ax1
		
		if show_axis == False:
			ax1.axis('off')
		
		if ax is None:
			if save:
				if savepath is None:
					savepath = os.path.join('figures',sample+'.svg')
				plt.savefig(savepath,dpi=600,format='svg', transparent=True,bbox_inches='tight')

			plt.show()

	def plot_polygons_zoom(
			self, 
			sample:str,
			clusters:list,
			xlim, #= (11000, 13000)
			ylim, #= (5000, 7000)
			palette,
			cluster_key='PredictedClass',
			grey_clusters=[],
			area_min_size=25,
			facecolor=(0,0,0), #white
			figsize=(10,10),
			alpha=0.75,
			show_axis:bool=False,
			save:bool=False,
			savepath=None,
			image_downscale:int=5,
			annotate:bool=False,
			show_scalebar:bool=True,
			image:np.array=None,
			flipy:bool=False,
			flipx:bool=False,
			ax=None,
			):
		scale_factor = 1
		adata = self.adata[self.adata.obs['Sample'] == sample]
		adata = adata[(adata.obs['Area'] > area_min_size), :]
		adata = adata[adata.obs[cluster_key].isin(clusters + grey_clusters), :]
		logging.info('First filter, {} cells left'.format(adata.shape[0]))

		PClass = adata.obs[cluster_key]
		MNgh = adata.obs['MolecularNgh']
		polygons = adata.obs['Polygons']
		
		gray_color = '#ececec'
		geometry = gpd.GeoSeries.from_wkt(polygons)
		if image is not None:
			scale_factor = 0.27*image_downscale
			geometry = geometry.affine_transform([1/scale_factor, 0, 0, 1/scale_factor, 0, 0])

		gdf = gpd.GeoDataFrame(geometry=geometry,
			data=
				{
					cluster_key:adata.obs[cluster_key],
					'PredictedClass':PClass,
					'MolecularNgh':MNgh, 
					'Area':adata.obs['Area'],
					'color':[palette[p] for p in adata.obs[cluster_key]]
			}
		)

		gdf = gdf[gdf.loc[:,'geometry'].apply(lambda p: inside(p, xlim=xlim, ylim=ylim))]
		logging.info('Zoom filter, {} cells left'.format(gdf.shape[0]))


		translated_geom = gdf.loc[:,'geometry'].translate(xoff=-xlim[0], yoff=-ylim[0])
		gdf.loc[:,'geometry'] = translated_geom
		gdf_col = gdf[gdf[cluster_key].isin(clusters)]

		if ax is None:
			fig, ax1 = plt.subplots(figsize=figsize)
		else:
			ax1 = ax
		ax1.set_facecolor((1.,1.,1.))
		if image is not None:
			if flipy:
				image = np.flipud(image)
			if flipx:
				image = np.fliplr(image)
			ax1.imshow(image[ylim[0]:ylim[1], xlim[0]:xlim[1]])

		im_gray = gdf[gdf[cluster_key].isin(grey_clusters)].plot(color=gray_color,edgecolor='black',linewidth=0.05,ax=ax1,rasterized=True,facecolor=(0,0,0), alpha=0.25)
		for c in clusters:
			gdf_ = gdf_col[gdf_col[cluster_key] == c]
			im = gdf_.plot(color= gdf_['color'], edgecolor='black',linewidth=0.05,ax=ax1,rasterized=True,facecolor=(0,0,0),alpha=alpha)

		ax1.set_rasterization_zorder(0)
		if annotate:
			ann = gdf_col[(gdf_col.Area > 200) & (gdf_col.Area <= 1000)]
			for m in ann[cluster_key].unique():
				x = ann[ann[cluster_key] == m].iloc[0,:]
				ax1.annotate(text='{}: {}'.format(x['PredictedClass'],x['MolecularNgh']), fontsize=6,xy=x.geometry.centroid.coords[0], ha='center', alpha=0.9, rotation=-90)

		if show_scalebar:
			scalebar = ScaleBar(
				scale_factor,
				units='um',
				length_fraction=.1,
				location='lower right'
			) # 1 pixel = 0.2 meter
			plt.gca().add_artist(scalebar)

		ax1.axis('off')

		plt.tight_layout()

		if show_axis == False:
			ax1.axis('off')
		
		if ax is None:
			if save:
				if savepath is None:
					savepath = os.path.join('figures','{}_zoom{}.svg'.format(sample, clusters))
				plt.savefig(savepath,dpi=300,format='svg', transparent=True,bbox_inches='tight')
			
			plt.show()
