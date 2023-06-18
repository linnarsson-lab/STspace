import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.spatial import KDTree
import logging
logging.basicConfig(level=logging.INFO)

class STSmoothAdatas:
	def __init__(
		    self, 
	        adata,
            feature_key: str = "MolecularNgh",
            same_cluster: bool = True,
	        max_distance=100,
	        ) -> None:
		super().__init__()
		self.adata = adata
		self.feature_key = feature_key
		self.same_cluster = same_cluster
		self.max_distance = max_distance

	def fit(self, save: bool = False) -> None:
		clusters = self.adata.obs[self.feature_key].values
		ClusterID = np.unique(clusters)
		expression = self.adata.X
		gene =self.adata.var.index.values
		n_clusters = clusters.max() + 1
		x,y = self.adata.obs['X'].values, self.adata.obs['Y'].values
		sample = self.adata.obs['Sample']#[subsample]
		unique_samples = np.unique(sample)
		centroids = np.array([x,y]).T
		self.centroids = centroids
		hm = []
		
		for c in ClusterID:
				hm.append(expression[clusters ==c,:].mean(axis=0))
		hm = np.stack(hm)

		hm = pd.DataFrame(StandardScaler().fit_transform(hm).T, columns=ClusterID, index=gene)
		Z = scipy.cluster.hierarchy.linkage(hm.T, method='average', metric='correlation')
		merged_labels_short = scipy.cluster.hierarchy.fcluster(Z, 0.5, criterion='distance')
		logging.info(f"Reduced {n_clusters} clusters to {merged_labels_short.max()} clusters")
		dic_merged_clusters = dict(zip(ClusterID, merged_labels_short))
		clusters = np.array([dic_merged_clusters[c] for c in clusters])

		expression_smooth = []
		tree = KDTree(centroids)
		dst, nghs= tree.query(centroids, distance_upper_bound=self.max_distance, k=50,workers=-1,p=2)
		nghs = [n[n < clusters.shape[0]] for n in nghs]
		clusters_nghs= [clusters[n] for n in nghs]
		if self.same_cluster:
			nghs= [n[c == c[0]] for c, n in zip(clusters_nghs, nghs)]
		else:
			nghs= [n for c, n in zip(clusters_nghs, nghs)]

		expression_smooth = np.array([expression[n,:].sum(axis=0) for n in nghs])
		if len(expression_smooth.shape) == 3:
			self.adata.layers['smooth'] = expression_smooth[:,0,:]
		elif len(expression_smooth.shape) == 2:
			self.adata.layers['smooth'] = expression_smooth#[:,:]
		self.adata.obs['MergedClusters'] = clusters
