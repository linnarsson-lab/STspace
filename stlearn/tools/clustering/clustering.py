import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.spatial import KDTree
from tqdm import tqdm, trange
import logging
from sklearn.manifold import SpectralEmbedding
from itertools import permutations, product
from collections import Counter
logging.basicConfig(level=logging.INFO)

def leiden(
        adata,
        n_obs: int = 250000, # Number of samples to run leiden on
        n_neighbors: int = 15, # Number of neighbors to use for leiden
        resolution: float=1.0, # Resolution parameter for leiden
        use_rep: str = 'X_pca', # Use PCA embedding for clustering
        key: str = 'leiden', # Key to store clustering in
    ):
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    adata_mini = sc.pp.subsample(adata, n_obs=n_obs, copy=True)
    logging.info('Building neighbor graph for clustering...')
    sc.pp.neighbors(adata_mini, n_neighbors=n_neighbors,use_rep=use_rep)
    logging.info('Running Leiden clustering...')
    sc.tl.leiden(adata_mini, random_state=42, resolution=resolution)
    logging.info('Leiden clustering done.')
    clusters= adata_mini.obs['leiden'].values

    logging.info('Total of {} found'.format(len(np.unique(clusters))))
    clf = make_pipeline(StandardScaler(), SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3))
    clf.fit(adata_mini.obsm['X_pca'], clusters)
    clusters = clf.predict(adata.obsm['X_pca']).astype('int16')
    adata.obs[key] = clusters
    adata.obs[key] = adata.obs[key].astype('category')
    _renumber_clusters(adata, key=key)


def _renumber_clusters(
		adata,
		key='leiden',
		):
        """Aranges cluster numbers based on similarity"""
        u, c = np.unique(adata.obs[key].to_numpy().astype(int), return_counts=True)
        # Renumber clusters
        #make mean expression
        df_mean = res = pd.DataFrame(columns=adata.var_names, index=sorted(u))
        for clust in sorted(u):
            filt = adata.obs[key].isin([clust])
            df_mean.loc[clust] = adata[filt, :].X.mean(0)

        #Order  and renumber clusters
        manifold = SpectralEmbedding(n_components=1, n_jobs=-1).fit_transform(df_mean)
        even_spaced = np.arange(manifold.shape[0])
        even_spaced_dict = dict(zip(np.sort(manifold.ravel()), even_spaced))
        manifold_even = np.array([even_spaced_dict[i] for i in manifold.ravel()])
        manifold_even_dict = dict(zip(df_mean.index, manifold_even))
        reordered_labels = np.array([manifold_even_dict[i] for i in adata.obs[key].to_numpy().astype(int)])
        key_reordered = f'{key}_reordered'
        adata.obs[key_reordered] = reordered_labels
        adata.obs[key_reordered] = adata.obs[key_reordered].astype('category')

def clean_outliers(
		adata,
		key='birch',
		eps=0.125,
		min_samples=150,
		save=False,
		save_path=None,
		n_clusters=5,
		):
        """Removes outliers using DBscan on each leiden cluster"""

        from sklearn.cluster import Birch
        bir= Birch(n_clusters=n_clusters,).fit_predict(adata.obsm['X_umap'])
        adata.obs[key] = bir

        from sklearn.cluster import DBSCAN
        logging.info('Total cells before DBscan {}'.format(adata.shape[0]))
        adata.obs_names_make_unique()
        adata_leiden = [adata[adata.obs[key] == l] for l in np.unique(adata.obs[key])]
        ad_clean = []
        for ad in tqdm(adata_leiden):
            logging.info(f'Number of cells before DBscan {ad.shape[0]}')
            xy = ad.obsm['X_umap']
            db = DBSCAN(eps=eps,min_samples=min_samples).fit_predict(xy)
            a, c = np.unique(db, return_counts=True)
            c = c[c<= min_samples]
            ad = ad[np.isin(db, [-1]+c.tolist(), invert=True)]
            logging.info(f'Number of cells after DBscan {ad.shape[0]}')
            ad_clean.append(ad)
            
        logging.info('Total of {} datasets to concatenate.'.format(len(ad_clean)))
        ad_clean = ad_clean[0].concatenate(*ad_clean[1:])
        logging.info('Total cells after DBscan {}'.format(ad_clean.shape[0]))
        if save:
            ad_clean.write_h5ad(f'{save_path}')

        return ad_clean
