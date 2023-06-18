from sklearn.preprocessing import MinMaxScaler, Normalizer
import scanpy as sc
import numpy as np
import logging

def preprocess(
    adata,
    normalize_mode='l2', #Choose between 'total'/'scanpy'/'l1 or 'l2', None
    log=True,
    batch_correction=True,
    scaling=False, # False, 'minmax', 'scanpy' 
    keep_raw=True,
    target_sum=1e3,
    ):

    if keep_raw:
        adata.raw = adata

    if normalize_mode is not None:
        if normalize_mode == 'total' or normalize_mode == 'scanpy' or normalize_mode == 'l1':
            sc.pp.normalize_total(adata, target_sum=target_sum)
        elif normalize_mode == 'l2':
            adata.X = Normalizer(norm='l2').fit_transform(adata.X)
        else:
            raise ValueError('normalize_mode must be either "total" or "l2"')
        
    if log:
        sc.pp.log1p(adata)
    
    if batch_correction and scaling:    
        logging.info('Batch correction mode on. This will force minmax scaling per sample to  remove batch effects.')
        adatas = []
        for s in np.unique(adata.obs.Sample):
            ad = adata[adata.obs.Sample == s,]#.X.mean(axis=0)
            ad.X = MinMaxScaler().fit_transform(ad.X)
            adatas.append(ad)
        adata = adata[0].concatenate(*adatas[1:])

    elif batch_correction == False and scaling == 'minmax':
        adata.X = MinMaxScaler().fit_transform(adata.X)

    elif batch_correction == False and scaling == 'scanpy':
        sc.pp.scale(adata)
