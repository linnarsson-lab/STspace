import scanpy as sc
import logging
logging.basicConfig(level=logging.INFO)

def umap_largedata(
        adata,
        n_obs: int = 250000, # Number of samples to run leiden on
        n_neighbors: int = 15, # Number of neighbors to use for leiden
        n_epochs: int = 250, # Number of epochs to run UMAP
        ):
    try:
        from umap.parametric_umap import ParametricUMAP
        adata_mini = sc.pp.subsample(adata, n_obs=n_obs, copy=True)
        embedder = ParametricUMAP(n_epochs = n_epochs, verbose=True)
        embedding_mini = embedder.fit_transform(adata_mini.obsm['X_pca'])

        history = embedder._history
        embedding = embedder.fit_transform(adata.obsm['X_pca'])
        adata.obsm['X_umap_parametric'] = embedding

    except:
        raise ImportError("Please install umap-learn and parametric_umap to use this function")

