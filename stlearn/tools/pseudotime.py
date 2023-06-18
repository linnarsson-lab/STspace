import scanpy as sc
import numpy as np
import pandas as pd
import cellrank as cr
from stlearn.tools.pseudotime_utils.cellrank_utils import _correlation_test
from stlearn.pp import preprocess
from cellrank.pl._utils import (
    _fit_bulk,
    _get_backend,
    _create_models,
    _create_callbacks,
)

from cellrank.ul._utils import (
    _get_n_cores
)

def pseudotime_genes(
    adata,
    genes_list,
    cluster_key,
    clusters=None,
    latent_time_key = 'latent_time',
    n_bins=4,
    min_pval=0.005,
    min_logfold=0.1,
    highlight_genes=[],

    ):
    adata.obs['latent_time']
    adata_orig = adata.copy()

    if clusters is not None:
        b00l = np.isin(adata.obs[cluster_key], clusters)
        adata = adata[b00l,:]
    #sc.pp.normalize_per_cell(adata)
    #sc.pp.log1p(adata)

    model = cr.ul.models.GAM(adata)

    lineage_class = cr.tl.Lineage(
                                    input_array=adata.obs['latent_time'].values,
                                    names=['latent'],
                                )
    adata.obsm['to_terminal_states'] = lineage_class
    driver_adata = _correlation_test(
                                    adata.X,
                                    Y=lineage_class,
                                    gene_names=adata.var.index,
                                    )

    notnull=~(driver_adata['latent_corr'].isnull())
    driver_adata = driver_adata.loc[notnull,:]
    idx = np.argsort(driver_adata['latent_corr'])[::-1]
    top_cell = driver_adata.iloc[idx,:]

    b00l = np.isin(top_cell.index.values,genes_list)
    top_lineage_genes = top_cell.loc[b00l,:]
    b00l = np.isin(adata.obs[cluster_key],clusters)

    time_adata = adata[b00l,:] 
    cellid_bin_ =[]
    edges = [int((x)*(time_adata.shape[0]/n_bins )) for x in range(n_bins+1)]
    for n in range(n_bins):
        cellid_bin = time_adata.obs.iloc[np.argsort(time_adata.obs['latent_time'].values)][edges[n]:edges[n+1]].index
        cellid_bin_.append(cellid_bin)

    time_adata.obs['lineage_Clusters']=np.repeat(0,time_adata.shape[0])
    for i,v in enumerate(cellid_bin_):
        b00l = np.isin(time_adata.obs.index,v)
        time_adata.obs['lineage_Clusters'][b00l] =np.repeat(i,len(b00l))
    time_adata.obs['lineage_Clusters'] = time_adata.obs['lineage_Clusters'].astype('category')
    
    
    if np.array_equal(adata.obs_names, time_adata.obs_names):
        adata.obs['lineage_Clusters'] = time_adata.obs['lineage_Clusters'].values
        b00l_var_1 = np.isin(adata.var.index,top_lineage_genes.index)
        sub_adata = adata[:,b00l_var_1]
        #sc.pp.normalize_per_cell(sub_adata)
        #
        #sc.pp.log1p(sub_adata)
        #del sub_adata.raw

        sc.tl.rank_genes_groups(sub_adata, 'lineage_Clusters',use_raw=False)

        result = sub_adata.uns['rank_genes_groups']
        groups = result['names'].dtype.names
        rank_result = pd.DataFrame(
            {group + '_' + key: result[key][group]
            for group in groups for key in ['names', 'pvals_adj','logfoldchanges']})
    else:
        print(f'CELL ID NOT the SAME!')

    bins=n_bins
    gene_in_order = []
    for i in np.arange(bins):

        total_genes= 1000
        min_logfold_ = min_logfold
        counter = 0
        print('bin',i,)
        while total_genes > 75:
            #print('bin',i,)
            b00l_0 = rank_result[f'{i}_pvals_adj']<min_pval
            #print(b00l_0.sum(),'pval')

            b00l_1 = rank_result[f'{i}_logfoldchanges']>min_logfold_
            bool_names = rank_result[f'{i}_names'].isin(highlight_genes) & rank_result[f'{i}_logfoldchanges']>min_logfold
            b00l = b00l_1 | bool_names
            #print(b00l_1.sum(), 'lofgold')
            b00l = b00l_0 & b00l_1
            #print(b00l.sum(),'left genes')
            rank_result_thres = rank_result.loc[b00l,:]
            min_logfold_ += 0.2
            total_genes = rank_result_thres.shape[0]
            counter += 1
            if counter > 100:  
                break
        if rank_result_thres.shape[0]>50:
            if i == (bins-1):
                print(i)
                print(rank_result_thres)
                print(rank_result_thres[f'{i}_logfoldchanges'])
                thres = np.percentile(rank_result_thres[f'{i}_logfoldchanges'].values,50)
                b00l = rank_result_thres[f'{i}_logfoldchanges'].values > thres
                bool_names = rank_result_thres[f'{i}_names'].isin(highlight_genes)
                b00l = b00l | bool_names

                rank_result_thres = rank_result_thres.loc[b00l,:]
        else:
            if i == (bins-1):
                thres = np.percentile(rank_result_thres[f'{i}_logfoldchanges'].values,50)
                b00l = rank_result_thres[f'{i}_logfoldchanges'].values>thres
                bool_names = rank_result_thres[f'{i}_names'].isin(highlight_genes)
                b00l = b00l | bool_names

                rank_result_thres = rank_result_thres.loc[b00l,:]


        sort_ = np.argsort(rank_result_thres[f'{i}_logfoldchanges'].values)[::-1]
        result_sorted = rank_result_thres.iloc[sort_]
        #print(result_sorted.shape[0],'result_sorted')
        gene_in_order.append(result_sorted[f'{i}_names'].values)
        
        tidy_gene_in_order =[]
        for i, v in enumerate(gene_in_order):
            flat = [i for s in tidy_gene_in_order for i in s]
            b00l = np.isin(v,flat,invert=True)
            tidy_gene_in_order.append(v[b00l])
        tidy_gene_in_order = [i for s in tidy_gene_in_order for i in s]

    try:
        adata_orig.X = adata_orig.raw[:,adata_orig.var_names].X
    except:
        pass
    data_processed = _preprocess(
        model,
        top_lineage_genes,
        sub_adata,

    )
    lineages_out = dict(
            {
                'tidy_gene_in_order':tidy_gene_in_order,
                'time_adata':time_adata,
                'model':model,
                'top_lineage_genes':top_lineage_genes,
                'data_processed':data_processed,
                'sub_adata':sub_adata,
            }
        )
    
        
    return lineages_out

def _preprocess(
    model,
    top_lineage_genes,
    adata,
    ):
    lineages = ['latent']
    
    orig_ = adata

    models = _create_models(model, top_lineage_genes.index,lineages)
    callback = None
    time_range = None
    backend = "loky"
    n_jobs = 1
    show_progress_bar = True
    kwargs = dict()
    kwargs["backward"] = False
    kwargs["time_key"] = 'latent_time'
    # kwargs['n_test_points']=None
    all_models, data, genes, lineages = _fit_bulk(
        models,
        _create_callbacks(orig_, callback, top_lineage_genes.index, lineages, **kwargs),
        top_lineage_genes.index,
        lineages,
        time_range,
        return_models=True,  # always return (better error messages)
        filter_all_failed=True,
        parallel_kwargs={
            "show_progress_bar": show_progress_bar,
            "n_jobs": _get_n_cores(n_jobs, len(top_lineage_genes.index)),
            "backend": _get_backend(models, backend),
        },
        **kwargs,
    )
    return data