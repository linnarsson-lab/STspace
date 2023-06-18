### In the memory of Ka Wai Lee's dirty code. Rest in peace. ###

_N_XTICKS = 10
from collections import Iterable, defaultdict
import matplotlib.pyplot as plt
from cellrank.tl._utils import _min_max_scale
import pandas as pd
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from stlearn.plotting.color_utils import colorize
import matplotlib as mpl
## adapt from cellrank
mpl.rcParams['pdf.fonttype'] = 42

def pseudotime_heatmap(
    data,
    genes:np.array,
    figsize=(12,8),
    cmap='inferno',
    attr:dict=None,
    attrs_type:np.array=None,
    attrs_color:dict=None, 
    attr_strip_height:int=2,
    fontsize=6,
    highlight=None,
    skip:str='skip',
    plot_time=False,
    save=False ,
    savepath=None,
    **kwargs):
    
    width, height = figsize[0], figsize[1]
    num_strips =len(attr)
    strip_height=1
    attr_strip_height=attr_strip_height
    total_height = len(genes)*strip_height+ attr_strip_height * num_strips 
    fig, axes = plt.subplots(figsize=(width, height))

    grid = (total_height, 1)
    offset = 0
    x_min_=[]
    x_max_=[]
    
    data_t = defaultdict(dict)
    for gene, lns in data.items():
                for ln, y in lns.items():
                    data_t[ln][gene] = y
                    
    for lname, models in data_t.items():
        xs = np.array([m.x_test for m in models.values()])
        x_min, x_max = np.nanmin(xs), np.nanmax(xs)
        df = pd.DataFrame([m.y_test for m in models.values()], index=models.keys())

        max_sort = np.argsort(
            np.argmax(df.apply(_min_max_scale, axis=1).values, axis=1)
        )
        df = df.iloc[max_sort[::-1], :]
    
    sele_gene = np.isin(df.index,genes)
    x_min_,x_max_ = np.min(df.loc[sele_gene]),np.max(df.loc[sele_gene])

    attr_sorted = dict()
    
    if attr is not None:
        
        tmin, tmax = np.min(x_min_), np.max(x_max_)
        t = np.array(attr['Latent time'])
        b00l =(t>=tmin) & (t<=tmax)
        order = np.argsort(t[b00l])
        for i,(k,v) in enumerate(attr.items()):
            attr_sorted[k] = np.array(v)[b00l][order]
            
        
    if plot_time:
        attr_sorted = attr_sorted
    else:
        attr_sorted.pop('Latent time')

    for i,(k,v) in enumerate(attr_sorted.items()):
        if attrs_type[i] =='s':
            if attrs_color is None:
                d = colorize(np.nan_to_num(v))
            else:
                if k in list(attrs_color.keys()):
                    d = np.array([attrs_color[k][v_i] for v_i in v]) 
                else:
                    d = colorize(np.nan_to_num(v))
                    
            ax = plt.subplot2grid(grid, (offset, 0), rowspan=attr_strip_height)
            offset += attr_strip_height
            plt.imshow(np.expand_dims(d, axis=0), aspect='auto', extent=(np.min(x_min_),np.max(x_max_),0,1),
                       origin='lower',interpolation='nearest',rasterized=True,**kwargs)
                
            plt.text(-0.01,0.9 , k, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=fontsize, color="black")
        elif attrs_type[i] == 'f':
            d = np.copy(v)
            ax = plt.subplot2grid(grid, (offset, 0), rowspan=attr_strip_height)
            offset += attr_strip_height
            plt.imshow(np.expand_dims(d, axis=0), aspect='auto', cmap=attrs_color[k],  
                       origin='upper',interpolation='nearest',rasterized=True,**kwargs)

            plt.text(-0.01,0.9, k, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=fontsize, color="black")
        plt.axis("off")
    
#     offset+=strip_height
    gene_pos = 1
    for ax_i, gene in enumerate(df.index):
        if gene in genes:

            c = _min_max_scale(df.loc[gene]) 

            ax = plt.subplot2grid(grid, (offset, 0), rowspan=strip_height)
            offset += strip_height
            plt.imshow(np.expand_dims(c, axis=0), aspect='auto', cmap=cmap, vmin=0, vmax=1, \
                       extent=(np.min(x_min_),np.max(x_max_),0,1),\
                         origin='lower',interpolation='nearest',rasterized=True,**kwargs)
            color = 'black'
            labelleft=True
            if highlight is not None: 
                if gene in highlight:
                    color = '#cf2f74'
                    
            if skip=='skip':
                if (gene_pos%2)!=0:
                    ax.set_yticks([0.5])
                    ax.set_yticklabels([gene], ha="right",size=fontsize,color=color)
#                     plt.text(-0.01, 1.2, gene, horizontalalignment='right', verticalalignment='top', \
#                          transform=ax.transAxes, fontsize=fontsize, color=color)
                else:
                    ax.set_yticks([])
                    ax.set_yticklabels([])
                    labelleft=False
            elif skip=='all':
                ax.set_yticks([])
                ax.set_yticklabels([])
            else:
                ax.set_yticks([0.5])
                ax.set_yticklabels([gene], ha="right",size=fontsize,color=color)

            gene_pos+=1
            
            if highlight is not None: 
                if gene in highlight:
                    ax.yaxis.label.set_color('red')

            for pos in ["top", "left","bottom", "right"]:
                ax.spines[pos].set_visible(False)
        
            ax.tick_params('y', length=2, width=0.5, which='major',pad=0.5)
            ax.tick_params(
                top=False,
                bottom=False,
                left=labelleft,
                right=False,
                labelleft=labelleft,
                labelbottom=False,
            )

    plt.subplots_adjust(hspace=0)
    if save:
        plt.savefig(savepath, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()

