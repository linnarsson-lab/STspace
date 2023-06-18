from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union,
    TypeVar,
    Callable,
    Hashable,
    Iterable,
    Optional,
    Sequence,
)
from typing_extensions import Literal

import os
import wrapt
import warnings
from types import MappingProxyType
from inspect import signature
from functools import wraps, update_wrapper
from itertools import tee, product, combinations
from contextlib import contextmanager
from statsmodels.stats.multitest import multipletests

import scanpy as sc
from anndata import AnnData
from cellrank import logging as logg
from anndata.utils import make_index_unique

import numpy as np
import pandas as pd
from pandas import Series
from scipy.stats import norm
from numpy.linalg import norm as d_norm
from scipy.sparse import eye as speye
from scipy.sparse import diags, issparse, spmatrix, csr_matrix, isspmatrix_csr
from sklearn.cluster import KMeans
from pandas.api.types import infer_dtype, is_bool_dtype, is_categorical_dtype
from scipy.sparse.linalg import norm as sparse_norm

import matplotlib.colors as mcolors

ColorLike = TypeVar("ColorLike")
GPCCA = TypeVar("GPCCA")
CFLARE = TypeVar("CFLARE")
DiGraph = TypeVar("DiGraph")

EPS = np.finfo(np.float64).eps


import sys
sys.path.append(r'/data/proj/EA_KWL/rmd/from_git/off_grid_cytograph/')

from stlearn.tools.pseudotime_utils import _parallelize
from stlearn.tools.pseudotime_utils import _enum


class TestMethod(_enum.ModeEnum):  # noqa
    FISCHER = "fischer"
    PERM_TEST = "perm_test"



def _perm_test(
    ixs: np.ndarray,
    corr: np.ndarray,
    X: Union[np.ndarray, spmatrix],
    Y: np.ndarray,
    seed: Optional[int] = None,
    queue=None,
) -> Tuple[np.ndarray, np.ndarray]:
    rs = np.random.RandomState(None if seed is None else seed + ixs[0])
    cell_ixs = np.arange(X.shape[1])
    pvals = np.zeros_like(corr, dtype=np.float64)
    corr_bs = np.zeros((len(ixs), X.shape[0], Y.shape[1]))  # perms x genes x lineages

    mmc = _mat_mat_corr_sparse if issparse(X) else _mat_mat_corr_dense

    for i, _ in enumerate(ixs):
        rs.shuffle(cell_ixs)
        corr_i = mmc(X, Y[cell_ixs, :])
        pvals += np.abs(corr_i) >= np.abs(corr)

        bootstrap_ixs = rs.choice(cell_ixs, replace=True, size=len(cell_ixs))
        corr_bs[i, :, :] = mmc(X[:, bootstrap_ixs], Y[bootstrap_ixs, :])

        if queue is not None:
            queue.put(1)

    if queue is not None:
        queue.put(None)

    return pvals, corr_bs

def _mat_mat_corr_sparse(
    X: csr_matrix,
    Y: np.ndarray,
) -> np.ndarray:
    n = X.shape[1]

    X_bar = np.reshape(np.array(X.mean(axis=1)), (-1, 1))
    X_std = np.reshape(
        np.sqrt(np.array(X.power(2).mean(axis=1)) - (X_bar**2)), (-1, 1)
    )

    y_bar = np.reshape(np.mean(Y, axis=0), (1, -1))
    y_std = np.reshape(np.std(Y, axis=0), (1, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (X @ Y - (n * X_bar * y_bar)) / ((n - 1) * X_std * y_std)

def _mat_mat_corr_dense(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    from FISHspace.tools.utils._utils import np_std, np_mean

    n = X.shape[1]

    X_bar = np.reshape(np_mean(X, axis=1), (-1, 1))
    X_std = np.reshape(np_std(X, axis=1), (-1, 1))

    y_bar = np.reshape(np_mean(Y, axis=0), (1, -1))
    y_std = np.reshape(np_std(Y, axis=0), (1, -1))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (X @ Y - (n * X_bar * y_bar)) / ((n - 1) * X_std * y_std)


def _correlation_test_helper(
    X: Union[np.ndarray, spmatrix],
    Y: np.ndarray,
    method: TestMethod = TestMethod.FISCHER,
    n_perms: Optional[int] = None,
    seed: Optional[int] = None,
    confidence_level: float = 0.95,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the correlation between rows in matrix ``X`` columns of matrix ``Y``.
    Parameters
    ----------
    from _correlation_test
        X
            Array or sparse matrix of shape ``(n_cells, n_genes)`` containing the expression.
        Y
            Array of shape ``(n_cells, n_lineages)`` containing the absorption probabilities.
    
    X
        Array or matrix of `(M, N)` elements.
    Y
        Array of `(N, K)` elements.
    method
        Method for p-value calculation.
    n_perms
        Number of permutations if ``method='perm_test'``.
    seed
        Random seed if ``method = 'perm_test'``.
    confidence_level
        Confidence level for the confidence interval calculation. Must be in `[0, 1]`.
    kwargs
        Keyword arguments for :func:`cellrank._utils._parallelize.parallelize`.
    Returns
    -------
        Correlations, p-values, corrected p-values, lower and upper bound of 95% confidence interval.
        Each array if of shape ``(n_genes, n_lineages)``.
    """

    def perm_test_extractor(
        res: Sequence[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pvals, corr_bs = zip(*res)
        pvals = np.sum(pvals, axis=0) / float(n_perms)

        corr_bs = np.concatenate(corr_bs, axis=0)
        corr_ci_low, corr_ci_high = np.quantile(corr_bs, q=ql, axis=0), np.quantile(
            corr_bs, q=qh, axis=0
        )

        return pvals, corr_ci_low, corr_ci_high

    if not (0 <= confidence_level <= 1):
        raise ValueError(
            f"Expected `confidence_level` to be in interval `[0, 1]`, found `{confidence_level}`."
        )

    n = X.shape[1]  # genes x cells
    ql = 1 - confidence_level - (1 - confidence_level) / 2.0
    qh = confidence_level + (1 - confidence_level) / 2.0

    if issparse(X) and not isspmatrix_csr(X):
        X = csr_matrix(X)

    corr = _mat_mat_corr_sparse(X, Y) if issparse(X) else _mat_mat_corr_dense(X, Y)

    if method == TestMethod.FISCHER:
        # see: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Using_the_Fisher_transformation
        mean, se = np.arctanh(corr), 1.0 / np.sqrt(n - 3)
        z_score = (np.arctanh(corr) - np.arctanh(0)) * np.sqrt(n - 3)

        z = norm.ppf(qh)
        corr_ci_low = np.tanh(mean - z * se)
        corr_ci_high = np.tanh(mean + z * se)
        pvals = 2 * norm.cdf(-np.abs(z_score))

    elif method == TestMethod.PERM_TEST:
        if not isinstance(n_perms, int):
            raise TypeError(
                f"Expected `n_perms` to be an integer, found `{type(n_perms).__name__}`."
            )
        if n_perms <= 0:
            raise ValueError(f"Expcted `n_perms` to be positive, found `{n_perms}`.")

        pvals, corr_ci_low, corr_ci_high = _parallelize.parallelize(
            _perm_test,
            np.arange(n_perms),
            as_array=False,
            unit="permutation",
            extractor=perm_test_extractor,
            **kwargs,
        )(corr, X, Y, seed=seed)

    else:
        raise NotImplementedError(method)

    return corr, pvals, corr_ci_low, corr_ci_high

def _correlation_test(
    X: Union[np.ndarray, spmatrix],
    Y: "Lineage",  # noqa: F821
    gene_names: Sequence[str],
    method: TestMethod = TestMethod.FISCHER,
    confidence_level: float = 0.95,
    n_perms: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Perform a statistical test.
    Return NaN for genes which don't vary across cells.
    Parameters
    ----------
    X
        Array or sparse matrix of shape ``(n_cells, n_genes)`` containing the expression.
    Y
        Array of shape ``(n_cells, n_lineages)`` containing the absorption probabilities.
    gene_names
        Sequence of shape ``(n_genes,)`` containing the gene names.
    method
        Method for p-value calculation.
    confidence_level
        Confidence level for the confidence interval calculation. Must be in `[0, 1]`.
    n_perms
        Number of permutations if ``method = 'perm_test'``.
    seed
        Random seed if ``method = 'perm_test'``.
    %(parallel)s
    Returns
    -------
    Dataframe of shape ``(n_genes, n_lineages * 5)`` containing the following columns, one for each lineage:
        - ``{lineage}_corr`` - correlation between the gene expression and absorption probabilities.
        - ``{lineage}_pval`` - calculated p-values for double-sided test.
        - ``{lineage}_qval`` - corrected p-values using Benjamini-Hochberg method at level `0.05`.
        - ``{lineage}_ci_low`` - lower bound of the ``confidence_level`` correlation confidence interval.
        - ``{lineage}_ci_high`` - upper bound of the ``confidence_level`` correlation confidence interval.
    """

    corr, pvals, ci_low, ci_high = _correlation_test_helper(
        X.T,
        Y.X,
        method=method,
        n_perms=n_perms,
        seed=seed,
        confidence_level=confidence_level,
        **kwargs,
    )
    invalid = (corr < -1) | (corr > 1)
    if np.any(invalid):
        logg.warning(
            f"Found `{np.sum(invalid)}` correlation(s) that are not in `[0, 1]`. "
            f"This usually happens when gene expression is constant across all cells. "
            f"Setting to `NaN`"
        )
        corr[invalid] = np.nan
        pvals[invalid] = np.nan
        ci_low[invalid] = np.nan
        ci_high[invalid] = np.nan

    res = pd.DataFrame(corr, index=gene_names, columns=[f"{c}_corr" for c in Y.names])
    for idx, c in enumerate(Y.names):
        p = pvals[:, idx]
        valid_mask = ~np.isnan(p)

        res[f"{c}_pval"] = p
        res[f"{c}_qval"] = np.nan
        if np.any(valid_mask):
            res.loc[gene_names[valid_mask], f"{c}_qval"] = multipletests(
                p[valid_mask], alpha=0.05, method="fdr_bh"
            )[1]
        res[f"{c}_ci_low"] = ci_low[:, idx]
        res[f"{c}_ci_high"] = ci_high[:, idx]

    # fmt: off
    res = res[[f"{c}_{stat}" for c in Y.names for stat in ("corr", "pval", "qval", "ci_low", "ci_high")]]
    return res.sort_values(by=[f"{c}_corr" for c in Y.names], ascending=False)
    # fmt: on