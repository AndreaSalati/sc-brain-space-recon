import numpy as np
import pandas as pd

# functions for fitting with scipy


def func_2(x, a, b, c):
    """model that uses 1D coordinates to predict gene expression"""
    return np.exp(a * x**2 + b * x + c)


def func_2D(X, ax, bx, ay, by, c):
    """model that uses 2D coordinates to predict gene expression"""
    x, y = X
    return np.exp(ax * x**2 + bx * x + ay * y**2 + by * y + c)


def loss_2D(coord, Y, *args):
    Y_hat = func_2D(coord.T, *args)
    return np.sum((Y - Y_hat) ** 2)


def func_3D(X, ax, bx, ay, by, az, bz, c):
    """model that uses 3D coordinates to predict gene expression""" ""
    x, y, z = X
    return np.exp(
        ax * x**2 + bx * x + ay * y**2 + by * y + az * z**2 + bz * z + c
    )


def loss_3D(coord, Y, *args):
    Y_hat = func_3D(coord.T, *args)
    return np.sum((Y - Y_hat) ** 2)


####################


def ind(adata, gene_name):
    """
    adata: AnnData object
    takes in a gene name and returns the index of the gene in the adata.var.index
    """
    return adata.var.index.get_loc(gene_name)


def ind_z(list, gene_name):
    """
    takes just a list and returns the index of the gene in the list
    """
    return np.where(list == gene_name)[0][0]


def gene_pos(adata, gene_name, obs_field="x"):
    """
    avarages the expression of a gene across all cells in a given assigned field
    """
    v = adata[:, gene_name].X
    pos = adata.obs[obs_field]
    pos_un = pos.unique()
    out = np.zeros(pos_un.shape)

    for i, p in enumerate(pos_un):
        out[i] = np.mean(v[pos == p])

    return out
