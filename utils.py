import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit


# functions for fitting with scipy
def calculate_bic(n, LL, num_params):
    bic = n * LL + num_params * np.log(n)
    return bic


# Exponential line function
def exp_line(x, a, b):
    return np.exp(a * x + b)


# Exponential parabola function
def exp_parabola(x, a, b, c):
    return np.exp(a * x**2 + b * x + c)


def LL(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2 / y_true.var())


#############################################


def func_2(x, a, b, c):
    """model that uses 1D coordinates to predict gene expression"""
    return np.exp(a * x**2 + b * x + c)


def func_2D(X, ax, bx, ay, by, c):
    """model that uses 2D coordinates to predict gene expression"""
    x, y = X
    return np.exp(ax * x**2 + bx * x + ay * y**2 + by * y + c)


def loss_2D(coord, Y, *args):
    Y_hat = func_2D(coord.T, *args)
    return np.sum((Y - Y_hat) ** 2) / Y.var()


def func_3D(X, ax, bx, ay, by, az, bz, c):
    """model that uses 3D coordinates to predict gene expression""" ""
    x, y, z = X
    return np.exp(
        ax * x**2 + bx * x + ay * y**2 + by * y + az * z**2 + bz * z + c
    )


def loss_3D(coord, Y, *args):
    Y_hat = func_3D(coord.T, *args)
    return np.sum((Y - Y_hat) ** 2) / Y.var()


####################


def analyze_gene(adata, g, d):
    mean_y = np.mean(adata[:, g].layers["norm"].squeeze())
    mse1 = LL(
        adata[:, g].layers["norm"].squeeze(),
        np.full_like(adata[:, g].layers["norm"].squeeze(), mean_y),
    )
    bic1 = calculate_bic(n, mse1, 1)

    # model 2
    params2, _ = curve_fit(
        exp_line, coord[:, d], adata[:, g].layers["norm"].squeeze(), p0=(0, 0)
    )
    y_pred2 = exp_line(coord[:, d], *params2)
    mse2 = LL(adata[:, g].layers["norm"].squeeze(), y_pred2)
    bic2 = calculate_bic(n, mse2, 2)

    # model 3
    params3, _ = curve_fit(
        exp_parabola, coord[:, d], adata[:, g].layers["norm"].squeeze(), p0=(0, 0, 0)
    )
    y_pred3 = exp_parabola(coord[:, d], *params3)
    mse3 = LL(adata[:, g].layers["norm"].squeeze(), y_pred3)
    bic3 = calculate_bic(n, mse3, 3)
    return bic1, bic2, bic3, mse1, mse2, mse3


def select_model(bic1, bic2, bic3):
    bic1_ = 1
    bic2_ = bic2 / bic1
    bic3_ = bic3 / bic1

    bic = np.array([bic1_, bic2_, bic3_])
    best = np.argmin(bic)

    if bic[best] < 0.95:
        zonated = True
    else:
        zonated = False

    return best, bic[best], zonated


def analyze_gene2(adata, g, d, threshold=0.95):
    mean_y = np.mean(adata[:, g].layers["norm"].squeeze())
    mse1 = LL(
        adata[:, g].layers["norm"].squeeze(),
        np.full_like(adata[:, g].layers["norm"].squeeze(), mean_y),
    )
    bic1 = calculate_bic(n, mse1, 1)

    # model 2
    params2, _ = curve_fit(
        exp_line, coord[:, d], adata[:, g].layers["norm"].squeeze(), p0=(0, 0)
    )
    y_pred2 = exp_line(coord[:, d], *params2)
    mse2 = LL(adata[:, g].layers["norm"].squeeze(), y_pred2)
    bic2 = calculate_bic(n, mse2, 2)

    # model 3
    params3, _ = curve_fit(
        exp_parabola, coord[:, d], adata[:, g].layers["norm"].squeeze(), p0=(0, 0, 0)
    )
    y_pred3 = exp_parabola(coord[:, d], *params3)
    mse3 = LL(adata[:, g].layers["norm"].squeeze(), y_pred3)
    bic3 = calculate_bic(n, mse3, 3)

    # normalize the bic
    bic1_ = 1
    bic2_ = bic2 / bic1
    bic3_ = bic3 / bic1
    bic = np.array([bic1_, bic2_, bic3_])
    best = np.argmin(bic)
    zonated = bic[best] < threshold

    if best == 0:
        return best, bic[best], zonated, [0.0, 0.0, np.log(mean_y)]
    elif best == 1:
        return best, bic[best], zonated, np.insert(params2, 0, 0)
    else:
        return best, bic[best], zonated, params3


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
