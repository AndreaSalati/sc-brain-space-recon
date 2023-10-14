import numpy as np
import scanpy as sc
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


def analyze_gene2(adata, g, coord, d, n, threshold=[0.5, 0.8]):
    mean_y = np.mean(adata[:, g].layers["norm"].squeeze())
    mse1 = LL(
        adata[:, g].layers["norm"].squeeze(),
        np.full_like(adata[:, g].layers["norm"].squeeze(), mean_y),
    )
    bic1 = calculate_bic(n, mse1, 1)

    # model 2
    params2, _ = curve_fit(
        exp_line,
        coord[:, d],
        adata[:, g].layers["norm"].squeeze(),
        p0=(0, 0),
        maxfev=10000,
    )
    y_pred2 = exp_line(coord[:, d], *params2)
    mse2 = LL(adata[:, g].layers["norm"].squeeze(), y_pred2)
    bic2 = calculate_bic(n, mse2, 2)

    # model 3
    params3, _ = curve_fit(
        exp_parabola,
        coord[:, d],
        adata[:, g].layers["norm"].squeeze(),
        p0=(0, 0, 0),
        maxfev=10000,
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
    # and of two conditions
    zonated = bic[best] < threshold[1] and bic[best] > threshold[0]
    if best == 0:
        return best, bic[best], zonated, [0.0, 0.0, np.log(mean_y)]
    elif best == 1:
        return best, bic[best], zonated, np.insert(params2, 0, 0)
    else:
        return best, bic[best], zonated, params3
