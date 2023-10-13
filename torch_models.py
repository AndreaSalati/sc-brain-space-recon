import numpy as np
import pandas as pd
import torch


####################
# torch models
####################


def model_2D_NB(X, Y, theta, n_count, disp):
    """
    model that given the expressiony and the parameters theta
    returns the expected value of the expression. To do so it needs to
    optimize the position-parameter x. Than we un least square loss to fit the model
    """
    NC = Y.shape[0]

    lmbda = (
        theta[:, 0][None, :] * X[:, 0][:, None] ** 2
        + theta[:, 1][None, :] * X[:, 0][:, None]
        + theta[:, 2][None, :] * X[:, 1][:, None] ** 2
        + theta[:, 3][None, :] * X[:, 1][:, None]
        + theta[:, 4][None, :]
    )
    lmbda = torch.exp(lmbda) * n_count[:, None]
    alpha = torch.exp(disp)

    r = 1 / alpha
    p = alpha * lmbda / (1 + alpha * lmbda)

    NB = torch.distributions.NegativeBinomial(
        total_count=r, probs=p, validate_args=None
    )

    return -NB.log_prob(Y).sum()


def model_2D_NB_batch(X, Y, theta, n_count, disp, batch_size=64):
    """
    model that given the expressiony and the parameters theta
    returns the expected value of the expression. To do so it needs to
    optimize the position-parameter x. Than we un least square loss to fit the model
    """
    NC = Y.shape[0]
    idx = torch.randperm(NC)[:batch_size]

    lmbda = (
        theta[:, 0][None, :] * X[idx, 0][:, None] ** 2
        + theta[:, 1][None, :] * X[idx, 0][:, None]
        + theta[:, 2][None, :] * X[idx, 1][:, None] ** 2
        + theta[:, 3][None, :] * X[idx, 1][:, None]
        + theta[:, 4][None, :]
    )

    lmbda = torch.exp(lmbda) * n_count[idx, None]
    alpha = torch.exp(disp)

    r = 1 / alpha
    p = alpha * lmbda / (1 + alpha * lmbda)

    NB = torch.distributions.NegativeBinomial(
        total_count=r, probs=p, validate_args=None
    )

    return -NB.log_prob(Y[idx, :]).sum() * (NC / batch_size)


def model_2D_NB_like(X, Y, theta, n_count, disp):
    """
    this is not a loss function as the output is not a scalar
    """
    NC = Y.shape[0]

    lmbda = (
        theta[:, 0][None, :] * X[:, 0][:, None] ** 2
        + theta[:, 1][None, :] * X[:, 0][:, None]
        + theta[:, 2][None, :] * X[:, 1][:, None] ** 2
        + theta[:, 3][None, :] * X[:, 1][:, None]
        + theta[:, 4][None, :]
    )
    lmbda = torch.exp(lmbda) * n_count[:, None]
    alpha = torch.exp(disp)

    r = 1 / alpha
    p = alpha * lmbda / (1 + alpha * lmbda)

    NB = torch.distributions.NegativeBinomial(
        total_count=r, probs=p, validate_args=None
    )
    log_prob = -NB.log_prob(Y)
    return log_prob


# least square loss
def model_2D(X, Y, theta):
    """
    model that given the expressiony and the parameters theta
    returns the expected value of the expression. To do so it needs to
    optimize the position-parameter x. Than we un least square loss to fit the model
    """
    NC = Y.shape[0]

    lmbda = (
        theta[:, 0][None, :] * X[:, 0][:, None] ** 2
        + theta[:, 1][None, :] * X[:, 0][:, None]
        + theta[:, 2][None, :] * X[:, 1][:, None] ** 2
        + theta[:, 3][None, :] * X[:, 1][:, None]
        + theta[:, 4][None, :]
    )
    lmbda = torch.exp(lmbda)

    loss = torch.sum((Y - lmbda) ** 2)
    return loss


def make_design_matrix(cell_identifiers):
    """A function to create a generic design matrix from cell identifiers

    For example:
    input, cell_identifiers = [1,1,1,2,2,2]
    output, design_matrix: [[1,1,1,0,0,0][0,0,0,1,1,1]]

    Arguments
    ---------
    cell_identifiers: torch tensor
        List of cells identifiers, should be of type torch.int

    Returns
    -------
    design_matrix: torch tensor
        Design matrix of shape (num_cells, num_unique_identifiers)

    """
    design_matrix = torch.hstack(
        [
            (cell_identifiers == v).type(torch.float).reshape(len(cell_identifiers), 1)
            for i, v in enumerate(torch.unique(cell_identifiers))
        ]
    )
    return design_matrix


def model_2D_DM(X, Y, theta, sample, disp, n_count, dm, batch_size=64):
    """
    this model introduces a sample dependent parameter in the model
    in order to disentangle the effect of the sample from the effect of the position
    while before all gene related coefficients where constant,
    now they are sample dependent, and will be optimized
    """
    NC = Y.shape[0]
    idx = torch.randperm(NC)[:batch_size]

    lmbda = (
        theta[:, 0][None, :] * X[idx, 0][:, None] ** 2
        + theta[:, 1][None, :] * X[idx, 0][:, None]
        + theta[:, 2][None, :] * X[idx, 1][:, None] ** 2
        + theta[:, 3][None, :] * X[idx, 1][:, None]
        + dm @ sample[idx, :]
    )

    lmbda = torch.exp(lmbda) * n_count[idx, None]
    alpha = torch.exp(disp)

    r = 1 / alpha
    p = alpha * lmbda / (1 + alpha * lmbda)

    NB = torch.distributions.NegativeBinomial(
        total_count=r, probs=p, validate_args=None
    )

    return -NB.log_prob(Y[idx, :]).sum() * (NC / batch_size)


# def model_2D_NB_gene_norm(X, Y, theta, n_count, disp, batch_size=64):
#     """
#     model that given the expressiony and the parameters theta
#     returns the expected value of the expression. To do so it needs to
#     optimize the position-parameter x. Than we un least square loss to fit the model
#     """
#     NC = Y.shape[0]
#     idx = torch.randperm(NC)[:batch_size]

#     lmbda = (
#         theta[:, 0][None, :] * X[idx, 0][:, None] ** 2
#         + theta[:, 1][None, :] * X[idx, 0][:, None]
#         + theta[:, 2][None, :] * X[idx, 1][:, None] ** 2
#         + theta[:, 3][None, :] * X[idx, 1][:, None]
#         + theta[:, 4][None, :]
#     )

#     lmbda = torch.exp(lmbda) * n_count[idx, None]
#     alpha = torch.exp(disp)

#     r = 1 / alpha
#     p = alpha * lmbda / (1 + alpha * lmbda)

#     NB = torch.distributions.NegativeBinomial(
#         total_count=r, probs=p, validate_args=None
#     )
#     like = NB.log_prob(Y[idx, :])#.sum(axis=0)
#     like_no_grad = like.detach().clone().sum(axis=0)
#     like_corrected = like / like_no_grad[None, :]


#     return - like_corrected.sum()
