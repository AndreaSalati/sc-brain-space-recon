import numpy as np
import pandas as pd

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

####################
# numpyro models
####################


def model_2D(theta, n_counts, y=None):
    Nc = y.shape[0]
    Ng = y.shape[1]

    cell_plate = numpyro.plate("cells", size=Nc, dim=-2)
    gene_plate = numpyro.plate("genes", size=Ng, dim=-1)

    disp = numpyro.param(
        "disp",
        jnp.array(0.3),
        constraint=dist.constraints.positive,
    )

    with cell_plate:
        x_1 = numpyro.sample("x_1", dist.Normal(0.0, 1))
        x_2 = numpyro.sample("x_2", dist.Normal(0.0, 1))

    lmbda = (
        theta[:, 0][None, :] * x_1**2
        + theta[:, 1][None, :] * x_1
        + theta[:, 2][None, :] * x_2**2
        + theta[:, 3][None, :] * x_2
        + theta[:, 4][None, :]
    )
    # before taking the exp i want to avoid exploding values
    lmbda = jnp.clip(lmbda, a_min=-10, a_max=10)
    mu = jnp.exp(lmbda) * n_counts[:, None]

    ##Using the mean and dispersion parametrization
    conc = 1 / disp
    rate = 1 / (disp * mu)

    with cell_plate, gene_plate:
        numpyro.sample("obs_", dist.GammaPoisson(concentration=conc, rate=rate), obs=y)


def guide_2D(theta, n_counts, y=None):
    # define the plates
    Nc = y.shape[0]

    cell_plate = numpyro.plate("cells", size=Nc, dim=-2)

    # define the parameters
    x_1_loc = numpyro.param("x_1_loc", jnp.zeros(Nc).reshape(-1, 1))
    x_2_loc = numpyro.param("x_2_loc", jnp.zeros(Nc).reshape(-1, 1))

    x_1_scale = numpyro.param(
        "x_1_scale",
        0.1 * jnp.ones(Nc).reshape(-1, 1),
        constraint=dist.constraints.positive,
    )

    x_2_scale = numpyro.param(
        "x_2_scale",
        0.1 * jnp.ones(Nc).reshape(-1, 1),
        constraint=dist.constraints.positive,
    )

    with cell_plate:
        x_1 = numpyro.sample("x_1", dist.Normal(x_1_loc, x_1_scale))
        x_2 = numpyro.sample("x_2", dist.Normal(x_2_loc, x_2_scale))


def model_2D_batch(theta, n_counts, y=None, batch=64):
    Nc = y.shape[0]
    Ng = y.shape[1]

    cell_plate = numpyro.plate("cells", size=Nc, dim=-2, subsample_size=batch)
    gene_plate = numpyro.plate("genes", size=Ng, dim=-1)

    disp = numpyro.param(
        "disp",
        jnp.array(0.2),
        constraint=dist.constraints.positive,
    )

    with cell_plate as ind:
        x_1 = numpyro.sample("x_1", dist.Normal(0.0, 0.3))
        x_2 = numpyro.sample("x_2", dist.Normal(0.0, 0.3))

    lmbda = (
        theta[:, 0][None, :] * x_1[ind, :] ** 2
        + theta[:, 1][None, :] * x_1[ind, :]
        + theta[:, 2][None, :] * x_2[ind, :] ** 2
        + theta[:, 3][None, :] * x_2[ind, :]
        + theta[:, 4][None, :]
    )

    # alpha = jnp.exp(disp)
    lmbda = jnp.clip(lmbda, a_min=-10, a_max=10)
    mu = jnp.exp(lmbda) * n_counts[ind, None]

    ##Using the mean and dispersion parametrization
    conc = 1 / disp
    rate = 1 / (disp * mu)

    with cell_plate, gene_plate:
        numpyro.sample(
            "obs_", dist.GammaPoisson(concentration=conc, rate=rate), obs=y[ind, :]
        )


def guide_2D_batch(theta, n_counts, y=None, batch=64):
    # define the plates
    Nc = y.shape[0]

    cell_plate = numpyro.plate("cells", size=Nc, dim=-2, subsample_size=batch)

    # define the parameters
    x_1_loc = numpyro.param("x_1_loc", jnp.zeros(Nc).reshape(-1, 1))
    x_2_loc = numpyro.param("x_2_loc", jnp.zeros(Nc).reshape(-1, 1))
    # x_1_loc = numpyro.param("x_1_loc", jnp.array(y_guess).reshape(-1,1))
    # x_2_loc = numpyro.param("x_2_loc", jnp.array(z_guess).reshape(-1,1))

    x_1_scale = numpyro.param(
        "x_1_scale",
        1.0 * jnp.ones(Nc).reshape(-1, 1),
        constraint=dist.constraints.positive,
    )
    x_2_scale = numpyro.param(
        "x_2_scale",
        1.0 * jnp.ones(Nc).reshape(-1, 1),
        constraint=dist.constraints.positive,
    )

    with cell_plate as ind:
        x_1 = numpyro.sample("x_1", dist.Normal(x_1_loc[ind, :], x_1_scale[ind, :]))
        x_2 = numpyro.sample("x_2", dist.Normal(x_2_loc[ind, :], x_2_scale[ind, :]))
