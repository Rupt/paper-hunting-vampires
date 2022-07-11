"""Tools for analysing jet-momenta data with a jax net."""
import gzip
import itertools
import json
import os
from dataclasses import dataclass
from functools import partial

import haiku
import jax
import joblib
import numpy
import optax
from .jet_lib import META_NAME, N_PARTICLES, PARAMS_NAME

# network fitting


def net_fit(
    zeta,
    train_real,
    val_real,
    *,
    seed,
    batch_size,
    nsteps_max,
    nsteps_round,
    early_stopping_rounds,
    learning_rate,
    tag,
):
    """Return fitted net parameters and a jsonnable metadata pytree."""
    net = make_net(zeta)

    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    params = net.net.init(key, train_real[:1])

    # preprocessing
    pre_loc, pre_scale = fit_prescale(train_real)

    # training
    rng, key = jax.random.split(rng)

    opt = optax.adam(learning_rate)

    opt_state = opt.init(params)

    @jax.jit
    def train_step(rng, params, opt_state, batch_real):
        batch_real_scaled = prescale(batch_real, pre_loc, pre_scale)

        grad_loss = jax.grad(net.loss_dropout, argnums=0)

        rng, key = jax.random.split(rng)
        grad = grad_loss(params, key, batch_real_scaled)

        updates, opt_state = opt.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        return rng, params, opt_state

    @jax.jit
    def quality_with_std(params, real):
        real_scaled = prescale(real, pre_loc, pre_scale)
        return net.quality_with_std(params, real_scaled)

    def report(params, msg, real_pre):
        quality, std = quality_with_std(params, real_pre)
        print("%6r %.1f +- %.1f ppm" % (msg, 1e6 * quality, 1e6 * std))
        return quality, std

    batches = itertools.cycle(batch_split(train_real, batch_size))
    yell = range(nsteps_round, nsteps_max + 1, nsteps_round)

    # train loop
    best, best_std = report(params, 0, val_real)
    best_params = params
    down_count = 0

    for i in range(1, nsteps_max + 1):
        rng, params, opt_state = train_step(
            rng, params, opt_state, next(batches)
        )

        if i not in yell:
            continue

        quality, quality_std = report(params, i, val_real)
        if quality > best:
            down_count = 0
            best = quality
            best_std = quality_std
            best_params = params
            continue

        down_count += 1
        if down_count >= early_stopping_rounds:
            break

    assert best_params is not None

    net_meta = {
        "ntrain": len(train_real),
        "nval": len(val_real),
        "log_r_val": float(best) * len(val_real),
        "best_quality": {
            "mean": float(best),
            "sdev": float(best_std),
        },
        "zeta_func": zeta.__name__,
        "batch_size": batch_size,
        "nsteps_max": nsteps_max,
        "nsteps_round": nsteps_round,
        "early_stopping_rounds": early_stopping_rounds,
        "learning_rate": learning_rate,
        "seed": seed,
        "nsteps_performed": i,
        "prescale": {
            "loc": list(pre_loc.astype(float)),
            "scale": list(pre_scale.astype(float)),
        },
        "tag": tag,
    }
    return best_params, net_meta


def net_test(zeta, params, meta, test_real, tag):
    """Return statistics of testing a net on given data."""
    net = make_net(zeta)

    pre_loc, pre_scale = meta["prescale"].values()
    pre_loc = jax.numpy.array(pre_loc, dtype=numpy.float32)
    pre_scale = jax.numpy.array(pre_scale, dtype=numpy.float32)

    @jax.jit
    def quality_with_std(params, real):
        real_scaled = prescale(real, pre_loc, pre_scale)
        return net.quality_with_std(params, real_scaled)

    quality, quality_std = quality_with_std(params, test_real)

    ntest = len(test_real)
    log_r_test = float(quality) * ntest

    return {
        "ntest": ntest,
        "log_r_test": log_r_test,
        "quality": {"mean": float(quality), "sdev": float(quality_std)},
        "tag": tag,
    }


def fit_dump(params, meta, dirname, *, verbose=True):
    """Serialize model and metadata pytree into directory dirname."""
    os.makedirs(dirname, exist_ok=True)
    params_name = os.path.join(dirname, PARAMS_NAME)
    meta_name = os.path.join(dirname, META_NAME)

    joblib.dump(params, gzip.open(params_name, "w"))
    json.dump(meta, open(meta_name, "w"), indent=4)
    if verbose:
        print("wrote %r, %r" % (params_name, meta_name))


def fit_load(dirname):
    """Return model and metadata loaded from directory dirname."""
    params_name = os.path.join(dirname, PARAMS_NAME)
    meta_name = os.path.join(dirname, META_NAME)

    params = joblib.load(gzip.open(params_name))
    params = jax.device_put(params)
    meta = json.load(open(meta_name))
    return params, meta


# zeta functions


def zeta_100_100_d_10(drop_rate, x):
    x = haiku.Linear(100)(x)
    x = jax.nn.relu(x)
    x = haiku.Linear(100)(x)
    x = jax.nn.relu(x)
    x = dropout(drop_rate)(x)
    x = haiku.Linear(10)(x)
    x = jax.nn.relu(x)
    return haiku.Linear(1)(x)


def zeta_20_20_10(_, x):
    x = haiku.Linear(20)(x)
    x = jax.nn.relu(x)
    x = haiku.Linear(20)(x)
    x = jax.nn.relu(x)
    x = haiku.Linear(10)(x)
    x = jax.nn.relu(x)
    return haiku.Linear(1)(x)


# network implementation


def make_net(zeta, *, drop_rate=0.5):
    net_dropout = haiku.transform(partial(zeta, drop_rate))
    net = haiku.without_apply_rng(haiku.transform(partial(zeta, 0)))
    return Net(net_dropout, net)


@dataclass(frozen=True)
class Net:
    net_dropout: "haiku-like"
    net: "haiku-like without rng"

    def loss(self, params, x):
        return zeta_loss(self.net, params, None, x)

    def loss_dropout(self, params, rng, x):
        return zeta_loss(self.net_dropout, params, rng, x)

    def quality(self, params, x):
        return zeta_quality(self.net, params, None, x)

    def quality_dropout(self, params, rng, x):
        return zeta_quality(self.net_dropout, params, rng, x)

    def quality_with_std(self, params, x):
        return zeta_quality_with_std(self.net, params, None, x)

    def quality_with_std_dropout(self, params, rng, x):
        return zeta_quality_with_std(self.net_dropout, params, rng, x)


def zeta_loss(zeta, params, rng, x):
    return -zeta_quality(zeta, params, rng, x)


def zeta_quality_with_std(zeta, params, rng, x):
    qi = zeta_quality_i(zeta, params, rng, x)
    return qi.mean(), qi.std() / len(qi) ** 0.5


def zeta_quality(zeta, params, rng, x):
    return zeta_quality_i(zeta, params, rng, x).mean()


def zeta_quality_i(zeta, params, rng, x):
    if rng is None:
        delta = zeta.apply(params, x) - zeta.apply(params, parity_flip_jax(x))
    else:
        delta = zeta.apply(params, rng, x) - zeta.apply(
            params, rng, parity_flip_jax(x)
        )

    return jax.nn.log_sigmoid(delta) - numpy.log(0.5)


def dropout(drop_rate):
    if drop_rate == 0:
        return lambda x: x
    return lambda x: haiku.dropout(
        haiku.next_rng_key(), numpy.float16(drop_rate), x
    )


# utilities


def batch_split(source, batch_size):
    return (
        jax.device_put(source[i : i + batch_size])
        for i in range(0, len(source) - batch_size + 1, batch_size)
    )


def parity_flip_jax(inv_momenta):
    """Return an equivalent to invariant_momenta(-momenta) -- jax version."""
    # the effect of parity is to change the sign of y components
    # x z x y z ...
    signs = numpy.ones(inv_momenta.shape[-1], numpy.float32)
    nmomenta = 3 * N_PARTICLES - 1
    signs[range(3, nmomenta, 3)] *= -1
    return inv_momenta * signs


def fit_prescale(real):
    # momentum parts ignore zeros (missing particle => zero)
    nmomenta = 3 * N_PARTICLES - 1
    slice_p = (slice(None), slice(0, nmomenta))
    mask = real[slice_p] != 0
    mean_p = real[slice_p].mean(axis=0, where=mask)
    std_p = real[slice_p].std(axis=0, where=mask)

    # only y flips under parity - must be zero by symmety
    mean_p[3: nmomenta: 3] = 0

    # other parts
    mean_o = real[:, nmomenta:].mean(axis=0)
    std_o = real[:, nmomenta:].std(axis=0)

    # stitch together and convert
    mean = numpy.append(mean_p, mean_o)
    std = numpy.append(std_p, std_o)

    loc = mean.astype(numpy.float32)
    scale = (1 / (std + 1e-5)).astype(numpy.float32)

    return loc, scale


def prescale(x, loc, scale):
    return (x - loc) * scale


def make_phi_func(params, meta):
    net = make_net(zeta_100_100_d_10)

    pre_loc, pre_scale = meta["prescale"].values()
    pre_loc = jax.numpy.array(pre_loc, dtype=numpy.float32)
    pre_scale = jax.numpy.array(pre_scale, dtype=numpy.float32)

    @jax.jit
    def net_phi(params, x):
        x = prescale(x, pre_loc, pre_scale)
        zeta = net.net
        phi_1 = zeta.apply(params, x) - zeta.apply(params, parity_flip_jax(x))
        return phi_1.ravel()

    return net_phi
