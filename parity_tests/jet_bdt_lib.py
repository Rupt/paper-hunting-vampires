"""Tools for analysing jet-momenta data with xgboost."""
import gzip
import json
import os

import joblib
import numpy
import xgboost
from .jet_lib import META_NAME, MODEL_NAME, parity_flip

from sksym import sksym


def bdt_fit(seed, model_kwargs, train_real, val_real, *, nstep=50):
    """Return a bdt model fitted to given data."""
    which = sksym.WhichIsReal()

    model = xgboost.XGBRegressor(
        objective=which.objective(),
        random_state=seed,
        **model_kwargs,
    )

    train_pack = which.stack(train_real, [parity_flip(train_real)])
    sksym.fit(model, train_pack)

    # iteration search on validation
    val_pack = which.stack(val_real, [parity_flip(val_real)])

    iteration_to_quality = {}
    for i in range(nstep, model.n_estimators + 1, nstep):
        quality, quality_std = sksym.score(
            model,
            val_pack,
            and_std=True,
            iteration_range=(0, i),
        )

        # returned values can be numpy things; cast for compatibility
        iteration_to_quality[str(i)] = {
            "mean": float(quality),
            "sdev": float(quality_std),
        }

    return model, iteration_to_quality


def bdt_meta(model, iteration_to_quality, ntrain, nval, *, tag=""):
    """Return a pytree of BDT model info."""
    qualities = [q["mean"] for q in iteration_to_quality.values()]
    iterations = list(iteration_to_quality.keys())
    best_iteration = int(iterations[numpy.argmax(qualities)])

    best_quality, _ = iteration_to_quality[str(best_iteration)].values()
    log_r_val = best_quality * nval

    return {
        "ntrain": ntrain,
        "nval": nval,
        "log_r_val": log_r_val,
        "best_iteration": best_iteration,
        "best_quality": iteration_to_quality[str(best_iteration)],
        "iteration_to_quality": iteration_to_quality,
        "tag": tag,
    }


def bdt_test(model, meta, test_real, *, tag=""):
    """Return test results for this model on these test data."""
    which = sksym.WhichIsReal()
    test_pack = which.stack(test_real, [parity_flip(test_real)])

    quality, quality_std = sksym.score(
        model,
        test_pack,
        and_std=True,
        iteration_range=(0, meta["best_iteration"]),
    )

    ntest = len(test_real)
    log_r_test = quality * ntest

    return {
        "ntest": ntest,
        "log_r_test": log_r_test,
        "quality": {"mean": float(quality), "sdev": float(quality_std)},
        "tag": tag,
    }


def fit_dump(model, meta, dirname, *, verbose=True):
    """Serialize model and metadata pytree into directory dirname."""
    os.makedirs(dirname, exist_ok=True)
    model_name = os.path.join(dirname, MODEL_NAME)
    meta_name = os.path.join(dirname, META_NAME)

    joblib.dump(model, gzip.open(model_name, "w"))
    json.dump(meta, open(meta_name, "w"), indent=4)
    if verbose:
        print("wrote %r, %r" % (model_name, meta_name))


def fit_load(dirname):
    """Return model and metadata loaded from directory dirname."""
    model_name = os.path.join(dirname, MODEL_NAME)
    meta_name = os.path.join(dirname, META_NAME)

    model = joblib.load(gzip.open(model_name))
    meta = json.load(open(meta_name))
    return model, meta
