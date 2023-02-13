import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple, Type, Union
from warnings import filterwarnings

import numpy as np
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from error_consistency.consistency import ErrorConsistencyKFoldHoldout, ErrorConsistencyKFoldInternal

from analysis.constants import CLASSIFIERS, DATA
from analysis.downsampling_setup import argparse_setup, classifier_from_args, dataset_from_args

CLASSIFIER_CHOICES = ["knn1", "knn3", "knn5", "knn10", "lr", "svm", "rf", "ada", "mlp"]
DATASET_CHOICES = ["diabetes", "park", "trans", "spect"]
SCRIPT_OUTDIR = Path(__file__).resolve().parent.parent

FlatArray = Union[DataFrame, Series, ndarray]


def overall_cohen_d(X: DataFrame, y: FlatArray, statistic: str = "mean") -> float:
    """For each feature `x` in `X`, compute the absolute [see notes] Cohen's d value and return a
    summary statistic of those values.

    Parameters
    ----------
    X: DataFrame
        DataFrame with shape (n_samples, n_features), and no target variable included

    y: DataFrame | Series | ndarray
        DataFrame, Series, or ndarray with shape (n_samples,) and only two unique values (0, 1).

    statistic: "mean" | "median"
        How to summarize the resulting rho values.

    Notes
    -----
    Because we are summarizing multiple Cohen's d values from different features, retaining the sign
    is invalid (e.g. each feature has its own unit of measure, and allowing "-1 f1" to cancel out `1
    f2" for units "f1" and "f2" is invalid.
    """
    data = np.array(np.copy(X))
    x1, x2 = data[y == 0], data[y == 1]
    n1, n2 = len(x1) - 1, len(x2) - 1
    sd1, sd2 = np.std(x1, ddof=1, axis=0), np.std(x2, ddof=1, axis=0)
    sd_pools = np.sqrt((n1 * sd1 + n2 * sd2) / (n1 + n2))
    ds = np.abs(np.mean(x1, axis=0) - np.mean(x2, axis=0)) / sd_pools
    summary = np.mean if statistic == "mean" else np.median
    return float(summary(np.array(ds).ravel()))


def overall_auroc(X: DataFrame, y: FlatArray, statistic: str = "mean") -> float:
    """For each feature `x` in `X`, compute rho, the common-language effect size (see Notes) via the
    area-under-the-ROC curve (AUC) using `x` as the `y_score`, rescale this effect size to allow
    aggregation across features, and return a summary statistic of those rescaled values.

    Parameters
    ----------
    X: DataFrame
        DataFrame with shape (n_samples, n_features), and no target variable included

    y: DataFrame | Series | ndarray
        DataFrame, Series, or ndarray with shape (n_samples,) and only two unique values (0, 1).

    statistic: "mean" | "median"
        How to summarize the resulting rho values.

    Notes
    -----
    This is equivalent to calculating U / abs(y1*y2), where `y1` and `y2` are the subgroup sizes,
    and U is the Mann-Whitney U, and is also sometimes referred to as Herrnstein's rho [1] or "f",
    the "common-language effect size" [2].

    Note we also *must* rescale due to the implicit different units of each feature. E.g. as rho
    is implicity "signed", with values above 0.5 indicating separation in one
    direction, and values below 0.5 indicating separation in the other, taking a mean or median can
    be deeply misleading if some features result in rho values close to 1, and others have rho
    values close to zero.

    [1] Herrnstein, R. J., Loveland, D. H., & Cable, C. (1976). Natural concepts in pigeons.
        Journal of Experimental Psychology: Animal Behavior Processes, 2, 285-302

    [2] McGraw, K.O.; Wong, J.J. (1992). "A common language effect size statistic". Psychological
        Bulletin. 111 (2): 361–365. doi:10.1037/0033-2909.111.2.361.
    """
    if statistic not in ["mean", "median"]:
        raise ValueError("Must use either 'mean' or 'median' for `statistic`.")
    X = np.copy(X)
    aucs = [roc_auc_score(y, X[:, i]) for i in range(X.shape[1])]
    aucs = [np.abs(auc - 0.5) for auc in aucs]
    summary = np.mean if statistic == "mean" else np.median
    return float(summary(np.array(aucs).ravel()))


def mahalanobi(X: DataFrame, y: FlatArray, center: str = "mean") -> float:
    """

    Parameters
    ----------
    X: DataFrame
        DataFrame with shape (n_samples, n_features), and no target variable included

    y: DataFrame
        DataFrame, Series, or ndarray with shape (n_samples,) and only two unique values (0, 1).

    Notes
    -----
    Finding a measure to convey the distance of two subgroups given multiple features is essentially
    a search for a cluster similarity / distance metric. If we replace subgroups with their mean
    vectors (or median vectors), then we can use a number of natural multi-dimensional distance
    measures. However, only a few of these take into account variance and/or overlap. Using
    Mahalanobis distance on the subgroup means/medians has this partial advantage (naturally
    rescales the data). Otherwise, data could be feature-wise centered and then standard distance
    metrics used.
    """
    if center not in ["mean", "median"]:
        raise ValueError("Must use either 'mean' or 'median' for `center`.")
    raise NotImplementedError()


def scaled_distance(X: DataFrame, y: FlatArray, center: str = "mean") -> float:
    """Compute the Euclidean distance between the centers of the two subgroups after rescaling via
    statistics computed over all samples (both groups).

    Parameters
    ----------
    X: DataFrame
        DataFrame with shape (n_samples, n_features), and no target variable included

    y: DataFrame
        DataFrame, Series, or ndarray with shape (n_samples,) and only two unique values (0, 1).

    center: "mean" | "median"
        Which statistic of central tendency to use in re-scaling and for calculating cluster centers.
    """
    if center not in ["mean", "median"]:
        raise ValueError("Must use either 'mean' or 'median' for `center`.")
    x = np.array(np.copy(X), dtype=float)
    summary = np.mean if center == "mean" else np.median
    m, sd = summary(x, axis=0).astype(float), np.std(X, axis=0, ddof=1).astype(float)
    x -= m
    x /= sd
    x1, x2 = x[y == 0], x[y == 1]
    return float(np.linalg.norm(summary(x2, axis=0) - summary(x1, axis=0)))


def separations(X: DataFrame, y: Union[DataFrame, Series, ndarray]) -> DataFrame:
    """Return the area under the ROC curve of a perfect classifier, Mahalanobis D, and average
    Cohen's d values from data `X` with **two** subgroups defined in `y`

    Parameters
    ----------
    X: DataFrame
        DataFrame with shape (n_samples, n_features), and no target variable included

    y: DataFrame
        DataFrame, Series, or ndarray with shape (n_samples,) and only two unique values (0, 1).

    Returns
    -------
    val1: Any
    """
    return DataFrame(
        {
            "d": [overall_cohen_d(X, y, "mean")],
            "d-med": [overall_cohen_d(X, y, "median")],
            "AUC": [overall_auroc(X, y, "mean")],
            "AUC-med": [overall_auroc(X, y, "median")],
            "delta": [scaled_distance(X, y, "mean")],
            "delta-med": [scaled_distance(X, y, "median")],
        }
    )


def select_features(X: ndarray, percent: float) -> Tuple[ndarray, ndarray]:
    """Return a copy of X with only `percent` of the total features included"""
    if percent >= 1.0:
        return np.copy(X), np.arange(X.shape[1], dtype=int)
    if percent <= 0:
        raise ValueError("`percent` must be a float in [0, 1].")
    n_features = X.shape[1]
    n_select = np.ceil(percent * n_features).astype(int).item()
    idx = np.random.choice(np.arange(n_features, dtype=int), n_select, replace=False).ravel()
    return np.copy(X[:, idx]), idx


def get_percent_acc_consistency(
    model: Type,
    model_args: Dict,
    X: ndarray,
    y: ndarray,
    percent: float,
    kfold_reps: int,
    X_test: ndarray = None,
    y_test: ndarray = None,
    cpus: int = 4,
) -> Tuple[float, float, float, ndarray]:
    """
    Parameters
    ----------
    percent: float
        Between 0 and 100
    """
    EMPTY_UNIONS = "drop"
    # EMPTY_UNIONS = 1
    X = np.array(np.copy(X))
    X_select, idx = select_features(X, percent / 100)
    if X_test is not None:
        if isinstance(X_test, DataFrame):
            X_test = X_test.to_numpy()
        X_test = np.copy(X_test[:, idx])
    if X_select.shape[1] < 1:
        raise RuntimeError("Failed to select features.")

    if (X_test is not None) and (y_test is not None):
        errcon = ErrorConsistencyKFoldHoldout(
            model=model,
            x=X_select,
            y=y,
            n_splits=5,
            model_args=model_args,
            stratify=True,
            empty_unions=EMPTY_UNIONS,
        )
        results = errcon.evaluate(
            X_test,
            y_test,
            repetitions=kfold_reps,
            save_test_accs=True,
            save_fold_accs=False,
            parallel_reps=cpus,
            loo_parallel=cpus,
            turbo=True,
            show_progress=False,
        )
        return (
            np.mean(results.test_accs),
            np.mean(results.consistencies),
            np.std(results.consistencies, ddof=1),
            X_select,
        )
    errcon_internal = ErrorConsistencyKFoldInternal(
        model=model,
        x=X_select,
        y=y,
        n_splits=5,
        model_args=model_args,
        stratify=True,
        empty_unions=EMPTY_UNIONS,
    )
    results = errcon_internal.evaluate(
        repetitions=kfold_reps,
        save_test_accs=True,
        save_fold_accs=False,
        parallel_reps=cpus,
        loo_parallel=cpus,
        turbo=True,
        show_progress=False,
    )
    return (
        np.mean(results.test_accs),
        np.mean(results.consistencies),
        np.std(results.consistencies, ddof=1),
        X_select,
    )


def holdout_downsampling(args: Namespace) -> None:
    disable = not args.pbar
    dataset = dataset_from_args(args)
    classifier = classifier_from_args(args)
    percents = np.sort(np.random.uniform(args.percent_min, args.percent_max, args.n_percents))
    kfold_reps = args.kfold_reps
    n_rows = len(percents)
    outdir = args.results_dir
    if not outdir.exists():
        os.makedirs(outdir)
    cpus = args.cpus

    x, y = DATA[dataset]
    print(f"Preparing {dataset} data...")
    if args.validation == "external":
        x, x_test, y, y_test = train_test_split(x, y, test_size=0.2)
    else:
        x_test = y_test = None
    model, model_args_dict = CLASSIFIERS[classifier]
    model_args = model_args_dict[dataset]
    print(f"Testing {classifier} classifier on {dataset} data...")
    seps = separations(x, y)
    data = np.full([n_rows, 4 + seps.shape[1]], -1, dtype=float)
    desc_percent = "Downsampling at {:.1f}%"
    pbar_percent = tqdm(desc=desc_percent.format(percents[0]), total=len(percents), leave=False, disable=disable)
    row = 0
    for i, percent in enumerate(percents):
        pbar_percent.set_description(desc_percent.format(percent))
        acc, cons, sd, x_sel = get_percent_acc_consistency(
            model, model_args, x, y, percent, kfold_reps, x_test, y_test, cpus
        )
        seps = separations(x_sel, y)
        data[row] = [percent, acc, cons, sd, *(seps.to_numpy().ravel().tolist())]
        row += 1
        pbar_percent.update()
    pbar_percent.close()
    print("rows:", row)
    assert row == n_rows
    df = DataFrame(
        data=data,
        columns=["Percent", "Accuracy", "Consistency", "EC (sd)", *seps.columns],
        index=range(n_rows),
        dtype=float,
    )
    print(df)
    classifier = classifier.replace(" ", "_")
    val = "holdout" if args.validation == "external" else "internal"
    outfile = outdir / f"{dataset}_{classifier}__k-fold-{val}_feat.json"
    df.to_json(outfile)


if __name__ == "__main__":
    filterwarnings("error")
    filterwarnings("ignore", message="`np.int` is a deprecated alias", category=DeprecationWarning)
    parser = argparse_setup("feature")
    if os.environ.get("CC_CLUSTER") is not None:
        args = parser.parse_args()
        holdout_downsampling(args)
    else:
        cmd_args = (
            "--classifier lr "
            "--dataset diabetes "
            "--kfold-reps 10 "
            "--n-percents 100 "
            "--percent-min 10 "
            "--results-dir analysis/results/test_results "
            "--pbar "
            "--cpus 8 "
            "--validation external"
        )
        args = parser.parse_args(cmd_args.split(" "))
        holdout_downsampling(args)
