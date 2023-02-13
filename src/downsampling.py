import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple, Type
from warnings import filterwarnings

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from error_consistency.consistency import ErrorConsistencyKFoldHoldout, ErrorConsistencyKFoldInternal

from analysis.constants import (
    CLASSIFIERS,
    COLS,
    DATA,
    DOWNSAMPLE_RESULTS_DIR,
    KFOLD_REPS,
    N_ROWS,
    PERCENTS,
    REPS_PER_PERCENT,
)
from analysis.downsampling_setup import argparse_setup, classifier_from_args, dataset_from_args

CLASSIFIER_CHOICES = ["knn1", "knn3", "knn5", "knn10", "lr", "svm", "rf", "ada", "mlp"]
DATASET_CHOICES = ["diabetes", "park", "trans", "spect"]
SCRIPT_OUTDIR = Path(__file__).resolve().parent.parent


def get_percent_acc_consistency(
    model: Type,
    model_args: Dict,
    x: ndarray,
    y: ndarray,
    percent: float,
    kfold_reps: int,
    x_test: ndarray = None,
    y_test: ndarray = None,
    cpus: int = 4,
) -> Tuple[float, float]:
    x_down, y_down = None, None
    if percent >= 100:
        x_down, y_down = x, y
    else:
        for _ in range(100):
            try:
                x_down, _, y_down, _ = train_test_split(x, y, stratify=y, train_size=percent / 100)
                break
            except Exception:
                pass
    if x_down is None:
        raise RuntimeError("Failed to downsample")
    if (x_test is not None) and (y_test is not None):
        errcon = ErrorConsistencyKFoldHoldout(
            model=model,
            x=x_down,
            y=y_down,
            n_splits=5,
            model_args=model_args,
            stratify=True,
            empty_unions="drop",
        )
        results = errcon.evaluate(
            x_test,
            y_test,
            repetitions=kfold_reps,
            save_test_accs=True,
            save_fold_accs=False,
            parallel_reps=cpus,
            loo_parallel=cpus,
            turbo=True,
            show_progress=False,
        )
        return np.mean(results.test_accs), np.mean(results.consistencies)
    errcon_internal = ErrorConsistencyKFoldInternal(
        model=model,
        x=x_down,
        y=y_down,
        n_splits=5,
        model_args=model_args,
        stratify=True,
        empty_unions="drop",
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
    return np.mean(results.test_accs), np.mean(results.consistencies)


# NOTE: for parkinsons and SPECT data, are going to get
# ValueError: n_splits=5 cannot be greater than the number of members in each class.
# for the small downsampling percents
def holdout_downsampling_all(show_progress: bool = False) -> None:
    disable = not show_progress
    for dataset_name, (x, y) in DATA.items():
        print(f"Preparing {dataset_name} data...")
        x, x_test, y, y_test = train_test_split(x, y, test_size=0.2)
        for classifier_name, (model, args) in CLASSIFIERS.items():
            print(f"Testing {classifier_name} classifier on {dataset_name} data...")
            data = np.full([N_ROWS, 3], -1, dtype=float)
            desc_percent = "Downsampling at {}%"
            pbar_percent = tqdm(desc=desc_percent.format(COLS[0]), total=len(COLS), leave=False, disable=disable)
            row = 0
            for percent, col in zip(PERCENTS, COLS):
                pbar_percent.set_description(desc_percent.format(col))
                desc_reps = "Repetition {:d}"
                pbar_reps = tqdm(desc=desc_reps.format(0), total=REPS_PER_PERCENT, leave=False, disable=disable)
                for r in range(REPS_PER_PERCENT):
                    pbar_reps.set_description(desc_reps.format(r))
                    acc, cons = get_percent_acc_consistency(model, args, x, y, percent, KFOLD_REPS, x_test, y_test)
                    data[row] = [percent, acc, cons]
                    row += 1
                    pbar_reps.update()
                pbar_reps.close()
                pbar_percent.update()
            pbar_percent.close()
            assert row == N_ROWS
            df = DataFrame(
                data=data,
                columns=["Percent", "Accuracy", "Consistency"],
                index=range(N_ROWS),
                dtype=float,
            )
            print(df)
            outfile = DOWNSAMPLE_RESULTS_DIR / f"{dataset_name}_{classifier_name}__k-fold-holdout_downsample.json"
            df.to_json(outfile)


def holdout_downsampling(
    args: Namespace,
) -> None:
    disable = not args.pbar
    dataset = dataset_from_args(args)
    classifier = classifier_from_args(args)
    percents = np.sort(np.random.uniform(args.percent_min, args.percent_max, args.n_percents))
    # d = float(args.percent_spacing)
    # percents = np.arange(0, 100 + d, d)[1:]
    # percents = percents[percents >= 5]
    # cols = [f"{e:1.0f}" for e in percents]
    # reps_per_percent = args.percent_reps
    # n_rows = len(percents) * reps_per_percent
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
    data = np.full([n_rows, 3], -1, dtype=float)
    desc_percent = "Downsampling at {:.1f}%"
    pbar_percent = tqdm(desc=desc_percent.format(percents[0]), total=len(percents), leave=False, disable=disable)
    row = 0
    for i, percent in enumerate(percents):
        pbar_percent.set_description(desc_percent.format(percent))
        acc, cons = get_percent_acc_consistency(model, model_args, x, y, percent, kfold_reps, x_test, y_test, cpus)
        data[row] = [percent, acc, cons]
        row += 1
        pbar_percent.update()
    pbar_percent.close()
    print("row:", row)
    assert row == n_rows
    df = DataFrame(data=data, columns=["Percent", "Accuracy", "Consistency"], index=range(n_rows), dtype=float)
    print(df)
    classifier = classifier.replace(" ", "_")
    val = "holdout" if args.validation == "external" else "internal"
    outfile = outdir / f"{dataset}_{classifier}__k-fold-{val}.json"
    df.to_json(outfile)


if __name__ == "__main__":
    parser = argparse_setup()
    # args = parser.parse_args()
    args = parser.parse_args(
        "--classifier lr --dataset diabetes --kfold-reps 10 --n-percents 50 --results-dir analysis/results/test_results --pbar --cpus 8 --validation internal".split(
            " "
        )
    )
    filterwarnings("ignore", message="Got `batch_size`", category=UserWarning)
    filterwarnings("ignore", message="Stochastic Optimizer")
    filterwarnings("ignore", message="Liblinear failed to converge")
    holdout_downsampling(args)
