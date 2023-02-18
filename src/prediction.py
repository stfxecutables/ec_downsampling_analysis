from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import os
import sys
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame, Series
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.constants import DOWNSAMPLE_OUTDIR, PLAIN_OUTDIR, ensure_dir
from src.enumerables import ClassifierKind, Dataset, Metric
from src.hparams.gbt import XGBoostHparams
from src.hparams.hparams import Hparams
from src.hparams.logistic import SGDLRHparams
from src.hparams.nystroem import NystroemHparams
from src.hparams.rf import XGBRFHparams
from src.utils import (
    get_classifier,
    get_rand_hparams,
    is_tuned,
    load_tuning_params,
    save_tuning_params,
    tuning_outdir,
)


@dataclass
class ParallelArgs:
    kind: ClassifierKind
    dataset: Dataset
    fold: int
    run: int
    downsample: bool
    rng: Generator


def pred_outfile(
    dataset: Dataset,
    kind: ClassifierKind,
    run: int,
    fold: int,
    downsample: Optional[float],
) -> Path:
    root = PLAIN_OUTDIR if downsample is None else DOWNSAMPLE_OUTDIR
    down = "" if downsample is None else f"__{downsample:0.12f}"
    out = root / f"{dataset}/{kind}/run{run:03d}_fold{fold}{down}.parquet"
    ensure_dir(out.parent)
    return out


def compute_preds(args: ParallelArgs) -> None:
    """Compute predictions and save predicted labels to file"""

    try:
        dataset = args.dataset
        kind = args.kind
        run = args.run
        fold = args.fold
        downsample = args.downsample
        rng = args.rng
        down = rng.uniform(0.5, 0.99) if downsample else None

        out = pred_outfile(
            dataset=dataset, kind=kind, run=run, fold=fold, downsample=down
        )
        if out.exists():
            return

        cls = get_classifier(kind)
        hps = load_tuning_params(dataset=dataset, kind=kind)
        classifier = cls(hps)
        X, y = args.dataset.load()
        y_sub = y
        if down is not None:
            sub_splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=down, random_state=rng.integers(0, 2**32 - 1)
            )
            sub_idx = next(sub_splitter.split(y, y))[0]
            X, y_sub = X.iloc[sub_idx], y.iloc[sub_idx]
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        idx_train, idx_test = list(skf.split(y_sub, y_sub))[args.fold]
        X_tr, y_tr = X.iloc[idx_train], y.iloc[idx_train]
        X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]
        classifier.fit(X_tr, y_tr, rng=rng)
        y_pred = classifier.predict(X_test)
        preds = DataFrame({"idx": idx_test, "y_pred": y_pred, "y_true": y_test})
        preds.to_parquet(out)
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")
        return None


def evaluate_downsampling(
    classifier: ClassifierKind,
    dataset: Dataset,
    downsample: bool = True,
    n_runs: int = 200,
    max_workers: int = 1,
) -> None:
    K = 5
    xgb_jobs = 4 if dataset is Dataset.MimicIV else 8
    run_seeds = np.random.SeedSequence().spawn(n_runs)
    fold_rngs: List[List[Generator]] = []
    for seed in run_seeds:
        fold_rngs.append([np.random.default_rng(seed) for seed in seed.spawn(K)])

    pargs = []
    tqdm_args = dict(chunksize=1)
    for run in range(n_runs):
        n_cpu = 80 if os.environ.get("CC_CLUSTER") == "niagara" else 8
        if classifier in [ClassifierKind.GBT, ClassifierKind.RF]:
            n_workers = n_cpu // xgb_jobs
            tqdm_args["max_workers"] = n_workers
        else:
            tqdm_args["max_workers"] = max_workers

        for fold in range(K):
            pargs.append(
                ParallelArgs(
                    kind=classifier,
                    dataset=dataset,
                    fold=fold,
                    run=run,
                    rng=fold_rngs[run][fold],
                    downsample=downsample,
                )
            )

    process_map(
        compute_preds,
        pargs,
        desc=f"Predictions for {classifier.name} on {dataset.name}",
        **tqdm_args,
    )


if __name__ == "__main__":
    evaluate_downsampling(
        classifier=ClassifierKind.GBT,
        dataset=Dataset.Diabetes,
        n_runs=10,
        max_workers=8,
        downsample=True,
    )
    sys.exit()
    MAX_WORKERS = {
        Dataset.UTIResistance: {
            ClassifierKind.GBT: 30,
            ClassifierKind.SVM: 8,
            ClassifierKind.RF: 30,
            ClassifierKind.LR: 8,
        }
    }
    for dataset in Dataset:
        if dataset not in [Dataset.MimicIV, Dataset.UTIResistance]:
            continue
        for kind in ClassifierKind:
            # random_tune(classifier=kind, dataset=dataset, n_runs=250, force=False)
            evaluate_downsampling(
                classifier=kind,
                dataset=dataset,
                n_runs=250,
                max_workers=20
                if dataset is Dataset.MimicIV
                else MAX_WORKERS[dataset][kind],
            )
