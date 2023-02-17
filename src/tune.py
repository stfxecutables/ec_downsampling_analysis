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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame, Series
from sklearn.model_selection import StratifiedKFold
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.constants import RESULTS
from src.enumerables import ClassifierKind, Dataset, Metric
from src.hparams.hparams import Hparams
from src.utils import get_classifier, get_rand_hparams, is_tuned, save_tuning_params


@dataclass
class ParallelArgs:
    kind: ClassifierKind
    dataset: Dataset
    metric: Metric
    hparams: Hparams
    fold: int
    run: int
    rng: Generator


@dataclass
class ParallelResult:
    hparams: Hparams
    fold: int
    run: int
    score: float

    def to_df(self) -> DataFrame:
        return DataFrame(
            {"run": self.run, "fold": self.fold, "acc": self.score}, index=[0]
        )


def evaluate(args: ParallelArgs) -> Optional[ParallelResult]:
    try:
        cls = get_classifier(args.kind)
        classifier = cls(args.hparams)
        rng = args.rng
        X, y = args.dataset.load()
        # X, y = args.dataset.load()
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        idx_train, idx_test = list(skf.split(y, y))[args.fold]
        X_tr, y_tr = X.iloc[idx_train], y.iloc[idx_train]
        X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]
        classifier.fit(X_tr, y_tr, rng=rng)
        score = classifier.score(X_test, y_test, metric=args.metric)
        return ParallelResult(
            hparams=args.hparams,
            fold=args.fold,
            run=args.run,
            score=score,
        )
    except Exception as e:
        traceback.print_exc()
        print(f"Got error: {e}")
        return None


def random_tune(
    classifier: ClassifierKind,
    dataset: Dataset,
    metric: Metric = Metric.Accuracy,
    n_runs: int = 100,
    force: bool = False,
) -> None:
    if not force and is_tuned(dataset=dataset, kind=classifier):
        return
    K = 5
    xgb_jobs = 4 if dataset is Dataset.MimicIV else 8
    run_seeds = np.random.SeedSequence().spawn(n_runs)
    run_rngs = [np.random.default_rng(seed) for seed in run_seeds]
    fold_rngs: List[List[Generator]] = []
    for seed in run_seeds:
        fold_rngs.append([np.random.default_rng(seed) for seed in seed.spawn(K)])

    pargs = []
    tqdm_args = dict(chunksize=1)
    for run in range(n_runs):
        hps = get_rand_hparams(kind=classifier, rng=run_rngs[run])  # type: ignore
        if classifier in [ClassifierKind.GBT, ClassifierKind.RF]:
            if os.environ.get("CC_CLUSTER") == "niagara":
                n_workers = 80 // xgb_jobs
                hps.set_n_jobs(xgb_jobs)
                tqdm_args["max_workers"] = n_workers
            else:
                n_workers = 8 // xgb_jobs
                hps.set_n_jobs(xgb_jobs)
                tqdm_args["max_workers"] = n_workers
        for fold in range(K):
            pargs.append(
                ParallelArgs(
                    kind=classifier,
                    dataset=dataset,
                    metric=metric,
                    hparams=hps,
                    fold=fold,
                    run=run,
                    rng=fold_rngs[run][fold],
                )
            )

    results = process_map(
        evaluate, pargs, desc=f"Fitting {classifier.name} on {dataset.name}", **tqdm_args
    )
    results: List[ParallelResult] = [result for result in results if result is not None]
    dfs = [result.to_df() for result in results]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    means = df.drop(columns="fold").groupby("run").mean()
    max_run_idx = means.idxmax().item()

    best = None
    for result in results:
        if result.run == max_run_idx:
            best = result.hparams
            break
    if best is None:
        raise RuntimeError("Did not find run in results matching `max_run_idx`")

    save_tuning_params(dataset=dataset, kind=classifier, hps=best)
    print("Acc folds dist:")
    print(df["acc"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    print("Acc means dist:")
    print(means["acc"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    print(f"Best acc: {means.iloc[max_run_idx]} with hparams:")
    print(best)


if __name__ == "__main__":
    # random_tune(
    #     classifier=ClassifierKind.LR, dataset=Dataset.Diabetes, n_runs=100, force=False
    # )
    for dataset in Dataset:
        if dataset not in [Dataset.MimicIV, Dataset.UTIResistance]:
            continue
        for kind in ClassifierKind:
            # random_tune(classifier=kind, dataset=dataset, n_runs=250, force=False)
            random_tune(
                classifier=kind,
                dataset=dataset,
                metric=Metric.F1,
                n_runs=250,
                force=False,
            )
