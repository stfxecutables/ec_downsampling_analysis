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
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
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
from sklearn.model_selection import StratifiedKFold
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.constants import CKPT_OUTDIR, RESULTS, ensure_dir
from src.enumerables import ClassifierKind, Dataset, Metric
from src.hparams.gbt import XGBoostHparams
from src.hparams.hparams import Hparams
from src.hparams.logistic import SGDLRHparams
from src.hparams.nystroem import NystroemHparams
from src.hparams.rf import XGBRFHparams
from src.hparams.svm import SVMHparams
from src.utils import (
    get_classifier,
    get_rand_hparams,
    is_tuned,
    save_tuning_params,
    tuning_outdir,
)


@dataclass
class ParallelArgs:
    kind: ClassifierKind
    dataset: Dataset
    metric: Metric
    hparams: Hparams
    fold: int
    run: int
    rng: Generator
    use_ckpt: bool = True


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


def get_ckpt_dir(dataset: Dataset, kind: ClassifierKind, run: int, fold: int) -> Path:
    return ensure_dir(CKPT_OUTDIR / f"{dataset.value}/{kind.value}/{run}/{fold}/ckpt")


def get_hp_dir(dataset: Dataset, kind: ClassifierKind, run: int, fold: int) -> Path:
    ckpt_dir = CKPT_OUTDIR / f"{dataset.value}/{kind.value}/{run}/{fold}/ckpt"
    return ensure_dir(ckpt_dir / "hps")


def is_computed(dataset: Dataset, kind: ClassifierKind, run: int, fold: int) -> bool:
    ckpt_dir = CKPT_OUTDIR / f"{dataset.value}/{kind.value}/{run}/{fold}/ckpt"
    jsons = list(ckpt_dir.rglob("*.json"))
    n_jsons = len(jsons)
    return n_jsons > 0 and "score.json" in [p.name for p in jsons]


def load_run_fold_params(
    dataset: Dataset, kind: ClassifierKind, run: int, fold: int
) -> Hparams:
    root = get_hp_dir(dataset=dataset, kind=kind, run=run, fold=fold)
    if dataset in [
        Dataset.SPECT,
        Dataset.Diabetes,
        Dataset.Parkinsons,
        Dataset.Transfusion,
    ]:
        SVMParams = SVMHparams
    else:
        SVMParams = NystroemHparams
    kinds: Dict[ClassifierKind, Type[Hparams]] = {
        ClassifierKind.GBT: XGBoostHparams,
        ClassifierKind.LR: SGDLRHparams,
        ClassifierKind.RF: XGBRFHparams,
        ClassifierKind.SVM: SVMParams,
    }
    hp = kinds[kind]
    return hp.from_json(root)


def evaluate(args: ParallelArgs) -> Optional[ParallelResult]:
    try:
        dataset = args.dataset
        kind = args.kind
        run = args.run
        fold = args.fold
        ckpt = get_ckpt_dir(dataset=dataset, kind=kind, run=run, fold=fold)
        hp_dir = get_hp_dir(dataset=dataset, kind=kind, run=run, fold=fold)
        scorefile = ckpt / "score.json"
        if args.use_ckpt and is_computed(dataset, kind, run, fold):
            hps = load_run_fold_params(dataset=dataset, kind=kind, run=run, fold=fold)
            score = float(pd.read_json(scorefile, typ="series").to_numpy().item())
            return ParallelResult(hparams=hps, fold=fold, run=run, score=score)

        cls = get_classifier(args.kind, dataset=dataset)
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
        s = Series([score], name=args.metric.value)
        s.to_json(scorefile)
        args.hparams.to_json(hp_dir)
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


def get_xgb_tqdm_workers(
    classifier: ClassifierKind,
    dataset: Dataset,
) -> Tuple[int, int]:
    """
    Returns
    -------
    xgb_n_jobs: int
    tqdm_max_workers: int
    """
    n_cpu = 80 if os.environ.get("CC_CLUSTER") == "niagara" else 8
    fasts = [Dataset.Diabetes, Dataset.Parkinsons, Dataset.SPECT, Dataset.Transfusion]
    if dataset in fasts:  # classifier irrelevant
        xgb_workers = 1
        tqdm_max_workers = n_cpu
        return xgb_workers, tqdm_max_workers

    tqdm_max_workers = {
        # Mid
        Dataset.HeartFailure: {
            ClassifierKind.SVM: 20,
            ClassifierKind.LR: 20,
            ClassifierKind.GBT: 40,
            ClassifierKind.RF: 40,
        },
        Dataset.Diabetes130: {
            ClassifierKind.SVM: 20,
            ClassifierKind.LR: 20,
            ClassifierKind.GBT: 40,
            ClassifierKind.RF: 40,
        },
        # Slow + Memory
        Dataset.UTIResistance: {
            ClassifierKind.SVM: 8,
            ClassifierKind.LR: 8,
            ClassifierKind.GBT: 30,
            ClassifierKind.RF: 30,
        },
        Dataset.MimicIV: {
            ClassifierKind.SVM: 8,
            ClassifierKind.LR: 8,
            ClassifierKind.GBT: 20,
            ClassifierKind.RF: 20,
        },
    }[dataset][classifier]
    xgb_workers = n_cpu // tqdm_max_workers
    return xgb_workers, tqdm_max_workers


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
    xgb_jobs, tqdm_workers = get_xgb_tqdm_workers(classifier=classifier, dataset=dataset)
    run_seeds = np.random.SeedSequence().spawn(n_runs)
    run_rngs = [np.random.default_rng(seed) for seed in run_seeds]
    fold_rngs: List[List[Generator]] = []
    for seed in run_seeds:
        fold_rngs.append([np.random.default_rng(seed) for seed in seed.spawn(K)])

    pargs = []
    tqdm_args = dict(chunksize=1)
    for run in range(n_runs):
        hps = get_rand_hparams(kind=classifier, rng=run_rngs[run], data=dataset)  # type: ignore
        hps.set_n_jobs(xgb_jobs)
        tqdm_args["max_workers"] = tqdm_workers

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
            random_tune(
                classifier=kind,
                dataset=dataset,
                metric=Metric.Accuracy,
                n_runs=250,
                tqdm_max_workers=20
                if dataset is Dataset.MimicIV
                else MAX_WORKERS[dataset][kind],
                force=False,
            )
