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
from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.random import Generator
from pandas import DataFrame
from scipy.stats.qmc import Halton
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm.contrib.concurrent import process_map

from src.constants import DOWNSAMPLE_OUTDIR, PLAIN_OUTDIR, ensure_dir
from src.enumerables import ClassifierKind, Dataset
from src.utils import get_classifier, load_tuning_params


@dataclass
class ParallelArgs:
    kind: ClassifierKind
    dataset: Dataset
    fold: int
    run: int
    rep: int
    downsample: Optional[float]
    split: int  # split seed
    rng: Generator


def pred_root(
    dataset: Dataset,
    kind: ClassifierKind,
    downsample: bool,
) -> Path:
    root = PLAIN_OUTDIR if downsample is None else DOWNSAMPLE_OUTDIR
    out = root / f"{dataset.value}/{kind.value}"
    return ensure_dir(out)


def pred_outdir(
    dataset: Dataset,
    kind: ClassifierKind,
    downsample: bool,
    rep: int,
    run: int,
) -> Path:
    parent = pred_root(dataset=dataset, kind=kind, downsample=downsample)
    out = parent / f"rep{rep:03d}/run{run:03d}"
    return ensure_dir(out)


def pred_outfile(
    dataset: Dataset,
    kind: ClassifierKind,
    rep: int,
    run: int,
    fold: int,
    down: Optional[float],
) -> Path:
    outdir = pred_outdir(
        dataset=dataset, kind=kind, downsample=down is not None, rep=rep, run=run
    )
    d = "" if down is None else f"_{down:0.12f}"
    out = outdir / f"fold{fold}{d}.parquet"
    return out


def compute_preds(args: ParallelArgs) -> None:
    """Compute predictions and save predicted labels to file"""

    try:
        dataset = args.dataset
        kind = args.kind
        rep = args.rep
        run = args.run
        fold = args.fold
        down = args.downsample
        seed = args.split
        rng = args.rng

        out = pred_outfile(
            dataset=dataset, kind=kind, rep=rep, run=run, fold=fold, down=down
        )
        if out.exists():
            return

        cls = get_classifier(kind)
        hps = load_tuning_params(dataset=dataset, kind=kind)
        classifier = cls(hps)
        X, y = args.dataset.load()
        if down is not None:
            sub_splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=down, random_state=seed
            )
            sub_idx = next(sub_splitter.split(y, y))[0]
            X, y = X.iloc[sub_idx], y.iloc[sub_idx]
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        idx_train, idx_test = list(skf.split(y, y))[args.fold]
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
    n_reps: int = 100,
    n_runs: int = 10,
    max_workers: int = 1,
) -> None:
    """
    Parameters
    ----------
    n_reps: int
        Number of times to generate a percentage

    n_runs: int
        Number of times to run k-fold per repeat
    """
    K = 5
    # below just plucked from a random run of SeedSequence().entropy
    entropy = 285216691326606742260051019197268485321
    xgb_jobs = 4 if dataset is Dataset.MimicIV else 8
    ss = np.random.SeedSequence(entropy=entropy)
    seeds = ss.spawn(n_reps * n_runs * K)
    base_rng = np.random.default_rng(ss)
    p_seed = base_rng.integers(0, 2**32 - 1)
    percents = (
        np.array(Halton(d=1, seed=p_seed).random(n_reps)) * 0.40 + 0.50
    )  # [0.5, 0.9]
    percents = np.clip(percents, a_min=0.5, a_max=1.0)
    rngs = np.array([np.random.default_rng(seed) for seed in seeds])
    split_seeds = np.array([rng.integers(0, 2**32 - 1) for rng in rngs]).reshape(
        n_reps, n_runs, K
    )
    rngs = rngs.reshape(n_reps, n_runs, K)

    pargs = []
    tqdm_args = dict(chunksize=1)
    n_cpu = 80 if os.environ.get("CC_CLUSTER") == "niagara" else 8
    if classifier in [ClassifierKind.GBT, ClassifierKind.RF]:
        n_workers = n_cpu // xgb_jobs
        tqdm_args["max_workers"] = n_workers
    else:
        tqdm_args["max_workers"] = max_workers

    for rep in range(n_reps):
        for run in range(n_runs):
            for fold in range(K):
                pargs.append(
                    ParallelArgs(
                        kind=classifier,
                        dataset=dataset,
                        fold=fold,
                        rep=rep,
                        run=run,
                        rng=rngs[rep][run][fold],
                        split=split_seeds[rep][run][fold],
                        downsample=float(percents[rep]) if downsample else None,
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
        n_reps=50,
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
