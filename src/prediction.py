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
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.random import Generator
from pandas import DataFrame, Series
from scipy.stats.qmc import Halton
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
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
    fold_rng: Generator
    shuffle_rng: Generator


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
        rng = args.fold_rng
        shuffle_rng = args.shuffle_rng

        out = pred_outfile(
            dataset=dataset, kind=kind, rep=rep, run=run, fold=fold, down=down
        )
        if out.exists():
            return

        cls = get_classifier(kind, dataset=dataset)
        hps = load_tuning_params(dataset=dataset, kind=kind)
        classifier = cls(hps)
        X, y = args.dataset.load()
        if down is not None:
            sub_splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=down, random_state=seed
            )
            sub_idx = next(sub_splitter.split(y, y))[0]
            # we no longer care about original data or indices
            X = X.iloc[sub_idx].reset_index()
            y = Series(
                y.iloc[sub_idx].reset_index().drop(columns="index").to_numpy().ravel()
            )
        idx_shuffle = shuffle_rng.permutation(len(y))
        X, y = X.iloc[idx_shuffle], y.iloc[idx_shuffle]
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        # skf = KFold(n_splits=5, shuffle=False)
        idx_train, idx_test = list(skf.split(y, y))[args.fold]

        X_tr, y_tr = X.iloc[idx_train], y.iloc[idx_train]
        X_test, y_test = X.iloc[idx_test], y.iloc[idx_test]
        classifier.fit(X_tr, y_tr, rng=rng)
        y_pred = classifier.predict(X_test)

        # Ultimately we will concat along axis 0 all `preds` dfs like below to
        # re-assemble the original test set. For this to work, we need the
        # "y_true" and "y_pred" columns unshuffled. Thankfully, the Series
        # index is actually useful for once and will make this work

        # idx_unshuffle = np.ravel(np.argsort(y_test.index))
        # y_pred = np.ravel(y_pred)[idx_unshuffle]
        # y_test = np.ravel(y_test)[idx_unshuffle]
        preds = DataFrame(
            {"y_pred": y_pred, "y_true": y_test},
            index=y_test.index,
        )
        preds.to_parquet(out)
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
    fasts = [
        Dataset.Diabetes,
        Dataset.Parkinsons,
        Dataset.SPECT,
        Dataset.Transfusion,
        Dataset.HeartFailure,
        Dataset.Diabetes130,
        Dataset.Diabetes130Reduced,
    ]
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
        Dataset.UTIResistanceReduced: {
            ClassifierKind.SVM: 40,
            ClassifierKind.LR: 40,
            ClassifierKind.GBT: 40,
            ClassifierKind.RF: 40,
        },
        Dataset.MimicIV: {
            ClassifierKind.SVM: 8,
            ClassifierKind.LR: 8,
            ClassifierKind.GBT: 20,
            ClassifierKind.RF: 20,
        },
        Dataset.MimicIVReduced: {
            ClassifierKind.SVM: 40,
            ClassifierKind.LR: 40,
            ClassifierKind.GBT: 40,
            ClassifierKind.RF: 40,
        },
    }[dataset][classifier]
    xgb_workers = n_cpu // tqdm_max_workers
    return xgb_workers, tqdm_max_workers


def evaluate_downsampling(
    classifier: ClassifierKind,
    dataset: Dataset,
    downsample: bool = True,
    n_reps: int = 100,
    n_runs: int = 10,
    max_workers: int = 1,
    only_rep: Optional[int] = None,
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
    xgb_jobs, tqdm_workers = get_xgb_tqdm_workers(classifier=classifier, dataset=dataset)
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
    tqdm_args = dict(chunksize=1, max_workers=tqdm_workers)

    for rep in range(n_reps):
        if only_rep is not None:
            if rep != only_rep:
                continue

        for run in range(n_runs):
            for fold in range(K):
                pargs.append(
                    ParallelArgs(
                        kind=classifier,
                        dataset=dataset,
                        fold=fold,
                        rep=rep,
                        run=run,
                        # We (perhaps) need each fold to get a different rng
                        # for fitting, so rngs[rep][run][fold], below.
                        fold_rng=rngs[rep][run][fold],
                        # Within a *run* (i.e. 5 folds) shuffling needs to be
                        # the same, (or we can't re-assemble the folds), so we
                        # need the same rng for each fold, hence
                        # rngs[rep][run][0].
                        shuffle_rng=rngs[rep][run][0],
                        # Within a *rep* (i.e. 10 runs) the same downsampling
                        # set must be maintained, or the predictions across
                        # runs cannot be compared for EC. So rngs[rep][0][0].
                        split=split_seeds[rep][0][0],
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
