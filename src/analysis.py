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
from src.metrics import acc_pairs, accs, ecs
from src.prediction import pred_root
from src.utils import (
    get_classifier,
    get_rand_hparams,
    is_tuned,
    load_tuning_params,
    save_tuning_params,
    tuning_outdir,
)


@dataclass
class Run:
    df: DataFrame
    down: float


@dataclass
class Repeat:
    id: int
    runs: Dict[int, Run]
    down: float

    def full_df(self) -> DataFrame:
        y_true = self.runs[0].df["y_true"]
        preds = []
        for i, run in self.runs.items():
            pred = run.df["y_pred"]
            pred.name = f"pred{i}"
            preds.append(pred)
        return pd.concat([*preds, y_true], axis=1)

    def stats(self) -> Tuple[DataFrame, DataFrame]:
        """
        Returns
        -------
        pairwise: DataFrame
            DataFrame of length `k(k-1)/2`, where `k` is the number of runs.
            Columns are "acc" (pairwise mean accs) and "ec" (local ECs)

        runwise: DataFrame
            DataFrame of length `k`, where `k` is the number of runs.
            Columns are "acc" (each run accuracy).

        """
        df = self.full_df()
        y = df["y_true"].copy()
        preds = df.filter(regex="pred")
        acc_ps = acc_pairs(preds, y)
        acc = accs(preds, y)
        ec = ecs(preds, y)
        pairwise = DataFrame(
            {
                "acc": acc_ps,
                "ec": ec,
            }
        )
        runwise = DataFrame(data=acc, columns=["acc"])
        return pairwise, runwise


DfPlus = Dict[str, Union[DataFrame, float]]


def consolidate_folds(
    dfs: Dict[int, Dict[int, Dict[int, DfPlus]]]
) -> Dict[int, DataFrame]:
    consolidated = {}
    for rep, runs in dfs.items():
        for run, folds in runs.items():
            df = pd.concat(folds.values(), axis=0, ignore_index=True)
            df.sort_values(by="idx")
            consolidated[run] = df
    return consolidated


def load_preds(
    dataset: Dataset, kind: ClassifierKind, downsample: bool = True
) -> Dict[int, Repeat]:
    outdir = pred_root(dataset=dataset, kind=kind, downsample=True)
    rep_dirs = sorted(outdir.glob("rep*"))
    reps: Dict[int, Repeat] = {}
    for rep, rep_dir in enumerate(rep_dirs):
        run_dirs = sorted(rep_dir.glob("run*"))
        runs: Dict[int, Run] = {}
        for r, run_dir in enumerate(run_dirs):
            fold_pqs = sorted(run_dir.glob("*.parquet"))
            df = pd.concat(
                [pd.read_parquet(fold) for fold in fold_pqs], axis=0, ignore_index=True
            )
            down = float(fold_pqs[0].stem.split("_")[1]) if downsample else 1.0
            runs[r] = Run(df=df, down=down)
        rep_id = int(rep_dir.name.replace("rep", ""))
        reps[rep] = Repeat(id=rep_id, runs=runs, down=runs[0].down)
    return reps

    parquets = sorted(outdir.glob("*.parquet"))
    rep_ids = [int(p.stem.split("_")[0].replace("rep", "")) for p in parquets]
    fold_dfs: Dict[int, Dict[int, Dict[int, DfPlus]]]
    fold_dfs = {}
    reps: Dict[int, Repeat] = {rid: Repeat(id=rid, runs={}) for rid in rep_ids}
    for path in parquets:
        if downsample:
            rep_, run_, fold_, down = path.stem.split("_")
            ds = float(down)
        else:
            rep_, run_, fold_ = path.stem.split("_")
            ds = 100.0
        rep_id = int(rep_.replace("rep", ""))
        rep = reps[rep_id]

        run_id = int(run_.replace("run", ""))
        fold_id = int(fold_.replace("fold", ""))

        if rep not in fold_dfs:
            fold_dfs[rep] = {}
        if run not in fold_dfs[rep]:
            fold_dfs[rep][run] = {}
        fold_dfs[rep][run][fold] = {"df": pd.read_parquet(path), "down": ds}
    dfs = consolidate_folds(fold_dfs)
    return dfs


if __name__ == "__main__":
    dataset = Dataset.Diabetes
    kind = ClassifierKind.GBT
    reps = load_preds(dataset=dataset, kind=kind)
    print(reps[0])
