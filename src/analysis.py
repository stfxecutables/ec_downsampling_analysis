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
import seaborn as sbn
from joblib import Memory
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.random import Generator
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from scipy.stats import linregress, pearsonr
from seaborn import FacetGrid
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm
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

MEMOIZER = Memory(location=ROOT / "__JOBLIB_CACHE__", verbose=0)


@dataclass
class LinResult:
    def __init__(self, x: ArrayLike, y: ArrayLike) -> None:
        result = linregress(x=x, y=y)
        self.m: float = result.slope  # type: ignore
        self.p: float = result.pvalue  # type: ignore
        self.r: float = result.rvalue  # type: ignore
        self.b: float = result.intercept  # type: ignore
        self.stderr: float = result.stderr  # type: ignore
        self.x_min = np.min(x)
        self.x_max = np.max(x)

    def line(self) -> Tuple[ndarray, ndarray]:
        x = np.linspace(self.x_min, self.x_max, 1000)
        return x, self.b + self.m * x


@dataclass
class Run:
    df: DataFrame
    down: float


@MEMOIZER.cache
def full_df(runs: Dict[int, Run]) -> DataFrame:
    y_true = runs[0].df["y_true"]
    preds = []
    for i, run in runs.items():
        pred = run.df["y_pred"]
        pred.name = f"pred{i}"
        preds.append(pred)
    return pd.concat([*preds, y_true], axis=1)


@dataclass
class Repeat:
    id: int
    runs: Dict[int, Run]
    down: float

    def full_df(self) -> DataFrame:
        return full_df(self.runs)

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


@MEMOIZER.cache
def load_preds(
    dataset: Dataset, kind: ClassifierKind, downsample: bool = True
) -> List[Repeat]:
    outdir = pred_root(dataset=dataset, kind=kind, downsample=True)
    rep_dirs = sorted(outdir.glob("rep*"))
    reps: List[Repeat] = []
    for rep_dir in tqdm(rep_dirs, desc="Loading predictions"):
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
        reps.append(Repeat(id=rep_id, runs=runs, down=runs[0].down))
    return reps


@MEMOIZER.cache
def get_rep_dfs(reps: List[Repeat]) -> Tuple[DataFrame, DataFrame]:
    df_pairs, df_runs = [], []
    for rep in tqdm(reps, desc="Summarizing predictions"):
        pairwise, runwise = rep.stats()
        df_pair = pairwise.copy()
        df_pair["rep"] = rep.id
        df_pair["down"] = rep.down
        df_run = runwise.copy()
        df_run["rep"] = rep.id
        df_run["down"] = rep.down
        df_pairs.append(df_pair)
        df_runs.append(df_run)

    df_pair = pd.concat(df_pairs, axis=0, ignore_index=True)
    df_run = pd.concat(df_runs, axis=0, ignore_index=True)
    return df_pair, df_run


def summarize_results(
    dataset: Dataset, kind: ClassifierKind, downsample: bool = True
) -> Tuple[DataFrame, DataFrame]:
    LW = 1.0
    reps = load_preds(dataset=dataset, kind=kind, downsample=downsample)
    df_pair, df_run = get_rep_dfs(reps)
    df_pair["down"] *= 100
    df_run["down"] *= 100
    renames = {"acc": "Accuracy", "ec": "EC", "down": "Downsample (%)"}
    df_pair.rename(columns=renames, inplace=True)
    df_run.rename(columns=renames, inplace=True)

    means = df_pair.groupby("rep").mean()  # columns = ["acc", "ec", "down"]
    acc_ec = LinResult(x=means["Accuracy"], y=means["EC"])
    down_acc = LinResult(x=means["Downsample (%)"], y=means["Accuracy"])
    down_ec = LinResult(x=means["Downsample (%)"], y=means["EC"])

    grid: FacetGrid
    ax: Axes
    args = dict(height=3.5, aspect=1.5, color="black")

    grid = sbn.relplot(data=means, x="Accuracy", y="EC", size="Downsample (%)", **args)
    grid.fig.text(x=0.7, y=0.85, s=f"r={acc_ec.r:0.2f}, p={acc_ec.p:0.2e}")
    ax = grid.fig.axes[0]
    ax.plot(*acc_ec.line(), color="red", lw=LW)
    ax.set_title("Mean Pairwise Acc Means vs. Run Mean EC")
    ax.set_xlabel("Run Mean Pairwise Mean Accuracy")
    ax.set_ylabel("Run Mean EC")
    grid.fig.tight_layout()
    grid.fig.subplots_adjust(right=0.85)

    # No effect of sample size on EC
    grid = sbn.relplot(data=means, x="Downsample (%)", y="EC", size="Accuracy", **args)
    ax = grid.fig.axes[0]
    ax.plot(*down_ec.line(), color="red", lw=LW)
    ax.set_title("Mean Pairwise Accuracy Means vs. Run Mean EC")
    ax.set_xlabel("Downsampling proportion")
    ax.set_ylabel("Run Mean EC")
    grid.fig.text(x=0.7, y=0.85, s=f"r={down_ec.r:0.2f}, p={down_ec.p:0.2e}")
    grid.fig.tight_layout()
    grid.fig.subplots_adjust(right=0.85)

    # effect of sample size on acc
    grid = sbn.relplot(data=means, x="Downsample (%)", y="Accuracy", size="EC", **args)
    ax = grid.fig.axes[0]
    ax.set_title("Effect of Downsampling on Accuracy")
    ax.plot(*down_acc.line(), color="red", lw=LW)
    ax.set_title("Mean Pairwise Accuracy Means vs. Run Mean EC")
    ax.set_xlabel("Downsampling proportion")
    ax.set_ylabel("Run Mean Pairwise Mean Accuracy")
    grid.fig.text(x=0.7, y=0.85, s=f"r={down_acc.r:0.2f}, p={down_acc.p:0.2e}")
    grid.fig.tight_layout()
    grid.fig.subplots_adjust(right=0.85)

    plt.show(block=False)

    mean_accs = df_run.groupby("rep").mean()["Accuracy"]
    downs = df_run.groupby("rep").mean()["Downsample (%)"]
    downs.name = "Downsample (%)"
    mean_ecs = df_pair.groupby("rep").mean()["EC"]
    macc_mec = LinResult(x=mean_accs, y=mean_ecs)
    grid = sbn.relplot(x=mean_accs, y=mean_ecs, size=downs, **args)
    ax: Axes = grid.fig.axes[0]
    ax.plot(*macc_mec.line(), color="red", lw=LW)
    ax.set_title("Mean Accuracy vs. Mean EC")
    ax.set_xlabel("Run Mean Accuracy")
    ax.set_ylabel("Run Mean Error consistency")
    grid.fig.text(x=0.7, y=0.85, s=f"r={macc_mec.r:0.2f}, p={macc_mec.p:0.2e}")
    grid.fig.tight_layout()
    grid.fig.subplots_adjust(right=0.85)
    plt.show(block=False)

    all_accs = df_pair["Accuracy"]
    all_ecs = df_pair["EC"]
    downs = df_pair["Downsample (%)"]
    downs.name = "Downsample (%)"
    aacc_aec = LinResult(x=all_accs, y=all_ecs)
    grid = sbn.relplot(x=all_accs, y=all_ecs, size=downs, **args)
    ax: Axes = grid.fig.axes[0]
    ax.plot(*aacc_aec.line(), color="red", lw=LW)
    ax.set_title("Paired Accuracy vs. EC")
    ax.set_xlabel("Paired Mean Accuracy")
    ax.set_ylabel("Error consistency")
    grid.fig.text(x=0.7, y=0.85, s=f"r={aacc_aec.r:0.2f}, p={aacc_aec.p:0.2e}")
    grid.fig.tight_layout()
    grid.fig.subplots_adjust(right=0.85)
    plt.show(block=False)

    plt.show(block=True)


if __name__ == "__main__":
    dataset = Dataset.Diabetes
    kind = ClassifierKind.GBT
    summarize_results(dataset=dataset, kind=kind, downsample=True)
