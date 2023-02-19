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

from src.constants import DOWNSAMPLE_OUTDIR, PLAIN_OUTDIR, PLOTS, TABLES, ensure_dir
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

OUT = ensure_dir(PLOTS / "downsampling")
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
    for rep_dir in tqdm(
        rep_dirs, desc=f"Loading predictions (data={dataset.name}, cls={kind.name})"
    ):
        run_dirs = sorted(rep_dir.glob("run*"))
        runs: Dict[int, Run] = {}
        for r, run_dir in enumerate(run_dirs):
            fold_pqs = sorted(run_dir.glob("*.parquet"))
            dfs = []
            for fold in fold_pqs:
                try:
                    df = pd.read_parquet(fold)
                    dfs.append(df)
                except TypeError as e:
                    raise RuntimeError(
                        f"Likely corrupt data at {fold}. Details above."
                    ) from e

            df = pd.concat(dfs, axis=0, ignore_index=True)
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
) -> None:
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
    fig: Figure
    args = dict(height=3.5, aspect=1.5, color="black")
    subtitle = f"\n{dataset.name} - {kind.name}"
    outdir = ensure_dir(OUT / f"{dataset.value}/{kind.value}")

    grid = sbn.relplot(data=means, x="Accuracy", y="EC", size="Downsample (%)", **args)
    fig, ax = grid.fig, grid.fig.axes[0]
    ax.plot(*acc_ec.line(), color="red", lw=LW)
    ax.set_title(f"Mean Pairwise Acc Means vs. Run Mean EC{subtitle}")
    ax.set_xlabel("Run Mean Pairwise Mean Accuracy")
    ax.set_ylabel("Run Mean EC")
    fig.text(x=0.7, y=0.85, s=f"r={acc_ec.r:0.2f}, p={acc_ec.p:0.2e}")
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.savefig(str(outdir / "mean_pairwise_accs_vs_mean_ec.png"), dpi=600)
    plt.close()

    # No effect of sample size on EC
    grid = sbn.relplot(data=means, x="Downsample (%)", y="EC", size="Accuracy", **args)
    fig, ax = grid.fig, grid.fig.axes[0]
    ax.plot(*down_ec.line(), color="red", lw=LW)
    ax.set_title(f"Mean Pairwise Accuracy Means vs. Run Mean EC{subtitle}")
    ax.set_xlabel("Downsampling proportion")
    ax.set_ylabel("Run Mean EC")
    fig.text(x=0.7, y=0.85, s=f"r={down_ec.r:0.2f}, p={down_ec.p:0.2e}")
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.savefig(str(outdir / "downsample_vs_mean_ec_by_mean_pairwise_accs.png"), dpi=600)
    plt.close()

    # effect of sample size on acc
    grid = sbn.relplot(data=means, x="Downsample (%)", y="Accuracy", size="EC", **args)
    fig, ax = grid.fig, grid.fig.axes[0]
    ax.set_title("Effect of Downsampling on Accuracy")
    ax.plot(*down_acc.line(), color="red", lw=LW)
    ax.set_title(f"Mean Pairwise Accuracy Means vs. Run Mean EC{subtitle}")
    ax.set_xlabel("Downsampling proportion")
    ax.set_ylabel("Run Mean Pairwise Mean Accuracy")
    fig.text(x=0.7, y=0.85, s=f"r={down_acc.r:0.2f}, p={down_acc.p:0.2e}")
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.savefig(str(outdir / "downsample_vs_mean_pairwise_accs_by_mean_ec.png"), dpi=600)
    plt.close()

    mean_accs = df_run.groupby("rep").mean()["Accuracy"]
    downs = df_run.groupby("rep").mean()["Downsample (%)"]
    downs.name = "Downsample (%)"
    mean_ecs = df_pair.groupby("rep").mean()["EC"]
    macc_mec = LinResult(x=mean_accs, y=mean_ecs)
    grid = sbn.relplot(x=mean_accs, y=mean_ecs, size=downs, **args)
    fig, ax = grid.fig, grid.fig.axes[0]
    ax.plot(*macc_mec.line(), color="red", lw=LW)
    ax.set_title(f"Mean Accuracy vs. Mean EC{subtitle}")
    ax.set_xlabel("Run Mean Accuracy")
    ax.set_ylabel("Run Mean Error consistency")
    fig.text(x=0.7, y=0.85, s=f"r={macc_mec.r:0.2f}, p={macc_mec.p:0.2e}")
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.savefig(str(outdir / "run_mean_acc_vs_run_mean_ec.png"), dpi=600)
    plt.close()

    all_accs = df_pair["Accuracy"]
    all_ecs = df_pair["EC"]
    downs = df_pair["Downsample (%)"]
    downs.name = "Downsample (%)"
    aacc_aec = LinResult(x=all_accs, y=all_ecs)
    grid = sbn.relplot(x=all_accs, y=all_ecs, size=downs, **args)
    fig, ax = grid.fig, grid.fig.axes[0]
    ax.plot(*aacc_aec.line(), color="red", lw=LW)
    ax.set_title(f"Paired Accuracy vs. EC{subtitle}")
    ax.set_xlabel("Paired Mean Accuracy")
    ax.set_ylabel("Error consistency")
    fig.text(x=0.7, y=0.85, s=f"r={aacc_aec.r:0.2f}, p={aacc_aec.p:0.2e}")
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)
    fig.savefig(str(outdir / "pairwise_accs_vs_ecs.png"), dpi=600)
    plt.close()


def make_table(
    downsample: bool = True, force: bool = False
) -> Tuple[DataFrame, DataFrame]:
    pairs_out = TABLES / "all_pairs.json"
    runs_out = TABLES / "all_runs.json"
    if pairs_out.exists() and runs_out.exists() and not force:
        df_pair = pd.read_json(pairs_out)
        df_run = pd.read_json(runs_out)
        return df_pair, df_run

    df_pairs, df_runs = [], []
    for dataset in DATASETS:
        for kind in ClassifierKind:
            reps = load_preds(dataset=dataset, kind=kind, downsample=downsample)
            df_pair, df_run = get_rep_dfs(reps)
            df_pair["down"] *= 100
            df_run["down"] *= 100
            renames = {"acc": "Accuracy", "ec": "EC", "down": "Downsample (%)"}
            df_pair.rename(columns=renames, inplace=True)
            df_run.rename(columns=renames, inplace=True)
            df_pair["data"] = dataset.name
            df_run["data"] = dataset.name
            df_pair["classifier"] = kind.name
            df_run["classifier"] = kind.name
            df_pairs.append(df_pair)
            df_runs.append(df_run)
    df_pair = pd.concat(df_pairs, axis=0, ignore_index=True)
    df_run = pd.concat(df_runs, axis=0, ignore_index=True)
    df_pair.to_json(pairs_out)
    print(f"Saved pairs data to {pairs_out}")
    df_run.to_json(runs_out)
    print(f"Saved runs data to {runs_out}")
    return df_pair, df_run


def corr_stats(grp: DataFrame) -> DataFrame:
    acc = grp["Accuracy"]
    ec = grp["EC"]
    res = LinResult(x=acc, y=ec)
    return DataFrame({"r": res.r, "p": res.p}, index=[0])

def corr_ec_down(grp: DataFrame) -> DataFrame:
    down = grp["Downsample (%)"]
    ec = grp["EC"]
    res = LinResult(x=down, y=ec)
    return DataFrame({"r": res.r, "p": res.p}, index=[0])


def print_tabular_info() -> None:
    df_pair, df_run = make_table()
    pair_corrs = (
        df_pair.drop(columns=["Downsample (%)", "rep"])
        .groupby(["data", "classifier"])
        .apply(corr_stats)
    )

    print(
        pair_corrs.droplevel(2)
        .reset_index()
        .to_markdown(index=False, floatfmt=["", "", "0.3f", "0.3f"])
    )

    acc_means = (
        df_run.drop(columns="Downsample (%)")
        .groupby(["data", "classifier", "rep"])
        .mean()
    )
    ec_means = (
        df_pair.drop(columns=["Downsample (%)", "Accuracy"])
        .groupby(["data", "classifier", "rep"])
        .mean()
    )
    downs = (
        df_pair.drop(columns=["EC", "Accuracy"])
        .groupby(["data", "classifier", "rep"])
        .mean()
    )
    means = pd.concat([acc_means, ec_means], axis=1)
    mean_corrs = (
        means.droplevel(2).groupby(["data", "classifier"]).apply(corr_stats).droplevel(2)
    )
    print(
        mean_corrs.reset_index().to_markdown(
            index=False, floatfmt=["", "", "0.3f", "0.3f"]
        )
    )

    pair_down_corrs = (
        df_pair.drop(columns=["Accuracy", "rep"])
        .groupby(["data", "classifier"])
        .apply(corr_ec_down).droplevel(2)
    )
    print(
        pair_down_corrs.reset_index().to_markdown(
            index=False, floatfmt=["", "", "0.3f", "0.3f"]
        )
    )

    df = pd.concat([ec_means, downs], axis=1).droplevel(2)
    mean_down_corrs = (
        df.groupby(["data", "classifier"]).apply(corr_ec_down).droplevel(2)

    )
    print(
        mean_down_corrs.reset_index().to_markdown(
            index=False, floatfmt=["", "", "0.3f", "0.3f"]
        )
    )


if __name__ == "__main__":
    DATASETS = [
        Dataset.Diabetes,
        Dataset.Parkinsons,
        Dataset.SPECT,
        Dataset.Transfusion,
        Dataset.HeartFailure,
    ]
    print_tabular_info()
    # for dataset in DATASETS:
    #     for kind in ClassifierKind:
    #         summarize_results(dataset=dataset, kind=kind, downsample=True)
