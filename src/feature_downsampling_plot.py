import os
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from time import strftime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.stats import linregress
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

sys.path.append(str(Path(__file__).resolve().parent.parent))


CLASSIFIER_CHOICES = ["knn1", "knn3", "knn5", "knn10", "lr", "svm", "rf", "ada", "mlp"]
DATASET_CHOICES = ["diabetes", "park", "trans", "spect"]
ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "results/dfs/feat_2021-Oct07"
PLOT_DIR = ROOT / f"results/plots/feat_{strftime('%Y-%b%d')}"
PDF_DIR = PLOT_DIR / "pdf"
PNG_DIR = PLOT_DIR / "png"
if not PDF_DIR.exists():
    os.makedirs(PDF_DIR, exist_ok=True)
if not PNG_DIR.exists():
    os.makedirs(PNG_DIR, exist_ok=True)
JSONS = sorted(JSON_DIR.rglob("*.json"))
DATASETS = [file.name.split("_")[0] for file in JSONS]
DFS = [pd.read_json(json) for json in JSONS]

COLUMNS = [
    "Percent",
    "Accuracy",
    "Consistency",
    "EC (sd)",
    "d",
    "d-med",
    "AUC",
    "AUC-med",
    "delta",
    "delta-med",
]

# DELTA = "#ed00fa"
DELTA = "#000"


@dataclass
class PlotArgs:
    file: Path
    df: DataFrame
    loess: bool
    standardize: bool
    show: bool = False


def standardize_vals(df: DataFrame, acc: bool = False, con: bool = False) -> DataFrame:
    df = df.copy()
    fnames = ["d", "d-med", "AUC", "AUC-med", "delta", "delta-med"]
    if acc:
        fnames.append("Consistency")
    if con:
        fnames.append("Accuracy")
    for f in fnames:
        df[f] -= df[f].mean()
        df[f] /= df[f].std(ddof=1)
    return df


def regression_line(x: ndarray, y: ndarray, loess: bool = False) -> ndarray:
    if loess:  # REVERSED!
        return lowess(y, x, frac=0.85, return_sorted=False)
    res = linregress(x, y)
    m, b = res.slope, res.intercept
    return m * x + b


def plot_by_downsampling(
    file: Path, df: DataFrame, curve: bool = True, standardize: bool = True, show: bool = False
) -> None:
    fig: plt.Figure
    ax: plt.Axes
    pieces = file.name.split("_")
    ds = dataset = pieces[0]
    classifier = file.name[len(ds) + 1 : file.name.find("__")]
    if classifier == "":
        return
    df = df.copy()
    x = df["Percent"].to_numpy().astype(float)
    acc = df["Accuracy"]
    con = df["Consistency"]
    d, d_med = df["d"], df["d-med"]
    auc, auc_med = df["AUC"], df["AUC-med"]
    delta, delta_med = df["delta"], df["delta-med"]
    if standardize:
        for f in [d, d_med, auc, auc_med, acc, con, delta, delta_med]:
            f -= f.mean()
            f /= f.std(ddof=1)
    sbn.set_style("darkgrid")
    fig, ax = plt.subplots()
    largs = dict(lw=2, ax=ax, legend=False)
    sargs = dict(s=3, ax=ax, legend=False)
    if curve:
        ex = df["Percent"].to_numpy()
        acc_smooth = lowess(acc, ex, return_sorted=False)
        con_smooth = lowess(con, ex, return_sorted=False)
        d_smooth = lowess(d, ex, return_sorted=False)
        d_med_smooth = lowess(d_med, ex, return_sorted=False)
        auc_smooth = lowess(auc, ex, return_sorted=False)
        auc_med_smooth = lowess(auc_med, ex, return_sorted=False)
        # delta_smooth = lowess(delta, ex, return_sorted=False)
        # delta_med_smooth = lowess(delta_med, ex, return_sorted=False)
        sbn.lineplot(x=ex, y=d_smooth, color="#004aeb", label="Cohen's d (mean)", alpha=0.9, **largs)
        sbn.lineplot(x=ex, y=d_med_smooth, color="#004aeb", label="Cohen's d (median)", alpha=0.3, **largs)
        sbn.lineplot(x=ex, y=auc_smooth, color="#eb9100", label="AUC (mean)", alpha=0.9, **largs)
        sbn.lineplot(x=ex, y=auc_med_smooth, color="#eb9100", label="AUC (median)", alpha=0.3, **largs)
        # sbn.lineplot(x=ex, y=delta_smooth, color="#ed00fa", label="delta (mean)", alpha=0.9, **largs)
        # sbn.lineplot(x=ex, y=delta_med_smooth, color="#ed00fa", label="delta (median)", alpha=0.3, **largs)
        sbn.lineplot(x=ex, y=acc_smooth, color="black", label="Accuracy", alpha=0.9, **largs)
        sbn.lineplot(x=ex, y=con_smooth, color="red", label="Consistency", alpha=0.9, **largs)
    sbn.scatterplot(x=x, y=d, color="#004aeb", label="Cohen's d (mean)", alpha=0.9, **sargs)
    sbn.scatterplot(x=x, y=d_med, color="#004aeb", label="Cohen's d (median)", alpha=0.3, **sargs)
    sbn.scatterplot(x=x, y=auc, color="#eb9100", label="AUC (mean)", alpha=0.9, **sargs)
    sbn.scatterplot(x=x, y=auc_med, color="#eb9100", label="AUC (median)", alpha=0.3, **sargs)
    # sbn.scatterplot(x=x, y=delta, color="#ed00fa", label="delta (mean)", alpha=0.9, **sargs)
    # sbn.scatterplot(x=x, y=delta_med, color="#ed00fa", label="delta (median)", alpha=0.3, **sargs)
    sbn.scatterplot(x=x, y=acc, color="black", label=None if curve else "Accuracy", alpha=0.9, **sargs)
    sbn.scatterplot(x=x, y=con, color="red", label=None if curve else "Consistency", alpha=0.9, **sargs)
    ax.set_title(f"{dataset} - {classifier}")
    ax.set_xlabel("Feature Downsampling Percentage")
    s = "Standarized " if standardize else ""
    ax.set_ylabel(f"{s}Metric Value / {s}Feature Separation")
    ax.set_xlim(50, 100)
    fig.legend().set_visible(True)
    fig.set_size_inches(w=12, h=6)
    if show:
        plt.show()
        return
    s = "_standardized" if standardize else ""
    pdf = PDF_DIR / f"{file.stem}{s}.pdf"
    png = PNG_DIR / f"{file.stem}{s}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close()


def plot_feat_quality_fits(
    ax: plt.Axes, df: DataFrame, y_val: str = "Consistency", loess: bool = True, label: bool = False
):
    df = df.copy()
    y_src = df[y_val].copy()  # y-values

    d, d_med = df["d"], df["d-med"]
    auc, auc_med = df["AUC"], df["AUC-med"]
    delta, delta_med = df["delta"], df["delta-med"]

    largs = dict(ax=ax, lw=2, legend=False)
    info = {
        "d": {**dict(color="#004aeb", label="Cohen's d (mean)", alpha=0.9), **largs},
        # "d-med": {**dict(color="#004aeb", label="Cohen's d (median)", alpha=0.9), **largs},
        "AUC": {**dict(color="#eb9100", label="AUC (mean)", alpha=0.9), **largs},
        # "AUC-med": {**dict(color="#eb9100", label="AUC (median)", alpha=0.9), **largs},
        "delta": {**dict(color=DELTA, label="delta (mean)", alpha=0.9), **largs},
        # "delta-med": {**dict(color=DELTA, label="delta (median)", alpha=0.9), **largs},
    }
    if not label:
        for key, val in info.items():
            val["label"] = None
    for fname, args in info.items():
        x = df[fname].copy()
        if loess:
            x += np.random.normal(0, np.std(x, ddof=1))
        idx = np.argsort(x)
        x = x[idx]
        y = np.copy(y_src)[idx]
        smoothed = regression_line(x, y, loess=loess)
        sbn.lineplot(x=x, y=smoothed, **args)
    return


def plot_by_feat_quality(
    file: Path, df: DataFrame, curve: bool = True, standardize: bool = True, loess: bool = True, show: bool = False
) -> None:
    fig: plt.Figure
    ax: plt.Axes
    pieces = file.name.split("_")
    ds = dataset = pieces[0]
    classifier = file.name[len(ds) + 1 : file.name.find("__")]
    if classifier == "":
        return
    df_all = df.copy()
    df = df.copy()
    if standardize:
        df = standardize_vals(df)
    p = df["Percent"].to_numpy().astype(float)
    acc = df["Accuracy"]  # y-values
    con = df["Consistency"]  # y-values
    d, d_med = df["d"], df["d-med"]
    auc, auc_med = df["AUC"], df["AUC-med"]
    delta, delta_med = df["delta"], df["delta-med"]
    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(ncols=2)
    # sargs = dict(s=3, ax=ax, legend=False)
    sargs = dict(size=p / 2, alpha=0.3, legend=False)
    if curve:
        plot_feat_quality_fits(axes[0], df, "Consistency", loess=loess)
        plot_feat_quality_fits(axes[1], df, "Accuracy", loess=loess)

    # fmt: off
    sbn.scatterplot(ax=axes[0], y=con, x=d, color="#004aeb", label="Cohen's d (mean)",  **sargs)
    sbn.scatterplot(ax=axes[0], y=con, x=auc, color="#eb9100", label="AUC (mean)",  **sargs)
    sbn.scatterplot(ax=axes[0], y=con, x=delta, color=DELTA, label="delta (mean)",  **sargs)
    # sbn.scatterplot(ax=axes[0], y=con, x=d_med, color="#004aeb", label="Cohen's d (median)", alpha=0.3, **sargs)
    # sbn.scatterplot(ax=axes[0], y=con, x=auc_med, color="#eb9100", label="AUC (median)", alpha=0.3, **sargs)
    # sbn.scatterplot(ax=axes[0], y=con, x=delta_med, color=DELTA label="delta (median)", alpha=0.3, **sargs)

    sbn.scatterplot(ax=axes[1], y=acc, x=d, color="#004aeb",  **sargs)
    sbn.scatterplot(ax=axes[1], y=acc, x=auc, color="#eb9100",  **sargs)
    sbn.scatterplot(ax=axes[1], y=acc, x=delta, color=DELTA,  **sargs)
    # sbn.scatterplot(ax=axes[1], y=acc, x=d_med, color="#004aeb", label="Cohen's d (median)", alpha=0.3, **sargs)
    # sbn.scatterplot(ax=axes[1], y=acc, x=auc_med, color="#eb9100", label="AUC (median)", alpha=0.3, **sargs)
    # sbn.scatterplot(ax=axes[1], y=acc, x=delta_med, color=DELTA, label="delta (median)", alpha=0.3, **sargs)
    # fmt: on
    axes[0].set_title(f"{dataset} - {classifier}: AEC and Feature Quality")
    s = "Standarized " if standardize else ""
    axes[0].set_xlabel(f"{s}Feature Separation/Quality")
    axes[0].set_ylabel("AEC")
    axes[1].set_title(f"{dataset} - {classifier}: AEC and Accuracy")
    s = "Standarized " if standardize else ""
    axes[1].set_xlabel(f"{s}Feature Separation/Quality")
    axes[1].set_ylabel("Mean Accuracy")
    # axes[0].set_xlim(50, 100)
    fig.legend().set_visible(True)
    fig.set_size_inches(w=12, h=6)
    if show:
        plt.show()
        return
    s = "standardized" if standardize else "unstandardized"
    l = "loess" if loess else "linear"
    pdf = PDF_DIR / f"{s}/{l}/{file.stem}_{l}_{s}.pdf"
    png = PNG_DIR / f"{s}/{l}/{file.stem}_{l}_{s}.png"
    os.makedirs(pdf.parent, exist_ok=True)
    os.makedirs(png.parent, exist_ok=True)
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close()


def plot_percent(args: PlotArgs) -> None:
    plot_by_downsampling(args.file, args.df, standardize=args.standardize, show=args.show)


def plot_quality(args: PlotArgs) -> None:
    plot_by_feat_quality(args.file, args.df, standardize=args.standardize, show=args.show, loess=args.loess)


if __name__ == "__main__":
    args = []
    args.extend([PlotArgs(file, df, show=False, standardize=True, loess=True) for file, df in zip(JSONS, DFS)])
    args.extend([PlotArgs(file, df, show=False, standardize=False, loess=True) for file, df in zip(JSONS, DFS)])
    args.extend([PlotArgs(file, df, show=False, standardize=True, loess=False) for file, df in zip(JSONS, DFS)])
    args.extend([PlotArgs(file, df, show=False, standardize=False, loess=False) for file, df in zip(JSONS, DFS)])
    process_map(plot_quality, args, desc="Plotting")
    # for arg in args:
    #     plot_quality(arg)
    print(f"Saved plots to {PDF_DIR.parent}")
