import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from pandas import DataFrame
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
from typing_extensions import Literal

sys.path.append(str(Path(__file__).resolve().parent.parent))

Dataset = Literal["Diabetes", "Parkinsons", "Transfusion", "SPECT"]

ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_CHOICES = ["knn1", "knn3", "knn5", "knn10", "lr", "svm", "rf", "ada", "mlp"]
DATASET_CHOICES = ["Diabetes", "Parkinsons", "Transfusion", "SPECT"]
JSON_DIR = Path(__file__).resolve().parent / "results/compute_canada/dfs"
PLOT_DIR = ROOT / "results/downsample/plots"
PDF_DIR = PLOT_DIR / "pdf"
PNG_DIR = PLOT_DIR / "png"
if not PDF_DIR.exists():
    os.makedirs(PDF_DIR, exist_ok=True)
if not PNG_DIR.exists():
    os.makedirs(PNG_DIR, exist_ok=True)
JSONS = sorted(JSON_DIR.rglob("*.json"))
DATASETS = [file.name.split("_")[0] for file in JSONS]
DFS = [pd.read_json(json) for json in JSONS]


def plot_results(file: Path, df: DataFrame, jitter: bool = True, curve: bool = True, show: bool = False) -> None:
    fig: plt.Figure
    ax: plt.Axes
    pieces = file.name.split("_")
    ds = dataset = pieces[0]
    classifier = file.name[len(ds) + 1 : file.name.find("__")]
    if classifier == "":
        return
    sbn.set_style("darkgrid")
    fig, ax = plt.subplots()
    x = df["Percent"].to_numpy().astype(float)
    acc = df["Accuracy"]
    con = df["Consistency"]
    if jitter:
        d = np.mean(np.diff(np.unique(x)))  # spacing for jitter
        x += np.random.uniform(0, d, len(x))  # x-jitter
    if curve:
        ex = df["Percent"].to_numpy()
        acc_smooth = lowess(acc, ex, return_sorted=False)
        con_smooth = lowess(con, ex, return_sorted=False)
        sbn.lineplot(x=ex, y=acc_smooth, color="black", label="Accuracy", alpha=0.6, lw=1, ax=ax)
        sbn.lineplot(x=ex, y=con_smooth, color="red", label="Consistency", alpha=0.6, lw=1, ax=ax)
    sbn.scatterplot(x=x, y=acc, color="black", label=None if curve else "Accuracy", alpha=0.6, s=3, ax=ax)
    sbn.scatterplot(x=x, y=con, color="red", label=None if curve else "Consistency", alpha=0.6, s=3, ax=ax)
    ax.set_title(f"{dataset} - {classifier}")
    ax.set_xlabel("Downsampling Percentage")
    ax.set_ylabel("Accuracy (or Consistency)")
    ax.set_xlim(50, 100)
    ax.set_ylim(0.2, 1)
    fig.set_size_inches(w=6, h=4)
    if show:
        plt.show()
        return
    j = "" if jitter else "_no-jitter"
    pdf = PDF_DIR / f"{file.stem}{j}.pdf"
    png = PNG_DIR / f"{file.stem}{j}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close()


def plot_dataset_results(
    dataset: Dataset,
    y_lim: float = 0.2,
    jitter: bool = False,
    curve: bool = True,
    show: bool = False,
    scatter_size: float = 5,
    alpha: float = 0.6,
) -> None:
    files: List[Path] = []
    dfs: List[DataFrame] = []
    for file, df in zip(JSONS, DFS):
        if "KNN" in file.name or "MLP" in file.name:
            continue
        if dataset in file.name:
            dfs.append(df)
            files.append(file)

    fig: plt.Figure
    ax: plt.Axes

    sbn.set_style("darkgrid")
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        file = files[i]
        pieces = file.name.split("_")
        ds = pieces[0]
        classifier = file.name[len(ds) + 1 : file.name.find("__")]

        df = dfs[i]
        x = df["Percent"].to_numpy().astype(float)
        acc = df["Accuracy"]
        con = df["Consistency"]
        if jitter:
            d = np.mean(np.diff(np.unique(x)))  # spacing for jitter
            x += np.random.uniform(0, d, len(x))  # x-jitter
        if curve:
            ex = df["Percent"].to_numpy()
            acc_smooth = lowess(acc, ex, return_sorted=False)
            con_smooth = lowess(con, ex, return_sorted=False)
            sbn.lineplot(
                x=ex,
                y=acc_smooth,
                color="black",
                label="Accuracy" if i == 0 else None,
                alpha=0.6,
                lw=1,
                ax=ax,
                legend=False,
            )
            sbn.lineplot(
                x=ex,
                y=con_smooth,
                color="red",
                label="Consistency" if i == 0 else None,
                alpha=0.6,
                lw=1,
                ax=ax,
                legend=False,
            )
        sbn.scatterplot(
            x=x,
            y=acc,
            color="black",
            label=None if (curve or i != 0) else "Accuracy",
            alpha=alpha,
            s=scatter_size,
            ax=ax,
            legend=False,
        )
        sbn.scatterplot(
            x=x,
            y=con,
            color="red",
            label=None if (curve or i != 0) else "Consistency",
            alpha=alpha,
            s=scatter_size,
            ax=ax,
            legend=False,
        )
        ax.set_title(f"{str(classifier).replace('_', '')}", fontsize=10, ha="left", x=0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(50, 100)
        ax.set_ylim(y_lim, 1)
    fig.suptitle(dataset)
    fig.set_size_inches(w=6.5, h=4)
    fig.legend(loc="upper right", fontsize=8).set_visible(True)
    fig.text(x=0.5, y=0.03, s="Downsampling Percentage", ha="center", fontsize=10)
    fig.text(x=0.03, y=0.5, s="Accuracy (or Consistency)", va="center", rotation="vertical", fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(top=0.891, bottom=0.134, left=0.103, right=0.959, hspace=0.212, wspace=0.083)
    if show:
        plt.show()
        return
    j = "" if jitter else "_no-jitter"
    s = f"_size={scatter_size}"
    a = f"_transparency={alpha}"
    pdf = PDF_DIR / f"{dataset}{j}{s}{a}.pdf"
    png = PNG_DIR / f"{dataset}{j}{s}{a}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    print(f"Saved to {PDF_DIR.parent}")
    plt.close()


if __name__ == "__main__":
    for size in [3, 5, 10]:
        for alpha in [0.4, 0.5, 0.6]:
            plot_dataset_results("Diabetes", y_lim=0.5, show=False, scatter_size=size, alpha=alpha)
    sys.exit()
    for file, df in tqdm(zip(JSONS, DFS), total=len(DFS), desc="Plotting"):
        if "MLP" not in file.name:
            continue
        plot_results(file, df, jitter=False)
    print(f"Saved plots to {PDF_DIR.parent}")
    # FILE = Path("/home/derek/Desktop/error-consistency/analysis/results/test_results/Diabetes_Logistic_Regression__k-fold-holdout_downsample.json")
    # df = pd.read_json(FILE)
    # plot_results(FILE, df, jitter=False, curve=True, show=True)
