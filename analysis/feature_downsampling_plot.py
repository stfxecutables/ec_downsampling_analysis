import os
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sbn
from pandas import DataFrame
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

sys.path.append(str(Path(__file__).resolve().parent.parent))


CLASSIFIER_CHOICES = ["knn1", "knn3", "knn5", "knn10", "lr", "svm", "rf", "ada", "mlp"]
DATASET_CHOICES = ["diabetes", "park", "trans", "spect"]
ROOT = Path(__file__).resolve().parent.parent
JSON_DIR = ROOT / "results/dfs/feat_2021-Oct07"
PLOT_DIR = ROOT / "results/plots/feat_2021-Oct07"
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


@dataclass
class PlotArgs:
    file: Path
    df: DataFrame
    standardize: bool


def plot_results(file: Path, df: DataFrame, curve: bool = True, standardize: bool = True, show: bool = False) -> None:
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


def plot(args: PlotArgs) -> None:
    plot_results(args.file, args.df, standardize=args.standardize, show=False)


if __name__ == "__main__":
    # for file, df in tqdm(zip(JSONS, DFS), total=len(DFS), desc="Plotting"):
    #     show = False
    #     plot_results(file, df, standardize=True, show=show)
    #     plot_results(file, df, standardize=False, show=show)
    #     # sys.exit()

    args = []
    args.extend([PlotArgs(file, df, standardize=True) for file, df in zip(JSONS, DFS)])
    args.extend([PlotArgs(file, df, standardize=False) for file, df in zip(JSONS, DFS)])
    process_map(plot, args, desc="Plotting")
    print(f"Saved plots to {PDF_DIR.parent}")
