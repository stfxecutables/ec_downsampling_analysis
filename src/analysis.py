from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from joblib import Memory
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from scipy.stats import linregress
from seaborn import FacetGrid
from statsmodels.api import OLS
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.rolling import RollingOLS
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.constants import PLOTS, TABLES, ensure_dir
from src.enumerables import ClassifierKind, Dataset
from src.metrics import acc_pairs, accs, ecs
from src.prediction import pred_root

matplotlib.use("QtAgg")

OUT = ensure_dir(PLOTS / "downsampling")
DETAILED = ensure_dir(OUT / "detailed")
MONTAGES = ensure_dir(OUT / "montage")
INDIVIDUALS = ensure_dir(OUT / "individual")
CORRELATIONS = ensure_dir(OUT / "correlations")
MEMOIZER = Memory(location=ROOT / "__JOBLIB_CACHE__", verbose=0)
CLASSIFIER_ORDER = ["LR", "SVM", "RF", "GBT"]


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


# @MEMOIZER.cache
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
        for i in range(len(self.runs) - 1):
            assert np.array_equal(
                self.runs[i].df["y_true"], self.runs[i + 1].df["y_true"]
            ), "Repeat runs do not have identical test sets"

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


# @MEMOIZER.cache
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

            df = pd.concat(dfs, axis=0, ignore_index=False).sort_index()
            uq, counts = np.unique(df.index, return_counts=True)
            if not np.all(counts == 1):
                raise ValueError("Duplicate test indices/samples in folds...")
            down = float(fold_pqs[0].stem.split("_")[1]) if downsample else 1.0
            runs[r] = Run(df=df, down=down)
        rep_id = int(rep_dir.name.replace("rep", ""))
        reps.append(Repeat(id=rep_id, runs=runs, down=runs[0].down))
    return reps


# @MEMOIZER.cache
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
    # reps = load_preds(dataset=dataset, kind=kind, downsample=downsample)
    # df_pair, df_run = get_rep_dfs(reps)
    # df_pair["down"] *= 100
    # df_run["down"] *= 100
    df_pair, df_run = make_table(
        dataset=dataset, kind=kind, downsample=downsample, force=False
    )
    renames = {"acc": "Accuracy", "ec": "EC", "down": "Downsample (%)"}
    df_pair.rename(columns=renames, inplace=True)
    df_run.rename(columns=renames, inplace=True)

    means = df_pair.groupby("rep").mean(
        numeric_only=True
    )  # columns = ["acc", "ec", "down"]
    acc_ec = LinResult(x=means["Accuracy"], y=means["EC"])
    down_acc = LinResult(x=means["Downsample (%)"], y=means["Accuracy"])
    down_ec = LinResult(x=means["Downsample (%)"], y=means["EC"])

    grid: FacetGrid
    ax: Axes
    fig: Figure
    args = dict(height=3.5, aspect=1.5, color="black")
    subtitle = f"\n{dataset.name} - {kind.name}"
    outdir = ensure_dir(DETAILED / f"{dataset.value}/{kind.value}")

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

    mean_accs = df_run.groupby("rep").mean(numeric_only=True)["Accuracy"]
    downs = df_run.groupby("rep").mean(numeric_only=True)["Downsample (%)"]
    downs.name = "Downsample (%)"
    mean_ecs = df_pair.groupby("rep").mean(numeric_only=True)["EC"]
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
    dataset: Dataset, kind: ClassifierKind, downsample: bool = True, force: bool = False
) -> Tuple[DataFrame, DataFrame]:
    pairs_out = TABLES / f"{dataset.value}_{kind.name.lower()}_pairs.json"
    runs_out = TABLES / f"{dataset.value}_{kind.name.lower()}_runs.json"
    if pairs_out.exists() and runs_out.exists() and not force:
        df_pair = pd.read_json(pairs_out)
        df_run = pd.read_json(runs_out)
        if "index" in df_pair.columns:
            df_pair.drop(columns="index", inplace=True)
        if "index" in df_run.columns:
            df_run.drop(columns="index", inplace=True)
        return df_pair, df_run

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
    df_pair.to_json(pairs_out)
    print(f"Saved pairs data to {pairs_out}")
    df_run.to_json(runs_out)
    print(f"Saved runs data to {runs_out}")
    return df_pair, df_run


def make_tables_legacy(
    datasets: List[Dataset], downsample: bool = True, force: bool = False
) -> Tuple[DataFrame, DataFrame]:
    pairs_out = TABLES / "all_pairs.json"
    runs_out = TABLES / "all_runs.json"
    if pairs_out.exists() and runs_out.exists() and not force:
        df_pair = pd.read_json(pairs_out)
        df_run = pd.read_json(runs_out)
        if "index" in df_pair.columns:
            df_pair.drop(columns="index", inplace=True)
        if "index" in df_run.columns:
            df_run.drop(columns="index", inplace=True)
        return df_pair, df_run

    df_pairs, df_runs = [], []
    for dataset in datasets:
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


def make_tables(
    datasets: List[Dataset],
    kinds: List[ClassifierKind],
    downsample: bool = True,
    force: bool = False,
) -> Tuple[DataFrame, DataFrame]:
    df_pairs, df_runs = [], []
    for dataset in datasets:
        for kind in kinds:
            df_pair, df_run = make_table(
                dataset=dataset, kind=kind, downsample=downsample, force=force
            )
            df_pairs.append(df_pair)
            df_runs.append(df_run)
    df_pair = pd.concat(df_pairs, axis=0, ignore_index=True)
    df_run = pd.concat(df_runs, axis=0, ignore_index=True)
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


def print_tabular_info(datasets: List[Dataset]) -> None:
    df_pair, df_run = make_tables(datasets=datasets, kinds=[*ClassifierKind])
    pair_corrs = (
        df_pair.drop(columns=["Downsample (%)", "rep"])
        .groupby(["data", "classifier"])
        .apply(corr_stats)
        .droplevel(2)
        .reset_index()
    )

    print(pair_corrs.to_markdown(index=False, floatfmt=["", "", "0.3f", "0.3f"]))
    out = TABLES / "pairwise_correlations.csv"
    pair_corrs.to_csv(out)
    print(f"Saved pairwise correlations table to {out}")

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
        means.droplevel(2)
        .groupby(["data", "classifier"])
        .apply(corr_stats)
        .droplevel(2)
        .reset_index()
    )
    print(mean_corrs.to_markdown(index=False, floatfmt=["", "", "0.3f", "0.3f"]))
    out = TABLES / "mean_correlations.csv"
    mean_corrs.to_csv(out)
    print(f"Saved mean correlations table to {out}")

    pair_down_corrs = (
        df_pair.drop(columns=["Accuracy", "rep"])
        .groupby(["data", "classifier"])
        .apply(corr_ec_down)
        .droplevel(2)
        .reset_index()
    )
    print(pair_down_corrs.to_markdown(index=False, floatfmt=["", "", "0.3f", "0.3f"]))
    out = TABLES / "pairwise_downsample_correlations.csv"
    pair_down_corrs.to_csv(out)
    print(f"Saved pairwise downsampling correlations table to {out}")

    df = pd.concat([ec_means, downs], axis=1).droplevel(2)
    mean_down_corrs = (
        df.groupby(["data", "classifier"]).apply(corr_ec_down).droplevel(2).reset_index()
    )
    print(mean_down_corrs.to_markdown(index=False, floatfmt=["", "", "0.3f", "0.3f"]))
    out = TABLES / "mean_downsample_correlations.csv"
    mean_down_corrs.to_csv(out)
    print(f"Saved mean downsampling correlations table to {out}")


def print_data_tables() -> None:
    dfs = []
    for dataset in tqdm(Dataset, total=len(Dataset), desc="Collecting dataset stats"):
        X, y = dataset.load()
        unq, cnts = np.unique(y, return_counts=True)
        dfs.append(
            DataFrame(
                {
                    "name": dataset.name,
                    "n_samples": len(y),
                    "n_features": X.shape[1],
                    "n_class": len(unq),
                    "majority": np.round(np.max(cnts) / len(y), 2),
                },
                index=[0],
            )
        )
    df = pd.concat(dfs, axis=0, ignore_index=True).reset_index()
    print(df.to_markdown(index=False))


def get_lowess_fits(df: DataFrame) -> DataFrame:
    def fit_lowess(y: str) -> Callable[[DataFrame], DataFrame]:
        def _closure(grp: DataFrame) -> DataFrame:
            endog = grp[y]
            exog = grp["Downsample (%)"]
            fitted = np.ravel(lowess(endog=endog, exog=exog, return_sorted=False))
            table = DataFrame({"lowess_ec" if y == "EC" else "lowess_acc": fitted})
            # table["classifier"] = grp["classifier"]
            return table

        return _closure

    if ("classifier" in df.columns) and len(np.unique(df.classifier)) != 1:
        acc_lowess = (
            df.groupby("classifier")
            .apply(fit_lowess("Accuracy"))
            .droplevel(1)
            .reset_index()
            .drop(columns="classifier")
        )
        ec_lowess = (
            df.groupby("classifier")
            .apply(fit_lowess("EC"))
            .droplevel(1)
            .reset_index()
            .drop(columns="classifier")
        )
    else:
        acc_lowess = fit_lowess("Accuracy")(df)
        ec_lowess = fit_lowess("EC")(df)
    return pd.concat([df, acc_lowess, ec_lowess], axis=1)


def make_lowess_plot(
    df_sub: DataFrame,
    split_axes: bool = False,
    individual: bool = False,
    **kwargs,
) -> FacetGrid:
    col_args = (
        dict()
        if individual
        else dict(
            col="classifier",
            col_wrap=2,
            col_order=CLASSIFIER_ORDER,
        )
    )
    grid: FacetGrid = sbn.relplot(
        data=df_sub,
        height=3,
        aspect=1.25,  # type: ignore
        x="Downsample (%)",
        y="Accuracy",
        color="black",
        label="Accuracy",
        s=5.0,
        **col_args,
        **kwargs,
    )
    grid.map(sbn.lineplot, "Downsample (%)", "lowess_acc", color="black", label=None)

    if split_axes:
        for ax in grid.figure.axes:
            axx = ax.twinx()
            sbn.scatterplot(
                data=df_sub,
                x="Downsample (%)",
                y="EC",
                color="red",
                label="EC",
                s=5.0,
                ax=axx,
            )
            sbn.lineplot(
                data=df_sub,
                x="Downsample (%)",
                y="lowess_ec",
                color="red",
                label=None,
                ax=axx,
            )
    else:
        grid.map(
            sbn.scatterplot,
            "Downsample (%)",
            "EC",
            color="red",
            label="EC",
            s=5.0,
        )
        grid.map(sbn.lineplot, "Downsample (%)", "lowess_ec", color="red", label=None)

    grid.add_legend()
    sbn.move_legend(grid, loc="upper right")
    grid.set_axis_labels(y_var="Mean EC or Accuracy", x_var="Downsampling Percent")
    if individual:
        classifier = df_sub.classifier.unique().item()
        grid.set_titles(classifier)
    else:
        grid.set_titles("{col_name}")
    grid.tight_layout()
    return grid


def make_custom_lowess_plot(
    df_sub: DataFrame,
    split_axes: bool = False,
    individual: bool = False,
    **kwargs,
) -> Figure:
    col_args = (
        dict()
        if individual
        else dict(
            col="classifier",
            col_wrap=2,
            col_order=CLASSIFIER_ORDER,
        )
    )
    ax: Axes
    axt: Axes
    sbn.set_style("white")
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=False)
    for i, (ax, classifier) in enumerate(zip(axes.flat, CLASSIFIER_ORDER)):
        sbn.scatterplot(
            data=df_sub[df_sub.classifier == classifier],
            x="Downsample (%)",
            y="Accuracy",
            color="black",
            label="Accuracy" if i == 0 else None,
            legend=False,
            s=5.0,
            **kwargs,
            ax=ax,
        )
        sbn.lineplot(
            data=df_sub[df_sub.classifier == classifier],
            x="Downsample (%)",
            y="lowess_acc",
            color="black",
            label=None,
            **kwargs,
            ax=ax,
        )
        axt = ax.twinx() if split_axes else ax
        sbn.scatterplot(
            data=df_sub[df_sub.classifier == classifier],
            x="Downsample (%)",
            y="EC",
            color="red",
            label="EC" if i == 0 else None,
            legend=False,
            s=5.0,
            ax=axt if split_axes else ax,
        )
        sbn.lineplot(
            data=df_sub[df_sub.classifier == classifier],
            x="Downsample (%)",
            y="lowess_ec",
            color="red",
            label=None,
            ax=axt if split_axes else ax,
        )
        ax.set_title(classifier)
        ax.ticklabel_format(useOffset=False)
        axt.ticklabel_format(useOffset=False)
        if split_axes:
            if i in [0, 2]:
                ax.set_ylabel("Accuracy")
            else:
                ax.set_ylabel("")
        if split_axes:
            if i in [1, 3]:
                axt.set_ylabel("EC")
            else:
                axt.set_ylabel("")
        if not split_axes and (i in [0, 2]):
            ax.set_ylabel("Accuracy or EC")
        ax.set_xlabel("Downsampling Percent")
    # grid.set_axis_labels(y_var="Mean EC or Accuracy", x_var="Downsampling Percent")
    # if individual:
    #     classifier = df_sub.classifier.unique().item()
    #     grid.set_titles(classifier)
    # else:
    #     grid.set_titles("{col_name}")
    return fig


def make_montage_plots(datasets: List[Dataset], norm: bool = False) -> None:
    df_pairs, df_runs = make_tables(
        datasets=datasets, kinds=[*ClassifierKind], force=False
    )
    ec_means = (
        df_pairs.drop(columns=["Downsample (%)", "Accuracy"])
        .groupby(["data", "classifier", "rep"])
        .mean()
    )
    acc_means = df_runs.groupby(["data", "classifier", "rep"]).mean()
    df_means = pd.concat([acc_means, ec_means], axis=1).reset_index()

    for data in datasets:
        dsname = data.name
        df = df_means.loc[df_means.data == dsname].reset_index()
        if norm:
            amean = df["Accuracy"].mean()
            ec_mean = df["EC"].mean()
            df["Accuracy"] = df["Accuracy"] - amean
            df["EC"] = df["EC"] - ec_mean
            df["Accuracy"] /= df["Accuracy"].std()
            df["EC"] /= df["EC"].std()
        dfl = get_lowess_fits(df)
        sbn.set_style("darkgrid")
        grid: FacetGrid = make_lowess_plot(dfl, split_axes=True)
        grid._sharey = False

        grid.fig.subplots_adjust(right=0.95)
        grid.fig.suptitle(data.name)
        grid.tight_layout()
        plt.show()
        out = MONTAGES / f"{data.value}.png"
        grid.savefig(out, dpi=600)
        print(f"Saved plot to {out}")
        plt.close()


def make_custom_montage_plots(datasets: List[Dataset]) -> None:
    df_pairs, df_runs = make_tables(
        datasets=datasets, kinds=[*ClassifierKind], force=False
    )
    ec_means = (
        df_pairs.drop(columns=["Downsample (%)", "Accuracy"])
        .groupby(["data", "classifier", "rep"])
        .mean()
    )
    acc_means = df_runs.groupby(["data", "classifier", "rep"]).mean()
    df_means = pd.concat([acc_means, ec_means], axis=1).reset_index()

    for data in datasets:
        dsname = data.name
        df = df_means.loc[df_means.data == dsname].reset_index()
        dfl = get_lowess_fits(df)
        sbn.set_style("darkgrid")
        fig: Figure = make_custom_lowess_plot(dfl, split_axes=True)
        fig.set_size_inches(w=12, h=8)
        fig.legend().set_visible(True)
        sbn.move_legend(fig, loc="upper right")
        fig.subplots_adjust(right=0.95)
        fig.suptitle(data.name)
        fig.tight_layout()
        # plt.show()
        out = MONTAGES / f"{data.value}_split_axes.png"
        fig.savefig(out, dpi=600)
        print(f"Saved plot to {out}")
        plt.close()


def make_individual_plots(datasets: List[Dataset], kinds: List[ClassifierKind]) -> None:
    df_pairs, df_runs = make_tables(datasets=datasets, kinds=kinds, force=False)
    grouper = ["rep"]
    if len(kinds) > 1:
        grouper = ["classifier"] + grouper
    if len(datasets) > 1:
        grouper = ["data"] + grouper
    ec_means = (
        df_pairs.drop(columns=["Downsample (%)", "Accuracy"]).groupby(grouper).mean()
    )
    acc_means = df_runs.groupby(grouper).mean()
    df_means = pd.concat([acc_means, ec_means], axis=1).reset_index()

    for data in datasets:
        for kind in kinds:
            dsname = data.name
            if len(datasets) > 1:
                df = df_means.loc[df_means.data == dsname].reset_index()
            else:
                df = df_means.copy()
            dfl = get_lowess_fits(df)
            if len(kinds) > 1:
                df_cls = dfl.loc[dfl.classifier == kind.name].reset_index()
            else:
                df_cls = dfl
                df_cls["classifier"] = kind.name
            grid = make_lowess_plot(df_cls, individual=True)

            grid.fig.suptitle(data.name)
            grid.tight_layout()
            grid.fig.subplots_adjust(right=0.95)
            out = INDIVIDUALS / f"{data.value}/{kind.name.lower()}.png"
            ensure_dir(out.parent)
            grid.savefig(out, dpi=600)
            print(f"Saved plot to {out}")
            plt.close()


@dataclass
class BootArgs:
    idx: ndarray
    x_w: ndarray
    acc_s: ndarray
    ec_s: ndarray
    n_boot: ndarray
    window_width: int


@dataclass
class BootResult:
    r_acc_mean: float
    r_acc_025: float
    r_acc_975: float
    r_ec_mean: float
    r_ec_025: float
    r_ec_975: float
    w_mean: float


def boot_resample(boot_args: BootArgs) -> BootResult:
    idx = boot_args.idx
    x_w = boot_args.x_w
    acc_s = boot_args.acc_s
    ec_s = boot_args.ec_s
    n_boot = boot_args.n_boot
    window_width = boot_args.window_width

    ws, r_accs, r_ecs = [], [], []
    acc_w = acc_s[idx]
    ec_w = ec_s[idx]
    idx = np.arange(len(x_w))
    for _ in range(n_boot):
        idx_boot = np.random.choice(idx, size=window_width, replace=True)
        x_boot = x_w[idx_boot]
        acc_boot = acc_w[idx_boot]
        ec_boot = ec_w[idx_boot]
        r_accs.append(LinResult(x_boot, acc_boot).r)
        r_ecs.append(LinResult(x_boot, ec_boot).r)
        ws.append(np.mean(x_boot))
    r_acc_desc = Series(r_accs).describe(percentiles=[0.025, 0.975])
    r_ec_desc = Series(r_ecs).describe(percentiles=[0.025, 0.975])

    return BootResult(
        r_acc_mean=float(r_acc_desc["mean"]),
        r_acc_025=float(r_acc_desc["2.5%"]),
        r_acc_975=float(r_acc_desc["97.5%"]),
        r_ec_mean=float(r_ec_desc["mean"]),
        r_ec_025=float(r_ec_desc["2.5%"]),
        r_ec_975=float(r_ec_desc["97.5%"]),
        w_mean=float(np.mean(ws)),
    )


def plot_correlations(
    data: Dataset, means: bool = True, dpi: int = 300, force: bool = False
) -> None:
    ALPHA = 0.05
    dsname = data.name
    m = "means" if means else "pairwise"
    out = CORRELATIONS / f"{dsname.lower()}_{m}.png"
    if out.exists():
        return
    window_widths = [10, 25, 50] if means else [500, 1000, 2000]
    fig, axes = plt.subplots(
        nrows=len(ClassifierKind), ncols=1 + len(window_widths), sharex=True, sharey=False
    )
    fig.set_size_inches(w=20, h=20)
    fig.suptitle(
        f"Rolling Correlations between Downsampling and Accuracy / EC - {dsname}"
    )
    for row, kind in enumerate(ClassifierKind):
        df_pairs, df_runs = make_table(dataset=data, kind=kind, force=force)
        if means:
            ec_means = (
                df_pairs.drop(columns=["Downsample (%)", "Accuracy"])
                .groupby(["data", "classifier", "rep"])
                .mean()
            )
            acc_means = df_runs.groupby(["data", "classifier", "rep"]).mean()
            df_means = (
                pd.concat([acc_means, ec_means], axis=1).reset_index().drop(columns="rep")
            )
            df = df_means.loc[df_means.data == dsname].reset_index().drop(columns="index")
        else:
            df = df_pairs.loc[df_pairs.data == dsname]

        dfl = get_lowess_fits(df)
        df = dfl.loc[dfl.classifier == kind.name].reset_index()

        x = dfl["Downsample (%)"].to_numpy()
        y_acc, y_ec = dfl["Accuracy"].to_numpy(), dfl["EC"].to_numpy()
        idx_sort = np.argsort(x)
        x_s = x.copy()[idx_sort]
        acc_s = y_acc.copy()[idx_sort]
        ec_s = y_ec.copy()[idx_sort]

        sbn.scatterplot(
            x=x, y=y_acc, color="black", label="Accuracy", s=5.0, ax=axes[row][0]
        )
        sbn.scatterplot(x=x, y=y_ec, color="red", label="EC", s=5.0, ax=axes[row][0])
        sbn.lineplot(
            data=dfl,
            x=x,
            y="lowess_acc",
            color="black",
            label=None,
            ax=axes[row][0],
        )
        sbn.lineplot(
            data=dfl,
            x=x,
            y="lowess_ec",
            color="red",
            label=None,
            ax=axes[row][0],
        )
        axes[row][0].set_xlabel("Downsampling (%)")
        axes[row][0].set_ylabel("Accuracy or Consistency")

        n_boot = 500
        delta = 1 if means else 100

        for i, window_width in tqdm(enumerate(window_widths), leave=True):
            ax = axes[row][i + 1]
            w_mean = []
            r_acc_mean, r_acc_025, r_acc_975 = [], [], []
            r_ec_mean, r_ec_025, r_ec_975 = [], [], []
            start = 0
            idxs = []
            while True:
                idx = slice(start, start + min(window_width, len(x) + 1))
                x_w = x_s[idx]
                if len(x_w) < window_width:
                    break
                idxs.append(idx)
                start += 1
            boot_args = [
                BootArgs(
                    idx=idx,
                    x_w=x_s[idx],
                    acc_s=acc_s,
                    ec_s=ec_s,
                    n_boot=n_boot,
                    window_width=window_width,
                )
                for idx in idxs
            ]
            results: List[BootResult] = process_map(
                boot_resample,
                boot_args,
                total=len(idxs),
                desc="Bootstrapping",
                chunksize=1,
            )

            # for x_w, idx in tqdm(zip(x_ws, idxs), total=len(x_ws), desc="Bootstrapping"):
            #     ws, r_accs, r_ecs = [], [], []
            #     acc_w = acc_s[idx]
            #     ec_w = ec_s[idx]
            #     idx = np.arange(len(x_w))
            #     r_accs, r_ecs, ws = list(zip(*res))
            #     for _ in range(n_boot):
            #         idx_boot = np.random.choice(idx, size=window_width, replace=True)
            #         x_boot = x_w[idx_boot]
            #         acc_boot = acc_w[idx_boot]
            #         ec_boot = ec_w[idx_boot]
            #         r_accs.append(LinResult(x_boot, acc_boot).r)
            #         r_ecs.append(LinResult(x_boot, ec_boot).r)
            #         ws.append(np.mean(x_boot))
            #     r_acc_desc = Series(r_accs).describe(percentiles=[0.025, 0.975])
            #     r_acc_mean.append(r_acc_desc["mean"])
            #     r_acc_025.append(r_acc_desc["2.5%"])
            #     r_acc_975.append(r_acc_desc["97.5%"])

            #     r_ec_desc = Series(r_ecs).describe(percentiles=[0.025, 0.975])
            #     r_ec_mean.append(r_ec_desc["mean"])
            #     r_ec_025.append(r_ec_desc["2.5%"])
            #     r_ec_975.append(r_ec_desc["97.5%"])

            #     w_mean.append(np.mean(ws))

            r_acc_mean = [result.r_acc_mean for result in results]
            r_acc_025 = [result.r_acc_025 for result in results]
            r_acc_975 = [result.r_acc_975 for result in results]

            r_ec_mean = [result.r_ec_mean for result in results]
            r_ec_025 = [result.r_ec_025 for result in results]
            r_ec_975 = [result.r_ec_975 for result in results]

            w_mean = [result.w_mean for result in results]

            r_acc_mean_smooth = lowess(exog=w_mean, endog=r_acc_mean, return_sorted=False)
            r_acc_025_smooth = lowess(exog=w_mean, endog=r_acc_025, return_sorted=False)
            r_acc_975_smooth = lowess(exog=w_mean, endog=r_acc_975, return_sorted=False)

            r_ec_mean_smooth = lowess(exog=w_mean, endog=r_ec_mean, return_sorted=False)
            r_ec_025_smooth = lowess(exog=w_mean, endog=r_ec_025, return_sorted=False)
            r_ec_975_smooth = lowess(exog=w_mean, endog=r_ec_975, return_sorted=False)

            sbn.lineplot(
                x=w_mean,
                y=r_acc_mean_smooth,
                color="black",
                label="r(d, acc)",
                ax=ax,
            )
            ax.fill_between(
                x=w_mean,
                y1=r_acc_025_smooth,
                y2=r_acc_975_smooth,
                color="black",
                alpha=ALPHA,
            )
            sbn.lineplot(
                x=w_mean,
                y=r_ec_mean_smooth,
                color="red",
                label="r(d, EC)",
                ax=ax,
            )
            ax.fill_between(
                x=w_mean, y1=r_ec_025_smooth, y2=r_ec_975_smooth, color="red", alpha=ALPHA
            )
            ax.set_title(f"{kind.name}: window = {window_width}")
            ax.set_xlabel("Downsampling (%)")
            ax.set_ylabel(r"Correlation ($\rho$)")

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(str(out), dpi=dpi)
    plt.close()
    return

    sbn.scatterplot(
        x=w_mean,
        y=r_acc_desc,
        color="black",
        label="r(d, acc)",
        ax=ax2,
    )
    sbn.scatterplot(
        x=w_mean,
        y=r_ec_mean,
        color="red",
        label="r(d, EC)",
        ax=ax2,
    )

    sbn.scatterplot(
        x=w_mean,
        y=acc_ps,
        color="black",
        label="Correlation p-value",
        ax=ax3,
    )
    sbn.scatterplot(
        x=w_mean,
        y=ec_ps,
        color="red",
        # label="r(EC, acc)",
        ax=ax3,
    )

    sbn.lineplot(
        x=w_mean,
        y=r_acc_smooth,
        color="black",
        ax=ax2,
    )
    sbn.lineplot(
        x=w_mean,
        y=r_ec_smooth,
        color="red",
        ax=ax2,
    )
    sbn.lineplot(
        x=w_mean,
        y=r_acc_p_smooth,
        color="black",
        ax=ax3,
    )
    sbn.lineplot(
        x=w_mean,
        y=r_ec_p_smooth,
        color="red",
        ax=ax3,
    )

    # Correlation analysis
    x, y = df["Downsample (%)"], df["EC"]
    ols = OLS(endog=y, exog=x, missing="none")
    res = ols.fit()
    print(res.summary())
    print(res.summary2())

    grid.fig.suptitle(data.name)
    grid.tight_layout()
    grid.fig.subplots_adjust(right=0.95)
    out = INDIVIDUALS / f"{dsname}_{kind.name.lower()}_regression_stats.png"
    ensure_dir(out.parent)
    grid.savefig(out, dpi=600)
    print(f"Saved plot to {out}")
    # plt.show()


if __name__ == "__main__":
    DATASETS = [
        # Dataset.Diabetes,
        # Dataset.Parkinsons,
        # Dataset.SPECT,
        # Dataset.Transfusion,
        # Dataset.HeartFailure,
        Dataset.Diabetes130,
        Dataset.Diabetes130Reduced,
        # Dataset.MimicIV,
        # Dataset.UTIResistance,
    ]
    # make_tables(datasets=DATASETS, kinds=[*ClassifierKind], force=True)
    # make_montage_plots(DATASETS, norm=False)
    make_custom_montage_plots(DATASETS)
    # make_individual_plots(DATASETS, kinds=[*ClassifierKind])
    # for data in DATASETS:
    #     plot_correlations(data=data, force=False, dpi=300, means=False)
    # print_data_tables()
    # make_table(force=False)
    # print_tabular_info(datasets=DATASETS)
    # for dataset in tqdm(DATASETS, leave=True, desc="Plotting data", ncols=120):
    #     for kind in tqdm(
    #         ClassifierKind, leave=True, desc="Plotting classifiers", ncols=120
    #     ):
    #         summarize_results(dataset=dataset, kind=kind, downsample=True)
