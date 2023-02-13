import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from analysis.constants import (
    DOWNSAMPLE_PLOT_OUTDIR,
    DOWNSAMPLE_RESULTS_DIR,
    FEATURE_PLOT_OUTDIR,
    FEATURE_RESULTS_DIR,
    KFOLD_REPS,
    N_PERCENTS,
    PERCENT_MAX,
    PERCENT_MIN,
    Analysis,
    AnalysisType,
)

CLASSIFIER_CHOICES = ["knn1", "knn3", "knn5", "knn10", "lr", "svm", "rf", "ada", "mlp"]
DATASET_CHOICES = ["diabetes", "park", "trans", "spect"]
SCRIPT_OUTDIR = Path(__file__).resolve().parent.parent


def argparse_setup(analysis: AnalysisType = "downsample") -> ArgumentParser:
    analyse = Analysis(analysis)
    parser = ArgumentParser(description="Plot effect of downsampling on error consistency.")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="directory to save generated plots",
        default=DOWNSAMPLE_PLOT_OUTDIR if analyse is Analysis.downsample else FEATURE_PLOT_OUTDIR,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="directory to save generated data for plots",
        default=DOWNSAMPLE_RESULTS_DIR if analyse is Analysis.feature else FEATURE_RESULTS_DIR,
    )
    parser.add_argument("--kfold-reps", type=int, help="times to repeat k-fold per downsampling", default=KFOLD_REPS)
    parser.add_argument(
        "--n-percents",
        type=int,
        help="times to sample the percentages between --percent-min and --percent-max",
        default=N_PERCENTS,
    )
    parser.add_argument("--percent-min", type=int, help="smallest percentage to downsample to", default=PERCENT_MIN)
    parser.add_argument("--percent-max", type=int, help="largest percentage to downsample to", default=PERCENT_MAX)

    # parser.add_argument(
    #     "--percent-reps",
    #     type=int,
    #     help="times to downsample for each percentage",
    #     default=REPS_PER_PERCENT,
    # )
    # parser.add_argument(
    #     "--percent-spacing",
    #     type=int,
    #     help="space between percents. E.g.  5 means 5, 10, 15, ...",
    #     default=5,
    # )
    parser.add_argument("--classifier", choices=CLASSIFIER_CHOICES, required=True)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--validation", choices=["internal", "external"], default="external")
    parser.add_argument("--pbar", action="store_true")
    parser.add_argument("--cpus", type=int, help="number of cpus to use for parallelization", default=4)
    return parser


def classifier_from_args(args: Namespace) -> str:
    mappings = dict(
        knn1="KNN-1",
        knn3="KNN-3",
        knn5="KNN-5",
        knn10="KNN-10",
        lr="Logistic Regression",
        svm="SVM Classifier",
        rf="Random Forest",
        ada="AdaBoosted DTree",
        mlp="MLP",
    )
    return mappings[args.classifier]


def dataset_from_args(args: Namespace) -> str:
    mappings = dict(diabetes="Diabetes", park="Parkinsons", trans="Transfusion", spect="SPECT")
    return mappings[args.dataset]
