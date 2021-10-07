import os
import sys
from enum import Enum
from itertools import repeat
from pathlib import Path
from typing import Dict, Tuple, Type, Union

import numpy as np
from pandas import DataFrame
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

sys.path.append(str(Path(__file__).resolve().parent.parent))
from analysis.loading import load_diabetes, load_park, load_SPECT, load_trans

ROOT = Path(__file__).parent.resolve().parent
TEST_OUTDIR = ROOT / "test_results"
DOWNSAMPLE_OUTDIR = ROOT / "results/downsample"
FEATURE_SELECTION_OUTDIR = ROOT / "results/feature_selection"

DOWNSAMPLE_PLOT_OUTDIR = DOWNSAMPLE_OUTDIR / "plots"
DOWNSAMPLE_RESULTS_DIR = ROOT / "results/dfs"
FEATURE_PLOT_OUTDIR = FEATURE_SELECTION_OUTDIR / "plots"
FEATURE_RESULTS_DIR = ROOT / "results/dfs"

DIRS = [
    TEST_OUTDIR,
    DOWNSAMPLE_OUTDIR,
    FEATURE_SELECTION_OUTDIR,
    DOWNSAMPLE_PLOT_OUTDIR,
    DOWNSAMPLE_RESULTS_DIR,
    FEATURE_PLOT_OUTDIR,
    FEATURE_RESULTS_DIR,
]

for dir in DIRS:
    if not dir.exists():
        os.makedirs(dir, exist_ok=True)

REPS_PER_PERCENT = 50
REPS_PER_PERCENT = 10
KFOLD_REPS = 10
N_PERCENTS = 150
PERCENT_MIN = 50
PERCENT_MAX = 95

PERCENTS = np.linspace(0, 1, 21)[1:-1]  # 5, 10, ..., 95
COLS = [f"{e:1.0f}" for e in PERCENTS * 100]
N_ROWS = len(COLS) * REPS_PER_PERCENT


DATASET_NAMES = ["Diabetes", "Transfusion", "Parkinsons", "SPECT"]
KNN1_ARGS = {name: dict(n_neighbors=1) for name in DATASET_NAMES}
KNN3_ARGS = {name: dict(n_neighbors=3) for name in DATASET_NAMES}
KNN5_ARGS = {name: dict(n_neighbors=5) for name in DATASET_NAMES}
KNN10_ARGS = {name: dict(n_neighbors=10) for name in DATASET_NAMES}
LR_ARGS = {
    "Diabetes": dict(solver="liblinear", penalty="l1", C=1.7, max_iter=250),
    "Transfusion": dict(solver="liblinear", penalty="l1", C=1.6, max_iter=500),
    "Parkinsons": dict(solver="liblinear", penalty="l2", C=0.7, max_iter=500),
    "SPECT": dict(solver="liblinear", penalty="l1", C=0.4, max_iter=500),
}
SVC_ARGS: Dict = {  # basically doesn't matter, all perform about the same
    "Diabetes": dict(kernel="rbf", C=1.0),
    "Transfusion": dict(kernel="rbf", C=5.0),
    "Parkinsons": dict(kernel="rbf", C=52.0),
    "SPECT": dict(kernel="rbf", C=50.0),
}
RF_ARGS: Dict = {
    "Diabetes": dict(n_estimators=400, min_samples_leaf=4, max_features=0.25, max_depth=20),
    "Transfusion": dict(n_estimators=20, min_samples_leaf=1, max_features=0.25, max_depth=2),
    "Parkinsons": dict(n_estimators=20, min_samples_leaf=2, max_features="auto", max_depth=2),
    "SPECT": dict(n_estimators=20, min_samples_leaf=3, max_features=0.5, max_depth=8),
}
ADA_ARGS: Dict = {
    "Diabetes": dict(n_estimators=50, learning_rate=0.52),
    "Transfusion": dict(n_estimators=100, learning_rate=0.0436),
    "Parkinsons": dict(n_estimators=200, learning_rate=0.00452),
    "SPECT": dict(n_estimators=200, learning_rate=0.0285),
}
MLP_ARGS: Dict = {  # all found via random search with ray.tune
    "Diabetes": dict(hidden_layer_sizes=tuple(repeat(4, times=6)), alpha=0.0002, batch_size=32, max_iter=750),
    "Transfusion": dict(hidden_layer_sizes=tuple(repeat(4, times=6)), alpha=5e-6, batch_size=32, max_iter=750),
    "Parkinsons": dict(hidden_layer_sizes=tuple(repeat(4, times=6)), alpha=2e-5, batch_size=32, max_iter=1000),
    "SPECT": dict(hidden_layer_sizes=tuple(repeat(4, times=6)), alpha=0.0005, batch_size=32, max_iter=500),
}


CLASSIFIERS: Dict[str, Tuple[Type, Dict]] = {
    "KNN-1": (KNN, KNN1_ARGS),
    "KNN-3": (KNN, KNN3_ARGS),
    "KNN-5": (KNN, KNN5_ARGS),
    "KNN-10": (KNN, KNN10_ARGS),
    "Logistic Regression": (LR, LR_ARGS),
    "SVM Classifier": (SVC, SVC_ARGS),
    "Random Forest": (RF, RF_ARGS),
    "AdaBoosted DTree": (AdaBoost, ADA_ARGS),
    "MLP": (MLPClassifier, MLP_ARGS),
}
DATA: Dict[str, Tuple[DataFrame, DataFrame]] = {
    "Diabetes": load_diabetes(),
    "Parkinsons": load_park(),
    "Transfusion": load_trans(),
    "SPECT": load_SPECT(),
}

# Would prefer Literal, but clusters often don't have newer Python versions...
class Analysis(Enum):
    feature = "feature"
    downsample = "downsample"


AnalysisType = Union[Analysis, str]
