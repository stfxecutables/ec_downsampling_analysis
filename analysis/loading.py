from pathlib import Path
from typing import Dict, Tuple, Type

import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC

# fmt: off
DATAROOT = Path(__file__).resolve().parent.parent / "tests/datasets"

DIABETES    = DATAROOT / "diabetes/diabetes.csv"  # noqa
PARK        = DATAROOT / "parkinsons/parkinsons.data"  # noqa
SPECT_TRAIN = DATAROOT / "SPECT/SPECT.train"  # noqa
SPECT_TEST  = DATAROOT / "SPECT/SPECT.test"  # noqa
TRANSFUSION = DATAROOT / "transfusion/transfusion.data"  # noqa

DIABETES_TARGET = "Outcome"  # noqa
PARK_TARGET     = "status"  # noqa
SPECT_TARGET    = "diagnosis"  # noqa
TRANS_TARGET    = "donated"  # original dataset has an entire sentence for the column name... # noqa
# fmt: on


def load_df(path: Path, target: str) -> Tuple[DataFrame, Series]:
    df = pd.read_csv(path)
    x = df.drop(columns=[target])
    y = df[target].copy()
    return x, y


def load_diabetes() -> Tuple[DataFrame, Series]:
    return load_df(DIABETES, DIABETES_TARGET)


def load_park() -> Tuple[DataFrame, Series]:
    x, y = load_df(PARK, PARK_TARGET)
    x = x.drop(columns="name")
    return x, y


def load_trans() -> Tuple[DataFrame, Series]:
    return load_df(TRANSFUSION, TRANS_TARGET)


def load_SPECT() -> Tuple[DataFrame, Series]:
    x1, y1 = load_df(SPECT_TRAIN, SPECT_TARGET)
    x2, y2 = load_df(SPECT_TEST, SPECT_TARGET)
    return pd.concat([x1, x2], axis=0), pd.concat([y1, y2], axis=0)


KNN1_ARGS = dict(n_neighbors=1)
KNN3_ARGS = dict(n_neighbors=3)
KNN5_ARGS = dict(n_neighbors=5)
KNN10_ARGS = dict(n_neighbors=10)
LR_ARGS = dict(solver="liblinear")  # small datasets
SVC_ARGS = dict()
RF_ARGS = dict()
ADA_ARGS = dict()

CLASSIFIERS: Dict[str, Tuple[Type, Dict]] = {
    "KNN-1": (KNN, KNN1_ARGS),
    "KNN-3": (KNN, KNN3_ARGS),
    "KNN-5": (KNN, KNN5_ARGS),
    "KNN-10": (KNN, KNN10_ARGS),
    "Logistic Regression": (LR, LR_ARGS),
    "SVM": (SVC, SVC_ARGS),
    "Random Forest": (RF, RF_ARGS),
    "AdaBoosted DTree": (AdaBoost, ADA_ARGS),
}
DATA: Dict[str, Tuple[DataFrame, DataFrame]] = {
    "Diabetes": load_diabetes(),
    "Parkinsons": load_park(),
    "Transfusion": load_trans(),
    "SPECT": load_SPECT(),
}
OUTDIR = Path(__file__).resolve().parent / "results"
