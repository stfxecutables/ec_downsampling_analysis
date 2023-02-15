from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


from pathlib import Path
from typing import Dict, Tuple, Type

import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from typing_extensions import Literal

from src.constants import DATA


def load_df(path: Path, target: str) -> Tuple[DataFrame, Series]:
    df = pd.read_csv(path)
    x = df.drop(columns=[target])
    y = df[target].copy()
    return x, y


def load_diabetes() -> Tuple[DataFrame, Series]:
    source = DATA / "diabetes/diabetes.csv"
    target = "Outcome"
    return load_df(source, target)


def load_park() -> Tuple[DataFrame, Series]:
    source = DATA / "parkinsons/parkinsons.data"
    target = "status"
    x, y = load_df(source, target)
    x = x.drop(columns="name")
    return x, y


def load_trans() -> Tuple[DataFrame, Series]:
    # original dataset has an entire sentence for the column name...
    source = DATA / "transfusion/transfusion.data"
    target = "donated"
    return load_df(source, target)


def load_SPECT() -> Tuple[DataFrame, Series]:
    source = DATA / "SPECT/SPECT.train"
    source2 = DATA / "SPECT/SPECT.test"
    target = "diagnosis"
    x1, y1 = load_df(source, target)
    x2, y2 = load_df(source2, target)
    return pd.concat([x1, x2], axis=0), pd.concat([y1, y2], axis=0)


def load_heart_failure() -> Tuple[DataFrame, Series]:
    """
    >>> df.filter(regex="death|adm").describe().T.sort_index()
                                              count        mean         std
    death.within.28.days                     2008.0    0.018426    0.134521
    death.within.3.months                    2008.0    0.020916    0.143140
    death.within.6.months                    2008.0    0.028386    0.166116
    re.admission.time..days.from.admission.   901.0  126.711432  145.025343
    re.admission.within.28.days              2008.0    0.069721    0.254740
    re.admission.within.3.months             2008.0    0.248008    0.431964
    re.admission.within.6.months             2008.0    0.384960    0.486707
    time.of.death..days.from.admission.        44.0   29.522727   72.452226
    """
    source = DATA / "heart_failure/dat.csv"
    drops = [  # may allow trivial prediction
        "death.within.28.days",
        "death.within.3.months",
        "death.within.6.months",
        "re.admission.time..days.from.admission.",
        "re.admission.within.28.days",
        "re.admission.within.3.months",
        "re.admission.within.6.months",
        "time.of.death..days.from.admission.",
        "outcome.during.hospitalization",
        "DestinationDischarge",
    ]
    target = "re.admission.within.6.months"
    df = pd.read_csv(source)
    target = df[target]
    df = df.drop(columns=drops)
    return df, target


def load_diabetes130() -> Tuple[DataFrame, Series]:
    source = DATA / "diabetes130.csv"
    target = "readmit_30_days"
    df = pd.read_csv(source)
    target = df[target]
    df = df.drop(columns=target)
    # TODO: one-hot encode categoricals
    return df, target


def load_uti_resistance(
    style: Literal["binary", "ordinal", "multi"]
) -> Tuple[DataFrame, Series]:
    features = pd.read_csv(DATA / "uti_resistance/all_uti_features.csv")
    labels = pd.read_csv(DATA / "uti_resistance/all_uti_resist_labels.csv")
    # prescs = DATA / "uti_resitance/all_prescriptions.csv" not needed
    drops = ["is_train", "uncomplicated", "example_id"]
    targets = ["NIT", "SXT", "CIP", "LVX"]
    # could do one for each, or combine to get "any resistance"
    df = pd.merge(features.drop(columns=drops[:-1]), labels, on="example_id", how="inner")
    df.dropna(axis=0, inplace=True)
    df_targets = df.loc[:, targets].copy()
    df.drop(columns=[*targets, *drops], inplace=True)
    if style == "binary":
        y = df_targets.any(axis=1).astype(int)
    elif style == "ordinal":
        y = df_targets.sum(axis=1).astype(int)
    elif "multi" in style:
        y_str = (
            df_targets.NIT.apply(str)
            + df_targets.SXT.apply(str)
            + df_targets.CIP.apply(str)
            + df_targets.LVX.apply(str)
        )
        y = LabelEncoder().fit_transform(y_str)
    return df, y


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
OUTDIR = Path(__file__).resolve().parent / "results"

if __name__ == "__main__":
    load_uti_resistance()
