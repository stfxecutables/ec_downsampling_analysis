from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from time import strftime, time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)
from warnings import catch_warnings, filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from lightgbm import LGBMClassifier
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from pandas.errors import PerformanceWarning
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import HistGradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler as FourierApproximator
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit as SSSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm
from typing_extensions import Literal
from xgboost import XGBClassifier, XGBRFClassifier

from src.constants import CAT_REDUCED, DATA

if TYPE_CHECKING:
    from src.enumerables import Dataset


def load_df(path: Path, target: str) -> Tuple[DataFrame, Series]:
    df = pd.read_csv(path)
    x = df.drop(columns=[target])

    y = df[target].copy()
    return x, y


def standardize(x: DataFrame) -> DataFrame:
    return DataFrame(
        data=StandardScaler().fit_transform(x), columns=x.columns, index=x.index
    )


def normalize(x: DataFrame) -> DataFrame:
    return DataFrame(
        data=MinMaxScaler().fit_transform(x), columns=x.columns, index=x.index
    )


def reduce_categoricals(
    cats_one_hot: DataFrame,
    dataset: Dataset,
    n_components: int = 2,
    n_neighbours: int = 15,
) -> DataFrame:
    from src.enumerables import Dataset

    """
    Notes
    -----
    We follow the guides:

        https://github.com/lmcinnes/umap/issues/58
        https://github.com/lmcinnes/umap/issues/104
        https://github.com/lmcinnes/umap/issues/241

    in spirit, but just embed all dummified categoricals to two dimensions.
    """
    from umap import UMAP

    outfile = CAT_REDUCED / f"{dataset.name}_n{n_components}.parquet"
    if outfile.exists():
        reduced: DataFrame = pd.read_parquet(outfile)
        return reduced

    filterwarnings("ignore", category=PerformanceWarning)
    umap = UMAP(n_components=n_components, n_neighbors=n_neighbours, metric="jaccard", verbose=True)  # type: ignore
    with catch_warnings():
        filterwarnings("ignore", message="gradient function", category=UserWarning)
        array = umap.fit_transform(cats_one_hot)
    reduced = DataFrame(data=array, columns=[f"umap{i}" for i in range(n_components)])
    reduced.to_parquet(outfile)
    print(f"Saved reduced data to {outfile}")
    return reduced


def load_diabetes() -> Tuple[DataFrame, Series]:
    source = DATA / "diabetes/diabetes.csv"
    target = "Outcome"
    x, y = load_df(source, target)
    x = standardize(x)
    return x, y


def load_park() -> Tuple[DataFrame, Series]:
    source = DATA / "parkinsons/parkinsons.data"
    target = "status"
    x, y = load_df(source, target)
    x = x.drop(columns="name")
    x = standardize(x)
    return x, y


def load_trans() -> Tuple[DataFrame, Series]:
    # original dataset has an entire sentence for the column name...
    source = DATA / "transfusion/transfusion.data"
    target = "donated"
    x, y = load_df(source, target)
    x = standardize(x)
    return x, y


def load_SPECT() -> Tuple[DataFrame, Series]:
    source = DATA / "SPECT/SPECT.train"
    source2 = DATA / "SPECT/SPECT.test"
    target = "diagnosis"
    x1, y1 = load_df(source, target)
    x2, y2 = load_df(source2, target)
    x = pd.concat([x1, x2], axis=0, ignore_index=True)
    y = pd.concat([y1, y2], axis=0, ignore_index=True)
    # all one-hots, no need to standardize
    return x, y


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
    preprocessed = DATA / "heart_failure_preprocessed.json"
    if preprocessed.exists():
        df = pd.read_json(preprocessed)
        y = df["target"].copy()
        df = df.drop(columns="target")
        return df, y
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
        "return.to.emergency.department.within.6.months",
        "time.to.emergency.department.within.6.months",
        "DestinationDischarge",
        # all NaN for below features
        "cholinesterase",
        "body.temperature.blood.gas",
    ]
    target = "re.admission.within.6.months"
    df = pd.read_csv(source).iloc[:, 2:]
    y = df[target].copy()
    df = df.drop(columns=drops)
    # about a quarter of features are half-null
    ord_cols = ["ageCat"]
    bool_cols = [
        "admission.way",
        "gender",
        "type.II.respiratory.failure",
        "oxygen.inhalation",
    ]
    age_bins = {
        "(89,110]": 7.0,
        "(79,89]": 6.0,
        "(69,79]": 5.0,
        "(59,69]": 4.0,
        "(49,59]": 3.0,
        "(39,49]": 2.0,
        "(29,39]": 1.0,
        "(21,29]": 0.0,
    }

    cats = df.select_dtypes("object").drop(columns=ord_cols + bool_cols)
    bools = df.select_dtypes("object").loc[:, bool_cols]
    cats = pd.get_dummies(cats)
    bools = pd.get_dummies(bools).iloc[:, ::2]
    floats = df.select_dtypes("float")
    floats["age"] = df["ageCat"].apply(lambda a: age_bins[a])
    floats.fillna(floats.median(), inplace=True)
    floats = standardize(floats)
    df = pd.concat([floats, cats, bools], axis=1)
    df_all = df.copy()
    df_all["target"] = y
    df_all.to_json(preprocessed)
    return df, y


def load_diabetes130() -> Tuple[DataFrame, Series]:
    preprocessed = DATA / "diabetes130_preprocessed.json"
    if preprocessed.exists():
        df = pd.read_json(preprocessed)
        y = df["target"].copy().astype(int)
        df = df.drop(columns="target")
        return df, y

    source = DATA / "diabetes130.csv"
    target = "readmit_30_days"
    df = pd.read_csv(source)
    y = df.loc[:, target].copy()
    df.drop(columns=target, inplace=True)
    ages = {"30 years or younger": 0, "30-60 years": 1, "Over 60 years": 2}
    bools = df.select_dtypes("bool").astype(int)
    bools["diabetesMed"] = df["diabetesMed"].apply(lambda x: 0 if x == "No" else 1)
    bools["change"] = df["change"].apply(lambda x: 0 if x == "No" else 1)
    cats = df.select_dtypes("object").drop(columns=["age", "diabetesMed", "change"])
    cats = pd.get_dummies(cats)
    ords = df.select_dtypes("int64")
    ords["age"] = df["age"].apply(lambda age: ages[age])
    ords = standardize(ords)
    df = pd.concat([ords, cats, bools], axis=1)
    df_all = df.copy()
    df_all["target"] = y
    df_all.to_json(preprocessed)
    return df, y


def load_diabetes130_continuous(n_components: int) -> Tuple[DataFrame, Series]:
    from src.enumerables import Dataset

    preprocessed = DATA / f"diabetes130_preprocessed_cont{n_components}.json"
    if preprocessed.exists():
        df = pd.read_json(preprocessed)
        y = df["target"].copy().astype(int)
        df = df.drop(columns="target")
        return df, y

    source = DATA / "diabetes130.csv"
    target = "readmit_30_days"
    df = pd.read_csv(source)
    y = df.loc[:, target].copy()
    df.drop(columns=target, inplace=True)
    ages = {"30 years or younger": 0, "30-60 years": 1, "Over 60 years": 2}
    bools = df.select_dtypes("bool").astype(int)
    bools["diabetesMed"] = df["diabetesMed"].apply(lambda x: 0 if x == "No" else 1)
    bools["change"] = df["change"].apply(lambda x: 0 if x == "No" else 1)
    cats = df.select_dtypes("object").drop(columns=["age", "diabetesMed", "change"])
    cats = pd.get_dummies(cats)
    cats = reduce_categoricals(
        cats, dataset=Dataset.Diabetes130, n_components=n_components
    )

    ords = df.select_dtypes("int64")
    ords["age"] = df["age"].apply(lambda age: ages[age])
    ords = standardize(ords)

    df = pd.concat([ords, cats, bools], axis=1)
    df_all = df.copy()
    df_all["target"] = y
    df_all.to_json(preprocessed)
    return df, y


def load_uti_resistance(
    style: Literal["binary", "ordinal", "multi"] = "binary"
) -> Tuple[DataFrame, Series]:
    preprocessed = DATA / "uti_preprocessed.json"
    if preprocessed.exists():
        df = pd.read_json(preprocessed)
        y = df["target"].copy().astype(int)
        df = df.drop(columns="target")
        return df, y
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
    floats = [col for col in df.columns if len(np.unique(df[col])) > 2]
    bools = [col for col in df.columns if len(np.unique(df[col])) == 2]
    df_float = standardize(df[floats])
    df_bool = df[bools]
    df = pd.concat([df_float, df_bool], axis=1)
    df_all = df.copy()
    df_all["target"] = y
    df_all.to_json(preprocessed)
    return df, y


def load_uti_reduced(
    n_components: int, style: Literal["binary", "ordinal", "multi"] = "binary"
) -> Tuple[DataFrame, Series]:
    from src.enumerables import Dataset

    preprocessed = DATA / f"uti_preprocessed_cont{n_components}.parquet"
    if preprocessed.exists():
        df = pd.read_parquet(preprocessed).dropna(axis=0)  # one sample
        y = df["target"].copy().astype(np.int64)
        df = df.drop(columns="target")
        return df, y
    features = pd.read_csv(DATA / "uti_resistance/all_uti_features.csv")
    labels = pd.read_csv(DATA / "uti_resistance/all_uti_resist_labels.csv")
    # prescs = DATA / "uti_resitance/all_prescriptions.csv" not needed
    drops = ["is_train", "uncomplicated", "example_id"]
    targets = ["NIT", "SXT", "CIP", "LVX"]
    # could do one for each, or combine to get "any resistance"
    df = pd.merge(features.drop(columns=drops[:-1]), labels, on="example_id", how="inner")
    df.dropna(axis=0, inplace=True)
    df_targets = df.loc[:, targets].copy().astype(np.int64)
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
    floats = [col for col in df.columns if len(np.unique(df[col])) > 2]
    bools = [col for col in df.columns if len(np.unique(df[col])) == 2]
    df_float = standardize(df[floats])
    df_bool = df[bools].copy()
    df_bool = reduce_categoricals(
        df_bool, dataset=Dataset.UTIResistance, n_components=n_components, n_neighbours=30
    )

    df = pd.concat(
        [
            df_float.reset_index().drop(columns="index"),
            df_bool.reset_index().drop(columns="index"),
        ],
        axis=1,
    )
    df_all = df.copy()
    df_all["target"] = y.to_numpy()
    df_all.to_parquet(preprocessed)
    return df, y


def load_mimic_iv(n_components: Optional[int] = None) -> Tuple[DataFrame, Series]:
    from src.enumerables import Dataset

    """Could uses edstays.csv disposition == "EXPIRED" as a binary class
    See https://icd.who.int/browse10/2019/en#/I to convert ICD-10 codes to
    broad categories and
    https://www2.gov.bc.ca/gov/content/health/practitioner-professional-resources/msp/physicians/diagnostic-code-descriptions-icd-9
    for ICD-9

    But probably better to use triage.csv data + demographic data to predict leave, admitted, expired

    """
    root = DATA / "mimic-iv-ed"
    preproc = (
        DATA / "mimic_iv_preprocessed.json"
        if n_components is None
        else DATA / f"mimic_iv_preproc_cont{n_components}.parquet"
    )
    if preproc.exists():
        # use the version that can be uploaded to CC safely
        df = pd.read_json(preproc) if n_components is None else pd.read_parquet(preproc)
        target = df["target"].copy()
        df.drop(columns="target", inplace=True)
        return df, target

    triage = pd.read_csv(root / "triage.csv").rename(columns={"pain": "pain_sr"})
    stays = pd.read_csv(root / "edstays.csv")
    drops = ["subject_id_x", "subject_id_y", "hadm_id", "stay_id", "intime", "outtime"]
    cat_cols = ["gender", "race", "arrival_transport"]
    y_cats = {
        # 0 = will be untreated
        # 1 = receives urgent care
        # 2 = seen, but not critical
        "left without being seen": 0,
        "left against medical advice": 0,
        "eloped": 0,
        "admitted": 1,  # "urgent treatment"
        "transfer": 1,  # "urgent treatment"
        "home": 2,
        "expired": 3,
        "other": 4,
    }
    # 30 most common entries for triage "pain" column. My conversions here are
    # not psychometrically valid at all, but will have to do
    pains = {
        "0": 0.0,
        "0.5": 0.5,
        "1": 1.0,
        "1-2": 1.5,
        "2": 2.0,
        "3": 3.0,
        "3-4": 3.5,
        "4": 4,
        "4-5": 4.5,
        "5": 5.0,
        "5-6": 5.5,
        "6": 6.0,
        "6-7": 6.5,
        "7": 7.0,
        "7-8": 7.5,
        "8": 8.0,
        "8-9": 8.5,
        "8.5": 8.5,
        "9": 9.0,
        "9.5": 9.5,
        "10": 10.0,
        "10 ": 10.0,
        "11": 11.0,
        "12": 12.0,
        "13": 13.0,
        "20": 20.0,
        "Critical": 20.0,
        "denies": 0.0,
    }

    # this is still about 93.8% of the triage data
    pain_idx = triage["pain_sr"].apply(lambda x: x in pains)  # numeric pains
    triage = triage.loc[pain_idx]
    triage["pain_sr"] = triage["pain_sr"].apply(lambda x: pains[x])
    stays = stays[pain_idx]

    df = pd.merge(triage, stays, on="stay_id", how="inner")
    df.drop(columns=drops, inplace=True)
    df.dropna(axis=0, inplace=True)
    y = df["disposition"].str.lower().apply(lambda s: y_cats[s])

    # reduce to most common cases
    idx = y <= 2
    df = df.loc[idx].copy()
    y = y[idx].copy()

    complaints = df.chiefcomplaint.str.lower()
    df.drop(columns="chiefcomplaint", inplace=True)
    cats = df.loc[:, cat_cols].copy()
    df.drop(columns=cat_cols, inplace=True)
    df.drop(columns="disposition", inplace=True)
    # just some of most common words included in "complaint" column, reducing some replicates
    # fmt: off
    words = [
        "abd", "abnormal", "allergic", "ankle", "anxiety", "arm", "back", "bleeding",
        "body", "brbpr", "changes", "chest", "cough", "cough,", "diarrhea", "dizziness",
        "dyspnea", "dysuria", "ear", "epigastric", "etoh", "eval", "eye", "facial",
        "fall", "fever", "finger", "flank", "foot", "hand", "head", "headache",
        "hematuria", "hip", "hypertension", "ili", "injury", "knee", "labs", "laceration",
        "left", "leg", "llq", "lower", "mental", "mvc", "n/v", "nausea", "neck",
        "numbness", "pain", "palpitations", "rash", "reaction", "right", "seizure",
        "shoulder", "sore", "swelling", "syncope", "throat", "transfer", "unable",
        "urinary", "vaginal", "visual", "vomiting", "weakness", "wound", "wrist"
    ]
    # fmt: on
    word_cols = []
    for word in tqdm(words, desc="Converting complaints to one-hot"):
        word_cols.append(Series(complaints.apply(lambda s: int(word in s)), name=word))
    complaints = pd.concat(word_cols, axis=1)
    cats = pd.get_dummies(cats)
    if n_components is not None:
        cats = reduce_categoricals(cats, Dataset.MimicIV, n_components=n_components)
    df = pd.concat(
        [
            standardize(df).reset_index().drop(columns="index"),
            cats.reset_index().drop(columns="index"),
            complaints.reset_index().drop(columns="index"),
        ],
        axis=1,
    )
    js = df.copy()
    js["target"] = y.to_numpy()
    js.to_json(preproc) if n_components is None else js.to_parquet(preproc)

    return df, y


class SGDHparams:
    def __init__(
        self,
        eta0: float = 0.1,
        alpha: float = 1e-4,
        max_iter: int = 1500,
        early_stopping: bool = False,
    ) -> None:
        self.loss = "log_loss"
        self.max_iter = max_iter
        self.shuffle = True
        self.learning_rate = "adaptive"
        self.eta0 = eta0
        self.early_stopping = early_stopping
        # self.alpha=1e-4,  # pretty decent
        self.alpha = alpha  # mimic needs alpha < 1e-3, others benefit from >=1e-3

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            loss=self.loss,
            max_iter=self.max_iter,
            shuffle=self.shuffle,
            learning_rate=self.learning_rate,
            eta0=self.eta0,
            early_stopping=self.early_stopping,
            alpha=self.alpha,
        )


class ApproximateSVM(ABC):
    """These all do well on diabetes130 with minimal tuning"""

    def __init__(
        self,
        gamma: Optional[float] = None,
        n_components: int = 100,
        sgd: Optional[SGDHparams] = None,
    ) -> None:
        if sgd is None:
            sgd = SGDHparams()
        sgd.loss = "hinge"  # force implementation of linear SVM
        # sgd.loss = "modified_huber"  # seems not good
        # sgd.loss = "squared_hinge"  # no good
        self.gamma = gamma
        self.n_components = n_components
        self.sgd_hps = sgd
        self.classifier = SGDClassifier(**self.sgd_hps.as_dict())
        self.kernel_approximator: Union[Nystroem, FourierApproximator]

    @abstractmethod
    def fit(self, X: DataFrame, y: Series) -> None:
        Xt = self.kernel_approximator.fit_transform(X=X)
        self.classifier.fit(Xt, y)

    def score(self, X: DataFrame, y: Series) -> float:
        Xt = self.kernel_approximator.fit_transform(X=X)
        return float(self.classifier.score(Xt, y))


class RandomFourierSVM(ApproximateSVM):
    def __init__(
        self,
        gamma: Optional[float] = None,
        n_components: int = 100,
        sgd: Optional[SGDHparams] = None,
    ) -> None:
        super().__init__(gamma=gamma, n_components=n_components, sgd=sgd)

    def fit(self, X: DataFrame, y: Series) -> None:
        if X.shape[1] < self.n_components:
            self.n_components = X.shape[1]
        self.kernel_approximator = FourierApproximator(
            gamma=self.gamma or "scale",  # type: ignore
            n_components=self.n_components,
        )
        Xt = self.kernel_approximator.fit_transform(X=X)
        self.classifier.fit(Xt, y)


if __name__ == "__main__":
    # df = load_diabetes130_continuous(n_components=5)
    # df = load_uti_reduced(n_components=100)
    df = load_mimic_iv(n_components=5)
    sys.exit()

    datasets: Dict[str, Callable[[], Tuple[DataFrame, Series]]] = {
        "diabetes": load_diabetes,
        "park": load_park,
        "trans": load_trans,
        "spect": load_SPECT,
        "heart-failure": load_heart_failure,  # all instant
        "diabetes-130": load_diabetes130,  # all under 2mins even with 1 core, RF and GBT under 10s
        "uti-resist": load_uti_resistance,  # LR=<1min, SVC=, RF=<1min@1core, GBT=<20s
        "mimic-iv": load_mimic_iv,  # LR=<2min, RF=<90s@1core , GBT=<1min
    }
    X, y = load_mimic_iv()

    for loader in datasets.values():
        loader()
    sys.exit()
