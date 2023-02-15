from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from time import strftime, time
from typing import (
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import HistGradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit as SSSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm
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
    df = pd.concat([floats, cats, bools], axis=1)
    return df, y


def load_diabetes130() -> Tuple[DataFrame, Series]:
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
    df = pd.concat([ords, cats, bools], axis=1)

    # TODO: one-hot encode categoricals
    return df, y


def load_uti_resistance(
    style: Literal["binary", "ordinal", "multi"] = "binary"
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


def load_mimic_iv() -> Tuple[DataFrame, Series]:
    """Could uses edstays.csv disposition == "EXPIRED" as a binary class
    See https://icd.who.int/browse10/2019/en#/I to convert ICD-10 codes to
    broad categories and
    https://www2.gov.bc.ca/gov/content/health/practitioner-professional-resources/msp/physicians/diagnostic-code-descriptions-icd-9
    for ICD-9

    But probably better to use triage.csv data + demographic data to predict leave, admitted, expired

    """
    root = DATA / "mimic-iv-ed"
    preproc = root / "preproc.json"
    if preproc.exists():
        # use the version that can be uploaded to CC safely
        df = pd.read_json(preproc)
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
    df = pd.concat([df, cats, complaints], axis=1)
    js = df.copy()
    js["target"] = y
    js.to_json(preproc)

    return df, y


if __name__ == "__main__":
    classifiers = {
        "lr": lambda: LR(
            solver="sag", max_iter=1000
        ),  # time seems to double with sample double, O(n)
        # "rf": lambda: RF(n_jobs=1),
        # "gbt": lambda: GBC(),
        # "svc": lambda: SVC(),  # really bad this one
    }
    datasets: Dict[str, Callable[[], Tuple[DataFrame, Series]]] = {
        "heart-failure": load_heart_failure,  # all instant
        "diabetes-130": load_diabetes130,  # all under 2mins even with 1 core, RF and GBT under 10s
        "uti-resist": load_uti_resistance,  # LR=<1min, SVC=, RF=<1min@1core, GBT=<20s
        "mimic-iv": load_mimic_iv,  # LR=<2min, RF=<90s@1core , GBT=<1min
    }
    """
    ==============================================================================
    Estimating runtimes for heart-failure with RF
    Fitting rf@N=2008 took:  0.74 s (i.e. 0.0 minutes).   {acc=0.6443}
    ------------------------------------------------------------------------------
    Estimating runtimes for heart-failure with GBT
    Fitting gbt@N=2008 took: 0.97 s (i.e. 0.0 minutes).   {acc=0.6567}
    ------------------------------------------------------------------------------
    Estimating runtimes for heart-failure with LR
    Fitting lr@N=2008 took:  1.67 s (i.e. 0.0 minutes).   {acc=0.5995} - ConvergenceWarning

    ==============================================================================
    Estimating runtimes for diabetes-130 with RF
    Fitting rf@N=5000 took:   0.36 s (i.e. 0.0 minutes).  {acc=0.8890}
    Fitting rf@N=10000 took:  0.69 s (i.e. 0.0 minutes).  {acc=0.8860}
    Fitting rf@N=20000 took:  1.37 s (i.e. 0.0 minutes).  {acc=0.8860}
    Fitting rf@N=40000 took:  2.82 s (i.e. 0.0 minutes).  {acc=0.8894}
    Fitting rf@N=80000 took:  5.94 s (i.e. 0.1 minutes).  {acc=0.8873}
    ------------------------------------------------------------------------------
    Estimating runtimes for diabetes-130 with GBT
    Fitting gbt@N=5000 took:  1.12 s (i.e. 0.0 minutes).  {acc=0.8960}
    Fitting gbt@N=10000 took: 1.28 s (i.e. 0.0 minutes).  {acc=0.8835}
    Fitting gbt@N=20000 took: 0.72 s (i.e. 0.0 minutes).  {acc=0.8895}
    Fitting gbt@N=40000 took: 0.79 s (i.e. 0.0 minutes).  {acc=0.8890}
    Fitting gbt@N=80000 took: 1.40 s (i.e. 0.0 minutes).  {acc=0.8890}
    ------------------------------------------------------------------------------
    Estimating runtimes for diabetes-130 with LR
    Fitting lr@N=5000 took:   1.49 s (i.e. 0.0 minutes).  {acc=0.8870} - ConvergenceWarning
    Fitting lr@N=10000 took:  3.03 s (i.e. 0.1 minutes).  {acc=0.8870} - ConvergenceWarning
    Fitting lr@N=20000 took:  3.85 s (i.e. 0.1 minutes).  {acc=0.8882}
    Fitting lr@N=40000 took:  6.41 s (i.e. 0.1 minutes).  {acc=0.8885}
    Fitting lr@N=80000 took:  5.44 s (i.e. 0.1 minutes).  {acc=0.8883}

    ==============================================================================
    Estimating runtimes for uti-resist with RF
    Fitting rf@N=5000 took:   1.07 s (i.e. 0.0 minutes).  {acc=0.6070}
    Fitting rf@N=10000 took:  3.05 s (i.e. 0.1 minutes).  {acc=0.6625}
    Fitting rf@N=20000 took:  7.51 s (i.e. 0.1 minutes).  {acc=0.6585}
    Fitting rf@N=40000 took: 15.96 s (i.e. 0.3 minutes).  {acc=0.6626}
    Fitting rf@N=80000 took: 35.64 s (i.e. 0.6 minutes).  {acc=0.6613}
    ------------------------------------------------------------------------------
    Estimating runtimes for uti-resist with GBT
    Fitting gbt@N=5000 took:   5.03 s (i.e. 0.1 minutes). {acc=0.6570}
    Fitting gbt@N=10000 took:  6.34 s (i.e. 0.1 minutes). {acc=0.6540}
    Fitting gbt@N=20000 took:  3.54 s (i.e. 0.1 minutes). {acc=0.6645}
    Fitting gbt@N=40000 took:  6.88 s (i.e. 0.1 minutes). {acc=0.6715}
    Fitting gbt@N=80000 took: 13.51 s (i.e. 0.2 minutes). {acc=0.6693}
    ------------------------------------------------------------------------------
    Estimating runtimes for uti-resist with LR
    Fitting lr@N=5000 took:   20.64 s (i.e. 0.3 minutes). {acc=0.6430} - ConvergenceWarning
    Fitting lr@N=10000 took:  41.67 s (i.e. 0.7 minutes). {acc=0.6695} - ConvergenceWarning
    Fitting lr@N=20000 took:  84.40 s (i.e. 1.4 minutes). {acc=0.6577} - ConvergenceWarning
    Fitting lr@N=40000 took: 185.79 s (i.e. 3.1 minutes). {acc=0.6569} - ConvergenceWarning
    Fitting lr@N=80000 took: 364.93 s (i.e. 6.1 minutes). {acc=0.6705} - ConvergenceWarning


    ==============================================================================
    Estimating runtimes for mimic-iv with RF
    Fitting rf@N=5000 took:    0.53 s (i.e. 0.0 minutes).  {acc=0.7120}
    Fitting rf@N=10000 took:   1.08 s (i.e. 0.0 minutes).  {acc=0.7260}
    Fitting rf@N=20000 took:   2.18 s (i.e. 0.0 minutes).  {acc=0.7185}
    Fitting rf@N=40000 took:   4.54 s (i.e. 0.1 minutes).  {acc=0.7325}
    Fitting rf@N=80000 took:  10.51 s (i.e. 0.2 minutes).  {acc=0.7298}
    Fitting rf@N=160000 took: 24.24 s (i.e. 0.4 minutes).  {acc=0.7263}
    Fitting rf@N=320000 took: 58.42 s (i.e. 1.0 minutes).  {acc=0.7320}
    ------------------------------------------------------------------------------
    Estimating runtimes for mimic-iv with GBT
    Fitting gbt@N=5000 took:    2.70 s (i.e. 0.0 minutes). {acc=0.7080}
    Fitting gbt@N=10000 took:   2.97 s (i.e. 0.0 minutes). {acc=0.7100}
    Fitting gbt@N=20000 took:   1.56 s (i.e. 0.0 minutes). {acc=0.7358}
    Fitting gbt@N=40000 took:   1.93 s (i.e. 0.0 minutes). {acc=0.7328}
    Fitting gbt@N=80000 took:   6.50 s (i.e. 0.1 minutes). {acc=0.7434}
    Fitting gbt@N=160000 took: 12.47 s (i.e. 0.2 minutes). {acc=0.7374}
    Fitting gbt@N=320000 took: 19.91 s (i.e. 0.3 minutes). {acc=0.7413}
    """
    for dsname, loader in datasets.items():
        for classifier, fitter in classifiers.items():
            print("=" * 80)
            print(f"Estimating runtimes for {dsname} with {classifier.upper()}")
            X, y = loader()
            N = min(5000, len(X))
            while N <= len(X):
                idx = np.random.permutation(len(X))[:N]
                Xm, ym = X.iloc[idx, :], y.iloc[idx]
                splitter = SSSplit(n_splits=1, test_size=0.2)
                idx_train, idx_test = next(splitter.split(Xm, ym))
                X_tr, y_tr = Xm.iloc[idx_train], ym.iloc[idx_train]
                X_test, y_test = Xm.iloc[idx_test], ym.iloc[idx_test]
                model = fitter()
                # print(f"Starting fit for N={N} samples at ")
                start = time()
                model.fit(X_tr, y_tr)
                duration = time() - start
                print(
                    f"Fitting {classifier}@N={N} took: {duration:0.2f} s "
                    f"(i.e. {duration / 60:0.1f} minutes). "
                    f"[Finished at {strftime('%c')}]"
                    f"{{acc={model.score(X_test, y_test):0.4f}}}"
                )
                N *= 2
