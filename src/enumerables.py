from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from typing_extensions import Literal

from src.loading import (
    load_diabetes,
    load_diabetes130,
    load_diabetes130_continuous,
    load_heart_failure,
    load_mimic_iv,
    load_park,
    load_SPECT,
    load_trans,
    load_uti_reduced,
    load_uti_resistance,
)


class Dataset(Enum):
    Diabetes = "diab"
    Transfusion = "trans"
    Parkinsons = "park"
    SPECT = "spect"
    Diabetes130 = "diab130"
    Diabetes130Reduced = "diab130-reduced"
    HeartFailure = "heart"
    MimicIV = "mimic"
    MimicIVReduced = "mimic-reduced"
    UTIResistance = "uti"
    UTIResistanceReduced = "uti-reduced"

    def load(self) -> Tuple[DataFrame, Series]:
        loaders: Dict[Dataset, Callable[[], Tuple[DataFrame, Series]]] = {
            Dataset.Diabetes: load_diabetes,
            Dataset.Transfusion: load_trans,
            Dataset.Parkinsons: load_park,
            Dataset.SPECT: load_SPECT,
            Dataset.Diabetes130: load_diabetes130,
            Dataset.Diabetes130Reduced: lambda: load_diabetes130_continuous(
                self.n_reduce()
            ),
            Dataset.HeartFailure: load_heart_failure,
            Dataset.MimicIV: lambda: load_mimic_iv(),
            Dataset.MimicIVReduced: lambda: load_mimic_iv(n_components=self.n_reduce()),
            Dataset.UTIResistance: load_uti_resistance,
            Dataset.UTIResistanceReduced: lambda: load_uti_reduced(
                n_components=self.n_reduce()
            ),
        }
        return loaders[self]()

    def n_categoricals(self) -> int:
        counts: Dict[Dataset, int] = {
            Dataset.Diabetes: -1,
            Dataset.Transfusion: -1,
            Dataset.Parkinsons: -1,
            Dataset.SPECT: -1,
            Dataset.Diabetes130: 35,
            Dataset.Diabetes130Reduced: 35,
            Dataset.HeartFailure: 0,
            Dataset.MimicIV: 40,
            Dataset.MimicIVReduced: 40,
            Dataset.UTIResistance: 713,
            Dataset.UTIResistanceReduced: 713,
        }
        return counts[self]

    def n_reduce(self) -> int:
        counts: Dict[Dataset, int] = {
            Dataset.Diabetes: -1,
            Dataset.Transfusion: -1,
            Dataset.Parkinsons: -1,
            Dataset.SPECT: -1,
            Dataset.Diabetes130: 5,
            Dataset.Diabetes130Reduced: 5,
            Dataset.HeartFailure: 0,
            Dataset.MimicIV: 5,
            Dataset.MimicIVReduced: 5,
            Dataset.UTIResistance: 100,
            Dataset.UTIResistanceReduced: 100,
        }
        return counts[self]

    @staticmethod
    def fast() -> List[Dataset]:
        return [
            Dataset.Diabetes,
            Dataset.Transfusion,
            Dataset.Parkinsons,
            Dataset.SPECT,
            Dataset.HeartFailure,
            Dataset.Diabetes130,
        ]

    @staticmethod
    def very_fast() -> List[Dataset]:
        return [
            Dataset.Diabetes,
            Dataset.Transfusion,
            Dataset.Parkinsons,
            Dataset.SPECT,
        ]

    @staticmethod
    def slow() -> List[Dataset]:
        return [
            Dataset.MimicIV,
            Dataset.UTIResistance,
        ]


class ClassifierKind(Enum):
    GBT = "gbt"
    LR = "lr"
    RF = "rf"
    SVM = "svm"


class Metric(Enum):
    Accuracy = "acc"
    F1 = "f1"
    BalancedAccuracy = "acc-bal"

    def compute(
        self, y_true: Union[ndarray, Series], y_pred: Union[ndarray, Series]
    ) -> float:
        computer = {
            Metric.Accuracy: accuracy_score,
            Metric.F1: lambda y_true, y_pred: f1_score(
                y_true=y_true, y_pred=y_pred, average="weighted"
            ),
            Metric.BalancedAccuracy: balanced_accuracy_score,
        }[self]
        return float(computer(y_true, y_pred))
