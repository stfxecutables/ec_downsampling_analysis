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
from typing_extensions import Literal

from src.loading import (
    load_diabetes,
    load_diabetes130,
    load_heart_failure,
    load_mimic_iv,
    load_park,
    load_SPECT,
    load_trans,
    load_uti_resistance,
)


class Dataset(Enum):
    Diabetes = "diab"
    Transfusion = "trans"
    Parkinsons = "park"
    SPECT = "spect"
    Diabetes130 = "diab130"
    HeartFailure = "heart"
    MimicIV = "mimic"
    UTIResistance = "uti"

    def load(self) -> Tuple[DataFrame, Series]:
        loaders: Dict[Dataset, Callable[[], Tuple[DataFrame, Series]]] = {
            Dataset.Diabetes: load_diabetes,
            Dataset.Transfusion: load_trans,
            Dataset.Parkinsons: load_park,
            Dataset.SPECT: load_SPECT,
            Dataset.Diabetes130: load_diabetes130,
            Dataset.HeartFailure: load_heart_failure,
            Dataset.MimicIV: load_mimic_iv,
            Dataset.UTIResistance: load_uti_resistance,
        }
        return loaders[self]()


class ClassifierKind(Enum):
    LR = "lr"
    SVM = "svm"
    GBT = "gbt"
    RF = "rf"
