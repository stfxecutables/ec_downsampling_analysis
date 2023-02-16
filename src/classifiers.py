from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast, no_type_check

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
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler as FourierApproximator
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit as SSSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from typing_extensions import Literal

from src.hparams import Hparams, SGDHparams


class Classifier(ABC):
    def __init__(self, hparams: Hparams) -> None:
        self.hparams = hparams

    @abstractmethod
    def fit(self, X: DataFrame, y: Series) -> None:
        ...

    @abstractmethod
    def score(self, X: DataFrame, y: Series) -> float:
        ...


class NystroemSVM(Classifier):
    def __init__(
        self,
        gamma: Optional[float] = None,
        n_components: int = 100,
        sgd: Optional[SGDHparams] = None,
    ) -> None:
        if sgd is None:
            sgd = SGDHparams()
        sgd.loss = "hinge"  # force implementation of linear SVM
        self.gamma = gamma
        self.n_components = n_components
        self.sgd_hps = sgd
        self.classifier = SGDClassifier(**self.sgd_hps.as_dict())
        self.kernel_approximator: Nystroem

    def fit(self, X: DataFrame, y: Series) -> None:
        if X.shape[1] < self.n_components:
            self.n_components = X.shape[1]
        self.kernel_approximator = Nystroem(
            kernel="rbf",
            gamma=self.gamma,
            n_components=self.n_components,
        )
        Xt = self.kernel_approximator.fit_transform(X=X)
        self.classifier.fit(Xt, y)

    def score(self, X: DataFrame, y: Series) -> float:
        Xt = self.kernel_approximator.fit_transform(X=X)
        return float(self.classifier.score(Xt, y))
