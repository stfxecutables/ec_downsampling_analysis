from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from numpy import ndarray
from numpy.random import Generator
from pandas import DataFrame, Series
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import SVC

from src.classifiers.classifier import Classifier
from src.enumerables import Metric
from src.hparams.svm import SVMHparams


class ClassicSVM(Classifier):
    def __init__(
        self,
        hparams: SVMHparams,
    ) -> None:
        self.hparams: SVMHparams = hparams
        self.classifier = SVC(**self.hparams.to_dict())

    def fit(self, X: DataFrame, y: Series, rng: Optional[Generator]) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.classifier.fit(X, y)

    def score(self, X: DataFrame, y: Series, metric: Metric = Metric.Accuracy) -> float:
        y_pred = self.classifier.predict(X)
        return metric.compute(y_true=y.to_numpy(), y_pred=y_pred)

    def predict(self, X: DataFrame) -> ndarray:
        return np.ravel(self.classifier.predict(X))
