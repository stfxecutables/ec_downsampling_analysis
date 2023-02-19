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
from numpy.random import Generator
from pandas import DataFrame, Series
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier

from src.classifiers.classifier import Classifier
from src.enumerables import Metric
from src.hparams.nystroem import NystroemHparams


class NystroemSVM(Classifier):
    def __init__(
        self,
        hparams: NystroemHparams,
    ) -> None:
        self.hparams: NystroemHparams = hparams
        self.classifier = SGDClassifier(**self.hparams.sgd_dict())
        self.kernel_approximator: Nystroem

    def fit(self, X: DataFrame, y: Series, rng: Optional[Generator]) -> None:
        if rng is None:
            rng = np.random.default_rng()
        ny_args = self.hparams.ny_dict()
        n_components = ny_args["n_components"]
        if X.shape[1] < n_components:
            n_components = X.shape[1]
        self.kernel_approximator = Nystroem(
            kernel="rbf",
            gamma=ny_args["gamma"],
            n_components=n_components,
        )
        Xt = self.kernel_approximator.fit_transform(X=X.to_numpy())
        self.classifier.fit(Xt, y.to_numpy())

    def score(self, X: DataFrame, y: Series, metric: Metric = Metric.Accuracy) -> float:
        Xt = self.kernel_approximator.fit_transform(X=X.to_numpy())
        y_pred = self.classifier.predict(Xt)
        return metric.compute(y_true=y.to_numpy(), y_pred=y_pred)
