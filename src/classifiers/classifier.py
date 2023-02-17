from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from numpy.random import Generator
from pandas import DataFrame, Series
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier, XGBRFClassifier

from src.enumerables import Metric
from src.hparams.hparams import Hparams


class Classifier(ABC):
    def __init__(self, hparams: Hparams) -> None:
        self.hparams = hparams
        self.classifier: Union[SGDClassifier, XGBRFClassifier, XGBClassifier]

    @abstractmethod
    def fit(self, X: DataFrame, y: Series, rng: Optional[Generator]) -> None:
        ...

    def score(self, X: DataFrame, y: Series, metric: Metric = Metric.Accuracy) -> float:
        y_pred = self.classifier.predict(X)
        return metric.compute(y_true=y, y_pred=y_pred)
