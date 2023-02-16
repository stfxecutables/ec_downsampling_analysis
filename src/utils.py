from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from typing import Dict, Optional, Type, Union

from numpy.random import Generator

from src.classifiers.classifier import Classifier
from src.classifiers.gbt import XGBoostClassifier
from src.classifiers.logistic import LogisticRegression
from src.classifiers.nystroem import NystroemSVM
from src.classifiers.rf import XGBoostRFClassifier
from src.enumerables import ClassifierKind
from src.hparams.gbt import XGBoostHparams
from src.hparams.hparams import Hparams
from src.hparams.logistic import SGDLRHparams
from src.hparams.nystroem import NystroemHparams
from src.hparams.rf import XGBRFHparams


def get_classifier(kind: ClassifierKind) -> Type[Classifier]:
    kinds: Dict[ClassifierKind, Type[Classifier]] = {
        ClassifierKind.GBT: XGBoostClassifier,
        ClassifierKind.LR: LogisticRegression,
        ClassifierKind.RF: XGBoostRFClassifier,
        ClassifierKind.SVM: NystroemSVM,
    }
    return kinds[kind]


def get_rand_hparams(kind: ClassifierKind, rng: Optional[Generator]) -> Hparams:
    kinds: Dict[ClassifierKind, Hparams] = {
        ClassifierKind.GBT: XGBoostHparams(),
        ClassifierKind.LR: SGDLRHparams(),
        ClassifierKind.RF: XGBRFHparams(),
        ClassifierKind.SVM: NystroemHparams(),
    }
    hps = kinds[kind]
    return hps.random(rng)
