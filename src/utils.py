from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from typing import Dict, Optional, Type, Union, overload

from numpy.random import Generator
from typing_extensions import Literal

from src.classifiers.classifier import Classifier
from src.classifiers.gbt import XGBoostClassifier
from src.classifiers.logistic import LogisticRegression
from src.classifiers.nystroem import NystroemSVM
from src.classifiers.rf import XGBoostRFClassifier
from src.constants import HP_OUTDIR
from src.enumerables import ClassifierKind, Dataset
from src.hparams.gbt import XGBoostHparams
from src.hparams.hparams import Hparams
from src.hparams.logistic import SGDLRHparams
from src.hparams.nystroem import NystroemHparams
from src.hparams.rf import XGBRFHparams


@overload
def get_classifier(kind: Literal[ClassifierKind.GBT]) -> Type[XGBoostClassifier]:
    ...


@overload
def get_classifier(kind: Literal[ClassifierKind.LR]) -> Type[LogisticRegression]:
    ...


@overload
def get_classifier(kind: Literal[ClassifierKind.RF]) -> Type[XGBoostRFClassifier]:
    ...


@overload
def get_classifier(kind: Literal[ClassifierKind.SVM]) -> Type[NystroemSVM]:
    ...


def get_classifier(kind: ClassifierKind) -> Type[Classifier]:
    kinds: Dict[ClassifierKind, Type[Classifier]] = {
        ClassifierKind.GBT: XGBoostClassifier,
        ClassifierKind.LR: LogisticRegression,
        ClassifierKind.RF: XGBoostRFClassifier,
        ClassifierKind.SVM: NystroemSVM,
    }
    return kinds[kind]


@overload
def get_rand_hparams(
    kind: Literal[ClassifierKind.GBT], rng: Optional[Generator]
) -> XGBoostHparams:
    ...


@overload
def get_rand_hparams(
    kind: Literal[ClassifierKind.LR], rng: Optional[Generator]
) -> SGDLRHparams:
    ...


@overload
def get_rand_hparams(
    kind: Literal[ClassifierKind.RF], rng: Optional[Generator]
) -> XGBRFHparams:
    ...


@overload
def get_rand_hparams(
    kind: Literal[ClassifierKind.SVM], rng: Optional[Generator]
) -> NystroemHparams:
    ...


def get_rand_hparams(kind: ClassifierKind, rng: Optional[Generator]) -> Hparams:
    kinds: Dict[ClassifierKind, Hparams] = {
        ClassifierKind.GBT: XGBoostHparams(),
        ClassifierKind.LR: SGDLRHparams(),
        ClassifierKind.RF: XGBRFHparams(),
        ClassifierKind.SVM: NystroemHparams(),
    }
    hps = kinds[kind]
    return hps.random(rng)


def tuning_outdir(dataset: Dataset, kind: ClassifierKind) -> Path:
    return HP_OUTDIR / f"{dataset.value}/{kind.value}/best"


def is_tuned(dataset: Dataset, kind: ClassifierKind) -> bool:
    root = tuning_outdir(dataset=dataset, kind=kind)
    return len(list(root.rglob("*.json"))) > 0


def load_tuning_params(dataset: Dataset, kind: ClassifierKind) -> Hparams:
    root = tuning_outdir(dataset=dataset, kind=kind)
    kinds: Dict[ClassifierKind, Type[Hparams]] = {
        ClassifierKind.GBT: XGBoostHparams,
        ClassifierKind.LR: SGDLRHparams,
        ClassifierKind.RF: XGBRFHparams,
        ClassifierKind.SVM: NystroemHparams,
    }
    hp = kinds[kind]
    return hp.from_json(root)


def save_tuning_params(dataset: Dataset, kind: ClassifierKind, hps: Hparams) -> None:
    root = tuning_outdir(dataset=dataset, kind=kind)
    hps.to_json(root)
    print(f"Saved best hparams for {kind.name} on {dataset.name} to {root}")
