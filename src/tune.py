from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

import numpy as np

from src.classifiers.classifier import Classifier
from src.enumerables import ClassifierKind, Dataset
from src.utils import get_classifier, get_rand_hparams


def random_tune(classifier: Classifier, dataset: Dataset) -> None:
    X, y = dataset.load()


if __name__ == "__main__":
    rng = np.random.default_rng()
    for dataset in Dataset:
        for kind in ClassifierKind:
            cls = get_classifier(kind)
            hps = get_rand_hparams(kind, rng=rng)
            classifier = cls(hps)
            random_tune(classifier=classifier, dataset=dataset)
