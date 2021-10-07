from typing import List, Tuple, Union

import numpy as np
from numpy import ndarray
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def random_preds(
    length: int, n_classes: int = 10, n_preds: int = 1, dims: int = 1
) -> Tuple[Union[ndarray, List[ndarray]], ndarray]:
    if dims not in [1, 2]:
        raise ValueError("Invalid dimension")

    if n_preds == 1:
        pred_labels = np.random.randint(0, n_classes, length, dtype=int)
        true_labels = np.roll(pred_labels, 10)
        labels = np.arange(n_classes)
        if dims == 1:
            le = LabelEncoder().fit(labels)
            return le.transform(pred_labels), le.transform(true_labels)
        if dims == 2:
            onehot = OneHotEncoder(sparse=False, dtype=int).fit(labels.reshape(len(labels), 1))
            return (
                onehot.transform(pred_labels.reshape(-1, 1)),
                onehot.transform(true_labels.reshape(-1, 1)),
            )

    pred_labels = [np.random.randint(0, n_classes, length, dtype=int) for _ in range(n_preds)]
    true_labels = np.roll(pred_labels[0], 10)
    labels = np.arange(n_classes)
    if dims == 1:
        le = LabelEncoder().fit(labels)
        preds = [le.transform(pred_label) for pred_label in pred_labels]
        return preds, le.transform(true_labels)
    if dims == 2:
        onehot = OneHotEncoder(sparse=False, dtype=int).fit(labels.reshape(len(labels), 1))
        preds = [onehot.transform(pred_label.reshape(-1, 1)) for pred_label in pred_labels]
        return preds, onehot.transform(true_labels.reshape(-1, 1))
    raise ValueError("Invalid number of dimensions.")
