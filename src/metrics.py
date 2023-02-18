from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on


import os
import re
import sys
import traceback
from argparse import Namespace
from functools import reduce
from itertools import combinations
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from joblib import Parallel, delayed

# from irrCAC.raw import CAC
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy import ndarray
from numpy.random import Generator
from numpy.typing import ArrayLike
from pandas import DataFrame, Series
from scipy.stats.contingency import association, crosstab
from scipy.stats.distributions import uniform
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import confusion_matrix as confusion
from sklearn.model_selection import ParameterGrid, ParameterSampler
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal


def acc(y1: ndarray, y2: ndarray) -> float:
    return np.mean(y1 == y2)


def ec_local(y: ndarray, y1: ndarray, y2: ndarray) -> float:
    e1 = y1 != y
    e2 = y2 != y
    inter = e1 & e2
    union = e1 | e2
    if np.all(~union):
        return 0.0
    return float(np.sum(inter) / np.sum(union))


def ecs(preds: DataFrame, y: Series) -> List[float]:
    ys = [preds.iloc[:, i].to_numpy() for i in range(preds.shape[1])]
    return [ec_local(y, *yy) for yy in combinations(ys, 2)]


def accs(preds: DataFrame, y: Series) -> List[float]:
    ys = [preds.iloc[:, i].to_numpy() for i in range(preds.shape[1])]
    return [acc(y, yy) for yy in ys]


def acc_pairs(preds: DataFrame, y: Series) -> List[float]:
    yn = y.to_numpy()
    ys = [preds.iloc[:, i].to_numpy() for i in range(preds.shape[1])]
    return [
        float(np.mean([acc(yn, yy[0]), acc(yn, yy[1])])) for yy in combinations(ys, 2)
    ]
