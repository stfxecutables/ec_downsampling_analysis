from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from enum import Enum


class Dataset(Enum):
    Diabetes = "diab"
    Transfusion = "trans"
    Parkinsons = "park"
    SPECT = "spect"
    Diabetes130 = "diab130"
    HeartFailure = "heart"
    MimicIV = "mimic"
    UTIResistance = "uti"
