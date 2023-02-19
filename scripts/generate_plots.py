from __future__ import annotations

# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent  # isort: skip
sys.path.append(str(ROOT))  # isort: skip
# fmt: on

from src.analysis import summarize_results
from src.enumerables import ClassifierKind, Dataset

if __name__ == "__main__":
    for dataset in [
        # Dataset.Diabetes,
        # Dataset.Parkinsons,
        # Dataset.SPECT,
        # Dataset.Transfusion,
        # Dataset.HeartFailure,
        Dataset.Diabetes130,
    ]:
        for kind in ClassifierKind:
            summarize_results(dataset=dataset, kind=kind, downsample=True)
