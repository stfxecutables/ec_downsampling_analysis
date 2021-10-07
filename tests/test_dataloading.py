import numpy as np
import pytest
from typing_extensions import Literal

from analysis.loading import load_diabetes, load_park, load_SPECT, load_trans

Dataset = Literal["Diabetes", "Transfusion", "Parkinsons", "SPECT"]


@pytest.mark.fast
def test_loading_functions() -> None:
    x, y = load_diabetes()
    assert x.shape == (768, 8)
    assert y.shape == (768,)
    assert len(np.unique(y)) == 2

    x, y = load_park()
    assert x.shape == (195, 22)
    assert y.shape == (195,)
    assert len(np.unique(y)) == 2

    x, y = load_trans()
    assert x.shape == (748, 4)
    assert y.shape == (748,)
    assert len(np.unique(y)) == 2

    x, y = load_SPECT()
    assert x.shape == (267, 22)
    assert y.shape == (267,)
    assert len(np.unique(y)) == 2
