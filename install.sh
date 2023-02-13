#!/bin/bash
# if on MacOS with Apple Silicon, may need following to compile LightGBM
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade black mypy flake8 isort pytest numpy pandas matplotlib seaborn scipy scikit-learn xgboost lightgbm statsmodels tqdm typing_extensions