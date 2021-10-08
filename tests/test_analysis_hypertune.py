from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import ray
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame
from ray import tune
from ray.tune import Analysis
from sklearn.ensemble import AdaBoostClassifier as AdaBoost
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from typing_extensions import Literal

from analysis.loading import load_diabetes, load_park, load_SPECT, load_trans

Dataset = Literal["Diabetes", "Transfusion", "Parkinsons", "SPECT"]


def print_results(analysis: Analysis, cols: List[str], dataset: Dataset) -> DataFrame:
    df = analysis.dataframe("acc")
    df = df.loc[:, cols]
    df.sort_values(by="acc", ascending=False, inplace=True)
    renamed_cols = [col.replace("config/", "") for col in df.columns]
    df.columns = renamed_cols
    print("\n")
    print(df.to_markdown(tablefmt="pretty"))
    print(f"Best config for {dataset}: ", analysis.get_best_config(metric="acc", mode="max"))
    return df


def objective_function(model: Any, model_args: Dict = dict()) -> Callable:
    def objective(x: ndarray, y: ndarray, config: Dict) -> float:
        m = model(**model_args, **config)
        return float(np.mean(cross_val_score(m, x, y, cv=5)))
        # x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)
        # m = model(**model_args, **config)
        # m.fit(x_train, y_train)
        # return float(m.score(x_test, y_test))

    return objective


def train_function(objective: Callable, dataset: str) -> Callable:
    mapping = {
        "Diabetes": load_diabetes,
        "Parkinsons": load_park,
        "Transfusion": load_trans,
        "SPECT": load_SPECT,
    }
    x, y = mapping[dataset]()
    # x = StandardScaler().fit_transform(x, y)
    x = MinMaxScaler().fit_transform(x, y)

    def train(config: Dict) -> None:
        acc = objective(x, y, config)
        tune.report(acc=acc)

    return train


def test_mlp_params(capsys: Any) -> None:
    ray.init(num_cpus=8)
    for DATASET in ["Diabetes", "Transfusion", "Parkinsons", "SPECT"]:
        # for DATASET in ["Parkinsons", "SPECT"]:

        def objective(
            x: ndarray,
            y: ndarray,
            alpha: float,
            layer: int,
            # layer1: int,
            # layer2: int,
            # layer3: int,
            # layer4: int,
            # layer5: int,
            # layer6: int,
            iter: int,
        ) -> float:
            mlp = MLPClassifier(
                # (layer1, layer2, layer3, layer4, layer5, layer6),
                (layer, layer, layer, layer, layer, layer),
                batch_size=32,
                alpha=alpha,
                max_iter=iter,
            )
            return np.mean(cross_val_score(mlp, x, y, cv=5))
            x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)
            mlp = MLPClassifier(
                # (layer1, layer2, layer3, layer4, layer5, layer6),
                (layer, layer, layer, layer, layer, layer),
                batch_size=32,
                alpha=alpha,
                max_iter=iter,
            )
            mlp.fit(x_train, y_train)
            return float(mlp.score(x_test, y_test))

        def train(config: Any) -> None:
            x, y = load_diabetes()
            # x, y = load_trans()
            # x, y = load_park()
            # x, y = load_SPECT()
            x = StandardScaler().fit_transform(x, y)
            alpha = config["alpha"]
            layer = config["layer"]
            # layer1 = config["layer1"]
            # layer2 = config["layer2"]
            # layer3 = config["layer3"]
            # layer4 = config["layer4"]
            # layer5 = config["layer5"]
            # layer6 = config["layer6"]
            iters = config["iter"]
            # acc = objective(x, y, alpha, layer1, layer2, layer3, layer4, layer5, layer6, iters)
            acc = objective(x, y, alpha, layer, iters)
            tune.report(acc=acc)

        config = {
            "alpha": tune.qloguniform(1e-6, 1e-2, 5e-7),
            # "layer": tune.choice([4, 8, 16, 32, 64]),
            # "layer": tune.choice([4, 8, 16]),
            "layer": tune.choice([4, 8, 16, 32]),
            # "layer1": tune.choice([8, 16, 32, 64]),
            # "layer2": tune.choice([8, 16, 32, 64]),
            # "layer3": tune.choice([8, 16, 32, 64]),
            # "layer4": tune.choice([8, 16, 32, 64]),
            # "layer5": tune.choice([8, 16, 32, 64]),
            # "layer6": tune.choice([8, 16, 32, 64]),
            "iter": tune.choice([750, 1000]),
        }
        cols = ["acc", *list(map(lambda k: f"config/{k}", config.keys()))]

        # analysis = tune.run(train, config=config, num_samples=128)
        with capsys.disabled():
            analysis = tune.run(train, config=config, num_samples=64)
            print_results(analysis, cols, DATASET)


def test_rf_params(capsys: Any) -> None:
    ray.init(num_cpus=8)
    for DATASET in ["Diabetes", "Transfusion", "Parkinsons", "SPECT"]:

        objective = objective_function(RF)
        train = train_function(objective, DATASET)
        config = {
            "n_estimators": tune.choice([10, 20, 50, 100, 200, 400]),
            "min_samples_leaf": tune.randint(1, 5),
            "max_features": tune.choice(["auto", "log2", None, 0.1, 0.25, 0.5, 0.75]),
            "max_depth": tune.choice([None, 2, 4, 6, 8, 10, 20]),
        }
        cols = ["acc", *list(map(lambda k: f"config/{k}", config.keys()))]
        with capsys.disabled():
            analysis = tune.run(train, config=config, num_samples=250)
            print_results(analysis, cols, DATASET)


def test_ada_params(capsys: Any) -> None:
    ray.init(num_cpus=8, configure_logging=True, logging_level=ray.logging.WARNING)
    for DATASET in ["Diabetes", "Transfusion", "Parkinsons", "SPECT"]:
        objective = objective_function(AdaBoost)
        train = train_function(objective, DATASET)
        config = {
            "n_estimators": tune.choice([10, 50, 100, 200]),
            "learning_rate": tune.qloguniform(1e-5, 1, 5e-6),
        }
        cols = ["acc", *list(map(lambda k: f"config/{k}", config.keys()))]
        with capsys.disabled():
            analysis = tune.run(train, config=config, num_samples=250)
            print_results(analysis, cols, DATASET)


def test_lr_params(capsys: Any) -> None:
    ray.init(num_cpus=8, configure_logging=True, logging_level=ray.logging.WARNING)
    for DATASET in ["Diabetes", "Transfusion", "Parkinsons", "SPECT"]:

        objective = objective_function(LR, model_args=dict(solver="liblinear"))
        train = train_function(objective, DATASET)
        config = {
            "penalty": tune.choice(["l1", "l2"]),
            # "C": tune.qloguniform(1e-2, 10000, 5e-6),
            "C": tune.qloguniform(0.1, 2, 0.1),
            "max_iter": tune.choice([250, 500]),
        }
        cols = ["acc", *list(map(lambda k: f"config/{k}", config.keys()))]

        with capsys.disabled():
            analysis = tune.run(train, config=config, num_samples=250)
            print_results(analysis, cols, DATASET)


def test_svm_params(capsys: Any) -> None:
    ray.init(num_cpus=8, configure_logging=True, logging_level=ray.logging.WARNING)
    for DATASET in ["Diabetes", "Transfusion", "Parkinsons", "SPECT"]:

        objective = objective_function(SVC, model_args=dict(max_iter=500))
        train = train_function(objective, DATASET)
        config = {
            # "kernel": tune.choice(["linear", "poly", "rbf"]),
            "C": tune.qloguniform(10, 100, 0.5),
            # "C": tune.qloguniform(1, 5, 0.5),
            # "shrinking": tune.choice([True, False]),
        }
        cols = ["acc", *list(map(lambda k: f"config/{k}", config.keys()))]

        with capsys.disabled():
            analysis = tune.run(train, config=config, num_samples=250)
            df = print_results(analysis, cols, DATASET)
            ifg, ax = plt.subplots()
            sbn.scatterplot(data=df, x="C", y="acc", ax=ax)
            ax.set_xlabel("C")
            ax.set_ylabel("Accuracy")
            plt.show(block=False)
    plt.show()
