from typing import Any

import numpy as np
import pandas as pd
from error_consistency.consistency import ErrorConsistencyKFoldHoldout, ErrorConsistencyKFoldInternal
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN


class TestConsistency:
    def test_sanity(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            knn_args = dict(n_neighbors=10, n_jobs=-1)
            errcon = ErrorConsistencyKFoldHoldout(model=KNN, x=X_train, y=y_train, n_splits=5, model_args=knn_args)
            results = errcon.evaluate(
                x_test=X_test,
                y_test=y_test,
                repetitions=100,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                turbo=True,
            )

            print("Mean pairwise consistency ....... ", np.round(np.mean(results.consistencies), 4))
            print("Mean leave-one-out consistency .. ", np.round(results.leave_one_out_consistency, 4))
            print("Total consistency ............... ", np.round(results.total_consistency, 4))
            print("sd (pairwise) ................... ", np.std(results.consistencies, ddof=1))
            print(
                "sd (leave-one-out) .............. ",
                np.std(results.leave_one_out_consistency, ddof=1),
            )

    def test_numba(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            knn_args = dict(n_neighbors=10, n_jobs=-1)
            errcon = ErrorConsistencyKFoldHoldout(model=KNN, x=X_train, y=y_train, n_splits=5, model_args=knn_args)
            errcon.evaluate(
                x_test=X_test,
                y_test=y_test,
                repetitions=100,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                turbo=True,
            )

    def test_slow(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            knn_args = dict(n_neighbors=10, n_jobs=-1)
            errcon = ErrorConsistencyKFoldHoldout(model=KNN, x=X_train, y=y_train, n_splits=5, model_args=knn_args)
            # for _ in range(10):
            errcon.evaluate(
                x_test=X_test,
                y_test=y_test,
                repetitions=10,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                turbo=False,
            )

    def test_parallel_reps(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            knn_args = dict(n_neighbors=10, n_jobs=1)
            errcon = ErrorConsistencyKFoldHoldout(model=KNN, x=X_train, y=y_train, n_splits=5, model_args=knn_args)
            results = errcon.evaluate(
                x_test=X_test,
                y_test=y_test,
                repetitions=100,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                parallel_reps=True,
                turbo=True,
            )
            print("Mean pairwise consistency ....... ", np.round(np.mean(results.consistencies), 4))
            print("Mean leave-one-out consistency .. ", np.round(results.leave_one_out_consistency, 4))
            print("Total consistency ............... ", np.round(results.total_consistency, 4))
            print("sd (pairwise) ................... ", np.std(results.consistencies, ddof=1))
            print(
                "sd (leave-one-out) .............. ",
                np.std(results.leave_one_out_consistency, ddof=1),
            )

    def test_parallel_all(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            knn_args = dict(n_neighbors=10, n_jobs=1)
            errcon = ErrorConsistencyKFoldHoldout(model=KNN, x=X_train, y=y_train, n_splits=5, model_args=knn_args)
            results = errcon.evaluate(
                x_test=X_test,
                y_test=y_test,
                repetitions=100,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                parallel_reps=True,
                loo_parallel=True,
                turbo=True,
            )
            print("Mean pairwise consistency ....... ", np.round(np.mean(results.consistencies), 4))
            print("Mean leave-one-out consistency .. ", np.round(results.leave_one_out_consistency, 4))
            print("Total consistency ............... ", np.round(results.total_consistency, 4))
            print("sd (pairwise) ................... ", np.std(results.consistencies, ddof=1))
            print(
                "sd (leave-one-out) .............. ",
                np.std(results.leave_one_out_consistency, ddof=1),
            )


class TestInternalConsistency:
    def test_sanity(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            knn_args = dict(n_neighbors=10, n_jobs=-1)
            errcon = ErrorConsistencyKFoldInternal(model=KNN, x=X, y=y, n_splits=5, model_args=knn_args)
            results = errcon.evaluate(
                repetitions=100,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                save_test_predictions=True,
                turbo=True,
            )
            for preds in results.test_predictions:  # type: ignore
                assert -1 not in preds.ravel()

            print("Mean pairwise consistency ....... ", np.round(np.mean(results.consistencies), 4))
            print("Mean leave-one-out consistency .. ", np.round(results.leave_one_out_consistency, 4))
            print("Total consistency ............... ", np.round(results.total_consistency, 4))
            print("sd (pairwise) ................... ", np.std(results.consistencies, ddof=1))
            print(
                "sd (leave-one-out) .............. ",
                np.std(results.leave_one_out_consistency, ddof=1),
            )

    def test_numba(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            knn_args = dict(n_neighbors=10, n_jobs=-1)
            errcon = ErrorConsistencyKFoldInternal(model=KNN, x=X, y=y, n_splits=5, model_args=knn_args)
            results = errcon.evaluate(
                repetitions=100,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                save_test_predictions=True,
                turbo=True,
            )
            for preds in results.test_predictions:  # type: ignore
                assert -1 not in preds.ravel()

    def test_slow(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            knn_args = dict(n_neighbors=10, n_jobs=-1)
            errcon = ErrorConsistencyKFoldInternal(model=KNN, x=X, y=y, n_splits=5, model_args=knn_args)
            # for _ in range(10):
            results = errcon.evaluate(
                repetitions=10,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                save_test_predictions=True,
                turbo=False,
            )
            for preds in results.test_predictions:  # type: ignore
                assert -1 not in preds.ravel()

    def test_parallel_reps(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            knn_args = dict(n_neighbors=10, n_jobs=1)
            errcon = ErrorConsistencyKFoldInternal(model=KNN, x=X, y=y, n_splits=5, model_args=knn_args)
            results = errcon.evaluate(
                repetitions=100,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                save_test_predictions=True,
                parallel_reps=True,
                turbo=True,
            )
            for preds in results.test_predictions:  # type: ignore
                assert -1 not in preds.ravel()
            print("Mean pairwise consistency ....... ", np.round(np.mean(results.consistencies), 4))
            print("Mean leave-one-out consistency .. ", np.round(results.leave_one_out_consistency, 4))
            print("Total consistency ............... ", np.round(results.total_consistency, 4))
            print("sd (pairwise) ................... ", np.std(results.consistencies, ddof=1))
            print(
                "sd (leave-one-out) .............. ",
                np.std(results.leave_one_out_consistency, ddof=1),
            )

    def test_parallel_all(self, capsys: Any) -> None:
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            knn_args = dict(n_neighbors=10, n_jobs=1)
            errcon = ErrorConsistencyKFoldInternal(model=KNN, x=X, y=y, n_splits=5, model_args=knn_args)
            results = errcon.evaluate(
                repetitions=100,
                save_fold_accs=True,
                save_test_accs=True,
                save_test_errors=True,
                save_test_predictions=True,
                parallel_reps=True,
                loo_parallel=True,
                turbo=True,
            )
            for preds in results.test_predictions:  # type: ignore
                assert -1 not in preds.ravel()
            print("Mean pairwise consistency ....... ", np.round(np.mean(results.consistencies), 4))
            print("Mean leave-one-out consistency .. ", np.round(results.leave_one_out_consistency, 4))
            print("Total consistency ............... ", np.round(results.total_consistency, 4))
            print("sd (pairwise) ................... ", np.std(results.consistencies, ddof=1))
            print(
                "sd (leave-one-out) .............. ",
                np.std(results.leave_one_out_consistency, ddof=1),
            )


class TestClassifiersHoldout:
    def test_knn(self, capsys: Any) -> None:
        BIG_REPS = 3  # no need in this case, err-con is entirely function of test set
        # order below is important
        COLS = ["Mean (pairs)", "sd (pairs)", "Mean (LOO)", "sd (LOO)", "Total"]
        with capsys.disabled():
            X, y = load_iris(return_X_y=True)
            np.random.seed(2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
            knn_args = dict(n_neighbors=10, n_jobs=1)
            errcon = ErrorConsistencyKFoldHoldout(
                model=KNN, x=X_train, y=y_train, n_splits=5, model_args=knn_args, stratify=True
            )
            df = pd.DataFrame(index=pd.Index(range(BIG_REPS), name="REP"), columns=COLS)
            for i in range(BIG_REPS):
                results = errcon.evaluate(
                    x_test=X_test,
                    y_test=y_test,
                    repetitions=100,
                    save_fold_accs=True,
                    save_test_accs=True,
                    save_test_errors=True,
                    empty_unions=0,
                    parallel_reps=True,
                    loo_parallel=True,
                    turbo=True,
                    seed=42,
                )

                mean = np.mean(results.consistencies)
                sd_pair = np.std(results.consistencies, ddof=1)
                mean_loo = results.leave_one_out_consistency
                sd_loo = np.std(results.leave_one_out_consistency, ddof=1)
                total = results.total_consistency

                df.loc[i, COLS] = [mean, sd_pair, mean_loo, sd_loo, total]

                print("Mean pairwise consistency ....... ", np.round(mean, 4))
                print("Mean leave-one-out consistency .. ", np.round(mean_loo, 4))
                print("Total consistency ............... ", np.round(results.total_consistency, 4))
                print("sd (pairwise) ................... ", np.std(results.consistencies, ddof=1))
                print(
                    "sd (leave-one-out) .............. ",
                    np.std(results.leave_one_out_consistency, ddof=1),
                )
            print(df.round(4))
