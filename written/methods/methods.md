# Methods

## Classifiers

We fit four classifiers: support-vector machine with radial basis function
(SVM), logistic regression (LR), a random forest (RF), and histogram-based gradient-boosted
decision trees (GBT). The LR and SVM classifiers were fit with stochastic
gradient descent via the scikit-learn Python library
[@pedregosaScikitlearnMachineLearning2011b]. In order to make SVM compute times
tractable on the larger datasets, kernel approximation via the Nystroem method
[@Williams:161322] was employed. Both the RF and GBT models were implemented via XGBoost [@chenXGBoostScalableTree2016].



## Tuning
All classifiers were tuned via random search (250 iterations). The
parameter ranges searched were:

LR:

| Parameter  |         Range or Values          | Log Scale |
|:----------:|:--------------------------------:|:---------:|
|  `alpha`   |           [1e-1, 1e-7]           |     ✓     |
| `l1_ratio` |            [0.0, 1.0]            |           |
|   `eta0`   |           [1e-6, 1.0]            |     ✓     |
| `penalty`  | ["l1", "l2", "elasticnet", None] |           |
| `average`  |          [True, False]           |           |

SVM:

|   Parameter    |         Range or Values          | Log Scale |
|:--------------:|:--------------------------------:|:---------:|
|    `gamma`     |           [1e-10, 1e3]           |     ✓     |
| `n_components` |            [5, 1000]             |           |
|    `alpha`     |           [1e-1, 1e-7]           |     ✓     |
|   `l1_ratio`   |            [0.0, 1.0]            |           |
|     `eta0`     |           [1e-6, 1.0]            |     ✓     |
|   `penalty`    | ["l1", "l2", "elasticnet", None] |           |
|   `average`    |          [True, False]           |           |

GBT and RF (as recommended by [@kadraWelltunedSimpleNets2021]):

|      Parameter      | Range or Values | Log Scale |
|:-------------------:|:---------------:|:---------:|
|        `eta`        |  [0.001, 1.0]   |     ✓     |
|      `lambda`       |  [1e-10, 1.0]   |     ✓     |
|       `alpha`       |  [1e-10, 1.0]   |     ✓     |
|       `gamma`       |   [0.1, 1.0]    |     ✓     |
| `colsample_bylevel` |   [0.1, 1.0]    |           |
| `colsample_bytree`  |   [0.1, 1.0]    |           |
|     `max_depth`     |     [1, 20]     |           |
|  `max_delta_step`   |     [0, 10]     |           |
| `min_child_weight`  |    [0.1, 20]    |     ✓     |
|     `subsample`     |   [0.01, 1.0]   |           |




# References