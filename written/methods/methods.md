# Methods

## Datasets

| name          | n_samples | n_features | n_class | majority |
|:--------------|----------:|-----------:|--------:|---------:|
| Diabetes      |       768 |          8 |       2 |     0.65 |
| Transfusion   |       748 |          4 |       2 |     0.76 |
| Parkinsons    |       195 |         22 |       2 |     0.75 |
| SPECT         |       267 |         22 |       2 |     0.79 |
| Diabetes130   |    101766 |         48 |       2 |     0.89 |
| HeartFailure  |      2008 |        148 |       2 |     0.62 |
| MimicIV       |    369618 |        118 |       3 |      0.6 |
| UTIResistance |    100769 |        788 |       2 |     0.52 |

The "Diabetes130" dataset was originally collected by
@strackImpactHbA1cMeasurement2014, and is publicly available on the OpenML
[@vanschorenOpenMLNetworkedScience2014] platform. The data are derived from
130 US Hospitals over a period of 10 years, and the features comprise things
like race, gender, number of lab procedures, diagnoses, and other pieces of
medical information. We attempt to predict whether a subject is re-admitted
30 days following initial admission (and remove features that allow trivial
prediction of this variable).

The "HeartFailure" dataset [@zhangzhonghengHospitalizedPatientsHeart;
@zhangElectronicHealthcareRecords2021], made available through PhysioNet
[@goldbergerPhysioBankPhysioToolkitPhysioNet2000], contains a large number of
features related to medications, heart measurements, demographics, and other
medical factors. We predict whether a patient will be re-admitted within 6
months of the initial admission (and remove features that allow trivial
prediction of this variable).

The "UTIResistance" dataset [@oberstmichaelAMRUTIAntimicrobialResistance;
kanjilalDecisionAlgorithmPromote2020], also available from PhysioNet
[@goldbergerPhysioBankPhysioToolkitPhysioNet2000], concerns antibiotic
resistance observed in patients with urinary tract infections (UTI). The large
number of sparse, binary indicator features comprise information such as
hospital floor, ward, previous instance of resistance, medications
administered, demographic information, and other medically-relevant indicators.
Binary resistance categories are available for four antibiotic types, however,
we combine this into a single binary variable representing any observed
antibiotic resistance, and use this for prediction.

The "MimicIV" emergency stay (ES) dataset [@johnsonalistairMIMICIVED], also available from PhysioNet
[@goldbergerPhysioBankPhysioToolkitPhysioNet2000], is comprised of multiple tables
of data (previous medical history, demographics, triage information, etc). For each
ES, a patient has a final disposition, which we assign the following labels,
roughly according to severity / seriousness:

|          Disposition          | Label |
|:-----------------------------:|:-----:|
|   "left without being seen"   |   0   |
| "left against medical advice" |   0   |
|           "eloped":           |   0   |
|          "admitted":          |   1   |
|          "transfer":          |   1   |
|            "home":            |   2   |
|          "expired":           |   3   |
|           "other":            |   4   |

We use the triage information (which includes heartrate, blood pressure, oxygen
saturation, subjective pain reports, and attending comments). We included
attending comments by counting the most common words used (e.g. "pain",
"stomach", "abnormal", "injury", etc.) and then creating indicator features for
each word. Details can be found in the source code.




After removing or median-interpolating missing values, non-categorical features
were were standardized prior to fitting with predictive models.

## Classifiers

We fit four classifiers: support-vector machine with radial basis function
(SVM), logistic regression (LR), a random forest (RF), and histogram-based
gradient-boosted decision trees (GBT). The LR and SVM classifiers were fit with
stochastic gradient descent via the scikit-learn Python library
[@pedregosaScikitlearnMachineLearning2011b]. In order to make SVM compute times
tractable on the larger datasets, kernel approximation via the Nystroem method
[@Williams:161322] was employed. Both the RF and GBT models were implemented
via XGBoost [@chenXGBoostScalableTree2016].

The combination of parameters resulting in the highest overall accuracy were
then saved and used for all subsequent model evaluation.



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



## Downsampling Procedure

All models on all datasets were evaluated using the internal k-fold method. Evaluation
consists of $R=200$ *repetitions*, and $n=10$ *runs* per repetition. For each repetition,
a random downsampling proportion $r$ is chosen from $[50, 90]$ to be shared across all
runs in the repetition.


Given data $\mathcal{D} = (X_{\text{full}}, y_{\text{full}})$ with
$N_{\text{full}}$ samples, and downsampling proportion $r$, then *for each
run*, a random subset $(X, y)$ with $N = \lfloor r * N_{\text{full}} \rfloor$
samples is selected. That is, each run evaluates a *different* size N subset of
the data $\mathcal{D}$. On this subset $(X, y)$, models are fit and
evaluated via $k$-fold with $k=5$. The five test predictions are then concatenated
to yield a single prediction $\hat{y}$ over the entirety of $y$.
Thus each repetition results in $n=10$ predictions $\hat{y}_1, \dots, \hat{y}_n$

## Evaluation Metrics

The error consistency (EC) is computed as defined earlier. For the $n$
runs in a repetition, there are $n$ error vectors $E_i = [y \ne \hat{y}_i]$ of length
$N$, where $[]$ is the Iverson bracket. Then the EC is:

$$
\text{EC}_{ij}= \frac{\lvert E_i \cap E_j\rvert}{\lvert E_i \cup E_j \rvert} = \frac{\text{sum}(E_i \land E_j)}{\text{sum}(E_i \lor E_j)}
$$

for elementwise logical operators $\land$ and $\lor$, and the mean consistency $\overline{\text{EC}}$ is

$$
\overline{\text{EC}} = \left(\sum_{i > j} \text{EC}_{ij} \right) / \binom{n}{2}
$$


and *mean accuracy* $\bar{a}$ is

$$ \bar{a} = \frac{1}{n}\sum_{i=1}^{n}\text{acc}(y, \hat{y}_i) $$

We also compute the *pairwise mean accuracy* $\alpha_{ij}$


$$
\alpha_{ij} = \frac{\text{acc}(y, \hat{y}_i) + \text{acc}(y, \hat{y}_j)}{2}
$$

which is useful for directly relating individual EC values to individual accuracy values.

# Results

Correlations (Pearson's $r$) between $\alpha_{ij}$ and $\text{EC}_{ij}$ values:


| data         | classifier |      r |     p |
|:-------------|:-----------|-------:|------:|
| Diabetes     | GBT        | -0.881 | 0.000 |
|              | LR         | -0.904 | 0.000 |
|              | RF         | -0.497 | 0.000 |
|              | SVM        | -0.607 | 0.000 |
|              |            |        |       |
| Diabetes130  | GBT        |  0.776 | 0.000 |
|              | LR         |  0.000 | 1.000 |
|              | RF         |  0.000 | 1.000 |
|              | SVM        |  0.004 | 0.719 |
|              |            |        |       |
| HeartFailure | GBT        | -0.479 | 0.000 |
|              | LR         | -0.513 | 0.000 |
|              | RF         | -0.347 | 0.000 |
|              | SVM        |  0.116 | 0.000 |
|              |            |        |       |
| Parkinsons   | GBT        | -0.734 | 0.000 |
|              | LR         | -0.700 | 0.000 |
|              | RF         |  0.021 | 0.044 |
|              | SVM        | -0.792 | 0.000 |
|              |            |        |       |
| SPECT        | GBT        | -0.530 | 0.000 |
|              | LR         | -0.565 | 0.000 |
|              | RF         | -0.355 | 0.000 |
|              | SVM        |  0.126 | 0.000 |
|              |            |        |       |
| Transfusion  | GBT        |  0.251 | 0.000 |
|              | LR         |  0.229 | 0.000 |
|              | RF         |  0.632 | 0.000 |
|              | SVM        | -0.412 | 0.000 |

Correlations (Pearson's $r$) between $\bar{a}$ and $\overline{\text{EC}}$ values:

| data         | classifier |      r |     p |
|:-------------|:-----------|-------:|------:|
| Diabetes     | GBT        | -0.582 | 0.000 |
|              | LR         | -0.584 | 0.000 |
|              | RF         |  0.220 | 0.002 |
|              | SVM        | -0.917 | 0.000 |
|              |            |        |       |
| Diabetes130  | GBT        |  0.663 | 0.000 |
|              | LR         |  0.000 | 1.000 |
|              | RF         |  0.000 | 1.000 |
|              | SVM        | -0.038 | 0.596 |
|              |            |        |       |
| HeartFailure | GBT        | -0.086 | 0.225 |
|              | LR         | -0.098 | 0.169 |
|              | RF         | -0.081 | 0.254 |
|              | SVM        |  0.373 | 0.000 |
|              |            |        |       |
| Parkinsons   | GBT        | -0.480 | 0.000 |
|              | LR         | -0.405 | 0.000 |
|              | RF         |  0.507 | 0.000 |
|              | SVM        | -0.891 | 0.000 |
|              |            |        |       |
| SPECT        | GBT        | -0.114 | 0.107 |
|              | LR         | -0.111 | 0.116 |
|              | RF         |  0.102 | 0.150 |
|              | SVM        |  0.404 | 0.000 |
|              |            |        |       |
| Transfusion  | GBT        |  0.365 | 0.000 |
|              | LR         |  0.404 | 0.000 |
|              | RF         |  0.611 | 0.000 |
|              | SVM        | -0.635 | 0.000 |

Correlations between downsampling percentage ($r$) and $\text{EC}_{ij}$s:

| data         | classifier |      r |     p |
|:-------------|:-----------|-------:|------:|
| Diabetes     | GBT        | -0.015 | 0.166 |
|              | LR         |  0.011 | 0.302 |
|              | RF         | -0.284 | 0.000 |
|              | SVM        |  0.009 | 0.387 |
|              |            |        |       |
| Diabetes130  | GBT        |  0.095 | 0.000 |
|              | LR         |  0.000 | 1.000 |
|              | RF         |  0.000 | 1.000 |
|              | SVM        | -0.028 | 0.007 |
|              |            |        |       |
| HeartFailure | GBT        |  0.100 | 0.000 |
|              | LR         |  0.192 | 0.000 |
|              | RF         | -0.032 | 0.002 |
|              | SVM        |  0.152 | 0.000 |
|              |            |        |       |
| Parkinsons   | GBT        | -0.085 | 0.000 |
|              | LR         | -0.002 | 0.829 |
|              | RF         | -0.488 | 0.000 |
|              | SVM        | -0.018 | 0.094 |
|              |            |        |       |
| SPECT        | GBT        |  0.064 | 0.000 |
|              | LR         |  0.038 | 0.000 |
|              | RF         | -0.088 | 0.000 |
|              | SVM        | -0.152 | 0.000 |
|              |            |        |       |
| Transfusion  | GBT        | -0.006 | 0.542 |
|              | LR         |  0.302 | 0.000 |
|              | RF         |  0.162 | 0.000 |
|              | SVM        |  0.021 | 0.047 |

Correlations between downsampling percentage ($r$) and $\overline{\text{EC}}$s:

| data         | classifier |      r |     p |
|:-------------|:-----------|-------:|------:|
| Diabetes     | GBT        | -0.099 | 0.163 |
|              | LR         |  0.077 | 0.281 |
|              | RF         | -0.673 | 0.000 |
|              | SVM        |  0.032 | 0.652 |
|              |            |        |       |
| Diabetes130  | GBT        |  0.220 | 0.002 |
|              | LR         |  0.000 | 1.000 |
|              | RF         |  0.000 | 1.000 |
|              | SVM        | -0.063 | 0.373 |
|              |            |        |       |
| HeartFailure | GBT        |  0.288 | 0.000 |
|              | LR         |  0.558 | 0.000 |
|              | RF         | -0.092 | 0.196 |
|              | SVM        |  0.484 | 0.000 |
|              |            |        |       |
| Parkinsons   | GBT        | -0.325 | 0.000 |
|              | LR         | -0.009 | 0.894 |
|              | RF         | -0.800 | 0.000 |
|              | SVM        | -0.081 | 0.255 |
|              |            |        |       |
| SPECT        | GBT        |  0.227 | 0.001 |
|              | LR         |  0.148 | 0.037 |
|              | RF         | -0.243 | 0.001 |
|              | SVM        | -0.446 | 0.000 |
|              |            |        |       |
| Transfusion  | GBT        | -0.015 | 0.837 |
|              | LR         |  0.624 | 0.000 |
|              | RF         |  0.349 | 0.000 |
|              | SVM        |  0.068 | 0.339 |

# References