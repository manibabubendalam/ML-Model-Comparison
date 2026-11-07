# ML Model Comparison: Dry Bean Classification

Machine Learning, Assignment 2 (M.Tech AIML, BITS Pilani WILP).

Five classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) are trained on one
dataset, compared on six evaluation metrics, and served through a Streamlit web app.

## Problem statement

Dry beans are sold by variety, and sorting them by hand is slow and inconsistent. The task is to predict the **variety of a dry bean (7 classes)** from 16 shape measurements taken from a photograph of each grain. The aim is to implement five classification algorithms on one dataset, compare them on Accuracy, AUC, Precision, Recall, F1 and MCC, and pick the best one for this problem.

## Dataset description

| Item | Value |
|---|---|
| Source | [UCI Machine Learning Repository: Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| Instances | 13,611 (requirement: at least 500) |
| Features | 16, all numeric (requirement: at least 12) |
| Target | `Class`, 7 varieties: Dermason, Sira, Seker, Horoz, Cali, Barbunya, Bombay |
| Missing values | None |
| Problem type | Multi-class classification |

**Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1 to ShapeFactor4. These are shape measurements taken from a photograph of each grain.

**Class balance:** Dermason 3546, Sira 2636, Seker 2027, Horoz 1928, Cali 1630, Barbunya 1322, Bombay 522. Bombay is the smallest class at about 3.8% of the data.

The full dataset is stored as [`data/dry_bean.csv`](data/dry_bean.csv), converted from the `.xlsx` file published on UCI.

## Train/test split and preprocessing

The data is split 80/20 with stratification (`random_state=42`), so every class keeps the same proportion in both parts. The 20% hold-out (2,723 rows, features + `Class`) is saved as [`test_data.csv`](test_data.csv). The models never see it during training, and it is the file to upload in the Streamlit app.

`StandardScaler` is used for Logistic Regression, KNN and SVM, whose distance- and gradient-based methods need scaled inputs. Decision Tree and Gaussian Naive Bayes use the raw features.

Both steps are done by [`train_models.py`](train_models.py).

## GitHub repository and live app

- **GitHub repository:** <https://github.com/manibabubendalam/ML-Model-Comparison>
- **Live Streamlit app:** <https://manibabubendalam-ml-model-comparision.streamlit.app/>

## Models used

All five models are scikit-learn Pipelines trained on the same 80% training split and saved to [`model/`](model/) with `joblib`. The scaler is saved inside each Pipeline, so no separate preprocessing step is needed at prediction time. The metrics below are calculated on the 20% hold-out set and also saved to [`model/metrics.csv`](model/metrics.csv).

Because the target has 7 classes, **Precision, Recall and F1 are macro-averaged** and **AUC is one-vs-rest, macro-averaged**.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9214 | **0.9948** | 0.9354 | 0.9321 | 0.9335 | 0.9050 |
| Decision Tree | 0.9064 | 0.9777 | 0.9220 | 0.9193 | 0.9206 | 0.8867 |
| KNN | 0.9137 | 0.9858 | 0.9290 | 0.9245 | 0.9265 | 0.8956 |
| Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| Support Vector Machine | **0.9243** | 0.9946 | **0.9372** | **0.9350** | **0.9361** | **0.9085** |

Hyperparameters: Logistic Regression `max_iter=2000`; Decision Tree `max_depth=10, min_samples_leaf=5`; KNN `n_neighbors=7`; Gaussian Naive Bayes defaults; SVM RBF kernel with `C=10`.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Reaches 92.1% accuracy and the best AUC of the five (0.9948), despite being a linear model. After scaling, the bean varieties are close to linearly separable in this feature space. It is the fastest to train and its saved file is tiny (about 3 KB). Its weakest class is Sira (F1 0.864), which it mixes up with Dermason. |
| Decision Tree | Lowest accuracy of the four stronger models (90.6%) and a clearly lower AUC (0.978). Its probability estimates are coarse because each leaf gives a hard vote. Depth was limited to 10 to curb overfitting. It is easy to interpret and needs no scaling, but it is less accurate than the smoother models. |
| KNN | 91.4% accuracy, between the Decision Tree and Logistic Regression. It depends on feature scaling and must keep the whole training set, so its saved file is the largest (about 1.5 MB) and prediction cost grows with the data size. It handles the clearly distinct classes well but is not better than the linear model on the overlapping ones. |
| Naive Bayes | By far the weakest, at 76.4% accuracy and an MCC of 0.715. Many of the 16 features are strongly correlated (Area, Perimeter, ConvexArea, EquivDiameter and the axis lengths), which breaks the independence assumption, and the features are not Gaussian. Barbunya (F1 0.548) is hit hardest. Its AUC (0.967) is still high, so the probability ranking is reasonable while the final class decision is poor. |
| Support Vector Machine | Best on Accuracy (92.4%), Precision, Recall, F1 and MCC. The RBF kernel captures non-linear boundaries between similar-looking beans. It improves on Logistic Regression only slightly (+0.3 points of accuracy) and costs more to train. The remaining errors are mostly Sira vs Dermason (53 and 43 misclassifications) and Barbunya vs Cali. |
| **Overall winner** | **Support Vector Machine** has the best Accuracy, Precision, Recall, F1 and MCC. **Logistic Regression** is a very close second, with the highest AUC, and is much cheaper to train and store. Naive Bayes is clearly unsuitable for this feature set. Bombay is classified perfectly (F1 1.0) by all five models, and Sira vs Dermason is the main source of error for every model. |

## Streamlit app

Run it locally (after `pip install -r requirements.txt`):

```bash
streamlit run app.py
```

Features:

- **CSV upload** of the test data. A sample `test_data.csv` can be downloaded from the sidebar.
- **Model selection** dropdown for the five classifiers.
- **Evaluation metrics** (Accuracy, AUC, Precision, Recall, F1, MCC) for the selected model.
- **Confusion matrix** (counts or row-normalised) and **classification report**.
- **Compare all models** tab: the metrics of all five models on the uploaded data, side by side.

The uploaded CSV must contain the 16 feature columns and the `Class` column. The app loads the saved models from `model/` and only predicts. It does not retrain.

## Repository structure

```
ML-Model-Comparison/
├── app.py               # Streamlit app
├── model_utils.py       # model definitions + metric calculation
├── train_models.py      # trains and saves all five models
├── requirements.txt
├── README.md
├── test_data.csv        # hold-out set used for evaluation
├── data/dry_bean.csv    # full dataset
└── model/               # saved trained models + metrics.csv
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
