# ML Model Comparison: Dry Bean Classification

Machine Learning, Assignment 2 (M.Tech AIML, BITS Pilani WILP).

Five classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) are trained on one
dataset, compared on six evaluation metrics, and served through a Streamlit web app.

> Work in progress. This README is filled in as the project grows.

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

## Planned structure

```
ML-Model-Comparison/
├── app.py               # Streamlit app
├── model_utils.py       # model definitions + metric calculation
├── train_models.py      # trains and saves all five models
├── requirements.txt
├── README.md
├── test_data.csv        # hold-out set used for evaluation
├── data/                # dataset
└── model/               # saved trained models
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
