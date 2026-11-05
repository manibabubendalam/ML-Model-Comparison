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
