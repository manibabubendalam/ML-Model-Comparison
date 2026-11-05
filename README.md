# ML Model Comparison: Dry Bean Classification

Machine Learning, Assignment 2 (M.Tech AIML, BITS Pilani WILP).

Five classification models (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) are trained on one
dataset, compared on six evaluation metrics, and served through a Streamlit web app.

> Work in progress. This README is filled in as the project grows.

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
