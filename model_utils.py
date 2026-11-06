"""Shared model definitions and metric calculation (used by training and by the Streamlit app)."""
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def build_models(seed: int = 42) -> dict:
    """Each model is a Pipeline so the scaler is saved together with the classifier."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        ),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=seed)
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": make_pipeline(GaussianNB()),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=10, probability=True, random_state=seed)
        ),
    }


def evaluate(pipeline, X, y) -> dict:
    """Multi-class metrics: precision/recall/F1 are macro-averaged, AUC is one-vs-rest macro."""
    pred = pipeline.predict(X)
    proba = pipeline.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=pipeline.classes_),
        "Precision": precision_score(y, pred, average="macro", zero_division=0),
        "Recall": recall_score(y, pred, average="macro", zero_division=0),
        "F1": f1_score(y, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }
