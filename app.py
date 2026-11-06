"""Streamlit app: evaluate pre-trained classifiers on an uploaded Dry Bean test CSV."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model_utils import MODEL_FILES, evaluate

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"
TARGET = "Class"

st.set_page_config(page_title="Dry Bean Classifier Comparison", page_icon="🫘", layout="wide")


@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def check_columns(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [c for c in features + [TARGET] if c not in df.columns]


st.title("🫘 Dry Bean Classification: Model Comparison")
st.caption(
    "Five pre-trained classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM) "
    "predict the variety of a dry bean from 16 shape measurements. "
    "Models are loaded from disk. Nothing is retrained on your upload."
)

# ---- Sidebar: data upload + model choice -------------------------------------------------
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV (features + `Class`)", type="csv")
    if TEST_CSV.exists():
        st.download_button(
            "Download sample test_data.csv",
            TEST_CSV.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
        )
    st.header("2. Model")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))

if uploaded is None:
    st.info("Upload a test CSV in the sidebar to begin. You can download a sample first.")
    st.stop()

df = pd.read_csv(uploaded)
model = load_model(model_name)
features = list(model.feature_names_in_)

missing = check_columns(df, features)
if missing:
    st.error(f"Uploaded file is missing required column(s): {', '.join(missing)}")
    st.stop()

unknown = sorted(set(df[TARGET]) - set(model.classes_))
if unknown:
    st.error(f"`{TARGET}` contains labels the models were not trained on: {unknown}")
    st.stop()

X, y = df[features], df[TARGET]
st.success(f"Loaded {len(df):,} rows, {len(features)} features, {y.nunique()} classes.")

tab_single, tab_all, tab_data = st.tabs(["Selected model", "Compare all models", "Data preview"])

# ---- Tab 1: metrics + confusion matrix + report for the selected model ---------------------
with tab_single:
    st.subheader(model_name)
    scores = evaluate(model, X, y)
    cols = st.columns(6)
    for col, (metric, value) in zip(cols, scores.items()):
        col.metric(metric, f"{value:.4f}")
    st.caption("Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro.")

    pred = model.predict(X)
    labels = list(model.classes_)
    left, right = st.columns([1, 1])

    with left:
        st.markdown("**Confusion matrix**")
        normalise = st.toggle("Show row-normalised (recall) view", value=False)
        cm = confusion_matrix(y, pred, labels=labels, normalize="true" if normalise else None)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalise else "d",
            cmap="YlGnBu",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, pred, labels=labels, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

# ---- Tab 2: all five models side by side ------------------------------------------------
with tab_all:
    st.subheader("All models on the uploaded data")
    rows = [{"ML Model Name": n, **evaluate(load_model(n), X, y)} for n in MODEL_FILES]
    table = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(
        table.style.format("{:.4f}").highlight_max(
            axis=0, props="background-color: #c6efce; color: #111111; font-weight: 600"
        ),
        width="stretch",
    )
    best = table["MCC"].idxmax()
    st.success(f"Best model by MCC on this data: **{best}** ({table.loc[best, 'MCC']:.4f})")
    st.bar_chart(table[["Accuracy", "F1", "MCC"]], stack=False)

# ---- Tab 3: preview ---------------------------------------------------------------------
with tab_data:
    st.dataframe(df.head(100), width="stretch")
    st.caption("First 100 rows shown.")
