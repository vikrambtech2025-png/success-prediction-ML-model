import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load your trained model ---
@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pk")
    except Exception as e:
        st.error(f"Model loading error: {e}")
        st.info("Using a dummy fallback model for demo.")
        # Dummy model logic (you'll replace this with your real model later)
        class DummyModel:
            def predict_proba(self, X):
                # Simulate: more rows → higher success chance
                score = 0.6 + 0.3 * (X[:, 0] / X[:, 0].max())
                score = np.clip(score, 0, 1)  # 0–1
                return np.vstack([1 - score, score]).T
        return DummyModel()

model = load_model()

# --- 2. Streamlit UI ---
st.set_page_config(page_title="Dataset Success Rate", page_icon="📊", layout="wide")

st.title("📊 Dataset Success Rate Predictor")
st.markdown("Upload a CSV file and get its predicted **success rate** (data quality / readiness for ML).")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    with st.spinner("Analyzing your dataset..."):
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            n_rows, n_cols = df.shape
            n_missing = df.isnull().sum().sum()
            n_duplicates = df.duplicated().sum()
            num_cols = df.select_dtypes(include=[np.number]).shape[1]
            cat_cols = n_cols - num_cols

            # Feature vector (same as before)
            features = np.array([
                n_rows,
                n_cols,
                n_missing,
                n_duplicates,
                num_cols,
                cat_cols,
            ]).reshape(1, -1)

            # Predict success rate (class 1 = “success”)
            proba = model.predict_proba(features)[:, 1]
            success_rate = float(proba[0])
            percent = round(success_rate * 100, 1)

            # --- Show results ---
            st.success(f"✅ Predicted Success Rate: **{percent}%**")

            col1, col2 = st.columns(2)
            col1.metric("Rows", n_rows)
            col2.metric("Columns", n_cols)
            col1.metric("Missing Values", int(n_missing))
            col2.metric("Duplicates", int(n_duplicates))
            col1.metric("Numeric Columns", num_cols)
            col2.metric("Categorical Columns", cat_cols)

            # --- Show sample data ---
            if st.checkbox("Show first 10 rows"):
                st.dataframe(df.head(10))

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.code("Make sure your file is a valid CSV and try again.")