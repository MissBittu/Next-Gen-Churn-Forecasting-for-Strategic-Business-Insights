import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os


# import boto3  

try:
    from langchain.llms import OpenAI
    llm_available = True
except ImportError:
    llm_available = False

# -----------------------------
st.set_page_config(page_title="Customer Churn Predictor with AI Insights", page_icon=" ")
st.title(" Customer Churn Prediction App")
st.markdown("Enhanced with **LLM-based explanations** and **AI interpretability** for better insights.")

# ==============================
# 🔹 Load Saved Model Artifacts
# ==============================
def load_artifact(path):
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    else:
        st.warning(f" Missing artifact: {path}")
        return None

model = load_artifact("output_handcrafted/model.pkl")
scaler = load_artifact("output_handcrafted/scaler.pkl")
encoders = load_artifact("output_handcrafted/encoders.pkl")
feature_selector = load_artifact("output_handcrafted/feature_selector.pkl")

selected_features = []
if os.path.exists("output_handcrafted/selected_features.txt"):
    with open("output_handcrafted/selected_features.txt") as f:
        selected_features = [line.strip() for line in f]

# ===========================
# 🔹 Upload Section
# ===========================
uploaded_file = st.file_uploader(" Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, skipinitialspace=True)
    df = df.replace(['', ' ', '  '], np.nan)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()

    st.success("Data Uploaded Successfully!")
    st.dataframe(df.head())

    # ===========================
    # 🔹 Feature Engineering
    # ===========================
    numeric_cols = ["MonthlyCharges", "tenure", "TotalCharges", "SeniorCitizen"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    df["AvgMonthlyCharges"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["IsNewCustomer"] = (df["tenure"] < 6).astype(int)
    df["IsLoyalCustomer"] = (df["tenure"] > 24).astype(int)
    df["ContractValue"] = df["MonthlyCharges"] * df["tenure"]
    df["EngagementScore"] = df["tenure"] / (df["MonthlyCharges"] + 1)
    df["HighChargeNewCustomer"] = (
        (df["MonthlyCharges"] > df["MonthlyCharges"].median()) & (df["tenure"] < 6)
    ).astype(int)

    if "PaymentMethod" in df.columns:
        df["IsElectronicPayment"] = df["PaymentMethod"].apply(
            lambda x: 1 if "electronic" in str(x).lower() else 0
        )
    if "PaperlessBilling" in df.columns:
        df["PaperlessBillingFlag"] = df["PaperlessBilling"].apply(
            lambda x: 1 if str(x).lower() == "yes" else 0
        )

    if "Contract" in df.columns:
        df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
        df["IsOneYear"] = (df["Contract"] == "One year").astype(int)
        df["IsTwoYear"] = (df["Contract"] == "Two year").astype(int)

    # Encode categorical variables safely
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).fillna('Unknown')
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    X = df[selected_features].fillna(0)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    st.info(f"🔧 Processed {len(X)} rows × {len(X.columns)} features.")

    # ===========================
    # 🔹 Predictions
    # ===========================
    try:
        X_scaled = scaler.transform(X)
        X_selected = feature_selector.transform(X_scaled)

        predictions = model.predict(X_selected)
        probabilities = model.predict_proba(X_selected)[:, 1]

        df["Churn Prediction"] = predictions
        df["Churn Probability"] = probabilities

        result_df = df[["Churn Prediction", "Churn Probability"]].copy()
        result_df["Churn Prediction"] = result_df["Churn Prediction"].map({0: "No", 1: "Yes"})
        result_df["Churn Probability"] = result_df["Churn Probability"].round(4)

        st.subheader(" Prediction Results")
        st.dataframe(result_df)

        churn_count = int((predictions == 1).sum())
        st.metric("Customers Likely to Churn", f"{churn_count} ({churn_count / len(predictions) * 100:.1f}%)")

        # ===========================
        # 🔹 LLM-Based Explanation
        # ==========================
        st.subheader(" AI Insight (LLM Explanation)")
        if llm_available:
            st.caption("Powered by LangChain + OpenAI")
            try:
                llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                sample_customer = df.iloc[0].to_dict()
                prompt = f"Explain why this telecom customer might churn based on the data: {sample_customer}"
                explanation = llm(prompt)
                st.write(explanation)
            except Exception as e:
                st.warning(f"LLM explanation unavailable: {e}")
        else:
            st.info("LangChain not installed. Run `pip install langchain openai` to enable AI explanations.")

        # ===========================
        # 🔹 Download Results
        # ===========================
        st.download_button(
            " Download Results as CSV",
            df.to_csv(index=False).encode(),
            "churn_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Debug Info:")
        st.write("X shape:", X.shape)
        st.dataframe(X.head())
        st.title("Churn Prediction Results")
        st.button("Refresh Results")
