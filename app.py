import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Customer Churn Prediction App")

# Load saved artifacts
model = pickle.load(open("output_handcrafted/model.pkl", "rb"))
scaler = pickle.load(open("output_handcrafted/scaler.pkl", "rb"))
encoders = pickle.load(open("output_handcrafted/encoders.pkl", "rb"))
feature_selector = pickle.load(open("output_handcrafted/feature_selector.pkl", "rb"))

with open("output_handcrafted/selected_features.txt") as f:
    selected_features = [line.strip() for line in f]

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    # Read CSV with proper handling
    df = pd.read_csv(uploaded_file, skipinitialspace=True)
    
    # Clean whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # Replace empty strings with NaN
    df = df.replace(['', ' ', '  '], np.nan)
    
    st.write("Data Uploaded Successfully!")
    st.dataframe(df.head())
    
    # Check for essential columns
    if "MonthlyCharges" not in df.columns or "tenure" not in df.columns:
        st.error("Missing essential columns like MonthlyCharges or tenure")
        st.stop()
    
    # Convert numeric columns and handle missing values
    numeric_cols = ["MonthlyCharges", "tenure", "TotalCharges", "SeniorCitizen"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill missing values with median
            df[col] = df[col].fillna(df[col].median())
    
    # Feature Engineering (must match training pipeline)
    df["AvgMonthlyCharges"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    
    # Additional engineered columns
    df["IsNewCustomer"] = (df["tenure"] < 6).astype(int)
    df["IsLoyalCustomer"] = (df["tenure"] > 24).astype(int)
    df["ContractValue"] = df["MonthlyCharges"] * df["tenure"]
    
    service_cols = ["PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in service_cols:
        if col not in df.columns:
            df[col] = 0  # fill missing service columns with 0
    df["TotalServices"] = df[service_cols].notnull().sum(axis=1)
    
    df["IsElectronicPayment"] = df["PaymentMethod"].apply(
        lambda x: 1 if "electronic" in str(x).lower() else 0
    ) if "PaymentMethod" in df.columns else 0
    
    df["PaperlessBillingFlag"] = df["PaperlessBilling"].apply(
        lambda x: 1 if str(x).lower() == "yes" else 0
    ) if "PaperlessBilling" in df.columns else 0
    
    df["HighChargeNewCustomer"] = (
        (df["MonthlyCharges"] > df["MonthlyCharges"].median()) & (df["tenure"] < 6)
    ).astype(int)
    
    df["SeniorAlone"] = (
        (df["SeniorCitizen"] == 1) & (df["Dependents"] == "No")
    ).astype(int) if "SeniorCitizen" in df.columns and "Dependents" in df.columns else 0
    
    df["EngagementScore"] = df["tenure"] / (df["MonthlyCharges"] + 1)
    df["CompositeRiskScore"] = df["IsNewCustomer"] + df["PaperlessBillingFlag"] + df["HighChargeNewCustomer"]
    
    # Create contract type columns if missing
    if "Contract" in df.columns:
        df["IsMonthToMonth"] = (df["Contract"] == "Month-to-month").astype(int)
        df["IsOneYear"] = (df["Contract"] == "One year").astype(int)
        df["IsTwoYear"] = (df["Contract"] == "Two year").astype(int)
    
    # Encode categorical columns
    for col, le in encoders.items():
        if col in df.columns:
            try:
                # Handle unseen categories
                df[col] = df[col].astype(str).fillna('Unknown')
                df[col] = le.transform(df[col])
            except ValueError as e:
                st.warning(f"Warning: Unknown categories in column '{col}'. Using default encoding.")
                # Create a mapping for unknown values
                known_classes = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
                df[col] = le.transform(df[col])
    
    # Fill any remaining missing features with 0
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0
    
    # Select same features
    X = df[selected_features]
    
    # Final data cleaning before scaling
    # Replace any remaining NaN with 0
    X = X.fillna(0)
    
    # Convert all columns to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Display info about the data being scaled
    st.write(f"Processing {len(X)} rows with {len(X.columns)} features...")
    
    try:
        # Scale and select features
        X_scaled = scaler.transform(X)
        X_selected = feature_selector.transform(X_scaled)
        
        # Predict
        predictions = model.predict(X_selected)
        probabilities = model.predict_proba(X_selected)[:, 1]
        
        # Show results
        df["Churn Prediction"] = predictions
        df["Churn Probability"] = probabilities
        
        st.subheader("Predictions")
        result_df = df[["Churn Prediction", "Churn Probability"]].copy()
        result_df["Churn Prediction"] = result_df["Churn Prediction"].map({0: "No", 1: "Yes"})
        result_df["Churn Probability"] = result_df["Churn Probability"].round(4)
        
        st.dataframe(result_df)
        
        # Show summary statistics
        churn_count = (predictions == 1).sum()
        st.metric("Customers Likely to Churn", f"{churn_count} ({churn_count/len(predictions)*100:.1f}%)")
        
        st.download_button(
            "Download Results", 
            df.to_csv(index=False).encode(), 
            "churn_predictions.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.write("Debugging information:")
        st.write("X shape:", X.shape)
        st.write("X data types:", X.dtypes)
        st.write("Sample of X:")
        st.dataframe(X.head())
        st.write("Missing values in X:", X.isnull().sum().sum())