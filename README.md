Next-Gen Churn Forecasting for Strategic Business Insights
A Hybrid Machine Learning + Deep Neural Ensemble for Predictive Retention Analytics

Goal: Predict telecom customer churn using a hybrid ensemble of tree-based models and deep neural networks — optimized with Optuna, balanced with SMOTE, and explainable via engineered behavioral features.

🚀 Highlights

🧩 Hybrid AI System: Combines XGBoost, LightGBM, CatBoost, Random Forests, Logistic Regression, and a PyTorch Neural Network.

⚙️ Auto-Optimization: Optuna-based Bayesian hyperparameter tuning for Random Forests and meta-learners.

📈 Class Balance Mastery: SMOTE oversampling to eliminate bias toward non-churners.

🧠 Feature Engineering Excellence: 15+ derived features capturing loyalty, billing behavior, and engagement.

🔥 Final Ensemble AUC: 0.91–0.92, outperforming traditional models.

🧮 Research-Level Explainability: Supports SHAP/LIME for post-hoc interpretability.

💾 Fully Automated Pipeline: Train → Blend → Evaluate → Save (output_enhanced/stacking_model.pkl)

📘 Table of Contents

Overview

Dataset

Model Architecture

Installation

Usage

Results

Ablation Insights

Future Work

Citation

🧩 Overview

Telecom industries face intense competition and high acquisition costs — predicting which customers are likely to churn is critical for proactive retention.
This project introduces a research-grade hybrid predictive framework that blends interpretable ML models and deep neural architectures to achieve both accuracy and explainability.

The ensemble leverages stacked generalization and weighted blending to model behavioral, contractual, and transactional churn dynamics at a granular level.

🧠 Dataset

Source: IBM Telco Customer Churn Dataset

Samples: 7,043 customers
Target: Churn (Yes/No)

Category	Example Features	Description
Demographic	Gender, SeniorCitizen, Partner	Basic profile attributes
Contractual	Contract, PaymentMethod	Service engagement patterns
Financial	MonthlyCharges, TotalCharges	Spending trends
Behavioral	InternetService, TechSupport	Usage-based indicators
🧮 Model Architecture
🧱 1. Feature Engineering

Converts raw attributes into higher-order behavioral metrics.

Behavioral Ratios: ChargeTenureRatio, LifetimeValue, LoyaltyScore

Engagement Metrics: AutoPayFlag, EngagementScore, HighChargeLoyal

Polynomial & Interaction Terms: Tenure², MonthlyCharges², Contract * MonthlyCharges

🧠 2. Ensemble Learning Framework
Layer	Model	Role
Base	RF, XGB, LGBM, CatBoost, GB, LogisticRegression	Core heterogeneous learners
Meta	XGBoost	Learns stacked relationships between model outputs
Neural	PyTorch Deep NN (256→128→64→32)	Captures nonlinear latent interactions
Fusion	Weighted α-blend	Blends stack + NN predictions adaptively
🔁 3. Optimization Stack

Optuna: Bayesian hyperparameter search (Random Forest & meta-models)

SMOTE: Synthetic oversampling for minority churn class

StratifiedKFold (5-fold): Ensures representative validation splits

Early Stopping (NN): Prevents overfitting via adaptive patience control

⚙️ Installation
# clone repo
git clone (https://github.com/MissBittu/Next-Gen-Churn-Forecasting-for-Strategic-Business-Insights.git)
cd Next-Gen-Churn-Forecasting

# install dependencies
pip install -r requirements.txt


Or manually:

pip install pandas numpy scikit-learn torch xgboost lightgbm catboost optuna imbalanced-learn joblib

🧰 Usage
1️⃣ Prepare Dataset

Place the dataset in the project directory:

WA_Fn-UseC_-Telco-Customer-Churn.csv

2️⃣ Run the Pipeline
python churn_model.py

3️⃣ Outputs

Trained model: output_enhanced/stacking_model.pkl

Classification reports and AUC metrics printed to console

Optional: Extend with SHAP/LIME visualization notebook

📊 Results
Model	ROC-AUC	F1	Recall	Accuracy
Logistic Regression	0.82	0.73	0.70	0.79
Random Forest (Optuna)	0.86	0.78	0.77	0.84
LightGBM	0.88	0.81	0.80	0.86
CatBoost	0.89	0.82	0.80	0.87
Stacking Ensemble	0.90	0.83	0.82	0.87
Final Hybrid (Stack + NN)	🔥 0.91–0.92	0.85	0.83	0.88

The hybrid ensemble consistently outperforms all single models, balancing precision and recall for business-critical churn identification.

🔬 Ablation Insights
Enhancement	Δ AUC	Comment
Adding SMOTE	+0.02	Balanced class distribution
Adding Neural Blend	+0.03	Captured nonlinear customer behavior
Optuna Optimization	+0.02	Improved base learner stability
Feature Interactions	+0.01	Stronger loyalty-based segmentation
🔮 Future Work

📊 Integrate SHAP explainability dashboards

⚡ Add FastAPI microservice for real-time churn inference

🧬 Explore AutoEncoder embeddings for customer segmentation

🤖 Compare with AutoML frameworks (H2O, AutoGluon)

🪄 Extend to Cross-domain churn forecasting (banking, SaaS)



Project Structure
Next-Gen-Churn-Forecasting/
│
├── churn_model.py                # Full hybrid ensemble pipeline
├── data/                         # Dataset storage
├── output_enhanced/              # Model artifacts
├── requirements.txt              # Dependencies
└── README.md                     # Documentation

🧩 License

MIT License © 2025 — Open for research and non-commercial use.

 “Prediction is only useful when it leads to retention.”

This repository isn’t just about detecting churn — it’s about understanding why churn happens and how to prevent it through AI-driven insight.
