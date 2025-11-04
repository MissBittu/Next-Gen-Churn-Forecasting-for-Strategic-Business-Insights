Next-Gen Churn Forecasting for Strategic Business Insights
A Hybrid Machine Learning and Deep Neural Ensemble for Predictive Retention Analytics
Project Objective

This project develops a hybrid predictive framework to forecast telecom customer churn with high accuracy and interpretability. It combines tree-based ensemble models with a deep neural network to analyse behavioural, contractual, and transactional features, aiming to provide actionable insights for customer retention strategies.


Key Features
Hybrid Architecture: Integrates XGBoost, LightGBM, CatBoost, Random Forests, Logistic Regression, and a PyTorch Neural Network into a unified ensemble.

Automated Optimisation: Employs Optuna’s Bayesian search for optimal Random Forest and meta-model hyperparameters.

Class Rebalancing: Utilizes SMOTE for minority class oversampling to mitigate bias toward non-churners.

Feature Engineering Depth: Introduces more than fifteen engineered variables capturing loyalty, billing behaviour, and customer engagement.

Performance: Achieves ROC-AUC scores in the range of 0.91–0.92, consistently outperforming single-model baselines.

Explainability: Supports SHAP and LIME for model interpretation and business explainability.

End-to-End Automation: A complete workflow for data processing, training, model blending, evaluation, and model export.

Table of Contents
Overview
Dataset
Model Architecture
Installation
Usage
Results
Ablation Study
Future Improvements
Project Structure
License

1. Overview

Telecom service providers experience significant financial loss due to customer churn. This project introduces a research-oriented hybrid ensemble that captures both interpretable decision patterns and nonlinear feature interactions.

By integrating stacked generalisation, neural blending, and behavior-driven feature engineering, this framework enhances predictive power and provides a foundation for data-driven retention strategy development.

2. Dataset

Source: IBM Telco Customer Churn Dataset

Samples: 7,043 customer records

Target: Churn (binary classification: Yes/No)

Category	Example Features	Description
Demographic	Gender, SeniorCitizen, Partner	Basic customer attributes
Contractual	Contract, PaymentMethod	Service and billing contract details
Financial	Monthly Charges, Total Charges, Customer spending and billing metrics
Behavioral	Internet Service, TechSupport	Indicators of engagement and service usage

3. Model Architecture
Feature Engineering

Behavioural Ratios: ChargeTenureRatio, LifetimeValue, LoyaltyScore

Engagement Indicators: AutoPayFlag, EngagementScore, HighChargeLoyal

Polynomial & Interaction Terms: Tenure², MonthlyCharges², Contract × MonthlyCharges

These transformations enhance signal strength for the models by representing complex customer behaviours.

 Ensemble Learning Framework 
 | Layer        | Model                                                                              | Function                                                                 |
| ------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Base Models  | Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, Logistic Regression | Diverse learners capturing different statistical and structural patterns |
| Meta Learner | XGBoost                                                                            | Learns stacked relationships across base model predictions               |
| Neural Layer | PyTorch DNN (256→128→64→32)                                                        | Extracts nonlinear and latent representations                            |
| Fusion       | Weighted α-blend                                                                   | Blends stacked and neural predictions for final inference                |

Optimization Pipeline

Optuna: Automated Bayesian hyperparameter tuning for Random Forest and meta learners

SMOTE: Synthetic oversampling of the minority churn class

Stratified K-Fold Cross Validation (5-fold): Ensures balanced model evaluation

Early Stopping: Prevents overfitting in the neural component


4. Installation

Clone the repository:

git clone https://github.com/MissBittu/Next-Gen-Churn-Forecasting-for-Strategic-Business-Insights.git
cd Next-Gen-Churn-Forecasting-for-Strategic-Business-Insights


Install dependencies:

pip install -r requirements.txt


Or manually:

pip install pandas numpy scikit-learn torch xgboost lightgbm catboost optuna imbalanced-learn joblib


5. Usage

Place the dataset in the root directory:

WA_Fn-UseC_-Telco-Customer-Churn.csv


Run the pipeline:

python churn_model.py


Outputs:

Trained ensemble model: output_enhanced/stacking_model.pkl

Evaluation metrics (AUC, F1, Recall) printed in console

Optional SHAP/LIME notebooks for interpretability


6. Results
| Model                               |       ROC-AUC |       F1 |   Recall | Accuracy |
| ----------------------------------- | ------------: | -------: | -------: | -------: |
| Logistic Regression                 |          0.82 |     0.73 |     0.70 |     0.79 |
| Random Forest (Optuna)              |          0.86 |     0.78 |     0.77 |     0.84 |
| LightGBM                            |          0.88 |     0.81 |     0.80 |     0.86 |
| CatBoost                            |          0.89 |     0.82 |     0.80 |     0.87 |
| Stacking Ensemble                   |          0.90 |     0.83 |     0.82 |     0.87 |
| Hybrid (Stack + Neural Network)     |      0.91–0.92|     0.85 |     0.83 |     0.88 |

The hybrid ensemble demonstrates improved generalisation and robustness across validation splits, achieving superior recall without compromising precision.


7. Ablation Study
| Enhancement              | Δ AUC | Observation                                             |
| ------------------------ | ----- | ------------------------------------------------------- |
| SMOTE Oversampling       | +0.02 | Reduces majority bias and improves minority detection   |
| Neural Ensemble Blending | +0.03 | Captures nonlinear churn tendencies                     |
| Optuna Optimization      | +0.02 | Enhances parameter efficiency and convergence stability |
| Feature Interactions     | +0.01 | Improves segment-level churn segmentation accuracy      |


Integration of SHAP dashboards for managerial interpretability

Deployment via FastAPI and Docker for real-time inference services

Exploration of AutoEncoder embeddings for segmentation analysis

Benchmarking against AutoML frameworks such as H2O and AutoGluon

Domain extension to banking, insurance, and SaaS retention modelling


9. Project Structure
Next-Gen-Churn-Forecasting/
│
├── churn_model.py                # Full hybrid ensemble pipeline
├── data/                         # Dataset storage
├── output_enhanced/              # Model outputs and artefacts
├── requirements.txt              # Dependencies
├── Dockerfile                    # Containerization configuration
└── README.md                     # Documentation


10. License
MIT License © 2025
Open for academic research and non-commercial applications.

Summary:
This repository is a research-grade churn prediction system that not only identifies at-risk customers but also explains the behavioural and economic factors driving churn. The model is designed to bridge data science and strategic decision-making through explainable, reproducible AI.
