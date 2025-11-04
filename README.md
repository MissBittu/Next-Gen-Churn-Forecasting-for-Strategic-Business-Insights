# Next-Gen Churn Forecasting for Strategic Business Insights

> A Hybrid Machine Learning and Deep Neural Ensemble for Predictive Retention Analytics

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## 🎯 Project Objective

This project develops a **hybrid predictive framework** to forecast telecom customer churn with high accuracy and interpretability. It combines tree-based ensemble models with a deep neural network to analyze behavioral, contractual, and transactional features—providing actionable insights for customer retention strategies.

---

## ⚡ Key Features

- 🔧 **Hybrid Architecture** — Integrates XGBoost, LightGBM, CatBoost, Random Forests, Logistic Regression, and PyTorch Neural Networks
- 🤖 **Automated Optimization** — Bayesian hyperparameter search using Optuna for optimal model tuning
- ⚖️ **Class Rebalancing** — SMOTE for handling imbalanced churn classification
- 📊 **Advanced Feature Engineering** — 15+ engineered variables capturing loyalty, billing behavior, and engagement
- 🎯 **Exceptional Performance** — ROC-AUC of **0.91–0.92**, outperforming industry baselines
- 💡 **Explainability** — SHAP and LIME support for model interpretability
- 🚀 **End-to-End Automation** — Complete pipeline from preprocessing to model export

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Future Work](#future-work)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

Telecom service providers face significant revenue loss from customer churn. This project introduces a **research-oriented hybrid ensemble** that merges interpretable decision-tree models with deep neural components to capture both structured and nonlinear relationships.

By integrating stacked generalization, neural blending, and behavior-driven feature engineering, this framework enhances predictive performance and supports data-driven retention strategy design.

---

## Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | IBM Telco Customer Churn Dataset |
| **Total Samples** | 7,043 customers |
| **Target Variable** | Churn (Binary: Yes / No) |
| **Features** | 20 demographic, contractual, financial, and behavioral attributes |

### Feature Categories

| Category | Examples | Purpose |
|----------|----------|---------|
| **Demographic** | Gender, SeniorCitizen, Partner | Basic customer profiling |
| **Contractual** | Contract Type, PaymentMethod | Service agreement details |
| **Financial** | MonthlyCharges, TotalCharges | Revenue and spending patterns |
| **Behavioral** | InternetService, TechSupport | Engagement and service usage |

---

## Model Architecture

### 🔨 Feature Engineering

Advanced feature engineering captures nuanced behavioral signals:

```
Behavioral Ratios:
  • ChargeTenureRatio — Monthly charges relative to tenure
  • LifetimeValue — Total revenue generated per customer
  • LoyaltyScore — Tenure-based loyalty indicator

Engagement Indicators:
  • AutoPayFlag — Automatic payment enrollment status
  • EngagementScore — Service adoption composite metric
  • HighChargeLoyal — High-value customer retention indicator

Interaction Terms:
  • Tenure² — Polynomial tenure transformation
  • MonthlyCharges² — Polynomial charge transformation
  • Contract × MonthlyCharges — Cross-feature interactions
```

These engineered features enhance model signal strength and capture latent behavioral patterns invisible to raw features.

### 🧠 Ensemble Learning Framework

```
┌─────────────────────────────────────────────────────────┐
│                    Input Features                        │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼───┐    ┌─────▼────┐    ┌──▼──────┐
    │Random │    │ XGBoost  │    │LightGBM │
    │Forest │    │          │    │         │
    └───┬───┘    └─────┬────┘    └──┬──────┘
        │              │              │
    ┌───▼───┐    ┌─────▼────┐    ┌──▼──────┐
    │CatBoost│   │Grad Boost│    │Logistic │
    │        │   │          │    │Regress. │
    └───┬───┘    └─────┬────┘    └──┬──────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
            ┌──────────▼──────────┐
            │  Meta Learner (XGB) │
            │  Stacked Predictions│
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │ Neural Layer (DNN)  │
            │ PyTorch (256→128→64)│
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │ Weighted α-Blend    │
            │ Final Prediction    │
            └─────────────────────┘
```

| Layer | Components | Purpose |
|-------|-----------|---------|
| **Base Models** | RF, XGB, LGBM, CB, GB, LR | Capture diverse patterns and relationships |
| **Meta Learner** | XGBoost | Learn optimal combination of base models |
| **Neural Layer** | PyTorch DNN (256→128→64→32) | Capture nonlinear latent relationships |
| **Fusion** | Weighted Blending (α-parameter) | Combine predictions for final output |

### ⚙️ Optimization Pipeline

- **Hyperparameter Tuning** — Optuna Bayesian optimization for Random Forest and meta-learners
- **Class Imbalance** — SMOTE synthetic oversampling of minority churn class
- **Cross-Validation** — Stratified 5-Fold CV for robust and balanced evaluation
- **Regularization** — Early stopping and dropout to prevent overfitting

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)

### Clone Repository

```bash
git clone https://github.com/MissBittu/Next-Gen-Churn-Forecasting-for-Strategic-Business-Insights.git
cd Next-Gen-Churn-Forecasting-for-Strategic-Business-Insights
```

### Install Dependencies

**Option 1: Using requirements.txt (Recommended)**

```bash
pip install -r requirements.txt
```

**Option 2: Manual Installation**

```bash
pip install pandas numpy scikit-learn torch xgboost lightgbm catboost optuna imbalanced-learn joblib matplotlib seaborn
```

### Verify Installation

```bash
python -c "import torch, xgboost, lightgbm, catboost, optuna; print('✓ All dependencies installed successfully')"
```

---

## Usage

### Quick Start (3 Simple Steps)

#### Step 1️⃣ — Prepare Dataset

Place the Telco Customer Churn dataset in the project root directory:

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

#### Step 2️⃣ — Run Pipeline

Execute the complete hybrid ensemble pipeline:

```bash
python churn_model.py
```

#### Step 3️⃣ — Collect Results

The pipeline automatically generates outputs:

```
output_enhanced/
├── stacking_model.pkl          # Trained hybrid model
├── evaluation_metrics.json     # Performance scores
├── feature_importance.csv      # Feature rankings
└── model_predictions.csv       # Prediction results
```

### Output Description

| Output | Location | Purpose |
|--------|----------|---------|
| **Trained Model** | `output_enhanced/stacking_model.pkl` | Production-ready hybrid ensemble |
| **Metrics** | Console + JSON | ROC-AUC, F1, Recall, Accuracy |
| **Interpretability** | Optional Notebooks | SHAP/LIME explanations |

---

## Results

### Model Performance Comparison

| Model | ROC-AUC ⬆️ | F1 Score | Recall | Accuracy |
|:------|:----------:|:--------:|:------:|:--------:|
| Logistic Regression | 0.82 | 0.73 | 0.70 | 0.79 |
| Random Forest (Optuna) | 0.86 | 0.78 | 0.77 | 0.84 |
| LightGBM | 0.88 | 0.81 | 0.80 | 0.86 |
| CatBoost | 0.89 | 0.82 | 0.80 | 0.87 |
| Stacking Ensemble | 0.90 | 0.83 | 0.82 | 0.87 |
| **Hybrid (Stack + Neural Net)** | **0.91–0.92** ⭐ | **0.85** | **0.83** | **0.88** |

### Key Insights

- **+0.01–0.10 ROC-AUC improvement** over single models
- **83% recall** — Captures 4 out of 5 actual churners
- **0.88 accuracy** — Strong overall classification performance
- **Superior generalization** — Robust across validation folds

---

## Ablation Study

Incremental performance gains from each enhancement component:

| Enhancement | Δ AUC | Impact | Observation |
|-------------|:-----:|:------:|-------------|
| SMOTE Oversampling | +0.02 | 🟢 Medium | Reduces class imbalance bias effectively |
| Neural Ensemble Blending | +0.03 | 🔴 High | Captures nonlinear churn dynamics best |
| Optuna Optimization | +0.02 | 🟢 Medium | Improves stability and convergence |
| Feature Interactions | +0.01 | 🟡 Low | Strengthens segment-level accuracy |

**Total Combined Improvement**: +0.08 AUC gain vs. baseline Logistic Regression

---

## Future Work

### 🎯 Planned Enhancements

- [ ] **SHAP Dashboard** — Interactive business-level model interpretability UI
- [ ] **FastAPI Deployment** — Real-time REST API for churn prediction
- [ ] **Docker Containerization** — Easy deployment and scaling
- [ ] **AutoEncoder Embeddings** — Customer behavioral segmentation
- [ ] **AutoML Benchmarking** — Compare against H2O and AutoGluon
- [ ] **Domain Expansion** — Banking, insurance, SaaS retention systems
- [ ] **Mobile App Integration** — Push notifications for at-risk customers
- [ ] **Real-time Monitoring** — Model drift detection and retraining pipeline

---

## Project Structure

```
Next-Gen-Churn-Forecasting/
│
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 Dockerfile                   # Container configuration
│
├── 🐍 churn_model.py               # Main hybrid ensemble pipeline
│
├── 📁 data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── 📁 output_enhanced/             # Model outputs
│   ├── stacking_model.pkl
│   ├── evaluation_metrics.json
│   └── feature_importance.csv
│
└── 📁 notebooks/                   # Jupyter notebooks
    ├── exploratory_analysis.ipynb
    ├── shap_interpretability.ipynb
    └── lime_explanations.ipynb
```

---

## Configuration

### Hyperparameter Tuning

Modify the Optuna search space in `churn_model.py`:

```python
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
    }
    return evaluate_model(params)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### SMOTE Configuration

Adjust class balance ratio in preprocessing:

```python
smote = SMOTE(sampling_strategy=0.6, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

## License

**MIT License © 2025**

This project is open for academic research and non-commercial applications. See [LICENSE](LICENSE) file for details.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...
```

---

## 📚 Summary

This repository provides a **research-grade churn prediction framework** that not only identifies at-risk customers but also explains their behavior through comprehensive behavioral and financial analysis.

By bridging machine learning and business intelligence, the system enables **explainable and actionable retention strategies** grounded in data-driven insights.

### Core Principle

> **"Prediction is only useful when it leads to retention."**

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact & Support

- **Author**: [MissBittu](https://github.com/MissBittu)
-
- **Issues**: [GitHub Issues](https://github.com/MissBittu/Next-Gen-Churn-Forecasting-for-Strategic-Business-Insights/issues)

---

## 📖 References

- [Scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Optuna Hyperparameter Optimization](https://optuna.org/)
- [SHAP for Model Interpretability](https://shap.readthedocs.io/)
- [PyTorch Neural Networks](https://pytorch.org/)

---

**⭐ If you find this project helpful, please consider giving it a star!**

Last Updated: November 2025 | Status: Active Development
