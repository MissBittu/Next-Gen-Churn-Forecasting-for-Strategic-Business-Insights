# Next-Gen Churn Forecasting for Strategic Business Insights

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/) [![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen)](https://www.docker.com/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)

---

## Overview

This open-source project delivers a comprehensive hybrid machine learning and deep neural network pipeline designed to predict customer churn in telecom environments. By combining tree-based ensemble models (XGBoost, LightGBM, CatBoost) with a sophisticated stacking ensemble and PyTorch neural network, the system achieves high predictive performance while maintaining interpretability and production readiness.

### Core Capabilities

- **Balanced Classification** — SMOTE oversampling handles class imbalance in churn prediction
- **Advanced Feature Engineering** — Behavioural, billing, and loyalty metrics enhance predictive signals
- **Interactive UI** — Streamlit-based web application for easy data upload and prediction retrieval
- **Production-Ready Deployment** — Docker containerization ensures consistency across environments
- **High Performance** — ROC-AUC exceeding 0.90 with robust recall metrics

---

## Key Features

**Hybrid Machine Learning Architecture**
- Ensemble of six base models: Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, and Logistic Regression
- Meta-learner XGBoost for optimal combination of base model predictions
- PyTorch deep neural network capturing nonlinear patterns
- Weighted blending strategy for final prediction fusion

**Intelligent Optimisation**
- Optuna-based Bayesian hyperparameter tuning for Random Forest and meta-learners
- Stratified 5-fold cross-validation ensuring balanced evaluation
- Early stopping mechanisms preventing overfitting

**User-Friendly Interface**
- Web-based Streamlit application for data input and prediction
- Real-time feature engineering and preview
- CSV download functionality for batch prediction results
- Interactive visualisation of model predictions and probabilities

**Enterprise-Grade Deployment**
- Docker containerization for consistent reproducibility
- Support for local, on-premises, and cloud deployment
- Scalable architecture ready for production environments

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Optional: Docker Desktop (for containerised deployment)
- 4GB RAM minimum (8GB recommended)

### Local Installation

**Step 1: Clone the Repository**

```bash
git clone https://github.com/MissBittu/Next-Gen-Churn-Forecasting-for-Strategic-Business-Insights.git
cd Next-Gen-Churn-Forecasting-for-Strategic-Business-Insights
```

**Step 2: Navigate to Application Directory**

```bash
cd app
```

**Step 3: Create Virtual Environment**

```bash
# On Linux/macOS
python -m venv .venv
source .venv/bin/activate

# On Windows PowerShell
python -m venv .venv
.venv\Scripts\activate

# On Windows Command Prompt
python -m venv .venv
.venv\Scripts\activate.bat
```

**Step 4: Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Step 5: Run the Application**

```bash
streamlit run main.py
```

The application will be accessible at `http://localhost:8501` in your web browser.

---

## Usage Guide

### Using the Streamlit Web Interface

**Step 1: Access the Application**
Open your browser and navigate to the local Streamlit URL provided after running the application.

**Step 2: Upload Customer Data**
Select a CSV file containing customer information with the following expected columns:
- Tenure (months)
- Monthly Charges (currency)
- Total Charges (currency)
- Payment Method
- Contract Type
- Internet Service Type
- Technical Support Status
- And other relevant customer attributes

**Step 3: Review Data**
The application displays a preview of the uploaded data along with engineered features created from the raw input.

**Step 4: Generate Predictions**
Click the prediction button to process the data through the hybrid ensemble model. The system returns:
- Churn probability for each customer (0-1 scale)
- Binary churn prediction (Likely to Churn / Unlikely to Churn)
- Model confidence scores

**Step 5: Download Results**
Export predictions as a CSV file for further analysis, reporting, or integration with business systems.

---

## Model Performance

| Model | ROC-AUC | F1 Score | Recall | Accuracy |
|-------|---------|----------|--------|----------|
| Logistic Regression | 0.82 | 0.73 | 0.70 | 0.79 |
| Random Forest (Optuna) | 0.86 | 0.78 | 0.77 | 0.84 |
| LightGBM | 0.88 | 0.81 | 0.80 | 0.86 |
| CatBoost | 0.89 | 0.82 | 0.80 | 0.87 |
| Stacking Ensemble | 0.90 | 0.83 | 0.82 | 0.87 |
| **Hybrid (Stack + Neural Net)** | **0.91–0.92** | **0.85** | **0.83** | **0.88** |

The hybrid ensemble model demonstrates superior performance with 0.83 recall, capturing approximately 83% of actual churn cases. This metric is critical for retention campaigns where false negatives prove costly.

---

## Docker Deployment

### Local Docker Deployment

Docker containerization ensures the application runs consistently across different environments without dependency conflicts.

**Step 1: Prerequisites**

Ensure Docker Desktop is installed and running on your system. Verify installation with:

```bash
docker --version
```

**Step 2: Build Docker Image**

Navigate to the project directory and execute:

```bash
docker build -t churn-app .
```

This command creates a container image named `churn-app` with all dependencies pre-installed.

**Step 3: Run Docker Container**

```bash
docker run -p 8501:8501 churn-app
```

The application will be accessible at `http://localhost:8501`. The `-p 8501:8501` flag maps the container port to your local machine.

**Step 4: Stop the Container**

Press `CTRL+C` in your terminal or run:

```bash
docker stop <container_id>
```

### Cloud Deployment Options

**AWS (Amazon Web Services)**
- Deploy using AWS Elastic Container Service (ECS) for managed container orchestration
- Utilise AWS EC2 for virtual machine instances running the Docker container
- Store models in AWS S3 for centralised artefact management

**Render**
- Connect GitHub repository directly to Render
- Automatic deployments on repository updates
- Built-in Docker support with no configuration required
- Free tier available for initial testing

**Railway**
- Simple deployment with GitHub integration
- Automatic scaling based on traffic
- Pay-as-you-go pricing model

**Microsoft Azure**
- Deploy using Azure Container Instances (ACI) for quick containerised applications
- Azure App Service for managed web application hosting

**Google Cloud Platform (GCP)**
- Cloud Run for serverless container deployment
- Compute Engine for persistent virtual machines

### FastAPI REST Endpoint

For programmatic integration, deploy the model as a REST API:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

class CustomerData(BaseModel):
    tenure: float
    monthly_charges: float
    total_charges: float

@app.post("/predict")
def predict(data: CustomerData):
    prediction = model.predict([data.dict().values()])
    probability = model.predict_proba([data.dict().values()])
    
    return {
        "churn_prediction": bool(prediction[0]),
        "churn_probability": float(probability[0][1])
    }
```

---

## Project Structure

```
Next-Gen-Churn-Forecasting-for-Strategic-Business-Insights/
├── app/
│   ├── output_handcrafted/        # Pre-trained model artifacts
│   ├── Dockerfile                 # Container configuration
│   ├── main.py                    # Streamlit application entry point
│   └── requirements.txt           # Python dependencies
├── output_enhanced/               # Alternative model versions
├── catboost_info/                 # CatBoost training logs
├── churn_analysis.ipynb           # Exploratory data analysis notebook
├── README.md                      # Project documentation
└── WA_Fn-UseC_-Telco-Customer-Churn.csv    # Training dataset
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Programming Language** | Python 3.10 |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM, CatBoost |
| **Deep Learning** | PyTorch |
| **Hyperparameter Tuning** | Optuna |
| **Web Framework** | Streamlit |
| **Containerization** | Docker |
| **Data Processing** | Pandas, NumPy |
| **Imbalanced Learning** | imbalanced-learn (SMOTE) |

---

## Troubleshooting

### Docker Build Issues

**Problem**: Docker fails to build or connect
- Ensure Docker Desktop is running on your system
- On Windows, verify WSL2 (Windows Subsystem for Linux 2) is configured
- Check that you are in the correct directory before building

**Problem**: "exec: streamlit: executable file not found in $PATH"
- Verify `streamlit` is listed in `requirements.txt`
- Rebuild the Docker image after updating dependencies
- Check that pip installations completed successfully during build

### Model Version Warnings

**Problem**: `InconsistentVersionWarning` for scikit-learn versions
- This warning occurs when scikit-learn versions differ between model training and inference
- The model will still function correctly for predictions
- For production use, consider retraining models with the current scikit-learn version

### Streamlit Connection Issues

**Problem**: Cannot connect to `localhost:8501`
- Verify the Streamlit application started successfully (check console for error messages)
- Ensure port 8501 is not already in use by another application
- Try accessing `127.0.0.1:8501` instead of `localhost:8501`
- Check firewall settings if running on a corporate network

### Out of Memory Errors

**Problem**: Application crashes with memory errors
- Reduce batch size for predictions in the configuration
- Close other applications to free system memory
- Deploy on a machine with more available RAM (8GB+ recommended)

---

## Model Architecture

### Feature Engineering

The framework incorporates advanced feature engineering to strengthen predictive signals:

- **Behavioral Ratios**: ChargeTenureRatio, LifetimeValue, LoyaltyScore
- **Engagement Indicators**: AutoPayFlag, EngagementScore, HighChargeLoyal
- **Polynomial & Interaction Terms**: Tenure squared, MonthlyCharges squared, Contract interactions

### Ensemble Learning Framework

| Layer | Components | Purpose |
|-------|-----------|---------|
| **Base Models** | RF, XGB, LGBM, CB, GB, LR | Capture diverse patterns and relationships |
| **Meta Learner** | XGBoost | Learn optimal combination of base models |
| **Neural Layer** | PyTorch DNN (256→128→64→32) | Capture nonlinear latent relationships |
| **Fusion** | Weighted Blending | Combine predictions for final output |

### Optimisation Pipeline

- **Hyperparameter Tuning** — Optuna Bayesian optimisation for Random Forest and meta-learners
- **Class Imbalance** — SMOTE synthetic oversampling of minority churn class
- **Cross-Validation** — Stratified 5-Fold CV for robust and balanced evaluation
- **Regularisation** — Early stopping and dropout to prevent overfitting

---

## Configuration

### Feature Engineering Parameters

Adjust engineered feature calculations:

```python
charge_tenure_ratio = monthly_charges / max(tenure, 1)
lifetime_value = total_charges / max(tenure, 1)
loyalty_score = tenure / 72
```

### SMOTE Configuration

Fine-tune class balance parameters:

``` python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.6, random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

### Optuna Hyperparameter Search

Customise optimisation objectives:

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
    }
    return evaluate_model(params)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

## Future Work

**Enhanced Model Explainability**
- SHAP interactive dashboards for feature importance visualization
- LIME for individual prediction explanations
- Business intelligence reports with actionable retention recommendations

**Advanced Deployment Capabilities**
- REST API implementation using FastAPI
- Microservice architecture for independent scaling
- WebSocket support for real-time prediction streaming
- Authentication and rate limiting via API Gateway

**ML Ops and Monitoring**
- Model performance monitoring and drift detection
- Automated retraining pipelines when performance degrades
- A/B testing framework for model comparison
- Prediction logging and audit trails for compliance

**Industry Expansion**
- Banking sector customer retention prediction
- Insurance policy lapse forecasting
- Software-as-a-Service (SaaS) subscription churn prediction
- Healthcare patient dropout prediction

**Technical Enhancements**
- AutoML framework benchmarking (H2O, AutoGluon)
- Attention mechanisms for temporal customer behavior analysis
- Reinforcement learning for optimal retention strategy recommendations
- Customer segmentation using AutoEncoder embeddings

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

MIT License © 2025

This project is open for academic research and non-commercial applications. See the LICENSE file for details.

---

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation and the troubleshooting section
- Review the churn_analysis.ipynb notebook for detailed model analysis

---

## Summary

This repository provides a research-grade churn prediction framework that not only identifies at-risk customers but also explains their behavior through comprehensive behavioral and financial analysis. By bridging machine learning and business intelligence, the system enables explainable and actionable retention strategies grounded in data-driven insights.

**Key Principle**: Prediction is only useful when it leads to retention.

---

**Last Updated**: November 2025 | **Status**: Active Development | **Version**: 1.0.0
