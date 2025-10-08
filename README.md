Telco Customer Churn Prediction

This project builds a complete end-to-end machine learning pipeline to predict customer churn for a telecom company.
It covers data cleaning, feature engineering, model training, evaluation, and deployment through a Streamlit web app.

🚀 Project Highlights

Uses real-world Telco Customer Churn dataset from Kaggle

Feature engineering focused on interpretability and business relevance

Handles class imbalance using SMOTETomek

Selects the best features with SelectKBest

Trains multiple models: Logistic Regression, Random Forest, Gradient Boosting

Combines them with a Voting Ensemble for better performance

Evaluates performance with Accuracy, Precision, Recall, F1, and ROC-AUC

Deploys a Streamlit app for quick batch predictions and result downloads

🧠 Tech Stack

Language: Python 3.x

Libraries: Pandas, NumPy, Scikit-learn, Imbalanced-learn

Notebook: Jupyter

Deployment: Streamlit

⚙️ Setup Instructions
1. Clone the repository
git clone https://github.com/MissBittu/Next-Gen-Churn-Forecasting-for-Strategic-Business-Insights.git
cd Next-Gen-Churn-Forecasting-for-Strategic-Business-Insights

2. Install dependencies
pip install -r requirements.txt

3. (Optional) Train the model

If you want to retrain the model:

Jupyter Notebook churn_analysis.ipynb


This will create all model artifacts inside the output_handcrafted/ folder.

4. Run the app
streamlit run app.py

 Project Structure
Next-Gen-Churn-Forecasting/
│
├── churn_analysis.ipynb       # Training and model building
├── app.py                     # Streamlit web app for predictions
├── requirements.txt           # Dependencies
├── output_handcrafted/        # Trained model files
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── feature_selector.pkl
│   ├── selected_features.txt
│   └── churn_predictions.csv
└── README.md                  # Documentation

🧪 How It Works

Upload a CSV file with customer data

The app cleans and processes the data automatically

It applies the same transformations as the training pipeline

Predictions are generated with churn probabilities

You can download the results as a CSV file

📈 Insights from the Model

Customers with month-to-month contracts are most likely to churn

New users with high monthly charges have a higher risk

Multiple services and longer tenure reduce churn probability

Electronic check payments correlate with higher churn

These insights help identify at-risk customers and design better retention strategies.

📊 Dataset Source

Telco Customer Churn dataset(kaggle):https://www.kaggle.com/datasets/blastchar/telco-customer-churn
