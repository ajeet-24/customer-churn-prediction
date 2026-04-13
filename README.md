# 📉 Customer Churn Prediction System

A machine learning project that predicts telecom customer churn using **XGBoost** and **Random Forest**, with SMOTE for class imbalance handling, SHAP for model explainability, and dual deployment via **Flask** and **FastAPI**.

---

## 📌 Features

- Predicts whether a customer will churn or not
- Handles class imbalance using **SMOTE**
- Hyperparameter tuning with **GridSearchCV**
- Model explainability using **SHAP** (global + local)
- **Customer Lifetime Value (CLV)** integration and priority segmentation
- Business cost analysis with threshold optimization
- Deployed via **Flask** web app and **FastAPI** REST API

---

## 🗂️ Project Structure

```
customer-churn-prediction/
├── Customer_Churn_Prediction.ipynb       # Main ML notebook
├── app.py                                # Flask web application
├── fastapi_app.py                        # FastAPI REST API
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── best_models.pkl                       # Trained XGBoost model
├── encoders.pkl                          # Label encoders
├── scaler.pkl                            # StandardScaler
├── requirements.txt                      # Python dependencies
├── templates/
│   └── index.html                        # Flask HTML template
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Run the Notebook

Open `Customer_Churn_Prediction.ipynb` in Jupyter or Google Colab and run all cells to train and evaluate models, which will generate `best_models.pkl`, `encoders.pkl`, and `scaler.pkl`.

### 2. Run Flask App

```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

### 3. Run FastAPI

```bash
uvicorn fastapi_app:app --reload
```
Visit `http://localhost:8000/docs` for the interactive Swagger UI.

**Example API request:**
```json
POST /predict
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "tenure": 24,
  "MonthlyCharges": 65.0,
  "TotalCharges": 1560.0
}
```

---

## 🧠 ML Pipeline

```
Raw Data → EDA → Preprocessing → SMOTE → GridSearchCV → Best Model → SHAP → Deploy
```

1. **EDA** — distribution plots, correlation heatmap, class imbalance check
2. **Preprocessing** — Label Encoding for categoricals, StandardScaler for numericals
3. **Class Imbalance** — handled with SMOTE (oversampling minority class)
4. **Models** — Random Forest & XGBoost with GridSearchCV tuning
5. **Threshold Tuning** — optimized at 0.41 for best F1 score
6. **Explainability** — SHAP summary + waterfall plots

---

## 📊 Model Performance (XGBoost, threshold=0.41)

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.81  |
| F1 Score  | 0.652 |
| Recall    | 0.678 |
| Precision | 0.628 |

---

## 💰 CLV & Priority Segmentation

Customers are segmented based on churn probability and Customer Lifetime Value:

| Segment          | Condition                               |
|------------------|-----------------------------------------|
| 🔴 Critical       | Churn prob ≥ 0.7 AND CLV ≥ 75th pct   |
| 🟠 High Priority  | Churn prob ≥ 0.5 AND CLV ≥ 50th pct   |
| 🔵 Monitor        | Churn prob ≥ 0.3                       |
| 🟢 Safe           | Churn prob < 0.3                       |

---

## 📁 Dataset

- **Source:** [Telco Customer Churn – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Features:** 20 input features + 1 target (`Churn`)
- **Target:** Binary — `Yes` (churn) / `No` (no churn)

---

## 📦 Key Dependencies

```
scikit-learn
xgboost
imbalanced-learn
shap
flask
fastapi
uvicorn
pandas
numpy
```

---

## 📝 Notes

- `.pkl` files are included for quick deployment without retraining
- Flask app requires a `templates/index.html` file for the frontend form
- Model was trained on Google Colab; update file paths if running locally