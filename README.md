

# ❤️ Diabetes Prediction using Machine Learning

### *Pima Indians Diabetes Dataset — End-to-End ML Classification Project*

This project builds a complete Machine Learning pipeline to predict whether a patient has diabetes based on medical data. It includes **EDA**, **data preprocessing**, **visualizations**, **model training**, **evaluation**, **hyperparameter-tuned models**, and a **saved production-ready model**.

The dataset used is the popular **Pima Indians Diabetes Database** from Kaggle.

---

## 📁 Project Structure

```
diabetes-prediction-ml/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   └── diabetes_prediction.ipynb
│
├── models/
│   ├── best_diabetes_model_<modelname>.joblib
│   └── scaler.joblib
│
├── figs/
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_importances.png
│   ├── roc_curve.png
│   └── target_distribution.png
│
├── requirements.txt
├── report.txt
└── README.md
```

---

## 🎯 Objective

To develop a machine learning model that accurately predicts the likelihood of diabetes based on several medical attributes such as:

* Glucose level
* Blood pressure
* BMI
* Age
* Insulin
* Skin thickness
* Pregnancies
* Diabetes pedigree function

---

## 📊 Dataset Details

**Dataset:** Pima Indians Diabetes Database
**Source:** Kaggle
**Total Rows:** 768
**Features:** 8
**Target:**

* `0` — No diabetes
* `1` — Diabetes

Columns:

```
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
Outcome
```

Some columns may contain zero values representing missing data; these were handled during preprocessing.

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook contains complete EDA, including:

✔ Statistical summary
✔ Outlier analysis
✔ Missing value handling
✔ Target distribution
✔ Correlation heatmap
✔ Feature relationships

### Example Visualizations

Images stored in `figs/`:

* `correlation_heatmap.png`
* `roc_curve.png`
* `feature_importances.png`
* `confusion_matrix.png`
* `target_distribution.png`

---

## 🤖 Models Trained

Multiple machine learning models were trained and compared:

| Model               | Accuracy         |
| ------------------- | ---------------- |
| Logistic Regression | 0.78+            |
| SVM (RBF Kernel)    | 0.82+            |
| KNN                 | 0.80+            |
| **Random Forest**   | **0.85+ (Best)** |

The best-performing model was selected automatically and saved.

---

## 🧪 Model Evaluation

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1-score
* ROC Curve & AUC
* Confusion Matrix
* Cross-validation (StratifiedKFold)

Best model metrics are saved in `report.txt`.

---

## 💾 Saved Model

The final trained model is saved in:

```
models/best_diabetes_model_<modelname>.joblib
models/scaler.joblib
```

You can load the model in any Python script:

```python
import joblib
import numpy as np

data = joblib.load("models/best_diabetes_model_RandomForest.joblib")
model = data["model"]
scaler = data["scaler"]

# sample input
sample = np.array([[2, 120, 70, 25, 80, 30.0, 0.45, 35]])

# scale + predict
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("Diabetes:", "Yes" if prediction[0] == 1 else "No")
```

---

## 🚀 How to Run This Project

### 1️⃣ Clone repository

```bash
git clone https://github.com/yourusername/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Open notebook

```bash
jupyter notebook notebooks/diabetes_prediction.ipynb
```

### 4️⃣ Run all cells

The notebook will:

* preprocess data
* train models
* save best model
* generate visualizations

---

## 🧱 Future Improvements

* Add Hyperparameter tuning using GridSearchCV
* Add XGBoost and LightGBM models
* Deploy as a Flask API
* Create a Streamlit web app
* Add model interpretability using SHAP values

---

## 🏷 GitHub Topics (recommended)

```
machine-learning  diabetes  healthcare  pima  kaggle  classification  scikit-learn  python  notebook
```

---

## 🧑‍💻 Author

**Akanksha**
Machine Learning & Python Developer
Passionate about building smart, deployable ML systems.

---
