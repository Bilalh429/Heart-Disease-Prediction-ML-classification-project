## 📖 Project Overview

Cardiovascular disease remains one of the leading causes of death worldwide. Early detection plays a critical role in reducing mortality rates and improving patient outcomes.

In this project, I assume the role of a **Machine Learning Analyst in a healthcare analytics team**, tasked with developing a predictive model that can classify whether a patient has heart disease based on clinical attributes.

Using the **UCI Heart Disease dataset (~300 records, 14 features)**, the objective is to transform structured medical data into a reliable classification model and evaluate it rigorously using confusion matrix metrics within a real-world medical context.

The focus of this project is not only prediction — but **interpretability, risk evaluation, and healthcare-aware decision making**.

---

## 🎯 Objectives

* Build a supervised binary classification model to detect heart disease
* Perform structured Exploratory Data Analysis (EDA)
* Identify features most correlated with heart disease
* Interpret Logistic Regression coefficients
* Analyze confusion matrix components in medical context
* Manually compute Accuracy, Precision, Recall, and F1-score
* Evaluate the trade-off between Precision and Recall
* Determine which metric matters most for disease detection

---

## 📂 Dataset Overview

**Source:** Kaggle – UCI Heart Disease Dataset
**Records:** ~300 patients
**Features:** 14 clinical attributes
**Target Variable:** `target`

* `1` → Patient has heart disease
* `0` → Patient does not have heart disease

### Key Features

| Feature  | Description                  |
| -------- | ---------------------------- |
| age      | Age of patient               |
| sex      | Gender                       |
| cp       | Chest pain type              |
| trestbps | Resting blood pressure       |
| chol     | Serum cholesterol            |
| fbs      | Fasting blood sugar          |
| restecg  | Resting ECG results          |
| thalach  | Maximum heart rate achieved  |
| exang    | Exercise induced angina      |
| oldpeak  | ST depression                |
| slope    | Slope of ST segment          |
| ca       | Number of major vessels      |
| thal     | Thalassemia                  |
| target   | Heart disease presence (1/0) |

---

## 🛠️ Tools & Technologies

* Python
* Pandas & NumPy – data manipulation
* Matplotlib & Seaborn – visualization
* Scikit-learn – model building & evaluation
* Jupyter Notebook – documentation

---

## 🔄 Project Workflow

---

### 1️⃣ Phase 1: Data Loading & Inspection

* Loaded CSV into Pandas DataFrame
* Checked shape, `.head()`, and `.info()`
* Analyzed class distribution
* Identified whether dataset is balanced

### Dataset Characteristics

* Supervised learning problem
* Binary classification
* Slightly balanced class distribution

---

### 2️⃣ Phase 2: Exploratory Data Analysis (EDA)

Performed structured EDA to understand feature behavior:

* `.describe()` statistical summary
* Missing value verification
* Target variable distribution visualization
* Correlation heatmap
* Feature comparison using boxplots

### Key EDA Visualizations Included:

* Target distribution bar chart
* Correlation matrix heatmap
* Boxplots comparing disease vs non-disease groups

---

### 3️⃣ Phase 3: Data Preparation

* Separated features (X) and target (y)
* Train-test split (80/20)

  * `random_state=42`
  * `stratify=y`
* Applied `StandardScaler` for feature normalization

Feature scaling ensures Logistic Regression coefficients are comparable and optimization converges efficiently.

---

### 4️⃣ Phase 4: Model Training

**Model Used:** Logistic Regression

Why Logistic Regression?

* Interpretable coefficients
* Probabilistic output via sigmoid function
* Suitable for structured medical data
* Strong baseline classifier

The sigmoid function:

[
\sigma(z) = \frac{1}{1 + e^{-z}}
]

Maps linear outputs into probabilities between 0 and 1.

---

### 5️⃣ Phase 5: Model Evaluation

Evaluation focused heavily on confusion matrix analysis.

#### Confusion Matrix Interpretation

|                 | Predicted Healthy   | Predicted Diseased  |
| --------------- | ------------------- | ------------------- |
| Actual Healthy  | True Negative (TN)  | False Positive (FP) |
| Actual Diseased | False Negative (FN) | True Positive (TP)  |

### 📈 Metrics Calculated

* Accuracy
* Precision
* Recall
* F1 Score

Metrics were:

1. Manually calculated using confusion matrix values
2. Verified using `classification_report`

---

## 📊 Medical Risk Analysis

### False Negative (Most Dangerous)

A False Negative means:

> The patient HAS heart disease but the model predicts they are healthy.

This can delay treatment and increase mortality risk.

### False Positive

A False Positive means:

> The patient is healthy but predicted as diseased.

This may cause unnecessary testing but is less dangerous.

### Conclusion

For disease detection systems:

> **Recall is more important than Precision.**

A model with:

* 85% Accuracy
* 50% Recall

Is NOT suitable for medical screening because it misses half of actual patients.

---

## 🧠 Key Insights & Findings

* Certain clinical features show strong correlation with heart disease
* Logistic Regression provides interpretable risk factors
* Recall is the most critical metric in healthcare classification
* High accuracy alone does not guarantee a reliable medical model
* Confusion matrix interpretation is essential for real-world ML applications

---

## 📌 What I Learned

Through this project, I gained hands-on experience in:

* End-to-end classification workflow
* Logistic Regression mathematics
* Sigmoid function interpretation
* Confusion matrix deep analysis
* Precision–Recall tradeoff reasoning
* Healthcare-aware model evaluation
* Writing business-context interpretations of ML results

---

## 🚀 Project Outcomes

* Built a fully documented machine learning notebook
* Developed a medically interpretable classifier
* Performed manual metric derivations
* Demonstrated understanding beyond basic model training
* Created a portfolio-quality healthcare AI project

---

## 📎 Files in This Repository

```
heart-disease-classification-logistic-regression/
│
├── Heart_Disease_main.ipynb   # Complete ML analysis
├── heart.csv                  # Dataset
├── README.md                  # Project documentation
└── requirements.txt           # Dependencies
```

---

## 🔮 Future Improvements

* ROC-AUC analysis
* Cross-validation
* Hyperparameter tuning
* Decision Tree / Random Forest comparison
* Model deployment via Streamlit

---

## ⭐ Acknowledgements

Dataset sourced from Kaggle (UCI Heart Disease Dataset).
Inspired by real-world medical AI risk assessment systems.
