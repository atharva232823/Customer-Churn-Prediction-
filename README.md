# 📊 Customer Churn Prediction – Machine Learning Project

This project focuses on predicting customer churn using Machine Learning techniques.  
It includes complete steps from exploratory data analysis (EDA) to model training, preprocessing, evaluation, and deployment using a Streamlit application.

---

## 🚀 Project Overview

Customer churn is a key performance indicator for subscription-based industries such as telecom, SaaS, banking, and internet services.  
The goal of this project is to develop a model that can accurately predict whether a customer is likely to churn based on historical customer data.

This repository contains:

- Jupyter notebooks for training and experimentation  
- Python scripts used for inference and Streamlit app  
- Preprocessing objects (encoders, scaler, imputer)  
- Trained SVC model saved as `.pkl`  
- Project assets (images, banner)  

---

## 🧠 Machine Learning Pipeline

The ML workflow implemented includes:

### ✔ 1. Data Preprocessing  
- Handling missing values  
- Feature encoding  
- Scaling numerical columns  
- Outlier removal  
- Train-test split

### ✔ 2. Exploratory Data Analysis  
- Understanding churn patterns  
- Visualizing numerical and categorical features  
- Correlation heatmaps  
- Insights into key factors that influence churn

### ✔ 3. Model Development  
Models explored include:

- Support Vector Classifier (SVC) — **Final Model**
- Logistic Regression  
- Random Forest  
- Decision Tree  
- KNN  

The best-performing model (based on accuracy and F1 score) was saved as `svc_model.pkl`.

### ✔ 4. Model Deployment  
A Streamlit app is included so users can interact with the model:

```bash
streamlit run App.py
