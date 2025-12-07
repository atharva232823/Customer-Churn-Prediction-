# 📘 USER GUIDE – Customer Churn Prediction Project

This guide explains how to **download, open, run, and understand** the complete Customer Churn Prediction Machine Learning Project.

---

## 1. Project Purpose
This project predicts whether a customer will churn using a trained **Support Vector Classifier (SVC)** model.  
It includes:
- Data preprocessing  
- Feature engineering  
- EDA (Exploratory Data Analysis)  
- Model training & evaluation  
- Saving preprocessing and model files  
- A **Streamlit web application** for real-time prediction  

---

## 2. Project Folder Location
All essential project files are located inside:


This folder contains:
- Main Jupyter Notebook  
- Streamlit App  
- Trained ML model  
- Encoder and scaler files  
- Testing notebook  
- Banner image  

---

## 3. How to Download the Project

### ✅ Option A  — Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git

### ✅ Option B — Download ZIP
```bash
If you do not want to use Git, you can download the project directly as a ZIP file:

1. Go to your GitHub repository  
2. Click the green **Code** button  
3. Select **Download ZIP**  
4. Extract the ZIP file to any folder on your computer  

---

## 4. How to Install Requirements

Before running the project, install the required Python libraries.

### ✔ Option A — Using requirements.txt
If a `requirements.txt` file exists in the project:
```bash
pip install -r requirements.txt

### ✔ Option B — Install manually  
If you want to install required libraries manually, run:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit

---

## 5. How to Open and Run the Main Notebook

Follow these steps to open the primary Jupyter Notebook used for training and analysis:

### ▶ Step 1 — Open Terminal or Command Prompt  
Navigate to the project folder:
```bash
cd Machine_Learning_Model

### ▶ Step 2 — Launch Jupyter Notebook
Run the following command to start Jupyter Notebook:
```bash
jupyter notebook

### ▶ Step 3 — Open the Main Notebook File
When Jupyter Notebook opens in your browser, click on the following file:

This notebook contains:
- Data loading and cleaning  
- Exploratory Data Analysis (EDA)  
- Feature encoding and scaling  
- Training the SVC model  
- Evaluating model performance  
- Saving `.pkl` files for preprocessing and model prediction  

---

## 6. How to Run the Streamlit App

The Streamlit application provides an easy interface for generating churn predictions.

### ▶ Step 1 — Navigate to the Project Folder
```bash
cd Machine_Learning_Model

### ▶ Step 2 — Run Streamlit
Run the Streamlit app using the following command:
```bash
streamlit run App.py

### ▶ Step 3 — Open the Streamlit App in Your Browser
If the app does not open automatically, you can manually open it using the link below:


Once opened, the app will allow you to input customer information such as:
- Tenure  
- Payment Method  
- Contract Type  
- Monthly Charges  
- Internet Service  
- Senior Citizen status  
- Additional customer details  

After entering the details, click **Predict** to instantly see whether the customer is predicted to churn or not.

---

## 7. Important Project Files

The project contains the following important files:

| File Name | Purpose |
|-----------|---------|
| **App.py** | Streamlit UI application for live predictions |
| **Customer_Churn_SVC.ipynb** | Main Jupyter Notebook (training, preprocessing, EDA) |
| **Streamlit_run.ipynb** | Notebook for testing Streamlit logic |
| **svc_model.pkl** | Final trained SVC model |
| **imputer.pkl** | Handles missing values automatically |
| **scaler.pkl** | Scales all numerical columns |
| **ohe_encoder.pkl** | One-hot encoder for categorical features |
| **ordinal_encoder_contract.pkl** | Encoder for contract type |
| **ordinal_encoder_internet.pkl** | Encoder for internet service |
| **Banner.png** | Banner image used in the Streamlit UI |

---

## 8. How the ML Pipeline Works

Here’s how the entire churn prediction pipeline works step-by-step:

1. **User Input**  
   Customer details are entered through the Streamlit UI.

2. **Missing Value Handling**  
   `imputer.pkl` fills any missing values in the input.

3. **Categorical Data Encoding**  
   The app applies the saved encoders:
   - `ohe_encoder.pkl`
   - `ordinal_encoder_contract.pkl`
   - `ordinal_encoder_internet.pkl`

4. **Numerical Scaling**  
   The `scaler.pkl` file scales numerical columns.

5. **Prediction Using Trained Model**  
   The processed inputs are sent to `svc_model.pkl`, which returns the churn prediction.

6. **Output Display**  
   Streamlit shows the final result clearly on the screen.

All these steps run automatically inside **App.py** whenever the user clicks the Predict button.

---
