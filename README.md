# Loan Default Prediction System

A practical loan default risk assessment system built using an Artificial Neural Network (ANN) with embedding layers for categorical features.  
The project focuses on building a **realistic decision-support tool**, not just a high-accuracy demo.

The system is trained on a large tabular dataset and deployed as an interactive Streamlit web application.

---

## Overview

Loan default prediction is a critical problem in financial institutions where decisions must balance risk, fairness, and consistency.  
This project models loan default probability using structured customer and loan data, with careful preprocessing and a clean inference pipeline.

Instead of relying on heavy feature engineering or complex models, the emphasis is on:
- Stable training
- Proper handling of categorical variables
- Reusable preprocessing
- Deployment-ready architecture

---

## Key Features

- End-to-end pipeline from data preprocessing to deployment
- ANN model designed for tabular financial data
- Embedding layers for high-cardinality categorical features
- Consistent preprocessing between training and inference
- Streamlit based user interface for real-time predictions
- Clean project structure suitable for production style workflows

---

## Model Details

- **Model type**: Artificial Neural Network (ANN)
- **Output**: Binary classification (Default / No Default)
- **Loss function**: Binary Cross Entropy
- **Optimizer**: Adam
- **Regularization**: Dropout + Early Stopping

### Categorical Handling
- Ordinal encoding for ordered features (Education)
- Label encoding for binary features
- Embedding layers for:
  - Employment Type
  - Marital Status
  - Loan Purpose

### Numerical Features
- Scaled using `StandardScaler`
- Same scaler reused during inference to prevent data leakage

---

## Project Structure

```text
loan-default-prediction/
│
├── app/
│   └── app.py                     # Streamlit application
│   └── app(only for streamlit).py
├── model/
│   └── model.h5                   # Trained ANN model
│
├── pickle/
│   ├── scaler.pkl
│   ├── education_encoder.pkl
│   ├── HasMortgage_encoder.pkl
│   ├── HasDependents_encoder.pkl
│   ├── HasCoSigner_encoder.pkl
│   ├── employment_encoder.pkl
│   ├── marital_encoder.pkl
│   └── loanpurpose_encoder.pkl
│
├── Dataset/
│   └── Loan_default.csv           # Original dataset
│
├── Notebooks/
│   └── Loan Default Prediction.ipynb
│
├── requirements.txt
└── README.md
```

---

## Web Application

The Streamlit application allows users to:
- Enter applicant and loan details
- Run real-time predictions
- View default risk as a probability score
- Use the model as a **decision-support system**, not an automated decision maker

The UI is intentionally minimal to keep focus on clarity and usability.

---

## Deployment

The application is deployed using **Streamlit Community Cloud** directly from this GitHub repository.

**Main entry file**:
`app/app.py`

All model artifacts and encoders are loaded at runtime to ensure consistency with training.

---

## How to Run Locally

```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```
