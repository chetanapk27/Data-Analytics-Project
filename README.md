# Mobile Money / Digital Wallet Fraud Detection System

_IBM SkillsBuild Data Analytics with AI Internship Capstone Project_

## Project Overview

An end-to-end machine learning solution built to detect fraudulent digital cash-out transactions in mobile wallet systems, leveraging the PaySim dataset.

## Dataset

- **Source:** Kaggle (Synthetic Financial Datasets For PaySim - Reduced Optimized Version)
- **File Used:** `Paysim_small.csv`

## Architecture & Technologies

- **Data Processing & ML:** Python, Pandas, Scikit-Learn (Random Forest Classifier, 0.936 ROC-AUC)
- **Backend API:** Flask (`app.py`)
- **Frontend UI:** Streamlit Decision Dashboard (`ui.py`)

## Setup & Run Instructions

1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python train_model.py`
3. Run Backend API: `python app.py`
4. Run Frontend UI: `python -m streamlit run ui.py`
