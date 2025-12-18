# 🌊 Global Earthquake–Tsunami Risk Prediction System

An end-to-end machine learning application that predicts tsunami risk
using seismic parameters from global earthquakes recorded between 2001–2022.

## 🚀 Features
- Binary tsunami risk prediction
- 782 major earthquakes (≥6.5 magnitude)
- Balanced classification problem
- Interactive Streamlit web application
- Production-ready ML pipeline

## 🧠 Model
- Random Forest Classifier
- Stratified training
- ROC-AUC optimized
- Class imbalance handling

## 📊 Dataset
- Source: Kaggle Earthquake-Tsunami Dataset
- Coverage: Global
- Time Span: 22 years
- Missing Values: None

## 🛠️ Tech Stack
Python, Pandas, NumPy, Scikit-learn, Streamlit, Git

## ▶️ Run Locally
```bash
pip install -r requirements.txt
python src/train.py
streamlit run app/streamlit_app.py
