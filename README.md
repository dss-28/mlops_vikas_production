# 🚀 End-to-End MLOps Pipeline for Income Classification

A production-oriented Machine Learning system that demonstrates the complete ML lifecycle — from data ingestion and preprocessing to model training, experiment tracking, deployment, and CI/CD automation.

The goal of this project is to build a reproducible and scalable ML workflow that bridges the gap between experimentation and production deployment.

---

## 🎯 Problem Statement

Develop an ML classification system to predict whether an individual's income exceeds a given threshold using demographic and employment features.

The project focuses on production ML engineering practices rather than only model development.

---

## 🏗️ Architecture

Raw Data  
↓  
Data Ingestion  
↓  
Data Validation  
↓  
Data Preprocessing  
↓  
Feature Engineering  
↓  
Model Training  
↓  
Model Evaluation  
↓  
MLflow Tracking  
↓  
Model Artifact Management  
↓  
FastAPI Deployment  
↓  
Docker Containerization  
↓  
CI/CD Automation  

---

## ✨ Key Features

### End-to-End ML Pipeline

Implemented a complete machine learning workflow:

• Data ingestion  
• Data validation  
• Data preprocessing  
• Feature engineering  
• Model training  
• Model evaluation  
• Model serialization  
• Prediction pipeline  

### Production ML Practices

• Modular pipeline components  
• Configuration-driven execution  
• Logging and exception handling  
• Reproducible workflows  
• Automated testing  

### Experiment Tracking

Integrated MLflow for:

• Experiment tracking  
• Parameter logging  
• Metric tracking  
• Model artifact management  

### Model Deployment

Developed a FastAPI inference service for real-time predictions.

Flow:

User Input → API Endpoint → Preprocessing Pipeline → ML Model → Prediction

### Deployment & Automation

Implemented:

• Docker containerization  
• GitHub Actions CI/CD workflow  
• Automated validation checks  

---

## 🛠️ Tech Stack

Programming:
Python

Machine Learning:
Scikit-learn, Pandas, NumPy

MLOps:
MLflow, Docker, GitHub Actions

Deployment:
FastAPI, Uvicorn

Tools:
Git, GitHub, YAML

---

## 📂 Project Structure

mlops_vikas_production/

├── .github/
│   └── workflows/
│       └── ci_cd.yml

├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── logger.py

├── artifacts/
│   ├── model.joblib
│   ├── pipeline.joblib
│   ├── scaler/
│   ├── metrics.json
│   └── evaluation_metrics.yaml

├── tests/
│   ├── test_pipeline.py
│   └── test_training.py

├── app.py
├── pipeline.py
├── config.yaml
├── Dockerfile
├── requirements.txt
└── README.md

---

## 🚀 How to Run

Install dependencies:

pip install -r requirements.txt

Run training pipeline:

python pipeline.py

Start API:

uvicorn app:app --reload

---

## 📌 Engineering Outcomes

✓ Built complete ML lifecycle pipeline  
✓ Converted ML workflow into deployable service  
✓ Implemented experiment tracking  
✓ Added containerized deployment  
✓ Created automated CI/CD workflow  
✓ Followed production ML engineering practices  

---

## 👨‍💻 Author

Darshan Shirsat

M.Tech AI & Data Science

Focused on Applied Machine Learning, MLOps, Industrial AI, and Production ML Systems
