# 🏥 Machine Learning & Deep Learning Clinical Diagnostic Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end repository featuring clinical machine learning and deep learning applications for **Parkinson's Disease Detection** and **Heart Disease Risk Stratification**. Both projects feature exploratory data analysis, cross-validated hyperparameter tuning (`GridSearchCV`), **TensorFlow / Keras Deep Neural Networks**, model serialization, and interactive **Streamlit** clinical decision-support web applications.

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Project 1: Parkinson's Disease Detection](#-project-1-parkinsons-disease-detection)
  - [Clinical Background & Dataset](#clinical-background--dataset)
  - [Machine Learning & TensorFlow Models](#machine-learning--tensorflow-models)
  - [Model Benchmarks](#model-benchmarks)
  - [Streamlit Web Application](#streamlit-web-application)
- [Project 2: Heart Disease Risk Stratification](#-project-2-heart-disease-risk-stratification)
  - [Clinical Features & Dataset](#clinical-features--dataset)
  - [Ensemble & TensorFlow Deep Architecture](#ensemble--tensorflow-deep-architecture)
  - [Diagnostic Benchmarks](#diagnostic-benchmarks)
  - [Streamlit Clinical Interface](#streamlit-clinical-interface)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Applications](#-running-the-applications)
- [License](#-license)

---

## 🔬 Project Overview

Biomedical datasets require robust preprocessing, feature scaling, handling of non-linear biomarker interactions, and reliable uncertainty quantification. This repository implements:
1. **Classical Machine Learning Baselines & Ensembles**: Support Vector Machines (Linear & RBF kernels), Logistic Regression, and Random Forest.
2. **Deep Learning with TensorFlow / Keras**: Multi-layer artificial neural networks with Batch Normalization, Dropout regularization, and Early Stopping.
3. **Interactive Clinical Decision-Support Dashboards**: Streamlit UIs that execute simultaneous model evaluations, present continuous probability estimates, and evaluate which architecture offers the superior clinical prediction.

---

## Project 1: Parkinson's Disease Detection

### Clinical Background & Dataset
Parkinson's disease (PD) affects motor function and speech production. Vocal impairment (dysphonia) is one of the earliest measurable clinical indicators:
* **Dataset**: Oxford Parkinson's Disease Detection Dataset (`parkinsons.data`).
* **Instances**: 195 acoustic voice recordings across 31 individuals (23 diagnosed with PD, 8 healthy controls).
* **Biomarkers (22 Features)**:
  - *Fundamental Frequencies*: `MDVP:Fo(Hz)`, `MDVP:Fhi(Hz)`, `MDVP:Flo(Hz)`
  - *Jitter Measures (Frequency Variation)*: `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`
  - *Shimmer Measures (Amplitude Variation)*: `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`
  - *Harmonicity & Noise*: `NHR` (Noise-to-Harmonics Ratio), `HNR` (Harmonics-to-Noise Ratio)
  - *Non-Linear Measures*: `RPDE`, `DFA`, `spread1`, `spread2`, `D2`, `PPE`

### Machine Learning & TensorFlow Models
* **`Perkinson.ipynb`**: Original data exploration and linear Support Vector Machine (SVM) baseline model.
* **`Parkinson_v2.ipynb`**:
  - Stratified 80/20 train/test split with `StandardScaler`.
  - **Support Vector Classifier**: Calibrated decision hyperplane.
  - **GridSearchCV Hyperparameter Optimization**: Exhaustive search over hidden layers `(64, 32)`, `(128, 64)`, activation functions (`relu`, `tanh`), and regularization alpha.
  - **TensorFlow / Keras Deep Neural Network (ANN)**:
    - `Dense(64, relu)` + `BatchNormalization()` + `Dropout(0.3)`
    - `Dense(32, relu)` + `BatchNormalization()` + `Dropout(0.2)`
    - `Dense(16, relu)` + `Dense(1, sigmoid)`
    - Trained with `Adam(lr=0.001)` and `EarlyStopping(monitor='val_loss', patience=15)`.

### Model Benchmarks
| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM Baseline** | 87.2% | 0.900 | 0.931 | 0.915 | 0.817 |
| **GridSearch Tuned MLP** | 89.7% | 0.903 | 0.966 | 0.933 | 0.897 |
| **TensorFlow Deep ANN** | **89.7%** | **0.903** | **0.966** | **0.933** | **0.910** |

### Streamlit Web Application
* **File**: `Parkinson's Disease Detection/app.py`
* **Features**:
  - One-click clinical profile presets (e.g., Confirmed PD vocal tremor vs. Healthy vocal control).
  - Simultaneous dual-model inference: Linear SVM vs. **TensorFlow Deep Neural Network**.
  - **Superior Model Assessment Engine**: Dynamically evaluates which model delivers the higher decision confidence for each patient sample, explains the underlying clinical rationale, and flags consensus or divergence.

---

## Project 2: Heart Disease Risk Stratification

### Clinical Features & Dataset
* **Dataset**: 1,025 patient records (`heart.csv`) across 14 vital signs, symptoms, and diagnostic tests:
  - `age`, `sex` (1 = male; 0 = female)
  - `cp`: Chest Pain type (Typical Angina, Atypical Angina, Non-Anginal, Asymptomatic)
  - `trestbps`: Resting blood pressure (mm Hg)
  - `chol`: Serum cholesterol (mg/dl)
  - `fbs`: Fasting blood sugar > 120 mg/dl
  - `restecg`: Resting electrocardiographic results (Normal, ST-T wave abnormality, LVH)
  - `thalach`: Maximum heart rate achieved during exercise
  - `exang`: Exercise-induced angina
  - `oldpeak`: ST depression induced by exercise relative to rest
  - `slope`: Slope of the peak exercise ST segment
  - `ca`: Number of major vessels (0-3) colored by fluoroscopy
  - `thal`: Thalassemia defect (Normal, Fixed, Reversible)
  - `target`: Diagnosis of heart disease (1 = presence; 0 = absence)

### Ensemble & TensorFlow Deep Architecture
* **`Heart_Disease_Prediction.ipynb`**:
  - **Random Forest Classifier**: Champion ensemble with tuned `n_estimators`, `max_depth`, and `min_samples_split`.
  - **Support Vector Classifier (SVC)**: Tuned RBF kernel with probability calibration.
  - **Logistic Regression**: Elastic/L2-regularized baseline.
  - **TensorFlow / Keras Deep Neural Network**:
    - Multi-layer deep representation learning with Batch Normalization and Dropout layers.
    - Continuous probability calibration for cardiovascular risk scoring.

### Diagnostic Benchmarks
| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **99.5%** | **1.000** | **0.990** | **0.995** | **1.000** |
| **Support Vector Classifier** | 88.8% | 0.865 | 0.923 | 0.893 | 0.942 |
| **Logistic Regression** | 86.3% | 0.842 | 0.894 | 0.867 | 0.924 |
| **TensorFlow Deep ANN** | 89.3% | 0.870 | 0.925 | 0.897 | 0.951 |

### Streamlit Clinical Interface
* **File**: `Heart Disease Prediction/app.py`
* **Features**:
  - High-risk vs. Healthy patient profile presets.
  - Dual predictions from Random Forest and **TensorFlow Deep Neural Network**.
  - **AHA Clinical Risk Badges**: Real-time evaluation of Stage 1/2 Hypertension, Borderline/High Cholesterol, Exercise Angina, and Coronary Vessel Blockage.

---

## 📂 Repository Structure

```text
Machine-Learning-Projects/
├── .gitignore                               # Ignore rules for .venv, __pycache__, checkpoints
├── LICENSE                                  # MIT License
├── README.md                                # Comprehensive documentation
├── requirements.txt                         # Global project dependencies
│
├── Parkinson's Disease Detection/
│   ├── Perkinson.ipynb                      # Cleaned EDA & SVM baseline notebook
│   ├── Parkinson_v2.ipynb                   # Complete TensorFlow ANN & GridSearch notebook
│   ├── app.py                               # Comparative Streamlit web application
│   ├── parkinsons.data                      # Original UCI vocal biomarker dataset
│   ├── Parkinsson_disease.csv               # CSV format dataset
│   ├── parkinsons_svm_model.pkl             # Trained Support Vector Machine model
│   ├── parkinsons_tf_model.keras            # Saved TensorFlow / Keras Deep Neural Network
│   ├── parkinsons_scaler.pkl                # Fitted StandardScaler
│   └── requirements.txt                     # Project-specific dependencies
│
└── Heart Disease Prediction/
    ├── Heart_Disease_Prediction.ipynb       # Tuning, TensorFlow ANN & feature importance notebook
    ├── app.py                               # Clinical risk stratification Streamlit app
    ├── heart.csv                            # Cardiovascular clinical dataset (1,025 samples)
    ├── heart_disease_model.pkl              # Champion Random Forest Classifier
    ├── heart_disease_tf_model.keras         # Saved TensorFlow / Keras Deep Neural Network
    ├── heart_disease_scaler.pkl             # Fitted StandardScaler
    └── requirements.txt                     # Project-specific dependencies
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/DEBNILPAL/Machine-Learning-Projects.git
cd Machine-Learning-Projects
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Applications

### Launch Parkinson's Disease Detection App
```bash
streamlit run "Parkinson's Disease Detection/app.py"
```

### Launch Heart Disease Risk Prediction App
```bash
streamlit run "Heart Disease Prediction/app.py"
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE) - Copyright (c) 2026 Debnil Pal.
