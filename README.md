# Malaria Diagnosis Prediction Using Stochastic Gradient Descent (SGD)

A machine learning project that predicts malaria diagnosis from patient symptoms and demographic data using **Logistic Regression built from scratch**, optimized with **Stochastic Gradient Descent (SGD)**.

---

## Project Overview

Malaria remains one of the leading causes of illness globally. This project explores how machine learning can support diagnosis by classifying patients as **Malaria Positive (1)** or **Negative (0)** based on clinical and demographic features.

The core objective was to implement the SGD optimization algorithm manually — without relying on library abstractions like `model.fit()` — to gain a deeper understanding of how gradient-based learning works under the hood.

---

## Dataset

- **Records:** 1,622 patients
- **Source regions:** Mangalore, Shimoga, Chickmagalur, Udupi, Kasargod
- **Features used:**
  - **Demographic:** Age, Sex, Residence Area
  - **Symptoms (binary):** Fever, Headache, Abdominal Pain, General Body Malaise, Dizziness, Vomiting, Confusion, Backache, Chest Pain, Coughing, Joint Pain
  - **Other:** Risk Score
- **Target:** 1 = Malaria, 0 = No Malaria (72% positive, 28% negative)

---

## Preprocessing Steps

- Verified no missing values in the dataset
- Dropped columns that could cause data leakage (`Primary_Code`, `Diagnosis_Type`) and non-predictive columns (`IP_Number`, `DOA`, `Discharge_Date`)
- Encoded categorical features (`Sex`, `Residence_Area`) using LabelEncoder
- Scaled all features using StandardScaler
- Split data into 80% training and 20% testing sets

---

## Model Implementation

Logistic Regression was implemented from scratch using NumPy:

- **Sigmoid activation** to output probabilities
- **Binary Cross-Entropy (Log Loss)** as the cost function
- **Stochastic Gradient Descent** with per-sample weight updates
- **Hyperparameters:** Learning rate = 0.01, Epochs = 100

---

## Results

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.9969 |
| Precision | 0.9957 |
| Recall    | 1.0000 |
| F1 Score  | 0.9979 |

---

## Visualizations

The project includes three key visualizations:

1. **Loss Curve** — Binary cross-entropy loss across 100 epochs showing model convergence
2. **Confusion Matrix** — Breakdown of true/false positives and negatives
3. **Predicted vs Actual** — Scatter plot comparing model predictions against ground truth

---

## Tech Stack

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn (for preprocessing and evaluation metrics only)

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Okosaedwin/malaria-prediction-sgd.git
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Run all cells sequentially.

---
