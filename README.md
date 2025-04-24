# 🌸 Iris Flower Classification

This project focuses on building a machine learning model that classifies Iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — based on petal and sepal measurements. This is a classic supervised learning classification problem.

---

## 📁 Dataset Overview

- **Source**: `Iris.csv`
- **Features**:
  - `SepalLengthCm`
  - `SepalWidthCm`
  - `PetalLengthCm`
  - `PetalWidthCm`
- **Target**:
  - `Species`

This dataset contains **150** rows, with **50 instances of each species**.

---

## 🔍 Objective

Develop a machine learning classification model that:
- Predicts the **species of an Iris flower** based on its measurements.
- Identifies the **most important features**.
- Evaluates model performance using accuracy, precision, recall, and F1-score.

---

## ⚙️ Workflow

### 1️⃣ Data Preprocessing
- Handled missing values (if any).
- Verified data types and checked class balance.
- Encoded target labels.

### 2️⃣ Data Visualization
- Used **Seaborn pairplot** and **heatmaps** to visualize relationships and correlations.
- Plotted **species distribution** and **feature importance**.

### 3️⃣ Model Building
- Trained multiple models:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree
  - Logistic Regression
- Selected the **best performing model** based on accuracy and cross-validation.

### 4️⃣ Evaluation
- Used:
  - Confusion Matrix
  - Classification Report
  - Accuracy Score

---

## 📈 Results

| Model               | Accuracy |
|--------------------|----------|
| KNN                | 0.96     |
| SVM (Best Model)   | 0.97     |
| Decision Tree      | 0.93     |
| Logistic Regression| 0.94     |

✅ **Support Vector Machine (SVM)** gave the highest accuracy with minimal overfitting.

---

## 🧪 Requirements

Install the dependencies with:

```bash
pip install


git clone https://github.com/vishnu99617/Iris-Flower-Classification.git
cd Iris-Flower-Classification



 -r requirements.txt

python iris_flower_classification.py


🛠 Tools & Libraries
Python

Pandas

Numpy

Seaborn

Matplotlib

Scikit-learn

📚 Reference
Iris Dataset - UCI Machine Learning Repository

Hands-On Machine Learning with Scikit-Learn and Python


