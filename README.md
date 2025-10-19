# CCMACLRL_-COM232-_PROJECT
This project applies machine learning algorithms to predict students’ academic outcomes based on demographic, social, and academic factors.

#  Predicting Student Performance Using Machine Learning Techniques

## Project Overview
This project focuses on predicting **student academic performance** using supervised **machine learning algorithms**.  
By analyzing demographic, social, and academic factors, the model predicts whether a student will **pass** or **fail**.  
The dataset is sourced from the **UCI Machine Learning Repository** (`student-por.csv`), and several algorithms are implemented and compared, including:

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression (LR)**
- **Random Forest Classifier (RFC)**

Among these, the **Random Forest Classifier** achieved the highest predictive accuracy and was further improved through **hyperparameter optimization** using GridSearchCV.

---

## Tech Stack

| Category | Tools & Libraries |
|:----------|:------------------|
| **Language** | Python 3.10 |
| **Libraries** | scikit-learn, pandas, numpy, seaborn, matplotlib |
| **Development Environment** | Google Colab / Jupyter Notebook |
| **Visualization Tools** | Matplotlib, Seaborn, Excel |
| **Dataset** | UCI Student Performance Dataset (`student-por.csv`) |

---

##  Methodology

### 1. Data Preprocessing
- **Data Cleaning:** Checked for null values and outliers using IQR filtering.  
- **Feature Encoding:** Applied one-hot encoding to categorical variables.  
- **Feature Scaling:** Standardized numerical features using `StandardScaler()`.  
- **Feature Selection:** Used correlation heatmaps and feature importance analysis.  
- **Data Splitting:** Split into 70% training and 30% testing sets.

### 2. Algorithms Implemented
- **K-Nearest Neighbors (KNN):** Instance-based learner using Euclidean distance.  
- **Logistic Regression (LR):** Linear model for binary classification.  
- **Random Forest Classifier (RFC):** Ensemble model of decision trees for robust prediction.  

### 3. Model Optimization
Performed **Grid Search Cross-Validation (GridSearchCV)** for parameter tuning:
```python
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}
