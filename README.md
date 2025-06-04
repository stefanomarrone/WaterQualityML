# 💧 Water Potability Prediction

This project focuses on building and evaluating various machine learning models to predict the **potability of water** based on its physicochemical features. It includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and explainability using SHAP.

## 📁 Dataset

The dataset used is `water_potability.csv`, which contains measurements of water quality indicators, including:

- pH
- Hardness
- Solids
- Chloramines
- Sulphate
- Conductivity
- Organic Carbon
- Trihalomethanes
- Turbidity

The target variable is `Potability` (0: not potable, 1: potable).

---

## 🔧 Main Steps

### 1. Data Preprocessing

- **Imputation**: Missing values are handled using `KNNImputer`.
- **Transformation**: Skewed features (`Solids`, `Trihalomethanes`) are log-transformed.
- **Standardization**: Features are scaled using `StandardScaler`.

### 2. Exploratory Data Analysis (EDA)

- Histograms for all features.
- Correlation heatmap.
- Outputs saved as:
  - `eda_analysis.pdf`
  - `feature_correlation.pdf`

### 3. Class Imbalance Handling

Used **SMOTEENN** to oversample and clean the training set.

### 4. Models Trained

Several models are trained and evaluated using **Stratified K-Fold Cross-Validation**:

- XGBoost
- Balanced Random Forest
- CatBoost
- LightGBM
- SVM
- AdaBoost
- Gradient Boosting

### 5. Evaluation Metrics

For each model:

- ROC AUC
- PR AUC
- Accuracy
- Precision
- Recall
- F1-Score

**ROC curves** are plotted and saved as `roc_plot.pdf`.

### 6. Feature Importance

- Calculated via **Permutation Importance**.
- Saved as `feature_importance.pdf`.

### 7. Explainability with SHAP

SHAP is used on the best-performing model (**XGBoost**) to understand feature impact:

- SHAP bar plot (global importance)
- SHAP beeswarm plot (per-instance explanation)

---

## 📦 Dependencies

Install the required packages with:

```bash
pip install -r requirements.txt
