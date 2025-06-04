# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, accuracy_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

import shap
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
data = pd.read_csv('water_potability.csv')
data.rename(columns={'Sulfate': 'Sulphate'}, inplace=True)  # Rename column here
data['Potability'] = data['Potability'].astype('category')

# Feature engineering
data['Solids_log'] = np.log1p(data['Solids'])
data['Trihalomethanes_log'] = np.log1p(data['Trihalomethanes'])

features = ['ph', 'Hardness', 'Chloramines', 'Sulphate', 'Conductivity',
            'Organic_carbon', 'Turbidity', 'Solids_log', 'Trihalomethanes_log']

# Imputation
imputer = KNNImputer(n_neighbors=7)
data[features] = imputer.fit_transform(data[features])

# Exploratory Data Analysis
plt.figure(figsize=(14, 10))
for idx, col in enumerate(features):
    plt.subplot(3, 3, idx + 1)
    sns.histplot(data[col], kde=True, bins=30, color='skyblue')
    plt.title(col)
plt.tight_layout()
plt.savefig("eda_analysis.pdf")

plt.figure(figsize=(10, 8))
corr_matrix = data[features + ['Potability']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Feature Correlation')
plt.tight_layout()
plt.savefig("feature_correlation.pdf")

# Data Preparation
X = data[features]
y = data['Potability'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Resampling
resampler = SMOTEENN(random_state=42)
X_res, y_res = resampler.fit_resample(X_train_scaled, y_train)

# Model Definition
models = {
    'XGBoost': XGBClassifier(
        scale_pos_weight=sum(y_res==0)/sum(y_res==1),
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    ),
    'RandomForest': BalancedRandomForestClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(
        auto_class_weights='Balanced',
        silent=True,
        random_state=42
    ),
    'LightGBM': LGBMClassifier(
        class_weight='balanced',
        random_state=42
    ),
    'SVM': SVC(
        class_weight='balanced',
        probability=True,
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'GradientBoost': GradientBoostingClassifier(random_state=42)
}

#  ROC Curves plot for evaluation
results = {}
plt.figure(figsize=(10, 8))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nEvaluating {name}...")

    y_proba = cross_val_predict(model, X_res, y_res, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_res, y_proba)
    roc_auc = roc_auc_score(y_res, y_proba)

    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

    optimal_idx = np.argmax(tpr - fpr)
    y_pred = (y_proba >= thresholds[optimal_idx]).astype(int)

    results[name] = {
        'ROC AUC': roc_auc,
        'PR AUC': average_precision_score(y_res, y_proba),
        'Precision': precision_score(y_res, y_pred),
        'Recall': recall_score(y_res, y_pred),
        'F1': f1_score(y_res, y_pred),
        'Accuracy': accuracy_score(y_res, y_pred)
    }

    print(classification_report(y_res, y_pred))

plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_plot.pdf")


# Results Summary
results_df = pd.DataFrame(results).T.sort_values('ROC AUC', ascending=False)
print("\n=== Model Performance ===")
print(results_df)

# Feature Importance (Permutation)
feature_importance_all = {}
for name, model in models.items():
    model.fit(X_res, y_res)
    result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)
    feature_importance_all[name] = result.importances_mean

importance_df = pd.DataFrame(feature_importance_all, index=features).T
mean_importance = importance_df.mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=mean_importance.values, y=mean_importance.index)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.pdf")


# SHAP Analysis
print("\n=== SHAP Analysis ===")

# Use XGBoost for SHAP analysis 
best_model_name = 'XGBoost'
best_model = models[best_model_name]
best_model.fit(X_res, y_res)

# Create SHAP explainer and compute SHAP values
explainer = shap.Explainer(best_model, X_res)
shap_values = explainer(X_test_scaled)

# SHAP bar plot 
shap.summary_plot(shap_values, X_test, feature_names=features, plot_type='bar')

# SHAP beeswarm plot 
shap.summary_plot(shap_values, X_test, feature_names=features)