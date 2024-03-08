# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:18:34 2023

@author: begon
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt
import joblib

def clean_feature_names(df):
    df.columns = ["".join(c if c.isalnum() or c == '_' else '_' for c in str(col)) for col in df.columns]
    return df


# Load your DataFrame
datafile_path = "database.csv"
df = pd.read_csv(datafile_path)
df = df[df["party_winning"] <= 1]

# Select only numeric columns
df = df.select_dtypes(include=['float64', 'int64'])
df = df.drop(['case_disposition'], axis=1)
df = df.drop(['decision_direction','lower_disposition_direction','cert_reason'], axis=1)

for col in df:
    mode_col = df[col].mode()[0]
    df[col].fillna(mode_col, inplace=True)

df = clean_feature_names(df)


# Define your features (X) and target (y)
X = df.drop("party_winning", axis=1)
y = df["party_winning"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Configure LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Create and train the LightGBM classifier
lgb_model = lgb.LGBMClassifier(**params)
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

# =============================================================================
# # Make predictions using the LightGBM classifier
# y_pred_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
# 
# # Calculate accuracy for the LightGBM classifier
# accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
# print("Accuracy (LightGBM):", accuracy_lgb)
# 
# # Calculate AUROC
# auroc_lgb = roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1])
# print("AUROC (LightGBM):", auroc_lgb)
# 
# # Generate and print the classification report
# class_report_lgb = classification_report(y_test, y_pred_lgb)
# print("Classification Report (LightGBM):\n", class_report_lgb)
# =============================================================================

# Make predictions using the LightGBM classifier
y_pred_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

# Convert probabilities to binary predictions
y_pred_lgb = (y_pred_prob_lgb > 0.5).astype(int)

# Calculate accuracy for the LightGBM classifier
accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
print("Accuracy (LightGBM):", accuracy_lgb)

# Calculate AUROC
auroc_lgb = roc_auc_score(y_test, y_pred_prob_lgb)
print("AUROC (LightGBM):", auroc_lgb)

# Generate and print the classification report
class_report_lgb = classification_report(y_test, y_pred_lgb)
print("Classification Report (LightGBM):\n", class_report_lgb)

# Save the LightGBM model using joblib
#joblib.dump(lgb_model, 'best_model_lgb.joblib')

prompt = input("Shap?")
if prompt == 'y':
    # Calculate SHAP values
    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_test)

    # Visualize SHAP values using a bar graph
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    # Show the plot
    plt.show()
