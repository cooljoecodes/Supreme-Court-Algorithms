# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:51:31 2023

@author: begon
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import shap
import matplotlib.pyplot as plt

# Load your DataFrame
datafile_path = "database.csv"
df = pd.read_csv(datafile_path)
df = df[df["party_winning"] <= 1]
print("#### Before Preprocessing ####")
print("Shape: ", df.shape[0])
print("Number of Columns: ", df.shape[1])

# Select only numeric columns
df = df.select_dtypes(include=['float64', 'int64'])
df = df.drop(['case_disposition'], axis=1)

for col in df:
    mode_col = df[col].mode()[0]  # Calculate the mode
    df[col].fillna(mode_col, inplace=True)  # Replace NaN with the mode

print("#### After Preprocessing ####")
print("Shape: ", df.shape[0])
print("Number of Columns: ", df.shape[1])

#prompt = input("Continue (y/n): ")
prompt = 'y' #input("Continue (y/n): ")
if prompt == 'y':

    # Define your features (X) and target (y)
    X = df.drop("party_winning", axis=1)
    y = df["party_winning"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the XGBoost classifier
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)

    # Make predictions using the XGBoost classifier
    y_pred_xgb = xgb_model.predict(X_test)

    # Calculate accuracy for the XGBoost classifier
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    print("Accuracy (XGBoost):", accuracy_xgb)

    # Calculate AUROC
    auroc_xgb = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])
    print("AUROC (XGBoost):", auroc_xgb)

    # Generate and print the classification report
    class_report_xgb = classification_report(y_test, y_pred_xgb)
    print("Classification Report (XGBoost):\n", class_report_xgb)
    
    #prompt = input("Shap?")
    prompt = 'n'
    if prompt == 'y':
        # Calculate SHAP values
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        
        # Visualize SHAP values using a bar graph
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        
        # Show the plot
        plt.show()
