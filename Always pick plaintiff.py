# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 13:31:08 2023

@author: begon
"""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings

# Load your DataFrame
datafile_path = "database.csv"
df = pd.read_csv(datafile_path)
df = df[df["party_winning"] <= 1]

# Select only numeric columns
df = df.select_dtypes(include=['float64', 'int64'])
df = df.drop(['case_disposition'], axis=1)

for col in df:
    mode_col = df[col].mode()[0]
    df[col].fillna(mode_col, inplace=True)

# Define target variable
y = df["party_winning"]

# (guessing 1 for every entry)
import numpy as np
size = len(y)

# Create a NumPy array of all 1s
ones_array = np.ones(size, dtype=int)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
# Calculate accuracy for predictions
accuracy = accuracy_score(y, ones_array)
print("Accuracy (Guessing the Petitioner Wins):", accuracy)

# Calculate AUROC for random predictions
auroc = roc_auc_score(y, ones_array)
print("AUROC (Guessing the Petitioner Wins):", auroc)

# Generate and print the classification report for random guessing
class_report = classification_report(y, ones_array)
print("Classification Report (Guessing the Petitioner Wins):\n", class_report)
