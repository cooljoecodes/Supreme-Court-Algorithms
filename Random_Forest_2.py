import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

prompt = 'y' #input("Continue (y/n): ")
if prompt == 'y':

    # Define your features (X) and target (y)
    X = df.drop("party_winning", axis=1)
    y = df["party_winning"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and train the Random Forest classifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # Make predictions using the Random Forest classifier
    y_pred_rf = rf_model.predict(X_test)

    # Calculate accuracy for the Random Forest classifier
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print("Accuracy (Random Forest):", accuracy_rf)

    # Calculate AUROC
    auroc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
    print("AUROC (Random Forest):", auroc_rf)

    # Generate and print the classification report
    class_report_rf = classification_report(y_test, y_pred_rf)
    print("Classification Report (Random Forest):\n", class_report_rf)
    
    prompt = 'n' #input("Shap?")
    if prompt == 'y':
        # Calculate SHAP values
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)
        
        # Visualize SHAP values using a bar graph
        shap.summary_plot(shap_values, X_test, plot_type="bar")
        
        # Show the plot
        plt.show()