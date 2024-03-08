import pandas as pd
import joblib

def clean_feature_names(df):
    df.columns = ["".join(c if c.isalnum() or c == '_' else '_' for c in str(col)) for col in df.columns]
    return df

# Load the LightGBM model from the joblib file
loaded_model = joblib.load('best_model_lgb.joblib')

# Read the data from the CSV file
data = pd.read_csv("database.csv")

#data = data.drop("party_winning", axis=1, inplace=False)


# Select only numeric columns
data = data.select_dtypes(include=['float64', 'int64'])
data = data.drop(['case_disposition'], axis=1)

for col in data:
    mode_col = data[col].mode()[0]
    data[col].fillna(mode_col, inplace=True)

data = clean_feature_names(data)

# Select the first row for prediction
#df = data.head(1)

# Create a DataFrame with all zeroes to get the appellee to win (0)
# Create a Dataframe with all 200s to get the petitioner to win (1)
df = pd.DataFrame([[0]*len(data.columns)], columns=data.columns)

result = df["party_winning"].iloc[0]
df = df.drop("party_winning", axis=1, inplace=False)

print(df)

# Make predictions for the single data point
y_pred_single_data_point_prob = loaded_model.predict_proba(df)[:, 1]

# Convert probability to a binary prediction
y_pred_single_data_point = (y_pred_single_data_point_prob > 0.5).astype(int)

# Display or use the prediction for the single data point
print("Prediction for the single data point:", y_pred_single_data_point[0])
print(len(df.columns))
print (df.columns)
#print("Actual result: ", result)
