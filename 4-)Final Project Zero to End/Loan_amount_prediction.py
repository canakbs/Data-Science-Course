import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---------------------------------------------------------
FILE_PATH = r"C:\Users\LENOVO\OneDrive\Masaüstü\Çalışmalar\Data Science Basics\Data-Science-Course\4-)Final Project Zero to End\original_dataset.csv"
SAVE_PATH = r"C:\Users\LENOVO\OneDrive\Masaüstü\Çalışmalar\Data Science Basics\Data-Science-Course\4-)Final Project Zero to End"

# 1. Load the data
# ---------------------------------------------------------

df = pd.read_csv(FILE_PATH)

# ---------------------------------------------------------
# 2. Data Filtering according to Business Rules
# ---------------------------------------------------------

df.loc[df['credit_score'] < 400, 'loan_paid_back'] = 0

df.loc[(df['employment_status'] == 'Unemployed') & (df['credit_score'] < 600), 'loan_paid_back'] = 0

df.loc[df['loan_amount'] > df['annual_income'], 'loan_paid_back'] = 0

# 3. Data Preprocessing
# ---------------------------------------------------------
print("\n[INFO] 3. Data Preprocessing...")

selected_features = [
    'age', 'gender', 'marital_status', 'education_level', 
    'annual_income', 'employment_status', 'credit_score', 
    'loan_amount', 'loan_purpose', 'loan_term', 'loan_paid_back'
]

df = df[selected_features]

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=[
    'gender', 'marital_status', 'education_level', 
    'employment_status', 'loan_purpose'
], drop_first=True)

X = df_encoded.drop('loan_paid_back', axis=1)
y = df_encoded['loan_paid_back']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
print("\n[INFO] 4. Model Training...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Saving
# ---------------------------------------------------------
print("\n[INFO] 5. Model Saving...")
joblib.dump(model, os.path.join(SAVE_PATH, 'model.pkl'))
joblib.dump(X.columns, os.path.join(SAVE_PATH, 'model_columns.pkl'))

print(f"[SUCCESS] Model saved to: {SAVE_PATH}")