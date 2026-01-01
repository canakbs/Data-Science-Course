# app.py
from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Model klasörünün yolu
model_path = r"C:\Users\LENOVO\OneDrive\Masaüstü\Çalışmalar\Data Science Basics\Data-Science-Course\4-)Final Project Zero to End"

# Dosyaları oradan yükle
model = joblib.load(os.path.join(model_path, 'model.pkl'))
model_columns = joblib.load(os.path.join(model_path, 'model_columns.pkl'))

@app.route('/')
def home():
    # Renders the main form page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get Data from Form
        input_data = {
            'age': int(request.form['age']),
            'gender': request.form['gender'],
            'marital_status': request.form['marital_status'],
            'education_level': request.form['education_level'],
            'annual_income': float(request.form['annual_income']),
            'employment_status': request.form['employment_status'],
            'credit_score': int(request.form['credit_score']),
            'loan_amount': float(request.form['loan_amount']),
            'loan_purpose': request.form['loan_purpose'],
            'loan_term': int(request.form['loan_term'])
        }
        credit_score = int(request.form['credit_score']) # Skoru değişkene al

        # --- SİGORTA KURALI (Business Rule Layer) ---
        # Eğer skor 400'den küçükse, modele sormadan direkt RED ver.
        if credit_score < 400:
            return render_template('result.html', 
                                   prediction_text="LOAN REJECTED ❌", 
                                   probability="Reason: Credit Score too low (Policy Rule)",
                                   result_class="danger")

        # 2. Convert to DataFrame
        df_input = pd.DataFrame([input_data])

        # 3. One-Hot Encoding (Same process as training)
        df_encoded = pd.get_dummies(df_input, columns=[
            'gender', 'marital_status', 'education_level', 
            'employment_status', 'loan_purpose'
        ], drop_first=True)

        # 4. Align Columns with Training Data
        # Fills missing columns (e.g., if 'Male' is not chosen) with 0
        df_final = df_encoded.reindex(columns=model_columns, fill_value=0)

        # 5. Make Prediction
        prediction = model.predict(df_final)[0] # Returns 0 or 1
        probability = model.predict_proba(df_final)[0][1] # Probability of paying back

        # 6. Interpret Result
        if prediction == 1:
            result_text = "LOAN APPROVED ✅"
            result_class = "success"
        else:
            result_text = "LOAN REJECTED ❌"
            result_class = "danger"

        return render_template('result.html', 
                               prediction_text=result_text, 
                               probability=f"Approval Probability: {probability*100:.1f}%",
                               result_class=result_class)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    app.run(debug=True)