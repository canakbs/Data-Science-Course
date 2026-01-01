import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# AYARLAR
# ---------------------------------------------------------
# Senin Kaggle dosyanın yolu
FILE_PATH = r"C:\Users\LENOVO\OneDrive\Masaüstü\Çalışmalar\Data Science Basics\Data-Science-Course\4-)Final Project Zero to End\original_dataset.csv"
SAVE_PATH = r"C:\Users\LENOVO\OneDrive\Masaüstü\Çalışmalar\Data Science Basics\Data-Science-Course\4-)Final Project Zero to End"

# 1. VERİYİ YÜKLE
# ---------------------------------------------------------
print("[INFO] 1. Kaggle Verisi Yükleniyor...")
df = pd.read_csv(FILE_PATH)

# ÖNCEKİ DURUMU GÖRELİM
print("\nDüzeltme Öncesi 400 Puan Altı Durumu:")
print(df[df['credit_score'] < 400]['loan_paid_back'].value_counts())
# Eğer burada '1' (Ödedi) görüyorsan, Kaggle verisi mantıksız demektir.

# ---------------------------------------------------------
# 2. VERİ MANTIĞINI DÜZELTME (DATA CLEANING / LOGIC FIX)
# ---------------------------------------------------------
print("\n[INFO] 2. Bankacılık Kuralları Uygulanıyor (Veri Temizliği)...")

# KURAL 1: Kredi skoru 400'ün altındaysa bu kişi ödeyemez (0).
# Kaggle verisinde '1' yazsa bile bunu zorla '0' yapıyoruz.
df.loc[df['credit_score'] < 400, 'loan_paid_back'] = 0

# KURAL 2: İşsizse (Unemployed) ve skoru 600 altındaysa reddet.
df.loc[(df['employment_status'] == 'Unemployed') & (df['credit_score'] < 600), 'loan_paid_back'] = 0

# KURAL 3: İstenen kredi, yıllık gelirden fazlaysa reddet.
df.loc[df['loan_amount'] > df['annual_income'], 'loan_paid_back'] = 0

print(" Veri seti mantıklı hale getirildi.")
print("Düzeltme Sonrası 400 Puan Altı Durumu:")
print(df[df['credit_score'] < 400]['loan_paid_back'].value_counts())
# Artık burada sadece '0' görmelisin.

# 3. VERİ ÖN İŞLEME
# ---------------------------------------------------------
print("\n[INFO] 3. Veri Ön İşleme...")

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

# X ve y ayırma (Düzeltilmiş veri üzerinden)
X = df_encoded.drop('loan_paid_back', axis=1)
y = df_encoded['loan_paid_back']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODEL EĞİTİMİ
# ---------------------------------------------------------
print("\n[INFO] 4. Model Eğitiliyor...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. TEST ETME (320 PUANLI ADAM)
# ---------------------------------------------------------
print("\n[INFO] 5. 320 Puanlı Müşteri Test Ediliyor...")

# Manuel bir test verisi oluşturuyoruz
test_case = pd.DataFrame([{
    'age': 30, 'annual_income': 50000, 'loan_amount': 10000, 
    'credit_score': 320, 'loan_term': 36, # <--- DİKKAT: SKOR 320
    'gender': 'Male', 'marital_status': 'Single', 'education_level': 'High School',
    'employment_status': 'Employed', 'loan_purpose': 'Car'
}])

# One-Hot Encoding işlemini test verisine de uygula
test_encoded = pd.get_dummies(test_case, columns=[
    'gender', 'marital_status', 'education_level', 
    'employment_status', 'loan_purpose'
], drop_first=True)

# Eksik sütunları tamamla
test_final = test_encoded.reindex(columns=X.columns, fill_value=0)

prediction = model.predict(test_final)[0]

if prediction == 0:
    print(" BAŞARILI! Model mantığı öğrendi: 320 Skor -> RED (0)")
else:
    print(" BAŞARISIZ! Model hala 320 skoru kabul ediyor.")

# 6. KAYDETME
# ---------------------------------------------------------
print("\n[INFO] 6. Model Kaydediliyor...")
joblib.dump(model, os.path.join(SAVE_PATH, 'model.pkl'))
joblib.dump(X.columns, os.path.join(SAVE_PATH, 'model_columns.pkl'))

print(f"[SUCCESS] Model saved to: {SAVE_PATH}")