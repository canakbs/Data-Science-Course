# 🚀 Data Science & Machine Learning Course (From Zero to Deployment)

Welcome to my Data Science portfolio repository! This project documents my complete journey from understanding the fundamental concepts of Data Science to building and deploying a real-world **End-to-End Machine Learning Web Application**.

This repository is designed as a comprehensive course structure, broken down into 4 main modules, culminating in a full-stack Flask application.

---

## 📂 Course Structure & Modules

### 1️⃣ Introduction to Data Science
* **Core Concepts:** Understanding what Data Science is and the **CRISP-DM** methodology.
* **Data Types:** Distinguishing between Structured and Unstructured data.
* **Tools:** Introduction to the Python ecosystem (Anaconda, Jupyter, etc.).

### 2️⃣ Data Analysis (EDA)
* **Data Manipulation:** Mastering **Pandas** and **NumPy** for data cleaning and processing.
* **Visualization:** Creating meaningful insights using **Matplotlib** and **Seaborn**.
* **Real-world Scenarios:** * Handling missing values.
    * **Outlier Detection:** Analyzing edge cases (e.g., handling a 150kg weight entry in a dataset) to understand how anomalies affect distribution.

### 3️⃣ Machine Learning Concepts
* **Supervised Learning:**
    * **Regression:** Predicting continuous values (Linear Regression).
    * **Classification:** Predicting categories (Logistic Regression, KNN, SVM, Decision Trees, Random Forest).
* **Unsupervised Learning:**
    * **Clustering:** Grouping data without labels (K-Means).
* **Model Evaluation:** Understanding Accuracy, Confusion Matrix, Precision, Recall, and F1-Score to truly judge model performance.

---

## 🏆 Final Project: Loan Approval Prediction System

The highlight of this repository is the **"Zero to End" Final Project**. It is a fully functional web application that predicts whether a bank customer's loan application should be **Approved** or **Rejected**.

### 🌟 Key Features
* **End-to-End Workflow:** The project covers the entire pipeline: Data Generation -> Preprocessing -> Model Training -> Web Deployment.
* **Smart Data Logic:** Unlike standard datasets, this project uses a custom-built data generation script that enforces **realistic banking rules**.
    * *Example:* A credit score below 400 is automatically rejected.
    * *Example:* High debt-to-income ratios result in rejection.
    * This ensures the model learns logical patterns, not random noise.
* **Machine Learning Model:** Utilizes a **Random Forest Classifier** for robust and accurate predictions.
* **Web Interface:** A user-friendly frontend built with **HTML5 & CSS3**, powered by a **Flask** backend.

### 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Science:** Pandas, NumPy, Scikit-Learn, Joblib
* **Web Framework:** Flask
* **Frontend:** HTML, CSS

---

## ⚙️ Installation & Usage Guide

Follow these steps to run the final project on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/canakbs/Data-Science-Course.git](https://github.com/canakbs/Data-Science-Course.git)
cd Data-Science-Course
2. Navigate to the Project Folder
Bash

cd "4-)Final Project Zero to End"
3. Install Dependencies
Ensure you have Python installed. Then, install the required libraries:

Bash

pip install -r requirements.txt
4. Train the Model (Important!)
Before running the app, you must generate the data and train the model to create the .pkl files.

Bash

python train_model.py
Output: You will see a success message indicating model.pkl and model_columns.pkl have been created.

5. Run the Application
Start the Flask web server:

Bash

python app.py
6. Access the Interface
Open your web browser and navigate to: http://127.0.0.1:5000

📊 Project Logic & Business Rules
To ensure the AI doesn't make "silly" mistakes (like approving a Credit Score of 320), the system implements a Hybrid Decision Layer:

Hard Rules (Policy Layer): The Flask app checks critical thresholds immediately. If a score is too low, it rejects the application before even asking the AI.

AI Prediction (Model Layer): If the applicant passes the hard rules, the Random Forest model analyzes complex patterns (Income vs. Loan Amount, Term, Education) to make the final decision.

🤝 Contributing
Feel free to fork this repository, open issues, or submit pull requests. Any feedback is appreciated!

📬 Contact
If you have any questions about the code or the concepts, feel free to reach out.

Created with ❤️ by Can Akbaş
