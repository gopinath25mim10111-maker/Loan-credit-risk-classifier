import joblib
import pandas as pd

model = joblib.load(
    "models/random_forest.pkl"
)

application = {
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 1500,
    "LoanAmount": 120,
    "Loan_Amount_Term": 360,
    "Credit_History": 1.0,
    "Property_Area": "Urban"
}

data = pd.DataFrame([application])

prediction = model.predict(data)[0]

probability = model.predict_proba(data)[0][1]

if prediction == 1:
    print("Loan Prediction: APPROVED")
else:
    print("Loan Prediction: REJECTED")

print(f"Approval Probability: {probability:.2%}")