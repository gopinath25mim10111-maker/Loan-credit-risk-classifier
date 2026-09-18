# Loan Credit Risk Classifier

A student-level machine learning project that predicts whether a loan application is likely to be approved.

## Dataset
The project uses a 614-row loan approval dataset with `Loan_Status` as the target.

## Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest

## Workflow
Dataset → preprocessing → missing-value handling → one-hot encoding → train/test split → model training → evaluation → saved models → prediction.

## Evaluation
Accuracy, Precision, Recall and F1 Score are calculated. Confusion matrices are saved in the `results` folder.

## Run
Activate the virtual environment and run:

```powershell
python main.py
```

For a sample prediction:

```powershell
python src/predict.py
```

This project is for educational use and should not be used as the sole basis for real lending decisions.