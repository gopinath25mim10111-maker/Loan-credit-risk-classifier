# Loan Credit Risk Classifier

## 1. Project Overview

Loan applications contain several details such as applicant income, loan amount, credit history, education, employment status, and other information. Evaluating these details manually can be time-consuming.

This project is a student-level machine learning project that predicts whether a loan application is likely to be approved or not.

The project uses three machine learning classification models:

* Logistic Regression
* Decision Tree
* Random Forest

The models are trained using a loan approval dataset and evaluated using common classification metrics.

## 2. Objective

The main objectives of this project are:

* To preprocess loan application data.
* To handle missing values.
* To convert categorical data into numerical form.
* To train different machine learning classification models.
* To evaluate the trained models.
* To save the trained models for later prediction.
* To make a sample prediction using the trained model.

## 3. Dataset

The project uses a loan approval dataset containing 614 records.

The target variable is:

```text
Loan_Status
```

The dataset contains applicant and loan-related information such as:

* Gender
* Married
* Dependents
* Education
* Self Employment
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area

The dataset is stored in:

```text
data/loan_data.csv
```

## 4. Machine Learning Models

### Logistic Regression

Logistic Regression is used as a classification model to predict the loan approval class.

### Decision Tree

Decision Tree uses decision rules based on the input features to classify loan applications.

### Random Forest

Random Forest combines multiple decision trees to make the final classification.

The trained models are stored in the `models` folder.

## 5. Project Workflow

The project follows this workflow:

```text
Dataset
   ↓
Data Preprocessing
   ↓
Missing Value Handling
   ↓
Categorical Feature Encoding
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Save Trained Models
   ↓
Prediction
```

## 6. Project Structure

```text
Loan-credit-risk-classifier/
│
├── data/
│   └── loan_data.csv
│
├── models/
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── notebooks/
│
├── results/
│   ├── decision_tree_confusion_matrix.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── random_forest_confusion_matrix.png
│   ├── model_comparison.csv
│   └── test_data.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── train_model.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 7. Requirements

The project requires:

* Python 3.x
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* joblib

The required Python packages are listed in:

```text
requirements.txt
```

## 8. Environment Setup

### Step 1: Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/gopinath25mim10111-maker/Loan-credit-risk-classifier.git
```

Move into the project directory:

```bash
cd Loan-credit-risk-classifier
```

### Step 2: Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

### Step 3: Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

If activation is successful, the terminal will show:

```text
(.venv)
```

### Step 4: Install dependencies

Run:

```powershell
pip install -r requirements.txt
```

## 9. Running the Project

After installing the required dependencies, run:

```powershell
python main.py
```

The main program performs the project workflow, including preprocessing, model training, evaluation, and saving the trained models and results.

## 10. Making a Prediction

A sample prediction can be executed using:

```powershell
python src/predict.py
```

The prediction script uses the saved trained model to make a prediction from the available input.

## 11. Evaluation

The trained models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The generated evaluation results are stored in the:

```text
results/
```

folder.

The model comparison results are available in:

```text
results/model_comparison.csv
```

The confusion matrices are saved as image files.

## 12. Saved Models

The trained models are saved using Joblib in the `models` folder:

```text
models/logistic_regression.pkl
models/decision_tree.pkl
models/random_forest.pkl
```

These files can be loaded later without training the models again.

## 13. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Git and GitHub

## 14. Limitations

This project is developed for educational purposes.

The dataset is relatively small, and the prediction performance depends on the quality and characteristics of the available data.

The model should not be used as the sole basis for making real-world lending decisions.

## 15. Future Improvements

The project can be improved by:

* Using a larger and more diverse dataset.
* Testing additional machine learning algorithms.
* Performing more detailed feature engineering.
* Applying hyperparameter tuning.
* Adding cross-validation.
* Improving the prediction interface.
* Monitoring model performance on new data.

## 16. Conclusion

This project demonstrates a basic machine learning workflow for loan approval classification. It covers data preprocessing, model training, model evaluation, model saving, and prediction using Python and Scikit-learn.
