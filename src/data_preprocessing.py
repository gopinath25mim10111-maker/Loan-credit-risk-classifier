import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

TARGET = "Loan_Status"

def load_data(path):
    return pd.read_csv(path)

def prepare_data(df):
    df = df.copy()
    df = df.drop_duplicates()

    y = df[TARGET].map({"Y": 1, "N": 0})

    X = df.drop(columns=[TARGET, "Loan_ID"], errors="ignore")

    categorical_columns = X.select_dtypes(include=["object"]).columns
    numerical_columns = X.select_dtypes(exclude=["object"]).columns

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    return X, y, preprocessor

def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )