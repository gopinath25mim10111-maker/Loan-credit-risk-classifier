import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import load_data, prepare_data, split_data

DATA_PATH = "data/loan_data.csv"

def main():

    df = load_data(DATA_PATH)

    X, y, preprocessor = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        "logistic_regression": LogisticRegression(max_iter=3000),
        "decision_tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=8
        )
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        joblib.dump(
            pipeline,
            f"models/{name}.pkl"
        )

        print(f"Saved: models/{name}.pkl")

    test_data = X_test.copy()
    test_data["Loan_Status"] = y_test.values

    test_data.to_csv(
        "results/test_data.csv",
        index=False
    )

    print("\nTraining completed successfully.")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))


if __name__ == "__main__":
    main()