import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def main():

    test_data = pd.read_csv("results/test_data.csv")

    X_test = test_data.drop(columns=["Loan_Status"])
    y_test = test_data["Loan_Status"]

    results = []

    for filename in os.listdir("models"):

        if not filename.endswith(".pkl"):
            continue

        model = joblib.load(
            os.path.join("models", filename)
        )

        predictions = model.predict(X_test)

        results.append({
            "Model": filename.replace(".pkl", ""),
            "Accuracy": accuracy_score(
                y_test, predictions
            ),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1 Score": f1_score(
                y_test, predictions, zero_division=0
            )
        })

        cm = confusion_matrix(
            y_test,
            predictions
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Rejected", "Approved"]
        )

        display.plot()

        plt.title(
            filename.replace(".pkl", "").replace("_", " ").title()
        )

        plt.tight_layout()

        plt.savefig(
            f"results/{filename.replace('.pkl', '')}_confusion_matrix.png"
        )

        plt.close()

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "results/model_comparison.csv",
        index=False
    )

    print("\nMODEL PERFORMANCE")
    print(results_df.to_string(index=False))

    print("\nResults saved in the results folder.")


if __name__ == "__main__":
    main()