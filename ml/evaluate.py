import pandas as pd
import yaml
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def load_config():
    with open("ml/config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = pd.read_csv(config["data"]["path"])

    X = df.drop(columns=[config["data"]["target"]])
    y = df[config["data"]["target"]]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    model = joblib.load("models/model.joblib")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
