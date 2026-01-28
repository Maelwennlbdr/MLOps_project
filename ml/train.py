import pandas as pd
import yaml
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def load_config():
    with open("ml/config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # Load data
    df = pd.read_csv(config["data"]["path"])

    X = df.drop(columns=[config["data"]["target"]])
    y = df[config["data"]["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=y
    )

    # Build pipeline
    steps = []
    if config["training"]["scale"]:
        steps.append(("scaler", StandardScaler()))

    steps.append((
        "model",
        LogisticRegression(**config["model"]["params"])
    ))

    pipeline = Pipeline(steps)

    # Train
    pipeline.fit(X_train, y_train)

    # Save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(pipeline, "models/model.joblib")

    print("✅ Model trained and saved to models/model.joblib")


if __name__ == "__main__":
    main()