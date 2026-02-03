import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# ---- CONFIG ----
DATA_PATH = "data/raw/diabetes.csv"
MODEL_DIR = "ml/models"
MODEL_PATH = os.path.join(MODEL_DIR, "logreg_model.joblib")

# Crée le dossier du modèle s'il n'existe pas
os.makedirs(MODEL_DIR, exist_ok=True)

# ---- CHARGEMENT DES DONNÉES ----
data = pd.read_csv(DATA_PATH)

# Colonnes features et target
X = data.drop(columns="Outcome")
y = data["Outcome"]

# ---- SPLIT ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- ENTRAÎNEMENT ----
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---- ÉVALUATION ----
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# ---- SAUVEGARDE DU MODÈLE ----
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
