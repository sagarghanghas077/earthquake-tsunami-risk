import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.preprocess import load_data


DATA_PATH = "D:/earthquake-tsunami-risk/data/raw/earthquake_data_tsunami.csv"
MODEL_DIR = "models"


def train_and_evaluate():
    X, y, scaler = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", probability=True),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            eval_metric="logloss",
            random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            random_state=42
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "ROC_AUC": roc_auc,
            "Object": model
        })

        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC : {roc_auc:.4f}")

    results_df = pd.DataFrame(results).sort_values(
        by="ROC_AUC", ascending=False
    )

    print("\n🏆 Model Comparison:")
    print(results_df[["Model", "Accuracy", "ROC_AUC"]])

    best_row = results_df.iloc[0]
    best_model = best_row["Object"]
    best_model_name = best_row["Model"]

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, f"{MODEL_DIR}/tsunami_classifier.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    print(f"\n✅ Best Model Saved: {best_model_name}")


if __name__ == "__main__":
    train_and_evaluate()
