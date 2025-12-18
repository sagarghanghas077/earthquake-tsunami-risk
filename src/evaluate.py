import joblib
from sklearn.metrics import classification_report, roc_auc_score
from preprocess import load_and_clean_data, preprocess_data
from sklearn.model_selection import train_test_split

MODEL_PATH = "models/tsunami_classifier.pkl"

def evaluate():
    df = load_and_clean_data("data/raw/earthquake_data_tsunami.csv")
    X, y, _ = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

if __name__ == "__main__":
    evaluate()
