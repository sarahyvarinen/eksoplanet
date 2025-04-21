import joblib # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    def load_model():
        return joblib.load('model/random_forest_model.joblib')