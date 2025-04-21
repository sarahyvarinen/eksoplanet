# mypy ei analysoi joblib-kirjastoa
from sklearn.ensemble import RandomForestClassifier # type: ignore
import joblib # type: ignore

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'model/random_forest_model.joblib')
    return model