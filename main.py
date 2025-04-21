from sklearn.model_selection import train_test_split # type: ignore
from data import load_and_preprocess_data
from model import train_model
from predict import evaluate_model

X, y = load_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
