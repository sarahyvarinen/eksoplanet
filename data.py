import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
from sklearn.impute import SimpleImputer # type: ignore

def load_and_preprocess_data():
 df = pd.read_csv('datasets/cumulative.csv')
 y = df['koi_disposition']
 X = df.drop(['koi_disposition', 'rowid'], axis=1)
 X = pd.get_dummies(X)
 imputer = SimpleImputer(strategy='mean')
 X = imputer.fit_transform(X)
 return X,y



