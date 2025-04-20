import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
from sklearn.impute import SimpleImputer # type: ignore


df = pd.read_csv('datasets/cumulative.csv')

#target
y = df['koi_disposition']

X = df.drop(['koi_disposition', 'rowid'], axis=1)



print(df.head())