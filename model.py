import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.ensemble import RandomForestClassifier #type: ignore
from sklearn.impute import SimpleImputer #type: ignore
import joblib # type: ignore


df = pd.read_csv('datasets/cumulative.csv')

# Pudotetaan ei-numeeriset ja tarpeettomat sarakkeet
drop_columns = ['koi_disposition', 'rowid', 'kepid', 'kepoi_name', 'kepler_name']
for col in drop_columns:
    if col in df.columns:
        df = df.drop(col, axis=1)

# Pidetään vain numeeriset sarakkeet
numeric_df = df.select_dtypes(include=[np.number])

# Täytetään puuttuvat arvot
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(numeric_df)

# 5. Koska datassa ei ole suoraan asuttavuuden targetia
# luodaan "fake target" käyttämällä olemassa olevaa 'koi_score' arvoa.
# Tämä tehdään siten, että määritellään planeetta asuttavaksi, jos koi_score > 0.5.
# tällöin 'target' on 1. 
# Jos 'koi_score' ei ole datassa, käytetään oletuksena targetiksi nollia kaikille riveille. 
# Tämä on osin myös siksi, etten ole matemaatikko ja haluan vain mallintaa olemassa olevaa dataa.
target = (numeric_df['koi_score'] > 0.5).astype(int) if 'koi_score' in numeric_df.columns else np.zeros(len(numeric_df))


X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

# 7. Koulutetaan malli. Tämähän jäi vain kokeiluksi, kun en osannut saattaa sitä loppuun saakka.
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 8. Tallennetaan malli
joblib.dump(model, 'model/random_forest_model.joblib')
