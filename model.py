import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.ensemble import RandomForestClassifier #type: ignore
from sklearn.impute import SimpleImputer #type: ignore
import joblib #type: ignore

# 1. Ladataan data
df = pd.read_csv('datasets/cumulative.csv')

# 2. Pudotetaan ei-numeeriset ja tarpeettomat sarakkeet
drop_columns = ['koi_disposition', 'rowid', 'kepid', 'kepoi_name', 'kepler_name']
for col in drop_columns:
    if col in df.columns:
        df = df.drop(col, axis=1)

# 3. Pidetään vain numeeriset sarakkeet
numeric_df = df.select_dtypes(include=[np.number])

# 4. Täytetään puuttuvat arvot
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(numeric_df)

# 5. Koska tässä ei ole suoraan targettia (esim. onko asuttava), 
# tehdään mallinnus "fake targetilla" -> tässä demossa käytetään esim. koi_score > 0.5
target = (numeric_df['koi_score'] > 0.5).astype(int) if 'koi_score' in numeric_df.columns else np.zeros(len(numeric_df))

# 6. Jaetaan data (tässä ei ole pakko, koska tehdään vain demo)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

# 7. Koulutetaan malli
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 8. Tallennetaan malli
joblib.dump(model, 'model/random_forest_model.joblib')
