import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier# type: ignore
from sklearn.metrics import accuracy_score, classification_report# type: ignore
from sklearn.impute import SimpleImputer# type: ignore
import joblib# type: ignore

def main():
    # Ladataan data
    print("Ladataan data...")
    df = pd.read_csv('datasets/cumulative.csv')

    # Tarkistetaan data
    print("Data ensimmäiset rivit:")
    print(df.head())

    # Kategoristen muuttujien käsittely (yksi-hot-enkoodaus)
    print("Käsitellään kategoriset muuttujat...")
    X = df.drop(['koi_disposition', 'rowid'], axis=1)
    X = pd.get_dummies(X)

    # Täytetään puuttuvat arvot
    print("Täytetään puuttuvat arvot...")
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Target muuttuja
    y = df['koi_disposition']

    # Jaetaan data koulutus- ja testidatoihin
    print("Jaetaan data koulutukseen ja testaukseen...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Mallin luominen ja koulutus
    print("Koulutetaan malli...")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Ennusteiden tekeminen
    print("Tekemään ennusteita...")
    y_pred = model.predict(X_test)

    # Suorituskyvyn arviointi
    print("Mallin suorituskyvyn arviointi...")
    print(f"Tarkkuus: {accuracy_score(y_test, y_pred)}")
    print("Luokitteluraportti:")
    print(classification_report(y_test, y_pred))

    # Tallennetaan malli
    print("Tallennetaan malli...")
    joblib.dump(model, 'model/random_forest_model.joblib')
    print("Malli tallennettu!")

if __name__ == "__main__":
    main()
