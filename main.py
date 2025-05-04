import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
import joblib # type: ignore

# Tämä kaikki on vain jätetty tänne dokumentaatiota varten, koska en saanut koneoppimismallia toimimaan
def main():
    
    print("Ladataan data...")  # Tulostetaan viesti, kun dataa ladataan
    df = pd.read_csv('datasets/cumulative.csv') 

    # Tarkistetaan data
    print("Data ensimmäiset rivit:")  # Tulostetaan datan ensimmäiset rivit tarkistusta varten
    print(df.head())  # Näytetään DataFramen ensimmäiset rivit

    # Kategoristen muuttujien käsittely (yksi-hot-enkoodaus)
    print("Käsitellään kategoriset muuttujat...")  # Viesti, että kategoriset muuttujat käsitellään
    # Poistetaan 'koi_disposition' ja 'rowid' sarakkeet, koska ne eivät ole syötteitä mallille
    X = df.drop(['koi_disposition', 'rowid'], axis=1)  # Poistetaan ei-tarpeelliset sarakkeet
    # Kategoriset muuttujat muunnetaan numeerisiksi yksi-hot-enkoodauksella

    # Täytetään puuttuvat arvot
    print("Täytetään puuttuvat arvot...")  # Viesti, että puuttuvat arvot täytetään
    imputer = SimpleImputer(strategy='mean')  # Käytetään keskiarvoa täyttämään puuttuvat arvot
    X = imputer.fit_transform(X)  # Täytetään puuttuvat arvot DataFramessa

    # Target muuttuja
    y = df['koi_disposition']  # 'koi_disposition' on se, mitä ennustetaan (target)

    

    # Mallin luominen ja koulutus
    print("Koulutetaan malli...")  # Viesti, että malli koulutetaan
    model = RandomForestClassifier(random_state=42)  

    
    

    # Tallennetaan malli
    print("Tallennetaan malli...")  # Viesti, että malli tallennetaan
    joblib.dump(model, 'model/random_forest_model.joblib')  # Tallennetaan malli tiedostoon
    print("Malli tallennettu!")  # Viesti, että malli on tallennettu

#if __name__ == "__main__":
  #  main()  
