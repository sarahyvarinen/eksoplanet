import pandas as pd

# Ladataan CSV
df = pd.read_csv('datasets/cumulative.csv')

# Pääohjelma
def main():
    planeetta = input("Anna planeetan nimi (esim. K00752.01): ").strip()

    # Etsitään rivi, jossa kepoi_name vastaa annettua nimeä
    planeetta_rivi = df[df['kepoi_name'] == planeetta]

    if planeetta_rivi.empty:
        print(f"Planeettaa '{planeetta}' ei löytynyt tiedostosta.")
    else:
        # Haetaan koi_score (asuttavuuden todennäköisyys)
        koi_score = planeetta_rivi.iloc[0]['koi_score']

        # Jos koi_score on NaN, ilmoitetaan siitä
        if pd.isna(koi_score):
            print(f"Asuttavuuspisteet (koi_score) eivät ole saatavilla planeetalle {planeetta}.")
        else:
            prosentti = round(koi_score * 100, 2)
            print(f"Planeetta {planeetta} on todennäköisyydellä {prosentti}% eksoplaneetta.")

if __name__ == "__main__":
    main()
