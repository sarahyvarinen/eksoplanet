import pandas as pd

data = pd.read_csv("datasets/cumulative.csv")

def arvioi_asuttavuus(planeetta):
    rivi = data[data['kepoi_name'] == planeetta]

    if rivi.empty:
        print(f"Planeettaa {planeetta} ei löytynyt.")
        return

    teq = rivi['koi_teq'].values[0]       # Arvioitu lämpötila (Kelvin)
    srad = rivi['koi_srad'].values[0]     # Tähden säde (auringon säteinä, R☉)
    prad = rivi['koi_prad'].values[0]     # Planeetan säde (maan säteinä, R⊕)

    pisteet = 0
    selitys = []

    # Lämpötila (koi_teq)
    if pd.notna(teq):
        if 180 <= teq <= 310:
            pisteet += 40
            selitys.append(f"Lämpötila: {teq:.0f} K ✅ (ihanteellinen elämän kannalta)")
        else:
            selitys.append(f"Lämpötila: {teq:.0f} K ❌ (epäihanteellinen)")
    else:
        selitys.append("Lämpötila: ei tietoa ❌")

    # Tähden säde (koi_srad)
    if pd.notna(srad):
        if 0.7 <= srad <= 1.4:
            pisteet += 30
            selitys.append(f"Tähden säde: {srad:.2f} R☉ ✅ (sopiva, verrattuna Auringon säteeseen)")
        else:
            selitys.append(f"Tähden säde: {srad:.2f} R☉ ❌ (epäsopiva, verrattuna Auringon säteeseen)")
    else:
        selitys.append("Tähden säde: ei tietoa ❌")

    # Planeetan säde (koi_prad)
    if pd.notna(prad):
        if 0.8 <= prad <= 1.8:
            pisteet += 30
            selitys.append(f"Planeetan säde: {prad:.2f} R⊕ ✅ (maankaltainen, verrattuna Maan säteeseen)")
        else:
            selitys.append(f"Planeetan säde: {prad:.2f} R⊕ ❌ (ei maankaltainen, verrattuna Maan säteeseen)")
    else:
        selitys.append("Planeetan säde: ei tietoa ❌")

    print(f"\nPlaneetta {planeetta} arvioitu asuttavuus on {pisteet}%")
    print("Perustelut:")
    for kohta in selitys:
        print(" -", kohta)

# Käyttäjän syöte
planeetta = input("Syötä planeetan nimi (esim. K00752.01): ")
arvioi_asuttavuus(planeetta.strip())
