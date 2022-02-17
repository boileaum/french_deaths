"""Process INSEE data"""

from pathlib import Path
import pandas as pd

# Le fichier est fourni au format txt

# Nom et Prénom - Longueur : 80 - Position : 1-80 - Type : Alphanumérique
# La forme générale est NOM*PRENOMS

# Sexe - Longueur : 1 - Position : 81 - Type : Numérique
# 1 = Masculin; 2 = féminin

# Date de naissance - Longueur : 8 - Position : 82-89 - Type : Numérique
# Forme : AAAAMMJJ - AAAA=0000 si année inconnue; MM=00 si mois inconnu; JJ=00 si jour inconnu

# Code du lieu de naissance - Longueur : 5 - Position : 90-94 - Type : Alphanumérique
# Code Officiel Géographique en vigueur au moment de la prise en compte du décès

# Commune de naissance en clair - Longueur : 30 - Position : 95-124 - Type : Alphanumérique

# DOM/TOM/COM/Pays de naissance en clair - Longueur : 30 - Position : 125-154 - Type : Alphanumérique

# Date de décès - Longueur : 8 - Position : 155-162 - Type : Numérique
# Forme : AAAAMMJJ - AAAA=0000 si année inconnue; MM=00 si mois inconnu; JJ=00 si jour inconnu

# Code du lieu de décès - Longueur : 5 - Position : 163-167 - Type : Alphanumérique
# Code Officiel Géographique en vigueur au moment de la prise en compte du décès

# Numéro d'acte de décès - Longueur : 9 - Position : 168-176 - Type : Alphanumérique
# NOTA : Certains enregistrements peuvent contenir en toute fin des caractères non significatifs. Il est donc important, pour lire correctement ce champ, de bien respecter sa longueur ou sa borne de fin.


def txt2csv(data_file: str = "deces-2022-m01.txt"):
    """Convert INSEE format to csv"""

    csv_file = f"{Path(data_file).stem}.csv"

    with open(csv_file, 'w') as cf:
        with open(data_file) as f:
            cf.write("sex,birthdate,deathdate\n")
            # 1,1922-09-25,2022-01-24
            for line in f:
                m = re.match(
                    r".*/\s+(\d)(\d{4})(\d{2})(\d{2}).{65}(\d{4})(\d{2})(\d{2})", line)
                groups = (m.group(i) for i in range(1, 8))
                cf.write("{},{}-{}-{},{}-{}-{}\n".format(*groups))

    print(f"written {csv_file}")


def load_csv(csv_file: str):
    """Load csv as pandas dataframe"""

    df = pd.read_csv(csv_file, parse_dates=['birthdate', 'deathdate'],
                     infer_datetime_format=True,
                     date_parser=lambda x: pd.to_datetime(x, errors='coerce'))
    df['age'] = df['deathdate'] - df['birthdate']
    df['age'].dropna(axis='index', inplace=True)
    df['age'] = df['age'] / np.timedelta64(1, 'Y')
    ax = df['age'].hist(bins=120, fill=False, density=True)
    ax.set_xlabel('Âge (années)')
    return ax
