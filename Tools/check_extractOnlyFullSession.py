import pandas as pd
import numpy as np


def analyser_sessions(df):
    if 'SessionStartEnd' not in df.columns or 'timeStampOpeningConvertedtoDate' not in df.columns:
        raise ValueError(
            "Les colonnes 'SessionStartEnd' et 'timeStampOpeningConvertedtoDate' doivent être présentes dans le DataFrame.")

    session_start = None
    sessions = []
    erreurs = []

    for index, row in df.iterrows():
        if row['SessionStartEnd'] == 10:
            if session_start is not None:
                erreurs.append(f"Erreur: Deux débuts de session consécutifs à l'index {index}")
            session_start = row['timeStampOpeningConvertedtoDate']
        elif row['SessionStartEnd'] == 20:
            if session_start is None:
                erreurs.append(f"Erreur: Fin de session sans début correspondant à l'index {index}")
            else:
                session_end = row['timeStampOpeningConvertedtoDate']
                duree = (session_end - session_start).total_seconds() / 60  # Durée en minutes
                sessions.append((session_start, session_end, duree))
                if duree < 1380 * 0.95:
                    erreurs.append(f"Erreur: Session trop courte à l'index {index}. Durée: {duree} minutes")
                elif duree > 1380:
                    erreurs.append(f"Erreur: Session trop longue à l'index {index}. Durée: {duree} minutes")
                session_start = None
        elif row['SessionStartEnd'] == 15:
            # Ignorer les valeurs 15
            pass
        else:
            erreurs.append(f"Erreur: Valeur inattendue {row['SessionStartEnd']} à l'index {index}")

    if session_start is not None:
        erreurs.append("Erreur: La dernière session n'a pas de fin")

    return sessions, erreurs

# Charger les données
file_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL\merge\MergedAllFile_030619_300824_merged_extractOnlyFullSession.csv"
df = pd.read_csv(file_path, sep=';')

# Convertir la colonne timeStampOpening en datetime dans une nouvelle colonne
df['timeStampOpeningConvertedtoDate'] = pd.to_datetime(df['timeStampOpening'])  # Supprimé l'argument unit='s'

# Analyser les sessions
try:
    sessions, erreurs = analyser_sessions(df)

    print(f"\nNombre total de sessions : {len(sessions)}")

    if erreurs:
        print("\nErreurs détectées:")
        for erreur in erreurs:
            print(erreur)
    else:
        print("\nAucune erreur détectée.")

    # Afficher quelques statistiques sur les sessions
    if sessions:
        durees = [duree for _, _, duree in sessions]
        print(f"\nDurée moyenne des sessions : {np.mean(durees):.2f} minutes")
        print(f"Durée minimale des sessions : {np.min(durees):.2f} minutes")
        print(f"Durée maximale des sessions : {np.max(durees):.2f} minutes")

except ValueError as e:
    print(f"Erreur lors de l'analyse des sessions : {e}")