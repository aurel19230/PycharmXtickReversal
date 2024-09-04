import os
import pandas as pd
import re

def generate_output_filename(files):
    if not files:
        raise ValueError("La liste des fichiers est vide.")

    # Trier les fichiers
    sorted_files = sorted(files)

    # Trouver le fichier qui se termine par "_0.csv"
    first_file = next((f for f in sorted_files if f.endswith('_0.csv')), None)
    if first_file is None:
        raise ValueError("Aucun fichier ne se termine par '_0.csv'.")

    # Extraire la date du début à partir du fichier _0.csv (partie avant le premier '_')
    start_date = first_file.split('_')[0]

    # Trouver le fichier avec le plus grand X dans _X.csv
    last_file = max(sorted_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Extraire la date de fin à partir du dernier fichier (partie après le dernier '_', sans '.csv')
    end_date = last_file.split('_')[-1].split('.')[0]

    # Utiliser "MergedAllFile" comme nom de base du fichier de sortie
    output_filename = f"MergedAllFile_{start_date}_{end_date}_merged.csv"
    return output_filename

# Le reste du code reste inchangé



def merge_files(directory):
    # Récupérer tous les fichiers CSV dans le répertoire
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Filtrer les fichiers qui se terminent par _X.csv où X est un nombre
    files = [f for f in all_files if re.match(r'.+_\d+\.csv$', f)]

    # Afficher les noms des fichiers à fusionner
    print("Fichiers à fusionner :")
    for file in files:
        print(f" - {file}")
    print()  # Ligne vide pour la lisibilité

    # Vérifier si des fichiers correspondants ont été trouvés
    if not files:
        raise ValueError(
            f"Aucun fichier CSV correspondant au format *_X.csv n'a été trouvé dans le répertoire : {directory}")

    # Extraire et trier les numéros X
    file_numbers = sorted([int(re.findall(r'_(\d+)\.csv$', f)[0]) for f in files])

    # Vérifier si les fichiers sont consécutifs
    if file_numbers != list(range(len(file_numbers))):
        raise ValueError("Les fichiers ne sont pas consécutifs ou il manque file_0.")

    # Lire tous les fichiers
    dataframes = []
    for i, file_number in enumerate(file_numbers):
        file = next(f for f in files if f.endswith(f'_{file_number}.csv'))
        df = pd.read_csv(os.path.join(directory, file), delimiter=';')
        if i > 0:  # Supprimer l'en-tête pour tous les fichiers sauf le premier
            df = df.iloc[1:]
        dataframes.append(df)

    # Concaténer tous les dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Assurez-vous que timeStampOpening est de type numérique (int64 ou float64)
    merged_df['timeStampOpening'] = pd.to_numeric(merged_df['timeStampOpening'])

    # Colonnes pour la vérification des doublons
    check_columns = ['candleDir', 'candleSizeTicks', 'close', 'open', 'high', 'low', 'pocPrice', 'volPOC', 'deltaPOC',
                     'volume', 'delta', 'VolBlw', 'DeltaBlw', 'VolAbv', 'DeltaAbv', 'VolBlw_6Tick', 'DeltaBlw_6Tick',
                     'VolAbv_6Tick', 'DeltaAbv_6Tick', 'bidVolLow', 'askVolLow', 'bidVolLow_1', 'askVolLow_1',
                     'bidVolLow_2', 'askVolLow_2', 'bidVolLow_3', 'askVolLow_3', 'bidVolHigh', 'askVolHigh',
                     'bidVolHigh_1', 'askVolHigh_1', 'bidVolHigh_2', 'askVolHigh_2', 'bidVolHigh_3', 'askVolHigh_3',
                     'VWAP', 'VWAPsd1Top', 'VWAPsd2Top', 'VWAPsd3Top', 'VWAPsd4Top', 'VWAPsd1Bot', 'VWAPsd2Bot',
                     'VWAPsd3Bot', 'VWAPsd4Bot', 'bandWidthBB', 'perctBB', 'atr']

    # Trier le DataFrame par timeStampOpening
    merged_df = merged_df.sort_values('timeStampOpening')

    # Supprimer les doublons en gardant la première occurrence, uniquement pour les timeStampOpening égaux
    merged_df = merged_df.drop_duplicates(
        subset=['timeStampOpening'] + check_columns, keep='first'
    )

    # Vérification finale de l'ordre chronologique
    if not merged_df['timeStampOpening'].is_monotonic_increasing:
        print("Attention : Les timeStampOpening ne sont pas dans un ordre strictement croissant après la fusion.")
        print("Tri final du DataFrame par timeStampOpening...")
        merged_df = merged_df.sort_values('timeStampOpening', ignore_index=True)

        # Vérification après le tri
        if not merged_df['timeStampOpening'].is_monotonic_increasing:
            raise ValueError(
                "Erreur critique : Impossible d'obtenir un ordre chronologique strict des timeStampOpening.")
        else:
            print("Tri effectué avec succès. Les timeStampOpening sont maintenant dans un ordre strictement croissant.")
    else:
        print("Les timeStampOpening sont dans un ordre strictement croissant.")

    return merged_df


# Chemin spécifique pour les fichiers CSV
directory = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL\merge"

# Utilisation de la fonction
try:
    result = merge_files(directory)

    # Récupérer la liste des fichiers CSV dans le répertoire pour générer le nom de sortie
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    files = [f for f in all_files if re.match(r'.+_\d+\.csv$', f)]

    # Générer le nom du fichier de sortie
    output_filename = generate_output_filename(files)
    output_file = os.path.join(directory, output_filename)

    result.to_csv(output_file, index=False, sep=';')
    print(f"Fusion terminée. Résultat sauvegardé dans {output_file}")
except ValueError as e:
    print(f"Erreur : {str(e)}")
except Exception as e:
    print(f"Une erreur inattendue s'est produite : {str(e)}")