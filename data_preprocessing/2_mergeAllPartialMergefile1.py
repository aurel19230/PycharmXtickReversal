import pandas as pd
import os
import re


def configurer_pandas():
    """Configure les options d'affichage de pandas."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.precision', 10)
    pd.set_option('display.float_format', lambda x: '%.10f' % x if abs(x) < 1e10 else '%.0f' % x)


def generate_output_filename(files, xtickRev_config_dir):
    """Génère le nom du fichier de sortie basé sur les fichiers d'entrée."""
    if not files:
        raise ValueError("La liste des fichiers est vide.")

    # Trier les fichiers
    sorted_files = sorted(files)

    # Trouver le fichier qui se termine par "_0.csv"
    first_file = next((f for f in sorted_files if f.endswith('_0.csv')), None)
    if first_file is None:
        raise ValueError("Aucun fichier ne se termine par '_0.csv'.")

    # Extraire la date du début à partir du fichier _0.csv
    start_date = first_file.split('_')[1]

    # Trouver le fichier avec le plus grand X dans _X.csv
    last_file = max(sorted_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Extraire la date de fin à partir du dernier fichier
    end_date = last_file.split('_')[2]

    # Générer le nom du fichier de sortie
    output_filename = f"Step2_{xtickRev_config_dir}_{start_date}_{end_date}.csv"
    return output_filename


def nettoyer_dernieres_lignes(df, nom_fichier):
    """Supprime les dernières lignes à partir du dernier SessionStartEnd=10."""
    taille_initiale = len(df)

    # Vérifier si SessionStartEnd=10 existe dans le DataFrame
    if 10 in df['SessionStartEnd'].values:
        # Trouver l'index de la dernière occurrence de SessionStartEnd=10
        dernier_index = df.loc[df['SessionStartEnd'] == 10].index[-1]
        # Garder seulement les lignes jusqu'à cet index (non inclus)
        df_nettoye = df.loc[:dernier_index - 1].copy()
        lignes_supprimees = taille_initiale - len(df_nettoye)
        print(f"Fichier {nom_fichier}: {lignes_supprimees} lignes supprimées")
        return df_nettoye
    else:
        print(f"Fichier {nom_fichier}: Aucun SessionStartEnd=10 trouvé")
        return df


def verifier_ordre_chronologique(df):
    """Vérifie si les timestamps sont dans l'ordre croissant."""
    if not df['timeStampOpening'].is_monotonic_increasing:
        raise ValueError("Les timeStampOpening ne sont pas dans un ordre strictement croissant")
    return True


def verifier_doublons(df):
    """Vérifie s'il y a des doublons dans le DataFrame fusionné."""
    colonnes_a_verifier = ['timeStampOpening', 'close', 'open', 'high', 'low',
                           'volume', 'atr', 'vaDelta_6periods', 'vaVol_16periods', 'perctBB', 'SessionStartEnd']

    doublons = df[df.duplicated(subset=colonnes_a_verifier, keep=False)]
    if len(doublons) > 0:
        print("\nDoublons trouvés :")
        print(doublons[colonnes_a_verifier].sort_values(by='timeStampOpening'))
        raise ValueError(f"Il y a {len(doublons)} lignes en double après la fusion")
    return True


def charger_et_traiter_fichiers(directory):
    """Charge, nettoie et fusionne tous les fichiers CSV du répertoire."""
    # Récupérer tous les fichiers CSV
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    files = [f for f in all_files if re.match(r'.+_\d+\.csv$', f)]

    # Trier les fichiers par leur numéro
    files.sort(key=lambda f: int(re.findall(r'_(\d+)\.csv$', f)[0]))

    # Vérifier l'ordre des fichiers
    file_numbers = [int(re.findall(r'_(\d+)\.csv$', f)[0]) for f in files]
    if file_numbers != list(range(len(file_numbers))):
        raise ValueError("Les fichiers ne sont pas consécutifs ou ne commencent pas par _0")

    print("Fichiers à traiter :")
    for file in files:
        print(f" - {file}")

    # Traiter chaque fichier
    dataframes = []
    for file in files:
        print(f"\nTraitement de {file}")
        df = pd.read_csv(os.path.join(directory, file), sep=";", encoding="ISO-8859-1")
        df['timeStampOpening'] = df['timeStampOpening'].astype('Int64')

        # Nettoyer les dernières lignes
        df_clean = nettoyer_dernieres_lignes(df, file)
        dataframes.append(df_clean)

    # Fusionner tous les DataFrames
    print("\nFusion des fichiers...")
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Vérifications finales
    print("\nVérification de l'ordre chronologique...")
    verifier_ordre_chronologique(merged_df)

    print("Vérification des doublons...")
    verifier_doublons(merged_df)

    return merged_df, files


def main():
    DIRECTORY_PATH = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\\\4_0_5TP_1SL_newBB\merge"

    # Extraire le nom du répertoire de configuration
    path_components = DIRECTORY_PATH.split(os.sep)
    merge_index = path_components.index('merge')
    xtickRev_config_dir = path_components[merge_index - 1]
    print(f"Configuration directory: {xtickRev_config_dir}")

    # Configuration de pandas
    configurer_pandas()

    try:
        # Charger et traiter les fichiers
        df_final, files = charger_et_traiter_fichiers(DIRECTORY_PATH)
        print(f"\nTraitement terminé avec succès !")
        print(f"Nombre total de lignes dans le fichier final : {len(df_final)}")

        # Générer le nom du fichier de sortie et sauvegarder
        output_filename = generate_output_filename(files, xtickRev_config_dir)
        output_path = os.path.join(DIRECTORY_PATH, output_filename)
        df_final.to_csv(output_path, index=False, sep=';')
        print(f"\nFichier sauvegardé : {output_filename}")

    except Exception as e:
        print(f"\nUne erreur s'est produite : {str(e)}")


if __name__ == "__main__":
    main()