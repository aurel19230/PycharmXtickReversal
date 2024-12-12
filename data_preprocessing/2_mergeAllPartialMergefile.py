import os
import pandas as pd
import re


# Chemin spécifique pour les fichiers CSV
directory = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_6TP_1SL\merge"
#merge les fichier terminant par _x dans l'ordre. Pour éviter les erreurs les _x sont à rajouter suite au merge du step 1

#Diviser le chemin en ses composants
path_components = directory.split(os.sep)

# Trouver l'index de 'merge'
merge_index = path_components.index('merge')

# Extraire le répertoire juste avant 'merge'
xtickRev_config_dir = path_components[merge_index - 1]

print(xtickRev_config_dir)


option = input("Appuyez sur 'd' dedeoublonner, entrée pour uniquement concatener : ").lower()


def generate_output_filename(files,xtickRev_config_dir):
    if not files:
        raise ValueError("La liste des fichiers est vide.")

    # Trier les fichiers
    sorted_files = sorted(files)

    # Trouver le fichier qui se termine par "_0.csv"
    first_file = next((f for f in sorted_files if f.endswith('_0.csv')), None)
    if first_file is None:
        raise ValueError("Aucun fichier ne se termine par '_0.csv'.")

    # Extraire la date du début à partir du fichier _0.csv (partie avant le premier '_')
    start_date = first_file.split('_')[1]

    # Trouver le fichier avec le plus grand X dans _X.csv
    last_file = max(sorted_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Extraire la date de fin à partir du dernier fichier (partie après le dernier '_', sans '.csv')
    end_date = last_file.split('_')[2]

    # Utiliser "MergedAllFile" comme nom de base du fichier de sortie
    output_filename = f"Step2_{xtickRev_config_dir}_{start_date}_{end_date}.csv"
    return output_filename

# Le reste du code reste inchangé



import os
import re
import pandas as pd


def merge_files(directory):
    # Récupérer tous les fichiers CSV dans le répertoire
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    # Filtrer les fichiers qui se terminent par *X.csv où X est un nombre
    files = [f for f in all_files if re.match(r'.+_\d+\.csv$', f)]

    # Trier les fichiers par leur numéro à la fin
    files.sort(key=lambda f: int(re.findall(r'_(\d+)\.csv$', f)[0]))

    # Afficher les noms des fichiers à fusionner
    print("Fichiers à fusionner :")
    for file in files:
        print(f" - {file}")
    print()  # Ligne vide pour la lisibilité

    # Vérifier si des fichiers correspondants ont été trouvés
    if not files:
        raise ValueError(
            f"Aucun fichier CSV correspondant au format *_X.csv n'a été trouvé dans le répertoire : {directory}")

    # Vérifier si les fichiers sont consécutifs et commencent par *0
    file_numbers = [int(re.findall(r'_(\d+)\.csv$', f)[0]) for f in files]
    if file_numbers != list(range(len(file_numbers))):
        raise ValueError("Les fichiers ne sont pas consécutifs ou ne commencent pas par *0.")

    # Lire tous les fichiers
    dataframes = []
    for i, file in enumerate(files):
        if i == 0:
            # Pour le premier fichier, lire normalement avec l'en-tête
            df = pd.read_csv(os.path.join(directory, file), delimiter=';')
        else:
            # Pour les fichiers suivants, lire sans l'en-tête
            df = pd.read_csv(os.path.join(directory, file), delimiter=';', header=0)

        print(f"Traitement du fichier {file} : {len(df)} lignes")
        dataframes.append(df)

    # Concaténer tous les dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    if option == 'd':
        # Conversion de timeStampOpening en numérique
        merged_df['timeStampOpening'] = pd.to_numeric(merged_df['timeStampOpening'])

        # Colonnes pour la vérification des doublons
        colonnes_a_verifier = ['close', 'open', 'high', 'low', 'volume', "atr", "vaDelta_6periods", 'vaVol_16periods']

        # Identifier les doublons avant suppression
        doublons = merged_df[merged_df.duplicated(subset=colonnes_a_verifier, keep=False)].copy()

        # Grouper les doublons pour analyse
        groupes_doublons = doublons.groupby(colonnes_a_verifier)

        print("\nAnalyse des groupes de doublons avant suppression:")
        for name, group in groupes_doublons:
            nb_doublons = len(group) - 1  # -1 car on garde une ligne
            min_timestamp = group['timeStampOpening'].min()
            print(f"\nGroupe de doublons:")
            print(f"Nombre de lignes à supprimer: {nb_doublons}")
            print(f"TimeStampOpening conservé: {min_timestamp}")
            print("Lignes du groupe:")
            print(group.sort_values('timeStampOpening'))
            print("-" * 80)

        # Supprimer les doublons en gardant celui avec le plus petit timeStampOpening
        merged_df_clean = merged_df.sort_values('timeStampOpening').drop_duplicates(
            subset=colonnes_a_verifier,
            keep='first'
        )

        # Statistiques finales
        nb_lignes_avant = len(merged_df)
        nb_lignes_apres = len(merged_df_clean)
        nb_doublons_supprimes = nb_lignes_avant - nb_lignes_apres

        print(f"\nStatistiques finales:")
        print(f"Nombre de lignes avant: {nb_lignes_avant}")
        print(f"Nombre de lignes après: {nb_lignes_apres}")
        print(f"Nombre total de doublons supprimés: {nb_doublons_supprimes}")

        # Assigner le résultat nettoyé
        merged_df = merged_df_clean

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

    print(f"\nNombre total de lignes après la fusion : {len(merged_df)}")

    return merged_df



# Utilisation de la fonction
try:
    result = merge_files(directory)

    # Récupérer la liste des fichiers CSV dans le répertoire pour générer le nom de sortie
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    files = [f for f in all_files if re.match(r'.+_\d+\.csv$', f)]

    # Générer le nom du fichier de sortie
    output_filename = generate_output_filename(files,xtickRev_config_dir)
    output_file = os.path.join(directory, output_filename)

    result.to_csv(output_file, index=False, sep=';')
    print(f"Fusion terminée. Résultat sauvegardé dans {output_file}")
except ValueError as e:
    print(f"Erreur : {str(e)}")
except Exception as e:
    print(f"Une erreur inattendue s'est produite : {str(e)}")