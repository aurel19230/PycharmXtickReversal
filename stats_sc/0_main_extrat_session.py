import pandas as pd
import os
import datetime
from func_standard import print_notification
import platform
import chardet

if platform.system() != "Darwin":
    directory_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_6SL\merge"
else:
    directory_path = "/Users/aurelienlachaud/Documents/trading_local/5_0_5TP_1SL_1/merge"

file_name = "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat.csv"
file_path = os.path.join(directory_path, file_name)


def detect_file_encoding(file_path):
    """
    Détecte l'encodage d'un fichier.

    Args:
        file_path: Chemin du fichier

    Returns:
        Encodage détecté (str)
    """
    with open(file_path, 'rb') as file:
        # Lire un échantillon du fichier (les premiers 100 Ko devraient suffire)
        raw_data = file.read(100000)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print_notification(f"Encodage détecté : {encoding} (confiance : {confidence:.2f})")
        return encoding


def load_csv_with_encoding(file_path, encodings=None):
    """
    Essaie de charger un fichier CSV avec différents encodages.

    Args:
        file_path: Chemin du fichier CSV
        encodings: Liste d'encodages à essayer (si None, détectera automatiquement)

    Returns:
        DataFrame pandas ou None si échec
    """
    if encodings is None:
        # Si aucun encodage fourni, essayer de détecter
        detected_encoding = detect_file_encoding(file_path)
        encodings = [detected_encoding, 'latin1', 'utf-8', 'cp1252', 'ISO-8859-1']

    # Supprimer les doublons et placer l'encodage détecté en premier
    encodings = list(dict.fromkeys(encodings))

    for encoding in encodings:
        try:
            print_notification(f"Tentative de chargement avec l'encodage: {encoding}")

            # Ajout de low_memory=False pour éviter les DtypeWarning
            df = pd.read_csv(file_path, sep=';', encoding=encoding, low_memory=False)

            # Afficher les types de données détectés pour les colonnes
            print_notification("Types de données détectés :")
            for col, dtype in df.dtypes.items():
                print(f"  - {col}: {dtype}")

            print_notification(f"Chargement réussi avec l'encodage: {encoding}")
            return df, encoding
        except UnicodeDecodeError:
            print_notification(f"Échec avec l'encodage: {encoding}")
        except Exception as e:
            print_notification(f"Autre erreur lors du chargement avec {encoding}: {e}")

    return None, None


def diviser_fichier_par_sessions(fichier_entree):
    """
    Divise un fichier en plusieurs parties selon les sessions spécifiées par l'utilisateur.

    Args:
        fichier_entree: Chemin du fichier d'entrée contenant les sessions complètes
    """
    print_notification("Début de l'analyse et de la division du fichier par sessions")

    # Charger le fichier avec détection d'encodage
    print_notification(f"Chargement du fichier {fichier_entree}")
    df, encoding_used = load_csv_with_encoding(fichier_entree)

    if df is None:
        print_notification("Impossible de charger le fichier avec les encodages disponibles. Fin du programme.")
        return

    print_notification(f"Fichier chargé avec succès : {len(df)} lignes (encodage: {encoding_used})")

    # Vérifier et convertir les types de données problématiques
    print_notification("Vérification et correction des types de données...")

    # Afficher les premières lignes pour comprendre la structure
    print("Aperçu des 5 premières lignes :")
    for col in df.columns:
        print(f"Colonne '{col}' - Premiers éléments: {df[col].head().tolist()}")

    # Conversion des colonnes si nécessaire
    try:
        # Essayer de convertir les colonnes numériques
        if 'SessionStartEnd' in df.columns:
            df['SessionStartEnd'] = pd.to_numeric(df['SessionStartEnd'], errors='coerce')
            print_notification(f"Colonne 'SessionStartEnd' convertie en numérique")

        # Forcer les conversions des colonnes qui posent problème (0, 4, 5)
        # Noter les indices de colonnes selon le message d'erreur
        problematic_columns = [df.columns[0]]
        if len(df.columns) > 4:
            problematic_columns.append(df.columns[4])
        if len(df.columns) > 5:
            problematic_columns.append(df.columns[5])

        for col in problematic_columns:
            # Essayer de déterminer le type approprié pour chaque colonne
            if df[col].str.contains('\.').any():  # Contient des points décimaux
                df[col] = pd.to_numeric(df[col], errors='ignore')
                print_notification(f"Colonne '{col}' traitée comme numérique")
            else:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                    print_notification(f"Colonne '{col}' traitée comme numérique")
                except:
                    print_notification(f"Colonne '{col}' conservée comme chaîne de caractères")
    except Exception as e:
        print_notification(f"Erreur lors de la conversion des types: {e}")
        print_notification("Poursuite du traitement avec les types actuels")

    # Vérifier que les colonnes nécessaires existent
    colonnes_requises = ['SessionStartEnd', 'timeStampOpening']
    for colonne in colonnes_requises:
        if colonne not in df.columns:
            print_notification(f"Erreur : La colonne '{colonne}' est absente du fichier")
            print_notification(f"Colonnes disponibles : {', '.join(df.columns)}")
            return

    # Convertir la colonne timeStampOpening en datetime si ce n'est pas déjà fait
    if 'timeStampOpeningConvertedtoDate' not in df.columns:
        print_notification("Conversion de la colonne timeStampOpening en datetime")
        df['timeStampOpeningConvertedtoDate'] = pd.to_datetime(df['timeStampOpening'], unit='s')

    # Vérifier l'intégrité des sessions (nombre de 10 = nombre de 20)
    nb_debuts = sum(df['SessionStartEnd'] == 10)
    nb_fins = sum(df['SessionStartEnd'] == 20)

    print_notification(f"Nombre de débuts de session (10) : {nb_debuts}")
    print_notification(f"Nombre de fins de session (20) : {nb_fins}")

    if nb_debuts != nb_fins:
        print_notification("ATTENTION : Le nombre de débuts de session ne correspond pas au nombre de fins de session")
        print_notification("Le traitement continue mais les résultats pourraient être incorrects")
    else:
        print_notification(f"Le fichier contient {nb_debuts} sessions complètes")

    # Identifier toutes les sessions
    sessions = []
    debut_session = None

    for index, row in df.iterrows():
        if row['SessionStartEnd'] == 10:
            debut_session = index, row['timeStampOpeningConvertedtoDate']
        elif row['SessionStartEnd'] == 20 and debut_session is not None:
            sessions.append((debut_session[0], index, debut_session[1], row['timeStampOpeningConvertedtoDate']))
            debut_session = None

    # Si la dernière session n'est pas complète, l'ignorer
    if debut_session is not None:
        print_notification("ATTENTION : La dernière session n'est pas complète et sera ignorée")

    # Afficher toutes les sessions identifiées de manière claire pour faciliter la sélection
    print_notification(f"\n{'#' * 5} LISTE DES SESSIONS DISPONIBLES ({len(sessions)}) {'#' * 5}")
    print(f"\n{'N°':^4}|{'DATE DÉBUT':^25}|{'DATE FIN':^25}|{'DURÉE':^15}")
    print("-" * 71)

    for i, (start_idx, end_idx, start_time, end_time) in enumerate(sessions):
        duree = end_time - start_time
        heures = duree.total_seconds() // 3600
        minutes = (duree.total_seconds() % 3600) // 60
        duree_str = f"{int(heures)}h {int(minutes)}m"
        print(
            f"{i + 1:^4}| {start_time.strftime('%Y-%m-%d %H:%M:%S')} | {end_time.strftime('%Y-%m-%d %H:%M:%S')} | {duree_str:^15}")

    print("-" * 71)

    # Étape 2: Demander combien de splits à effectuer avec valeur par défaut (4)
    default_nb_splits = 4
    print_notification(f"\nNombre de splits suggéré par défaut : {default_nb_splits}")

    try:
        user_input = input(f"Combien de splits souhaitez-vous effectuer ? [Par défaut: {default_nb_splits}] : ")

        # Si l'utilisateur appuie simplement sur Entrée, utiliser la valeur par défaut
        if user_input.strip() == "":
            nb_splits = default_nb_splits
            print(f"→ Utilisation de la valeur par défaut : {default_nb_splits} splits")
        else:
            nb_splits = int(user_input)

        if nb_splits <= 0:
            print_notification("Le nombre de splits doit être positif. Utilisation de la valeur par défaut.")
            nb_splits = default_nb_splits
        if nb_splits > len(sessions):
            print_notification(f"Attention : Il n'y a que {len(sessions)} sessions disponibles.")
            print_notification("Le nombre de splits sera limité au nombre de sessions.")
            nb_splits = len(sessions)
    except ValueError:
        print_notification("Entrée invalide. Utilisation de la valeur par défaut.")
        nb_splits = default_nb_splits

    # Valeurs par défaut pour les débuts des splits 2, 3 et 4
    default_values = [85, 186, 285]
    # S'assurer que nous avons exactement les valeurs pour les 3 splits (2, 3 et 4)
    if default_nb_splits == 4:
        # Utiliser les valeurs spécifiques 85, 186, 285 pour un découpage en 4
        default_values = [85, 186, 285]
    else:
        # Pour un autre nombre de splits, calculer des valeurs réparties uniformément
        default_values = []
        for i in range(1, nb_splits):
            # Répartir les sessions uniformément entre les splits
            val = min(round(i * len(sessions) / nb_splits), len(sessions))
            if val > 1:  # Éviter de commencer un split à la session 1
                default_values.append(val)

    points_de_division = [0]  # Le premier split commence toujours au début du fichier

    print_notification("\nLe premier split commencera au début du fichier.")

    # Afficher les valeurs par défaut suggérées pour les splits
    print_notification("\nValeurs suggérées pour les débuts des splits:")
    for i in range(1, nb_splits):
        if i - 1 < len(default_values):
            default_val = default_values[i - 1]
            if default_val > len(sessions):
                default_val = len(sessions)
            print_notification(f"- Split {i + 1} : session {default_val}")

    # Si le nombre de splits par défaut est 4, s'assurer que tous les splits sont affichés
    if nb_splits == 4:
        print_notification("- Split 2 : session 85")
        print_notification("- Split 3 : session 186")
        print_notification("- Split 4 : session 285")

    for i in range(1, nb_splits):
        while True:
            try:
                # Proposer la valeur par défaut et permettre à l'utilisateur de la modifier
                default_val = default_values[i - 1] if i - 1 < len(default_values) else min(
                    i * len(sessions) // nb_splits, len(sessions))
                if default_val > len(sessions):
                    default_val = len(sessions)

                print(f"\nConsultez la liste des sessions ci-dessus pour faire votre choix.")
                user_input = input(
                    f"Entrez le numéro de la session qui commence le split {i + 1} (2-{len(sessions)}) [Par défaut: {default_val}] : ")

                # Si l'utilisateur appuie simplement sur Entrée, utiliser la valeur par défaut
                if user_input.strip() == "":
                    session_num = default_val
                    print(f"→ Utilisation de la valeur par défaut : {default_val}")
                else:
                    session_num = int(user_input)

                if 2 <= session_num <= len(sessions):
                    # On utilise session_num - 1 car les indices commencent à 0 dans la liste
                    session_debut = sessions[session_num - 1][2]  # Date de début de la session
                    print(
                        f"→ Split {i + 1} commencera à la session {session_num} (débutant le {session_debut.strftime('%Y-%m-%d %H:%M:%S')})")
                    points_de_division.append(session_num - 1)
                    break
                else:
                    print(f"❌ Veuillez entrer un numéro entre 2 et {len(sessions)}")
            except (ValueError, IndexError):
                print("❌ Veuillez entrer un nombre valide")

    # Afficher les points de division pour débogage
    print_notification("\n[DEBUG] Points de division avant ajout de la fin: " + str(points_de_division))

    # Ajouter un point supplémentaire pour la fin du fichier
    points_de_division.append(len(sessions))

    print_notification("[DEBUG] Points de division après ajout de la fin: " + str(points_de_division))
    print_notification(f"[DEBUG] Nombre de points de division: {len(points_de_division)}")
    print_notification(f"[DEBUG] Nombre total de sessions: {len(sessions)}")

    # Créer les splits avec débogage
    splits = []
    for i in range(len(points_de_division) - 1):
        debut_split = points_de_division[i]
        # Utiliser directement le point de division suivant sans soustraction
        fin_split = points_de_division[i + 1]

        print_notification(
            f"[DEBUG] Vérification split {i + 1}: début={debut_split}, fin={fin_split}, len(sessions)={len(sessions)}")

        if fin_split > len(sessions):
            fin_split = len(sessions)
            print_notification(f"[DEBUG] Ajustement fin_split à {fin_split} car hors limites")

        if debut_split >= len(sessions):
            print_notification(
                f"[DEBUG] AVERTISSEMENT: Le split {i + 1} est invalide (début: {debut_split + 1}, fin: {fin_split}) - début hors limites")
            continue  # Cette ligne saute la création du split

        if debut_split >= fin_split:
            print_notification(
                f"[DEBUG] AVERTISSEMENT: Le split {i + 1} est invalide (début: {debut_split + 1}, fin: {fin_split}) - début ≥ fin")
            continue  # Cette ligne saute la création du split

        # Première session du split
        premiere_session = sessions[debut_split]
        # Dernière session du split
        derniere_session = sessions[fin_split - 1]  # -1 car fin_split est l'indice APRÈS la dernière session du split

        # Indices dans le DataFrame
        debut_idx = premiere_session[0]  # Indice de début de la première session
        fin_idx = derniere_session[1]  # Indice de fin de la dernière session

        # Dates de début et de fin
        date_debut = premiere_session[2]
        date_fin = derniere_session[3]

        splits.append((i + 1, debut_idx, fin_idx, date_debut, date_fin, fin_split - debut_split))
        print_notification(
            f"[DEBUG] Split {i + 1} créé: début={debut_split + 1}, fin={fin_split}, sessions incluses={fin_split - debut_split}")

    # Afficher les splits pour information de manière détaillée
    print_notification(f"\n{'#' * 5} SPLITS QUI SERONT CRÉÉS ({len(splits)}) {'#' * 5}")
    print(f"\n{'SPLIT':^5}|{'DATE DÉBUT':^25}|{'DATE FIN':^25}|{'NOMBRE SESSIONS':^15}|{'SESSIONS INCLUSES':^20}")
    print("-" * 93)

    for i, (num_split, debut_idx, fin_idx, date_debut, date_fin, nb_sessions) in enumerate(splits):
        # Déterminer les sessions incluses dans ce split
        debut_session_num = points_de_division[i] + 1  # +1 car l'affichage des sessions commence à 1
        fin_session_num = points_de_division[i + 1]  # Pas besoin de -1 car point_de_division[i+1] est déjà l'indice + 1
        sessions_incluses = f"{debut_session_num} à {fin_session_num if fin_session_num < len(sessions) + 1 else len(sessions)}"

        print(
            f"{num_split:^5}| {date_debut.strftime('%Y-%m-%d %H:%M:%S')} | {date_fin.strftime('%Y-%m-%d %H:%M:%S')} | {nb_sessions:^15} | {sessions_incluses:^20}")

    # Demander confirmation avec plus de détails
    print("\n" + "=" * 50)
    print("CONFIRMATION DE CRÉATION DES FICHIERS")
    print("=" * 50)
    print(f"Vous êtes sur le point de créer {len(splits)} fichiers distincts")
    print(f"Les fichiers seront enregistrés dans le même dossier que le fichier source")

    # Exemple avec le format de date DDMMYYYY pour la démonstration
    if splits:
        exemple_date_debut = splits[0][3].strftime('%d%m%Y')
        exemple_date_fin = splits[0][4].strftime('%d%m%Y')
        print(
            f"Exemple de nom de fichier: {os.path.basename(fichier_entree).split('.')[0]}__split1_{exemple_date_debut}_{exemple_date_fin}.{os.path.basename(fichier_entree).split('.')[-1]}")
    print("=" * 50)

    confirmation = input("\nConfirmez-vous la création de ces splits ? (o/n) : ")
    if confirmation.lower() not in ['o', 'oui', 'y', 'yes']:
        print_notification("Opération annulée par l'utilisateur.")
        return

    # Créer les fichiers de split sans demander confirmation
    for num_split, debut_idx, fin_idx, date_debut, date_fin, _ in splits:
        # Extraire les données pour ce split
        split_df = df.iloc[debut_idx:fin_idx + 1].reset_index(drop=True)

        # Créer le nom du fichier
        nom_fichier = os.path.basename(fichier_entree)
        nom_base, extension = os.path.splitext(nom_fichier)
        dossier = os.path.dirname(fichier_entree)

        # Formater les dates pour le nom de fichier au format DDMMYYYY (jour-mois-année)
        date_debut_str = date_debut.strftime('%d%m%Y')
        date_fin_str = date_fin.strftime('%d%m%Y')

        nouveau_fichier = f"{nom_base}__split{num_split}_{date_debut_str}_{date_fin_str}{extension}"
        chemin_sortie = os.path.join(dossier, nouveau_fichier)

        # Sauvegarder le split avec le même encodage que celui utilisé pour lire le fichier
        split_df.to_csv(chemin_sortie, sep=';', index=False, encoding=encoding_used)
        print_notification(f"Split {num_split} sauvegardé dans {chemin_sortie} ({len(split_df)} lignes)")

    print_notification("Division du fichier par sessions terminée avec succès")


if __name__ == "__main__":
    # Vérifier que le fichier existe
    if not os.path.isfile(file_path):
        print_notification(f"Erreur : Le fichier {file_path} n'existe pas")
    else:
        diviser_fichier_par_sessions(file_path)