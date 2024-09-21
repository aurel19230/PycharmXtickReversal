import calendar
from datetime import datetime
import seaborn as sns
from termcolor import colored

import matplotlib.pyplot as plt

import pandas as pd
import time
from colorama import Fore, Style, init
import time
def timestamp_to_date_utc(timestamp):
    date_format = "%Y-%m-%d %H:%M:%S"
    if isinstance(timestamp, pd.Series):
        return timestamp.apply(lambda x: time.strftime(date_format, time.gmtime(x)))
    else:
        return time.strftime(date_format, time.gmtime(timestamp))

def date_to_timestamp_utc(year, month, day, hour, minute, second): # from c++ encoding
    timeinfo = (year, month, day, hour, minute, second)
    return calendar.timegm(timeinfo)



# Fonction pour convertir une chaîne de caractères en timestamp Unix
def convert_to_unix_timestamp(date_string):
    date_format = "%Y-%m-%d %H:%M:%S"  # Modifier le format de date
    dt = datetime.strptime(date_string, date_format)
    return int(dt.timestamp())



# Fonction pour convertir un timestamp Unix en chaîne de caractères de date
def convert_from_unix_timestamp(unix_timestamp):
    date_format = "%Y-%m-%d %H:%M:%S"  # Modifier le format de date
    dt = datetime.fromtimestamp(unix_timestamp)
    return dt.strftime(date_format)

def plot_feature_histograms_by_class(data,className, column_settings, figsize=(32, 24)):
    columns = list(column_settings.keys())
    n_columns = len(columns)
    ncols = 6
    nrows = (n_columns + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, column in enumerate(columns):
        # Compter le nombre d'observations pour chaque classe
        class_0_count = data[data[className] == 0].shape[0]
        class_1_count = data[data[className] == 1].shape[0]

        # Tracer l'histogramme pour la classe 0 en rouge avec densité
        sns.histplot(data=data[data[className] == 0], x=column, color='red', label='Classe 0', bins=300, kde=True, stat='density', common_norm=True, ax=axes[i], alpha=0.5)

        # Tracer l'histogramme pour la classe 1 en bleu avec densité
        sns.histplot(data=data[data[className] == 1], x=column, color='blue', label='Classe 1', bins=300, kde=True, stat='density', common_norm=True, ax=axes[i], alpha=0.5)

        # Ajouter un titre
        axes[i].set_title(f'{column}')

        # Supprimer le label de l'axe des abscisses
        axes[i].set_xlabel('')

        # Ajouter le label de l'axe des ordonnées
        axes[i].set_ylabel('')

        # Ajouter une légende
        axes[i].legend()

        # Ajouter le décompte des classes au graphique
        axes[i].text(0.95, 0.95, f'Classe 0: {class_0_count}\nClasse 1: {class_1_count}', transform=axes[i].transAxes, fontsize=8, horizontalalignment='right', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()



import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def split_sessions(df, test_size=0.2, min_train_sessions=2, min_test_sessions=2):
    session_col = "SessionStartEnd"

    # Identifier les sessions complètes
    session_starts = df[df[session_col] == 10].index
    session_ends = df[df[session_col] == 20].index

    print(f"Nombre de débuts de session détectés : {len(session_starts)}")
    print(f"Nombre de fins de session détectées : {len(session_ends)}")

    sessions = []
    excluded_rows = 0
    orphan_starts = 0
    orphan_ends = 0

    for start, end in zip(session_starts, session_ends):
        if start < end:  # Vérifier que la session est complète
            session = df.loc[start:end]
            if session.iloc[0][session_col] == 10 and session.iloc[-1][session_col] == 20:
                sessions.append(session)
                #print(f"Session extraite de l'index {start} à {end}, comprenant {len(session)} lignes. "
                      #f"Valeur de {session_col} au début: {session.iloc[0][session_col]}, "
                      #f"à la fin: {session.iloc[-1][session_col]}")
            else:
                excluded_rows += len(session)
                print(colored(f"Session invalide ignorée de l'index {start} à {end}. "
                              f"Valeur de {session_col} au début: {session.iloc[0][session_col]}, "
                              f"à la fin: {session.iloc[-1][session_col]}", "red"))
        else:
            excluded_rows += end - start + 1
            orphan_starts += 1
            orphan_ends += 1

    # Gérer les débuts ou fins de session orphelins
    if len(session_starts) > len(session_ends):
        orphan_starts += len(session_starts) - len(session_ends)
    elif len(session_ends) > len(session_starts):
        orphan_ends += len(session_ends) - len(session_starts)

    total_sessions = len(sessions)
    print(f"Nombre de sessions complètes et valides extraites : {total_sessions}")

    # Compter les lignes avant la première session et après la dernière
    rows_before_first = session_starts[0] - df.index[0] if len(session_starts) > 0 else 0
    rows_after_last = df.index[-1] - session_ends[-1] if len(session_ends) > 0 else 0
    excluded_rows += rows_before_first + rows_after_last

    #print(f"Lignes avant la première session : {rows_before_first}")
    #print(f"Lignes après la dernière session : {rows_after_last}")
    #print(f"Débuts de session orphelins : {orphan_starts}")
    #print(f"Fins de session orphelines : {orphan_ends}")
    #print(f"Nombre total de lignes exclues : {excluded_rows}")

    # Vérifier s'il y a des sessions orphelines et lever une erreur si c'est le cas
    if orphan_starts > 0 or orphan_ends > 0:
        error_message = f"Erreur : Sessions orphelines détectées. Débuts orphelins : {orphan_starts}, Fins orphelines : {orphan_ends}"
        raise ValueError(error_message)

    if not sessions:
        raise ValueError("Aucune session complète et valide détectée dans les données.")

    # Calculer le nombre de sessions de test nécessaires
    test_sessions_count = max(min_test_sessions, int(total_sessions * test_size))
    min_required_sessions = min_train_sessions + test_sessions_count

    if total_sessions < min_required_sessions:
        raise ValueError(f"Nombre insuffisant de sessions. "
                         f"Trouvé : {total_sessions}, "
                         f"Minimum requis : {min_required_sessions} "
                         f"({min_train_sessions} pour l'entraînement, {test_sessions_count} pour le test). "
                         f"Veuillez fournir plus de données.")

    # Ajuster le nombre de sessions de test si nécessaire
    if total_sessions - test_sessions_count < min_train_sessions:
        test_sessions_count = total_sessions - min_train_sessions

    # Diviser les sessions en train et test
    train_sessions = sessions[:-test_sessions_count]
    test_sessions = sessions[-test_sessions_count:]

    # Vérification finale
    if len(train_sessions) < min_train_sessions:
        raise ValueError(f"Nombre insuffisant de sessions d'entraînement. "
                         f"Trouvé : {len(train_sessions)}, Minimum requis : {min_train_sessions}")
    if len(test_sessions) < min_test_sessions:
        raise ValueError(f"Nombre insuffisant de sessions de test. "
                         f"Trouvé : {len(test_sessions)}, Minimum requis : {min_test_sessions}")

    # Combiner les sessions en DataFrames
    train_df = pd.concat(train_sessions)
    test_df = pd.concat(test_sessions)

    # Afficher les informations
    print(f"Nombre total de lignes dans le DataFrame original : {len(df)}")
    print(f"Nombre de lignes dans le train : {len(train_df)}")
    print(f"Nombre de lignes dans le test : {len(test_df)}")
    print(f"Nombre de sessions dans le train : {len(train_sessions)}")
    print(f"Nombre de sessions dans le test : {len(test_sessions)}")

    total_included_rows = len(train_df) + len(test_df)
    print(f"Nombre total de lignes incluses : {total_included_rows}")
    print(f"Différence avec le total : {len(df) - total_included_rows}")

    return train_df, test_df

"""
# Utilisation de la fonction
train_df, test_df = split_sessions(data, test_size=0.2)

# Séparer les features et la target pour le modèle
X_train = train_df.drop(columns=['target'])
y_train = train_df['target']

X_test = test_df.drop(columns=['target'])
y_test = test_df['target']
"""


def print_notification(message, color=None):
    """
    Affiche un message avec un horodatage. Optionnellement, le message peut être affiché en couleur.

    Args:
    - message (str): Le message à afficher.
    - color (str, optionnel): La couleur du texte ('red', 'green', 'yellow', 'blue', etc.).
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Définir les couleurs selon le choix de l'utilisateur pour le message uniquement
    if color == 'red':
        color_code = Fore.RED
    elif color == 'green':
        color_code = Fore.GREEN
    elif color == 'yellow':
        color_code = Fore.YELLOW
    elif color == 'blue':
        color_code = Fore.BLUE
    else:
        color_code = ''  # Pas de couleur

    # Afficher le message avec le timestamp non coloré et le message coloré si nécessaire
    print(f"\n[{timestamp}] {color_code}{message}{Style.RESET_ALL}")

def load_data(file_path: str) -> pd.DataFrame:
    print_notification("Début du chargement des données")
    df = pd.read_csv(file_path, sep=';', encoding='iso-8859-1')
    print_notification("Données chargées avec succès")
    return df