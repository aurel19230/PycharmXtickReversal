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
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
def plot_calibration_curve(y_true, y_pred_proba, n_bins=200, strategy='uniform', optimal_threshold=None,
                           show_histogram=True):
    """
    Plots an improved calibration curve and distribution of predicted probabilities with TP, FP, TN, FN counts per bin.

    Parameters:
    - y_true: array-like, True binary labels (0 or 1).
    - y_pred_proba: array-like, Predicted probabilities for the positive class.
    - n_bins: int, Number of bins for the histogram and calibration curve (default=200).
    - strategy: str, Strategy for calibration curve ('uniform' or 'quantile').
    - optimal_threshold: float, The optimal decision threshold.
    - show_histogram: bool, Whether to display the histogram.

    Returns:
    - None
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    if optimal_threshold is None:
        raise ValueError("The 'optimal_threshold' parameter must be provided.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10, strategy=strategy)
    brier_score = brier_score_loss(y_true, y_pred_proba)

    ax1.plot(prob_pred, prob_true, marker='o', linewidth=1, color='blue', label='Calibration curve')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax1.set_title(f'Calibration Curve (Reliability Diagram)\nBrier Score: {brier_score:.4f}', fontsize=14)
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True)

    if show_histogram:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_width = bins[1] - bins[0]
        bin_centers = bins[:-1] + bin_width / 2

        # Initialize counts
        tp_counts = np.zeros(n_bins)
        fp_counts = np.zeros(n_bins)
        tn_counts = np.zeros(n_bins)
        fn_counts = np.zeros(n_bins)

        # Assign each sample to a bin
        bin_indices = np.digitize(y_pred_proba, bins) - 1  # Subtract 1 to get zero-based index
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Ensure indices are within valid range

        # Compute TP, FP, TN, FN counts per bin
        for i in range(len(y_true)):
            bin_idx = bin_indices[i]
            actual = y_true[i]
            predicted_proba = y_pred_proba[i]
            predicted_label = int(predicted_proba >= optimal_threshold)

            if actual == 1 and predicted_label == 1:
                tp_counts[bin_idx] += 1
            elif actual == 0 and predicted_label == 1:
                fp_counts[bin_idx] += 1
            elif actual == 0 and predicted_label == 0:
                tn_counts[bin_idx] += 1
            elif actual == 1 and predicted_label == 0:
                fn_counts[bin_idx] += 1

        # Stack the counts for plotting
        bottom_tn = np.zeros(n_bins)
        bottom_fp = tn_counts
        bottom_fn = tn_counts + fp_counts
        bottom_tp = tn_counts + fp_counts + fn_counts

        # Plotting
        ax2.bar(bin_centers, tn_counts, width=bin_width, label='TN', alpha=0.7, color='green')
        ax2.bar(bin_centers, fp_counts, width=bin_width, bottom=bottom_tn, label='FP', alpha=0.7, color='red')
        ax2.bar(bin_centers, fn_counts, width=bin_width, bottom=bottom_fp, label='FN', alpha=0.7, color='orange')
        ax2.bar(bin_centers, tp_counts, width=bin_width, bottom=bottom_fn, label='TP', alpha=0.7, color='blue')

        ax2.axvline(x=optimal_threshold, color='black', linestyle='--', label=f'Threshold ({optimal_threshold:.2f})')
        ax2.set_title('Distribution of Predicted Probabilities with TP, FP, TN, FN per Bin', fontsize=14)
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.legend(loc='upper center', fontsize=10, ncol=5)
        ax2.grid(True)

        # Add summary annotations
        total_tp = int(np.sum(tp_counts))
        total_fp = int(np.sum(fp_counts))
        total_tn = int(np.sum(tn_counts))
        total_fn = int(np.sum(fn_counts))
        total_samples = len(y_true)
        accuracy = (total_tp + total_tn) / total_samples
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        annotation_text = (f'Total Samples: {total_samples}\n'
                           f'TP: {total_tp}\nFP: {total_fp}\nTN: {total_tn}\nFN: {total_fn}\n'
                           f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1_score:.4f}')

        ax2.text(0.02, 0.98, annotation_text,
                 transform=ax2.transAxes, va='top', ha='left', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Adjust y-axis scale if necessary
        max_count = max(np.max(tn_counts + fp_counts + fn_counts + tp_counts), 1)
       # if max_count > 1000:
        #    ax2.set_yscale('log')
         #   ax2.set_ylabel('Count (log scale)', fontsize=12)

    plt.tight_layout()
    plt.savefig('improved_calibration_and_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()