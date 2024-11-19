import calendar
import seaborn as sns
from termcolor import colored
import cupy as cp
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, \
    average_precision_score, matthews_corrcoef,precision_recall_curve, precision_score
import pandas as pd
from colorama import Fore, Style, init
import time
from enum import Enum
import xgboost as xgb
from typing import Tuple
import shap
import seaborn as sns
import matplotlib.ticker as ticker
from PIL import Image
import sys
import csv
import logging
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import RFECV

import math
from functools import partial
from sklearn.model_selection import KFold, TimeSeriesSplit


# Définition des sections personnalisées
CUSTOM_SECTIONS = [
    {"name": "preAsian", "start": 0, "end": 240, "index": 0},
    {"name": "asianAndPreEurop", "start": 240, "end": 540, "index": 1},
    {"name": "europMorning", "start": 540, "end": 810, "index": 2},
    {"name": "europLunch", "start": 810, "end": 870, "index": 3},
    {"name": "preUS", "start": 870, "end": 930, "index": 4},
    {"name": "usMoning", "start": 930, "end": 1065, "index": 5},
    {"name": "usAfternoon", "start": 1065, "end": 1200, "index": 6},
    {"name": "usEvening", "start": 1200, "end": 1290, "index": 7},
    {"name": "usEnd", "start": 1290, "end": 1335, "index": 8},
    {"name": "closing", "start": 1335, "end": 1380, "index": 9},
]

class optuna_options(Enum):
    USE_OPTIMA_ROCAUC = 1
    USE_OPTIMA_AUCPR = 2
    USE_OPTIMA_F1 = 4
    USE_OPTIMA_PRECISION = 5
    USE_OPTIMA_RECALL = 6
    USE_OPTIMA_MCC = 7
    USE_OPTIMA_YOUDEN_J = 8
    USE_OPTIMA_SHARPE_RATIO = 9
    USE_OPTIMA_CUSTOM_METRIC_PROFITBASED = 10
    USE_OPTIMA_CUSTOM_METRIC_TP_FP = 11





def timestamp_to_date_utc(timestamp):
    date_format = "%Y-%m-%d %H:%M:%S"
    if isinstance(timestamp, pd.Series):
        return timestamp.apply(lambda x: time.strftime(date_format, time.gmtime(x)))
    else:
        return time.strftime(date_format, time.gmtime(timestamp))

import torch

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
    print("\n")
    return train_df,len(train_sessions), test_df,{len(test_sessions)}

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
    print_notification(f"Début du chargement des données de: \n "
                       f"{file_path}")
    df = pd.read_csv(file_path, sep=';', encoding='iso-8859-1')
    print(f"Colonnes chargées: {df.columns.tolist()}")
    print(f"Premières lignes:\n{df.head()}")
    print_notification("Données chargées avec succès")
    return df
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def check_gpu_availability():
    torch_available = torch.cuda.is_available()
    cupy_available = cp.cuda.is_available()

    if not (torch_available and cupy_available):
        print("Erreur : GPU n'est pas disponible pour PyTorch et/ou CuPy. Le programme va s'arrêter.")
        if not torch_available:
            print("PyTorch ne détecte pas de GPU.")
        if not cupy_available:
            print("CuPy ne détecte pas de GPU.")
        exit(1)

    print("GPU est disponible. Utilisation de CUDA pour les calculs.")
    print(f"GPU détecté par PyTorch : {torch.cuda.get_device_name(0)}")
    print(f"GPU détecté par CuPy : {cp.cuda.runtime.getDeviceProperties(cp.cuda.Device())['name'].decode()}")

    # Vérification de la version CUDA
    torch_cuda_version = torch.version.cuda
    cupy_cuda_version = cp.cuda.runtime.runtimeGetVersion()

    print(f"Version CUDA pour PyTorch : {torch_cuda_version}")
    print(f"Version CUDA pour CuPy : {cupy_cuda_version}")

    if torch_cuda_version != cupy_cuda_version:
        print("Attention : Les versions CUDA pour PyTorch et CuPy sont différentes.")
        print("Cela pourrait causer des problèmes de compatibilité.")

    # Affichage de la mémoire GPU disponible
    torch_memory = torch.cuda.get_device_properties(0).total_memory
    cupy_memory = cp.cuda.runtime.memGetInfo()[1]

    print(f"Mémoire GPU totale (PyTorch) : {torch_memory / 1e9:.2f} GB")
    print(f"Mémoire GPU totale (CuPy) : {cupy_memory / 1e9:.2f} GB")
def plot_calibrationCurve_distrib(y_true, y_pred_proba, n_bins=200, strategy='uniform',
                                  optimal_threshold=None, show_histogram=True, user_input=None, num_sessions=25,results_directory=None):
    y_true = np.array(y_true)  # This is fine if y_true is already a NumPy array

    # If y_pred_proba is a CuPy array, you need to explicitly convert it to NumPy
    if isinstance(y_pred_proba, cp.ndarray):  # Check if it's a CuPy array
        y_pred_proba = y_pred_proba.get()  # Convert CuPy array to NumPy array

    if optimal_threshold is None:
        raise ValueError("The 'optimal_threshold' parameter must be provided.")

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10, strategy=strategy)
    brier_score = brier_score_loss(y_true, y_pred_proba)

    axes[0].plot(prob_pred, prob_true, marker='o', linewidth=1, color='blue', label='Calibration curve')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    axes[0].set_title(f'Calibration Curve (Reliability Diagram)\nBrier Score: {brier_score:.4f}', fontsize=12)
    axes[0].set_xlabel('Mean Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Fraction of Positives', fontsize=12)
    axes[0].legend(loc='upper left', fontsize=10)
    axes[0].grid(True)

    if show_histogram:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_width = bins[1] - bins[0]
        bin_centers = bins[:-1] + bin_width / 2

        tp_counts = np.zeros(n_bins)
        fp_counts = np.zeros(n_bins)
        tn_counts = np.zeros(n_bins)
        fn_counts = np.zeros(n_bins)

        bin_indices = np.digitize(y_pred_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

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

        bottom_tn = np.zeros(n_bins)
        bottom_fp = tn_counts
        bottom_fn = tn_counts + fp_counts
        bottom_tp = tn_counts + fp_counts + fn_counts

        axes[1].bar(bin_centers, tn_counts, width=bin_width, label='TN', alpha=0.7, color='green')
        axes[1].bar(bin_centers, fp_counts, width=bin_width, bottom=bottom_tn, label='FP', alpha=0.7, color='red')
        axes[1].bar(bin_centers, fn_counts, width=bin_width, bottom=bottom_fp, label='FN', alpha=0.7, color='orange')
        axes[1].bar(bin_centers, tp_counts, width=bin_width, bottom=bottom_fn, label='TP', alpha=0.7, color='blue')

        axes[1].axvline(x=optimal_threshold, color='black', linestyle='--',
                        label=f'Threshold ({optimal_threshold:.2f})')
        axes[1].set_title('Repartition des TP, FP, TN, FN per Bin Test Set', fontsize=12)
        axes[1].set_xlabel('Proportion de prédictions négatives (fonction du choix du seuil)', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].legend(loc='upper center', fontsize=10, ncol=5)
        axes[1].grid(True)

        total_tp = int(np.sum(tp_counts))
        total_fp = int(np.sum(fp_counts))
        total_tn = int(np.sum(tn_counts))
        total_fn = int(np.sum(fn_counts))
        total_samples = len(y_true)
        accuracy = (total_tp + total_tn) / total_samples
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        total_trades=total_tp+total_fp
        Winrate=total_tp / total_trades * 100 if total_trades > 0 else 0
        annotation_text = (f'Total Samples: {total_samples}\n'
                           f'TP: {total_tp}\nFP: {total_fp}\nTN: {total_tn}\nFN: {total_fn}\nWinrate: {Winrate:.2f}%\n'
                           f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1_score:.4f}\n'
                           f'Nombre de Session: {num_sessions}')

        axes[1].text(0.02, 0.98, annotation_text,
                     transform=axes[1].transAxes, va='top', ha='left', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, 'calibration_and_distribution.png'), dpi=300, bbox_inches='tight')

    if user_input.lower() == 'd':
        plt.show()
    plt.close()


from datetime import datetime, timedelta


def plot_fp_tp_rates(X_test, y_true, y_pred_proba, feature_deltaTime_name, optimal_threshold, user_input=None, index_size=5,results_directory=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(35, 20))

    def index_to_time(index):
        start_time = datetime.strptime("22:00", "%H:%M")
        minutes = index * index_size
        time = start_time + timedelta(minutes=minutes)
        if time.hour < 22:
            time += timedelta(days=1)
        return time.strftime("%H:%M")


    def plot_rates(ax, n_bins_feature):
        # Convertir les tableaux CuPy en NumPy si nécessaire
        feature_values_np = feature_values.get() if isinstance(feature_values, cp.ndarray) else feature_values
        y_true_np = y_true.get() if isinstance(y_true, cp.ndarray) else y_true
        y_pred_proba_np = y_pred_proba.get() if isinstance(y_pred_proba, cp.ndarray) else y_pred_proba

        bins = pd.cut(feature_values_np, bins=n_bins_feature)
        rates = pd.DataFrame({
            'feature': feature_values_np,
            'y_true': y_true_np,
            'y_pred': y_pred_proba_np >= optimal_threshold
        }).groupby(bins, observed=True).apply(lambda x: pd.Series({
            'FP_rate': ((x['y_pred'] == 1) & (x['y_true'] == 0)).sum() / len(x) if len(x) > 0 else 0,
            'TP_rate': ((x['y_pred'] == 1) & (x['y_true'] == 1)).sum() / len(x) if len(x) > 0 else 0
        }))

        x = np.arange(len(rates))
        width = 0.35

        ax.bar(x - width / 2, rates['FP_rate'], width, label='Taux de Faux Positifs', color='red', alpha=0.7)
        ax.bar(x + width / 2, rates['TP_rate'], width, label='Taux de Vrais Positifs', color='green', alpha=0.7)

        ax.set_ylabel('Taux', fontsize=14)
        ax.set_xlabel(feature_deltaTime_name, fontsize=14)
        ax.set_title(f'Taux de FP et TP par {feature_deltaTime_name} (bins={n_bins_feature})', fontsize=14)
        ax.set_xticks(x)

        # Conversion des index en format heure
        time_labels = [index_to_time(int(b.left)) for b in rates.index]
        ax.set_xticklabels(time_labels, rotation=45, ha='right')

        ax.legend(fontsize=12)
        ax.grid(True)

    feature_values = X_test[feature_deltaTime_name].values

    # Graphique avec 25 bins
    plot_rates(ax1, 25)

    # Graphique avec 100 bins
    plot_rates(ax2, 100)

    plt.tight_layout()

    # Enregistrer le graphique
    plt.savefig(os.path.join(results_directory, 'fp_tp_rates_by_feature_dual_time.png'), dpi=300, bbox_inches='tight')


    # Afficher le graphique
    if user_input and user_input.lower() == 'd':
        plt.show()

    # Fermer la figure après l'affichage
    plt.close()

    feature_values = X_test[feature_deltaTime_name].values

    # Graphique avec 25 bins
    plot_rates(ax1, 25)

    # Graphique avec 100 bins
    plot_rates(ax2, 100)

    plt.tight_layout()

    # Afficher le graphique
    if user_input and user_input.lower() == 'd':
        plt.show()

    # Fermer la figure après l'affichage
    plt.close()


import numba as nb
@nb.njit
def calculate_session_duration(session_start_end, delta_timestamp):
    total_minutes = 0
    residual_minutes = 0
    session_start = -1
    complete_sessions = 0
    in_session = False

    # Gérer le cas où le fichier ne commence pas par un 10
    if session_start_end[0] != 10:
        first_20_index = np.where(session_start_end == 20)[0][0]
        residual_minutes += delta_timestamp[first_20_index] - delta_timestamp[0]

    for i in range(len(session_start_end)):
        if session_start_end[i] == 10:
            session_start = i
            in_session = True
        elif session_start_end[i] == 20:
            if in_session:
                # Session complète
                total_minutes += delta_timestamp[i] - delta_timestamp[session_start]
                complete_sessions += 1
                in_session = False
            else:
                # 20 sans 10 précédent, ne devrait pas arriver mais gérons le cas
                residual_minutes += delta_timestamp[i] - delta_timestamp[session_start if session_start != -1 else 0]

    # Gérer le cas où le fichier se termine au milieu d'une session
    if in_session:
        residual_minutes += delta_timestamp[-1] - delta_timestamp[session_start]
    elif session_start_end[-1] != 20:
        # Si la dernière valeur n'est pas 20 et qu'on n'est pas dans une session,
        # ajoutons le temps depuis le dernier 20 jusqu'à la fin
        last_20_index = np.where(session_start_end == 20)[0][-1]
        residual_minutes += delta_timestamp[-1] - delta_timestamp[last_20_index]

    return complete_sessions, total_minutes, residual_minutes

def calculate_and_display_sessions(df):
    session_start_end = df['SessionStartEnd'].astype(np.int32).values
    delta_timestamp = df['deltaTimestampOpening'].astype(np.float64).values
    ts_startSection = df['timeStampOpening'].head(1).values[0]
    ts_endSection = df['timeStampOpening'].tail(1).values[0]

    date_startSection = timestamp_to_date_utc(ts_startSection)
    date_endSection = timestamp_to_date_utc(ts_endSection)


    complete_sessions, total_minutes, residual_minutes = calculate_session_duration(session_start_end, delta_timestamp)

    session_duration_hours = 23
    session_duration_minutes = session_duration_hours * 60

    residual_sessions = residual_minutes / session_duration_minutes
    total_sessions = complete_sessions + residual_sessions

    #print(f"Nombre de sessions complètes : {complete_sessions}")
    #print(f"Minutes résiduelles : {residual_minutes:.2f}")
    #print(f"Équivalent en sessions des minutes résiduelles : {residual_sessions:.2f}")
    #print(f"Nombre total de sessions (complètes + résiduelles) : {total_sessions:.2f}")

    return total_sessions,date_startSection,date_endSection



def sigmoidCustom(x):
# Supposons que x est déjà un tableau CuPy
  return 1 / (1 + cp.exp(-x))
def sigmoidCustom_cpu(x):
    """Custom sigmoid function."""
    return 1 / (1 + np.exp(-x))
def compute_confusion_matrix_cupy(y_true_gpu, y_pred_gpu):
    tp = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 1))
    fp = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 1))
    tn = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 0))
    fn = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 0))
    return tn.item(), fp.item(), fn.item(), tp.item()

def weighted_logistic_gradient_cpu(predt: np.ndarray, dtrain: xgb.DMatrix, w_p: float, w_n: float) -> np.ndarray:
    """Calcule le gradient pour la perte logistique pondérée (CPU)."""
    y = dtrain.get_label()
    predt = 1.0 / (1.0 + np.exp(-predt))  # Fonction sigmoïde
    weights = np.where(y == 1, w_p, w_n)
    grad = weights * (predt - y)
    return grad

def weighted_logistic_hessian_cpu(predt: np.ndarray, dtrain: xgb.DMatrix, w_p: float, w_n: float) -> np.ndarray:
    """Calcule le hessien pour la perte logistique pondérée (CPU)."""
    y = dtrain.get_label()
    predt = 1.0 / (1.0 + np.exp(-predt))  # Fonction sigmoïde
    weights = np.where(y == 1, w_p, w_n)
    hess = weights * predt * (1.0 - predt)
    return hess

def calculate_weighted_adjusted_score_custom(scores, weight_split, nb_split_weight, std_penalty_factor=1.0):
    """
    Calcule le score ajusté pondéré avec gestion spéciale du cas nb_split_weight=0

    Args:
        scores: Liste des scores PNL pour chaque split
        weight_split: Poids à appliquer aux splits les plus anciens
        nb_split_weight: Nombre de splits auxquels appliquer le poids spécifique
        std_penalty_factor: Facteur de pénalité pour l'écart-type
    """
    if nb_split_weight > len(scores):
        raise ValueError("nb_split_weight ne peut pas être supérieur au nombre de scores.")

    scores = np.array(scores)

    if nb_split_weight == 0:
        # Cas sans pondération : calcul traditionnel
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)  # ddof=1 pour écart-type non biaisé
        return mean - std_penalty_factor * std,mean,std

    # Cas avec pondération
    weights = np.array([weight_split] * nb_split_weight + [1] * (len(scores) - nb_split_weight))
    weights = weights / np.sum(weights)

    weighted_mean = np.sum(scores * weights)
    squared_diff = (scores - weighted_mean) ** 2
    weighted_variance = np.sum(weights * squared_diff) / (1 - np.sum(weights ** 2))
    weighted_std = np.sqrt(weighted_variance)

    return weighted_mean - std_penalty_factor * weighted_std,weighted_mean,weighted_std





# Fonctions GPU mises à jour

def weighted_logistic_gradient_Cupygpu(predt, dtrain, w_p, w_n):
    predt_gpu = cp.asarray(predt)
    y_gpu = cp.asarray(dtrain.get_label())

    predt_sigmoid = sigmoidCustom(predt_gpu)
    grad = predt_sigmoid - y_gpu
    # Appliquer les poids après le calcul initial du gradient
    weights = cp.where(y_gpu == 1, w_p, w_n)
    grad *= weights

    return grad  # Retourner directement le tableau CuPy

def weighted_logistic_hessian_Cupygpu(predt, dtrain, w_p, w_n):
    predt_gpu = cp.asarray(predt)
    y_gpu = cp.asarray(dtrain.get_label())

    predt_sigmoid = sigmoidCustom(predt_gpu)
    hess = predt_sigmoid * (1 - predt_sigmoid)
    # Appliquer les poids après le calcul initial de la hessienne
    weights = cp.where(y_gpu == 1, w_p, w_n)
    hess *= weights

    return hess  # Retourner directement le tableau CuPy

def create_weighted_logistic_obj_gpu(w_p: float, w_n: float):
    def weighted_logistic_obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        grad = weighted_logistic_gradient_Cupygpu(predt, dtrain, w_p, w_n)
        hess = weighted_logistic_hessian_Cupygpu(predt, dtrain, w_p, w_n)
        return grad, hess
    return weighted_logistic_obj

# Fonction pour vérifier la disponibilité du GPU

def calculate_profitBased_gpu(y_true, y_pred_threshold, metric_dict):
    y_true_gpu = cp.array(y_true)
    y_pred_gpu = cp.array(y_pred_threshold)
    tp = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 1))
    fp = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 1))
    fn = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 0))
    profit_per_tp = metric_dict.get('profit_per_tp', 1.0)
    loss_per_fp = metric_dict.get('loss_per_fp', -1.1)
    penalty_per_fn = metric_dict.get('penalty_per_fn', -0.1)  # Include FN penalty
    total_profit = (tp * profit_per_tp) + (fp * loss_per_fp) + (fn * penalty_per_fn)
    total_trades_val = tp + fp  # Typically, total executed trades

    # Utiliser une condition pour éviter la division par zéro
    """"
    if total_trades_val > 0:
        normalized_profit = total_profit / total_trades_val
    else:
        normalized_profit = total_profit  # Reflect penalties from FNs when no trades are made



    return float(normalized_profit)  # Assurez-vous que c'est un float Python
    """
    return float(total_profit), int(tp), int(fp)

def custom_metric_Profit(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict, normalize: bool = False) -> Tuple[
    str, float]:
    """
    Fonction commune pour calculer les métriques de profit (normalisée ou non)

    Args:
        predt: prédictions brutes
        dtrain: données d'entraînement
        metric_dict: dictionnaire des paramètres de métrique
        normalize: si True, normalise le profit par le nombre de trades
    """
    y_true = dtrain.get_label()
    CHECK_THRESHOLD = 0.55555555

    threshold = metric_dict.get('threshold', CHECK_THRESHOLD)

    if 'threshold' not in metric_dict:
        logging.warning("Aucun seuil personnalisé n'a été défini. Utilisation du seuil par défaut de 0.55555555.")

    predt = cp.asarray(predt)
    predt = sigmoidCustom(predt)
    predt = cp.clip(predt, 0.0, 1.0)

    mean_pred = cp.mean(predt).item()
    std_pred = cp.std(predt).item()
    min_val = cp.min(predt).item()
    max_val = cp.max(predt).item()

    if min_val < 0 or max_val > 1:
        logging.warning(f"Les prédictions sont hors de l'intervalle [0, 1]: [{min_val:.4f}, {max_val:.4f}]")
        exit(12)

    y_pred_threshold = (predt > threshold).astype(int)

    # Calcul du profit et des TP/FP
    total_profit, tp, fp = calculate_profitBased_gpu(y_true, y_pred_threshold, metric_dict)

    if normalize:
        # Version normalisée
        total_trades_val = tp + fp
        if total_trades_val > 0:
            final_profit = total_profit / total_trades_val
        else:
            final_profit = 0.0
        metric_name = 'custom_metric_ProfitBased_norm'
    else:
        # Version non normalisée
        final_profit = total_profit
        metric_name = 'custom_metric_ProfitBased'

    return metric_name, float(final_profit)

# Création des deux fonctions spécifiques à partir de la fonction commune
def custom_metric_ProfitBased_gpu(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict) -> Tuple[str, float]:
    return custom_metric_Profit(predt, dtrain, metric_dict, normalize=False)



def calculate_profitBased_cpu(y_true, y_pred_threshold, metric_dict):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred_threshold)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    profit_per_tp = metric_dict.get('profit_per_tp', 1.0)
    loss_per_fp = metric_dict.get('loss_per_fp', -1.1)
    penalty_per_fn = metric_dict.get('penalty_per_fn', -0.1)
    total_profit = (tp * profit_per_tp) + (fp * loss_per_fp) + (fn * penalty_per_fn)
    return float(total_profit), int(tp), int(fp)

def custom_metric_Profit_cpu(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict, normalize: bool = False) -> Tuple[str, float]:
    y_true = dtrain.get_label()
    CHECK_THRESHOLD = 0.55555555
    threshold = metric_dict.get('threshold', CHECK_THRESHOLD)

    predt = np.array(predt)
    predt = sigmoidCustom_cpu(predt)
    predt = np.clip(predt, 0.0, 1.0)

    mean_pred = np.mean(predt)
    std_pred = np.std(predt)
    min_val = np.min(predt)
    max_val = np.max(predt)

    if min_val < 0 or max_val > 1:
        logging.warning(f"Les prédictions sont hors de l'intervalle [0, 1]: [{min_val:.4f}, {max_val:.4f}]")
        exit(12)

    y_pred_threshold = (predt > threshold).astype(int)

    total_profit, tp, fp = calculate_profitBased_cpu(y_true, y_pred_threshold, metric_dict)

    if normalize:
        total_trades_val = tp + fp
        if total_trades_val > 0:
            final_profit = total_profit / total_trades_val
        else:
            final_profit = 0.0
        metric_name = 'custom_metric_ProfitBased_norm'
    else:
        final_profit = total_profit
        metric_name = 'custom_metric_ProfitBased'

    return metric_name, float(final_profit)

def custom_metric_ProfitBased_cpu(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict) -> Tuple[str, float]:
    return custom_metric_Profit_cpu(predt, dtrain, metric_dict, normalize=False)

def create_weighted_logistic_obj_cpu(w_p: float, w_n: float):
    def weighted_logistic_obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        grad = weighted_logistic_gradient_cpu(predt, dtrain, w_p, w_n)
        hess = weighted_logistic_hessian_cpu(predt, dtrain, w_p, w_n)
        return grad, hess
    return weighted_logistic_obj

import traceback


def create_custom_importance_plot(shap_df, dataset_name, save_dir):
    """
    Crée un graphique personnalisé basé sur les valeurs SHAP du CSV
    avec code couleur rouge/bleu selon le signe
    """
    plt.figure(figsize=(12, 8))

    # Prendre les 20 premières features pour la lisibilité
    top_n = 20
    df_plot = shap_df.head(top_n).iloc[::-1]  # Inverse l'ordre pour l'affichage de bas en haut

    # Création des barres
    y_pos = np.arange(len(df_plot))
    bars = plt.barh(y_pos, df_plot['importance'].abs(),
                    color=[('red' if x < 0 else 'blue') for x in df_plot['importance']])

    # Personnalisation du graphique
    plt.yticks(y_pos, df_plot['feature'])
    plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
    plt.title(f'SHAP Feature Importance - {dataset_name}')

    # Ajout des valeurs sur les barres
    for i, bar in enumerate(bars):
        width = bar.get_width()
        value = df_plot['importance'].iloc[i]
        plt.text(width, bar.get_y() + bar.get_height() / 2,
                 f'{value:.2e}',
                 ha='left', va='center', fontsize=8)

    # Ajustements de la mise en page
    plt.tight_layout()

    # Sauvegarde
    plt.savefig(os.path.join(save_dir, f'shap_importance_{dataset_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def analyze_shap_values(model, X, y, dataset_name, create_dependence_plots=False, max_dependence_plots=3,
                        save_dir='./shap_dependencies_results/'):
    """
    Analyse les valeurs SHAP pour un ensemble de données et génère des visualisations.
    Version corrigée avec calcul approprié des importances SHAP et tri par valeur absolue.

    Parameters:
    -----------
    model : object
        Le modèle entraîné pour lequel calculer les valeurs SHAP.
    X : pandas.DataFrame
        Les features de l'ensemble de données.
    y : pandas.Series
        Les labels de l'ensemble de données.
    dataset_name : str
        Le nom de l'ensemble de données, utilisé pour nommer les fichiers de sortie.
    create_dependence_plots : bool, optional (default=False)
        Si True, crée des graphiques de dépendance pour les features les plus importantes.
    max_dependence_plots : int, optional (default=3)
        Le nombre maximum de graphiques de dépendance à créer si create_dependence_plots est True.
    save_dir : str, optional (default='./shap_dependencies_results/')
        Le répertoire où sauvegarder les graphiques générés et le fichier CSV.

    Returns:
    --------
    Tuple[numpy.ndarray, shap.Explanation]
        Les valeurs SHAP calculées et l'objet d'explication SHAP.
    """
    # Création du répertoire de sauvegarde
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Analyse SHAP pour {dataset_name} ===")
    print(f"Dimensions des données: {X.shape}")

    # Création de l'explainer SHAP
    try:
        explainer = shap.TreeExplainer(model)
        print("TreeExplainer créé avec succès")
    except Exception as e:
        print(f"Erreur lors de la création de l'explainer: {str(e)}")
        traceback.print_exc()
        raise

    # Calcul des valeurs SHAP
    try:
        shap_values_explanation = explainer(X)
        shap_values = explainer.shap_values(X)
        print("Valeurs SHAP calculées avec succès")

        # Affichage des dimensions des valeurs SHAP
        if isinstance(shap_values, list):
            print(f"Type de sortie: Liste de {len(shap_values)} tableaux")
            for i, sv in enumerate(shap_values):
                print(f"Shape des valeurs SHAP classe {i}: {sv.shape}")
        else:
            print(f"Shape des valeurs SHAP: {shap_values.shape}")

    except Exception as e:
        print(f"Erreur lors du calcul des valeurs SHAP: {str(e)}")
        traceback.print_exc()
        raise

    # Gestion des problèmes de classification binaire
    if isinstance(shap_values, list) and len(shap_values) == 2:
        print("Détection de classification binaire - utilisation des valeurs de la classe positive")
        shap_values = shap_values[1]

    # Création du graphique résumé
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {dataset_name}")
        plt.tight_layout()
        summary_path = os.path.join(save_dir, f'shap_importance_{dataset_name}.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graphique résumé SHAP créé et sauvegardé dans: {summary_path}")
    except Exception as e:
        print(f"Erreur lors de la création du graphique résumé: {str(e)}")
        traceback.print_exc()

    # Calcul des importances SHAP
    try:
        # Calcul des valeurs SHAP moyennes pour chaque feature
        feature_importance_signed = np.mean(shap_values, axis=0)

        # Création du DataFrame avec les importances
        shap_df = pd.DataFrame({
            'feature': X.columns,
            'importance': feature_importance_signed,
        })

        # Ajout de la colonne des valeurs absolues pour le tri
        shap_df['abs_importance'] = np.abs(shap_df['importance'])

        # Tri par importance absolue décroissante
        shap_df = shap_df.sort_values('abs_importance', ascending=False)

        # Calcul des métriques cumulatives
        total_abs_importance = shap_df['abs_importance'].sum()
        shap_df['importance_percentage'] = (shap_df['abs_importance'] / total_abs_importance) * 100
        shap_df['cumulative_importance_percentage'] = shap_df['importance_percentage'].cumsum()

        # Créer le graphique personnalisé au lieu du summary_plot
        create_custom_importance_plot(shap_df, dataset_name, save_dir)

        # Nettoyage avant sauvegarde
        shap_df = shap_df.drop('abs_importance', axis=1)

        # Sauvegarde du CSV
        csv_path = os.path.join(save_dir, f'shap_values_{dataset_name}.csv')
        shap_df.to_csv(csv_path, index=False, sep=';')
        print(f"\nValeurs SHAP sauvegardées dans: {csv_path}")

    except Exception as e:
        print(f"Erreur lors du calcul des importances: {str(e)}")
        traceback.print_exc()
        raise

    # Création des graphiques de dépendance si demandé
    if create_dependence_plots:
        print("\nCréation des graphiques de dépendance...")
        most_important_features = shap_df['feature'].head(max_dependence_plots)

        for feature in most_important_features:
            try:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature, shap_values, X, show=False)
                plt.title(f"SHAP Dependence Plot - {feature} - {dataset_name}")
                plt.tight_layout()
                dep_plot_path = os.path.join(save_dir, f'shap_dependence_{feature}_{dataset_name}.png')
                plt.savefig(dep_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Graphique de dépendance créé pour {feature}")
            except Exception as e:
                print(f"Erreur lors de la création du graphique de dépendance pour {feature}: {str(e)}")
                traceback.print_exc()

    print(f"\nAnalyse SHAP terminée - Tous les résultats sont sauvegardés dans: {save_dir}")

    return shap_values, shap_values_explanation




def compare_feature_importance(shap_values_train, shap_values_test, X_train, X_test,
                               save_dir='./shap_dependencies_results/', top_n=20):
    """
    Compare l'importance des features entre l'ensemble d'entraînement et de test.

    Parameters:
    -----------
    shap_values_train : numpy.ndarray
        Valeurs SHAP pour l'ensemble d'entraînement
    shap_values_test : numpy.ndarray
        Valeurs SHAP pour l'ensemble de test
    X_train : pandas.DataFrame
        DataFrame contenant les features d'entraînement
    X_test : pandas.DataFrame
        DataFrame contenant les features de test
    save_dir : str, optional
        Répertoire où sauvegarder le graphique (par défaut './shap_dependencies_results/')
    top_n : int, optional
        Nombre de top features à afficher dans le graphique (par défaut 20)

    Returns:
    --------
    pandas.DataFrame
        DataFrame contenant les importances des features et leurs différences
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    # Vérifier que les colonnes sont identiques dans X_train et X_test
    if not all(X_train.columns == X_test.columns):
        raise ValueError("Les colonnes de X_train et X_test doivent être identiques.")

    importance_train = np.abs(shap_values_train).mean(0)
    importance_test = np.abs(shap_values_test).mean(0)

    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance_Train': importance_train,
        'Importance_Test': importance_test
    })

    importance_df['Difference'] = importance_df['Importance_Train'] - importance_df['Importance_Test']
    importance_df = importance_df.sort_values('Difference', key=abs, ascending=False)

    # Sélectionner les top_n features pour la visualisation
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(12, 8))
    #sns.barplot(x='Difference', y='Feature', data=top_features, palette='coolwarm')
    sns.barplot(x='Difference', y='Feature', hue='Feature', data=top_features, palette='coolwarm', legend=False)
    plt.title(f"Top {top_n} Differences in Feature Importance (Train - Test)")
    plt.xlabel("Difference in Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance_difference.png'))
    plt.close()

    print(f"Graphique de différence d'importance des features sauvegardé dans {save_dir}")
    print("\nTop differences in feature importance:")
    print(importance_df.head(10))

    return importance_df




def compare_shap_distributions(shap_values_train=None, shap_explanation_train=None, shap_values_test=None, shap_explanation_test=None,
                             X_train=None, y_train_label=None, X_test=None, y_test_label=None,
                             top_n=10,
                             save_dir='./shap_dependencies_results/'):
    """
    Compare les distributions des valeurs SHAP entre l'ensemble d'entraînement et de test.

    Parameters:
    -----------
    shap_values_train : numpy.ndarray
        Valeurs SHAP pour l'ensemble d'entraînement
    shap_explanation_train : shap.Explanation
        Objet Explanation SHAP pour l'ensemble d'entraînement (pour les scatter plots)
    shap_values_test : numpy.ndarray
        Valeurs SHAP pour l'ensemble de test
    shap_explanation_test : shap.Explanation
        Objet Explanation SHAP pour l'ensemble de test (non utilisé actuellement)
    X_train : pandas.DataFrame
        DataFrame contenant les features d'entraînement
    y_train_label : numpy.ndarray
        Labels de l'ensemble d'entraînement
    X_test : pandas.DataFrame
        DataFrame contenant les features de test
    y_test_label : numpy.ndarray
        Labels de l'ensemble de test
    top_n : int, optional
        Nombre de top features à comparer (par défaut 10)
    save_dir : str, optional
        Répertoire où sauvegarder les graphiques (par défaut './shap_dependencies_results/')

    Returns:
    --------
    None
        Sauvegarde les graphiques dans le répertoire spécifié
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    # Vérifier que les colonnes sont identiques dans X_train et X_test
    if not all(X_train.columns == X_test.columns):
        raise ValueError("Les colonnes de X_train et X_test doivent être identiques.")

    # Calculer l'importance des features basée sur les valeurs SHAP absolues moyennes
    feature_importance = np.abs(shap_values_train).mean(0)
    top_features = X_train.columns[np.argsort(feature_importance)[-top_n:]]

    # Pour chaque feature importante
    for feature in top_features:
        # Créer une figure avec 3 sous-graphiques côte à côte
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        # 1. Distribution Train/Test
        sns.kdeplot(data=shap_values_train[:, X_train.columns.get_loc(feature)],
                    label='Train', fill=True, ax=ax1)
        sns.kdeplot(data=shap_values_test[:, X_test.columns.get_loc(feature)],
                    label='Test', fill=True, ax=ax1)
        ax1.set_title("Train/Test Distribution")
        ax1.set_xlabel("SHAP Value")
        ax1.set_ylabel("Density")
        ax1.legend()

        # 2. Distribution par classe
        feature_idx = X_train.columns.get_loc(feature)

        # Stats pour debug
        print(f"\nAnalyse pour {feature}:")
        print(
            f"Moyenne des valeurs SHAP pour classe 0: {shap_values_train[y_train_label == 0, feature_idx].mean():.6f}")
        print(
            f"Moyenne des valeurs SHAP pour classe 1: {shap_values_train[y_train_label == 1, feature_idx].mean():.6f}")
        print(f"Nombre d'exemples classe 0: {sum(y_train_label == 0)}")
        print(f"Nombre d'exemples classe 1: {sum(y_train_label == 1)}")

        # Histogramme à la place du KDE
        ax2.hist(shap_values_train[y_train_label == 0, feature_idx],
                 bins=50, alpha=0.5, color='blue', label='Classe 0', density=True)
        ax2.hist(shap_values_train[y_train_label == 1, feature_idx],
                 bins=50, alpha=0.5, color='red', label='Classe 1', density=True)

        ax2.set_title("Distribution par Classe")
        ax2.set_xlabel("SHAP Value")
        ax2.set_ylabel("Density")
        ax2.legend()

        # 3. Scatter plot SHAP
        feature_idx = list(X_train.columns).index(feature)
        shap.plots.scatter(shap_explanation_train[:, feature_idx], ax=ax3, show=False)
        ax3.set_title("SHAP Scatter Plot")

        # Ajout du titre global
        plt.suptitle(f"SHAP Analysis - {feature}", fontsize=16, y=1.05)
        plt.tight_layout()

        # Sauvegarde de la figure
        plt.savefig(os.path.join(save_dir, f'shap_combined_analysis_{feature}.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

    print(f"Les graphiques de distribution SHAP ont été sauvegardés dans {save_dir}")

    # Afficher un récapitulatif des features analysées
    print("Features analysées : " + " | ".join(f"{i}. {feature}" for i, feature in enumerate(top_features, 1)))
    # Dans la fonction compare_shap_distributions, avant la boucle des features
    # Calculer la somme des SHAP values pour chaque exemple
    shap_sums_train = shap_values_train.sum(axis=1)  # somme sur toutes les features

    print("\nAnalyse des sommes SHAP:")
    print(f"Moyenne des sommes SHAP pour classe 0: {shap_sums_train[y_train_label == 0].mean():.6f}")
    print(f"Moyenne des sommes SHAP pour classe 1: {shap_sums_train[y_train_label == 1].mean():.6f}")

    # Créer un nouveau plot pour la distribution des sommes
    plt.figure(figsize=(10, 6))
    plt.hist(shap_sums_train[y_train_label == 0], bins=50, alpha=0.5, color='blue', label='Classe 0', density=True)
    plt.hist(shap_sums_train[y_train_label == 1], bins=50, alpha=0.5, color='red', label='Classe 1', density=True)
    plt.title("Distribution des sommes des valeurs SHAP par classe")
    plt.xlabel("Somme des valeurs SHAP")
    plt.ylabel("Densité")
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'shap_sums_distribution_perClass.png'))

    # Créer une figure pour les top 20 features
    plt.figure(figsize=(16, 14))
    shap.summary_plot(shap_values_train, X_train, show=False, max_display=20)
    plt.title("Top 20 Features", fontsize=14, pad=20)
    plt.savefig(os.path.join(save_dir, 'shap_summary_plot_split1_top20.png'),
                bbox_inches='tight',
                dpi=300,
                pad_inches=1)
    plt.close()

    # Créer une figure séparée pour les features 21-40
    plt.figure(figsize=(16, 14))
    feature_importance = np.abs(shap_values_train).mean(0)
    sorted_idx = np.argsort(feature_importance)
    X_train_subset = X_train.iloc[:, sorted_idx[-40:-20]]
    shap_values_subset = shap_values_train[:, sorted_idx[-40:-20]]
    shap.summary_plot(shap_values_subset, X_train_subset, show=False, max_display=20)
    plt.title("Features 21-40", fontsize=14, pad=20)
    plt.savefig(os.path.join(save_dir, 'shap_summary_plot_split2_21_40.png'),
                bbox_inches='tight',
                dpi=300,
                pad_inches=1)
    plt.close()

    from PIL import Image

    def merge_images_horizontal(image1_path, image2_path, output_path, spacing=20):
        """
        Fusionner deux images horizontalement (côte à côte)

        Args:
            image1_path (str): Chemin vers la première image
            image2_path (str): Chemin vers la deuxième image
            output_path (str): Chemin pour sauvegarder l'image fusionnée
            spacing (int): Espace en pixels entre les images
        """
        # Ouvrir les images
        img1 = Image.open(image1_path)
        img2 = Image.open(image2_path)

        # Obtenir les dimensions
        width1, height1 = img1.size
        width2, height2 = img2.size

        # Utiliser la hauteur maximale
        max_height = max(height1, height2)

        # Redimensionner les images si nécessaire pour avoir la même hauteur
        if height1 != max_height:
            new_width1 = int(width1 * (max_height / height1))
            img1 = img1.resize((new_width1, max_height), Image.Resampling.LANCZOS)
            width1 = new_width1

        if height2 != max_height:
            new_width2 = int(width2 * (max_height / height2))
            img2 = img2.resize((new_width2, max_height), Image.Resampling.LANCZOS)
            width2 = new_width2

        # Créer une nouvelle image avec la largeur combinée (+ l'espacement)
        merged_image = Image.new('RGB', (width1 + spacing + width2, max_height), 'white')

        # Coller les images
        merged_image.paste(img1, (0, 0))
        merged_image.paste(img2, (width1 + spacing, 0))

        # Sauvegarder l'image fusionnée
        merged_image.save(output_path, 'PNG', dpi=(300, 300))

    # Utilisation
    image1_path = os.path.join(save_dir, 'shap_summary_plot_split1_top20.png')
    image2_path = os.path.join(save_dir, 'shap_summary_plot_split2_21_40.png')
    output_path = os.path.join(save_dir, 'shap_summary_plot_combined.png')

    merge_images_horizontal(image1_path, image2_path, output_path)
def compare_mean_shap_values(shap_values_train, shap_values_test, X_train, save_dir='./shap_dependencies_results/'):
    """
    Compare les valeurs SHAP moyennes entre l'ensemble d'entraînement et de test.

    Parameters:
    -----------
    shap_values_train : numpy.ndarray
        Valeurs SHAP pour l'ensemble d'entraînement
    shap_values_test : numpy.ndarray
        Valeurs SHAP pour l'ensemble de test
    X_train : pandas.DataFrame
        DataFrame contenant les features d'entraînement
    save_dir : str, optional
        Répertoire où sauvegarder le graphique (par défaut './shap_dependencies_results/')

    Returns:
    --------
    pandas.DataFrame
        DataFrame contenant la comparaison des valeurs SHAP moyennes
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    mean_shap_train = shap_values_train.mean(axis=0)
    mean_shap_test = shap_values_test.mean(axis=0)

    shap_comparison = pd.DataFrame({
        'Feature': X_train.columns,
        'Mean SHAP Train': mean_shap_train,
        'Mean SHAP Test': mean_shap_test,
        'Difference': mean_shap_train - mean_shap_test
    })

    shap_comparison = shap_comparison.sort_values('Difference', key=abs, ascending=False)

    plt.figure(figsize=(12, 8))
    plt.scatter(shap_comparison['Mean SHAP Train'], shap_comparison['Mean SHAP Test'], alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')  # ligne de référence y=x
    for i, txt in enumerate(shap_comparison['Feature']):
        plt.annotate(txt, (shap_comparison['Mean SHAP Train'].iloc[i], shap_comparison['Mean SHAP Test'].iloc[i]))
    plt.xlabel('Mean SHAP Value (Train)')
    plt.ylabel('Mean SHAP Value (Test)')
    plt.title('Comparison of Mean SHAP Values: Train vs Test')
    plt.tight_layout()

    # Sauvegarder le graphique dans le répertoire spécifié
    plt.savefig(os.path.join(save_dir, 'mean_shap_comparison.png'))
    plt.close()

    print("Top differences in mean SHAP values:")
    print(shap_comparison.head(10))

    return shap_comparison



def main_shap_analysis(final_model, X_train, y_train_label, X_test, y_test_label,
                       save_dir='./shap_dependencies_results/'):
    """Fonction principale pour l'analyse SHAP."""

    # Analyse SHAP sur l'ensemble d'entraînement et de test
    shap_values_train,shap_explanation_train = analyze_shap_values(final_model, X_train, y_train_label, "Training_Set",
                                            create_dependence_plots=True,
                                            max_dependence_plots=3, save_dir=save_dir)
    shap_values_test,shap_explanation_test = analyze_shap_values(final_model, X_test, y_test_label, "Test_Set", create_dependence_plots=True,
                                           max_dependence_plots=3, save_dir=save_dir)

    # Comparaison des importances de features et des distributions SHAP
    importance_df = compare_feature_importance(shap_values_train, shap_values_test, X_train, X_test, save_dir=save_dir)
    compare_shap_distributions(
        shap_values_train=shap_values_train,shap_explanation_train=shap_explanation_train, shap_values_test=shap_values_test,shap_explanation_test=shap_explanation_test,
        X_train=X_train,y_train_label=y_train_label, X_test=X_test,y_test_label=y_test_label ,
        top_n=40, save_dir=save_dir)

    # Comparaison des valeurs SHAP moyennes
    shap_comparison = compare_mean_shap_values(
        shap_values_train=shap_values_train, shap_values_test=shap_values_test, X_train=X_train, save_dir=save_dir)

    return importance_df, shap_comparison, shap_values_train, shap_values_test


def analyze_xgboost_trees(model, feature_names, nan_value, max_trees=None):
    """
    Analyse les arbres XGBoost pour identifier tous les splits et ceux impliquant des NaN.

    :param model: Modèle XGBoost entraîné (Booster ou XGBClassifier/XGBRegressor)
    :param feature_names: Liste des noms des features
    :param nan_value: Valeur utilisée pour remplacer les NaN (peut être np.nan)
    :param max_trees: Nombre maximum d'arbres à analyser (None pour tous les arbres)
    :return: Tuple (DataFrame de tous les splits, DataFrame des splits NaN)
    """
    # Si le modèle est un Booster, l'utiliser directement ; sinon, obtenir le Booster depuis le modèle
    if isinstance(model, xgb.Booster):
        booster = model
    else:
        booster = model.get_booster()

    # Obtenir les arbres sous forme de DataFrame
    trees_df = booster.trees_to_dataframe()

    if max_trees is not None:
        trees_df = trees_df[trees_df['Tree'] < max_trees].copy()

    # Filtrer pour ne garder que les splits (nœuds non-feuilles)
    all_splits = trees_df[trees_df['Feature'] != 'Leaf'].copy()

    # Check if 'Depth' is already present; if not, calculate it
    if pd.isna(nan_value):
        # Lorsque nan_value est np.nan, identifier les splits impliquant des valeurs manquantes
        nan_splits = all_splits[all_splits['Missing'] != all_splits['Yes']].copy()
        print(f"Utilisation de la condition pour np.nan. Nombre de splits NaN trouvés : {len(nan_splits)}")
    else:
        # Lorsque nan_value est une valeur spécifique, identifier les splits sur cette valeur
        all_splits.loc[:, 'Split'] = all_splits['Split'].astype(float)
        nan_splits = all_splits[np.isclose(all_splits['Split'], nan_value, atol=1e-8)].copy()
        print(
            f"Utilisation de la condition pour nan_value={nan_value}. Nombre de splits NaN trouvés : {len(nan_splits)}")
        if len(nan_splits) > 0:
            print("Exemples de valeurs de split considérées comme NaN:")
            print(nan_splits['Split'].head())

    return all_splits, nan_splits


def extract_decision_rules(model, nan_value, importance_threshold=0.01):
    """
    Extrait les règles de décision importantes impliquant la valeur de remplacement des NaN ou les valeurs NaN.
    :param model: Modèle XGBoost entraîné (Booster ou XGBClassifier/XGBRegressor)
    :param nan_value: Valeur utilisée pour remplacer les NaN (peut être np.nan)
    :param importance_threshold: Seuil de gain pour inclure une règle
    :return: Liste des règles de décision importantes
    """
    if isinstance(model, xgb.Booster):
        booster = model
    else:
        booster = model.get_booster()

    trees_df = booster.trees_to_dataframe()

    # Ajouter la colonne Depth en comptant le nombre de tirets dans 'ID'
    if 'Depth' not in trees_df.columns:
        trees_df['Depth'] = trees_df['ID'].apply(lambda x: x.count('-'))

    # Vérifiez que les colonnes attendues sont présentes dans chaque ligne avant de traiter
    expected_columns = ['Tree', 'Depth', 'Feature', 'Gain', 'Split', 'Missing', 'Yes']

    # Filtrer les lignes avec des NaN si c'est nécessaire
    trees_df = trees_df.dropna(subset=expected_columns, how='any')

    if pd.isna(nan_value):
        important_rules = trees_df[
            (trees_df['Missing'] != trees_df['Yes']) & (trees_df['Gain'] > importance_threshold)
            ]
    else:
        important_rules = trees_df[
            (trees_df['Split'] == nan_value) & (trees_df['Gain'] > importance_threshold)
            ]

    rules = []
    for _, row in important_rules.iterrows():
        try:
            # Check if necessary columns exist in the row before constructing the rule
            if all(col in row for col in expected_columns):
                rule = f"Arbre {row['Tree']}, Profondeur {row['Depth']}"
                feature = row['Feature']
                gain = row['Gain']

                if pd.isna(nan_value):
                    missing_direction = row['Missing']
                    rule += f": SI {feature} < {row['Split']} OU {feature} est NaN (valeurs manquantes vont vers le nœud {missing_direction}) ALORS ... (gain: {gain:.4f})"
                else:
                    rule += f": SI {feature} == {nan_value} ALORS ... (gain: {gain:.4f})"

                rules.append(rule)
            else:
                print(f"Colonnes manquantes dans cette ligne : {row.to_dict()}")
                continue

        except IndexError as e:
            # Gérer l'erreur d'index hors limites
            # print(f"Erreur lors de l'extraction des informations de la règle: {e}")
            continue
        except Exception as e:
            # Gérer toute autre erreur inattendue
            print(f"Erreur inattendue: {e}")
            continue

    return rules



def analyze_nan_impact(model, X_train, feature_names, nan_value, shap_values=None,
                       features_per_plot=35, verbose_nan_rule=False,
                       save_dir='./nan_analysis_results/'):
    """
    Analyse l'impact des valeurs NaN ou des valeurs de remplacement des NaN sur le modèle XGBoost.

    Parameters:
    -----------
    model : xgboost.Booster ou xgboost.XGBClassifier/XGBRegressor
        Modèle XGBoost entraîné
    X_train : pandas.DataFrame
        Données d'entrée
    feature_names : list
        Liste des noms des features
    nan_value : float ou np.nan
        Valeur utilisée pour remplacer les NaN
    shap_values : numpy.array, optional
        Valeurs SHAP pré-calculées
    features_per_plot : int, default 35
        Nombre de features à afficher par graphique
    verbose_nan_rule : bool, default False
        Si True, affiche les règles de décision détaillées
    save_dir : str, default './nan_analysis_results/'
        Répertoire où sauvegarder les résultats

    Returns:
    --------
    dict
        Dictionnaire contenant les résultats de l'analyse
    """
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    # 1. Analyser les splits impliquant les valeurs NaN
    all_splits, nan_splits = analyze_xgboost_trees(model, feature_names, nan_value)
    results['total_splits'] = len(all_splits)
    results['nan_splits'] = len(nan_splits)
    results['nan_splits_percentage'] = (len(nan_splits) / len(all_splits)) * 100

    # Stocker les distributions pour une utilisation ultérieure
    all_splits_dist = all_splits['Feature'].value_counts()
    nan_splits_dist = nan_splits['Feature'].value_counts()

    # 2. Visualiser la distribution des splits NaN
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Feature', data=nan_splits, order=nan_splits_dist.index)
    plt.title("Distribution des splits impliquant des valeurs NaN par feature")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nan_splits_distribution.png'))
    plt.close()

    # 3. Analyser la profondeur des splits NaN
    if 'Depth' not in nan_splits.columns and 'ID' in nan_splits.columns:
        nan_splits['Depth'] = nan_splits['ID'].apply(lambda x: x.count('-'))

    if 'Depth' in nan_splits.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Feature', y='Depth', data=nan_splits)
        plt.title("Profondeur des splits impliquant des valeurs NaN par feature")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'nan_splits_depth.png'))
        plt.close()

    # 4. Extraire les règles de décision importantes
    important_rules = extract_decision_rules(model, nan_value)
    results['important_rules'] = important_rules

    if verbose_nan_rule:
        print("\nRègles de décision importantes impliquant des valeurs NaN :")
        for rule in important_rules:
            print(rule)

    # 5. Analyser l'importance des features avec des valeurs NaN

    # Calculs pour l'analyse SHAP et NaN
    shap_mean = np.abs(shap_values).mean(axis=0)
    nan_counts = X_train.isna().sum() if pd.isna(nan_value) else (X_train == nan_value).sum()
    nan_percentages = (nan_counts / len(X_train)) * 100

    nan_fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_mean,
        'Total_NaN': nan_counts,
        'Percentage_NaN': nan_percentages
    }).sort_values('Importance', ascending=False)

    # Générer les graphiques par lots
    num_features = len(nan_fi_df)
    for i in range((num_features + features_per_plot - 1) // features_per_plot):
        subset_df = nan_fi_df.iloc[i * features_per_plot: (i + 1) * features_per_plot]

        fig, ax1 = plt.subplots(figsize=(14, 8))
        sns.barplot(x='Feature', y='Importance', data=subset_df, ax=ax1, color='skyblue')
        ax1.set_xlabel('Feature', fontsize=12)
        ax1.set_ylabel('Importance (SHAP)', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        sns.lineplot(x='Feature', y='Percentage_NaN', data=subset_df, ax=ax2, color='red', marker='o')
        ax2.set_ylabel('Pourcentage de NaN (%)', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter())

        plt.title(f"Importance des features (SHAP) et pourcentage de NaN (X_Train)\n"
                  f"(Features {i * features_per_plot + 1} à {min((i + 1) * features_per_plot, num_features)})",
                  fontsize=14)

        # Incliner les étiquettes de l'axe x à 45 degrés
        ax1.set_xticks(range(len(subset_df)))
        ax1.set_xticklabels(subset_df['Feature'], rotation=45, ha='right', va='top')

        # Ajuster l'espace pour améliorer la visibilité
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)  # Ajustez cette valeur si nécessaire

        plt.savefig(os.path.join(save_dir, f'nan_features_shap_importance_percentage_{i + 1}.png'))
        plt.close()

    # Calcul et visualisation de la corrélation
    correlation = nan_fi_df['Importance'].corr(nan_fi_df['Percentage_NaN'])
    results['shap_nan_correlation'] = correlation

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Percentage_NaN', y='Importance', data=nan_fi_df)
    plt.title('Relation entre le pourcentage de NaN et l\'importance des features (SHAP)')
    plt.xlabel('Pourcentage de NaN (%)')
    plt.ylabel('Importance (SHAP)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_importance_vs_percentage_nan.png'))
    plt.close()

    return results


# 1. Fonction pour analyser les erreurs

def analyze_errors(X_test, y_test_label, y_pred_threshold, y_pred_proba, feature_names, save_dir=None,
                   top_features=None):
    """
    Analyse les erreurs de prédiction du modèle et génère des visualisations.

    Parameters:
    -----------
    X_test : pd.DataFrame
        Les features de l'ensemble de test.
    y_test_label : array-like
        Les vraies étiquettes de l'ensemble de test.
    y_pred_threshold : array-like
        Les prédictions du modèle après application du seuil.
    y_pred_proba : array-like
        Les probabilités prédites par le modèle.
    feature_names : list
        Liste des noms des features.
    save_dir : str, optional
        Le répertoire où sauvegarder les résultats de l'analyse (par défaut './analyse_error/').
    top_features : list, optional
        Liste des 10 features les plus importantes (si disponible).

    Returns:
    --------
    tuple
        (results_df, error_df) : DataFrames contenant respectivement tous les résultats et les cas d'erreur.
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    # Vérifier et convertir les arrays CuPy en NumPy
    def convert_to_numpy(arr):
        return arr.get() if isinstance(arr, cp.ndarray) else arr

    # Convertir y_test_label, y_pred_threshold, y_pred_proba en NumPy si nécessaire
    y_test_label = convert_to_numpy(y_test_label)
    y_pred_threshold = convert_to_numpy(y_pred_threshold)
    y_pred_proba = convert_to_numpy(y_pred_proba)

    # Créer un dictionnaire pour stocker toutes les données
    data = {
        'true_label': y_test_label,
        'predicted_label': y_pred_threshold,
        'prediction_probability': y_pred_proba
    }

    # Ajouter les features au dictionnaire, en les convertissant en NumPy si nécessaire
    for feature in feature_names:
        data[feature] = convert_to_numpy(X_test[feature])

    # Créer le DataFrame en une seule fois
    results_df = pd.DataFrame(data)

    # Ajouter les colonnes d'erreur
    results_df['is_error'] = results_df['true_label'] != results_df['predicted_label']
    results_df['error_type'] = np.where(results_df['is_error'],
                                        np.where(results_df['true_label'] == 1, 'False Negative', 'False Positive'),
                                        'Correct')

    # Analyse des erreurs
    error_distribution = results_df['error_type'].value_counts(normalize=True)
    print("Distribution des erreurs:")
    print(error_distribution)

    # Sauvegarder la distribution des erreurs
    error_distribution.to_csv(os.path.join(save_dir, 'error_distribution.csv'), index=False, sep=';')

    # Analyser les features pour les cas d'erreur
    error_df = results_df[results_df['is_error']]

    print("\nMoyenne des features pour les erreurs vs. prédictions correctes:")
    feature_means = results_df.groupby('error_type')[feature_names].mean()
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(feature_means)

    # Sauvegarder les moyennes des features
    feature_means.to_csv(os.path.join(save_dir, 'feature_means_by_error_type.csv'), index=False, sep=';')

    # Visualiser la distribution des probabilités de prédiction pour les erreurs
    plt.figure(figsize=(10, 6))
    sns.histplot(data=error_df, x='prediction_probability', hue='true_label', bins=20)
    plt.title('Distribution des probabilités de prédiction pour les erreurs')
    plt.savefig(os.path.join(save_dir, 'error_probability_distribution.png'))
    plt.close()

    # Identifier les cas les plus confiants mais erronés
    most_confident_errors = error_df.sort_values('prediction_probability', ascending=False).head(5)
    print("\nLes 5 erreurs les plus confiantes:")
    print(most_confident_errors[['true_label', 'predicted_label', 'prediction_probability']])

    # Sauvegarder les erreurs les plus confiantes
    most_confident_errors.to_csv(os.path.join(save_dir, 'most_confident_errors.csv'), index=False, sep=';')

    # Visualisations supplémentaires
    plt.figure(figsize=(12, 10))
    sns.heatmap(error_df[feature_names].corr(), annot=False, cmap='coolwarm')
    plt.title('Corrélations des features pour les erreurs')
    plt.savefig(os.path.join(save_dir, 'error_features_correlation.png'))
    plt.close()

    # Identification des erreurs
    errors = X_test[y_test_label != y_pred_threshold]
    print("Nombre d'erreurs:", len(errors))

    # Créer un subplot pour chaque feature (si top_features est fourni)
    if top_features and len(top_features) >= 40:
        fig, axes = plt.subplots(5, 8, figsize=(60, 30))  # Grille 5x8
        fig.suptitle('Distribution des 40 features les plus importantes par type d\'erreur', fontsize=16)

        for i, feature in enumerate(top_features[:40]):
            row = i // 8  # Division entière par 8 (nombre de colonnes)
            col = i % 8  # Modulo 8 pour obtenir la colonne
            sns.boxplot(x='error_type', y=feature, data=results_df, ax=axes[row, col])
            axes[row, col].set_title(f'{i + 1}. {feature}')
            axes[row, col].set_xlabel('')
            if col == 0:
                axes[row, col].set_ylabel('Valeur')
            else:
                axes[row, col].set_ylabel('')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'top_40_features_distribution_by_error.png'),
                    dpi=300, bbox_inches='tight')
        print("Graphique combiné sauvegardé sous 'top_40_features_distribution_by_error.png'")
        plt.close()

    # Sauvegarder les DataFrames spécifiques demandés
    results_df.to_csv(os.path.join(save_dir, 'model_results_analysis.csv'), index=False, sep=';')
    error_df.to_csv(os.path.join(save_dir, 'model_errors_analysis.csv'), index=False, sep=';')
    print(f"\nLes résultats de l'analyse ont été sauvegardés dans le répertoire : {save_dir}")

    return results_df, error_df


# 2. Fonction pour analyser les erreurs confiantes
def analyze_confident_errors(shap_values, confident_errors, X_test, feature_names, important_features, n=5):
    for idx in confident_errors.index[:n]:
        print(f"-----------------> Analyse de l'erreur à l'index {idx}:")
        print(f"Vrai label: {confident_errors.loc[idx, 'true_label']}")
        print(f"Label prédit: {confident_errors.loc[idx, 'predicted_label']}")
        print(f"Probabilité de prédiction: {confident_errors.loc[idx, 'prediction_probability']:.4f}")

        print("\nValeurs des features importantes:")
        for feature in important_features:
            value = X_test.loc[idx, feature]
            print(f"{feature}: {value:.4f}")

        print("\nTop 5 features influentes (SHAP) pour ce cas:")
        case_feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values[0])
        }).sort_values('importance', ascending=False)

        print(case_feature_importance.head())
        print(f"<----------------- Fin Analyse de l'erreur à l'index {idx}:")


# 3. Fonction pour visualiser les erreurs confiantes
def plot_confident_errors(shap_values, confident_errors, X_test, feature_names, n=5,results_directory=None):
    # Vérifier le nombre d'erreurs confiantes disponibles
    num_errors = len(confident_errors)
    if num_errors == 0:
        print("Aucune erreur confiante trouvée.")
        return

    # Ajuster n si nécessaire
    n = min(n, num_errors)

    for i, idx in enumerate(confident_errors.index[:n]):
        plt.figure(figsize=(10, 6))

        # Vérifier si shap_values est une liste (pour les modèles à plusieurs classes)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Prendre les valeurs SHAP pour la classe positive

        shap.summary_plot(shap_values, X_test.loc[idx:idx], plot_type="bar", feature_names=feature_names, show=False)
        plt.title(f"Erreur {i + 1}: Vrai {confident_errors.loc[idx, 'true_label']}, "
                  f"Prédit {confident_errors.loc[idx, 'predicted_label']} "
                  f"(Prob: {confident_errors.loc[idx, 'prediction_probability']:.4f})")
        plt.tight_layout()
        plt.savefig(f'confident_error_shap_{i + 1}.png')

        plt.close()

    if n > 0:
        # Create a summary image combining all individual plots
        images = [Image.open(f'confident_error_shap_{i + 1}.png') for i in range(n)]
        widths, heights = zip(*(i.size for i in images))

        max_width = max(widths)
        total_height = sum(heights)

        new_im = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]

        new_im.save(os.path.join(results_directory, 'confident_errors_shap_combined.png'), dpi=(300, 300),
                    bbox_inches='tight')
        # Clean up individual images
        for i in range(n):
            os.remove(f'confident_error_shap_{i + 1}.png')

        print(f"Image combinée des {n} erreurs confiantes sauvegardée sous 'confident_errors_shap_combined.png'")
    else:
        print("Pas assez d'erreurs confiantes pour créer une image combinée.")


# 4. Fonction pour comparer les erreurs vs les prédictions correctes
def compare_errors_vs_correct(confident_errors, correct_predictions, X_test, important_features, results_directory):
    error_data = X_test.loc[confident_errors.index]
    correct_data = X_test.loc[correct_predictions.index]

    comparison_data = []
    for feature in important_features:
        error_mean = error_data[feature].mean()
        correct_mean = correct_data[feature].mean()
        difference = error_mean - correct_mean

        comparison_data.append({
            'Feature': feature,
            'Erreurs Confiantes (moyenne)': error_mean,
            'Prédictions Correctes (moyenne)': correct_mean,
            'Différence': difference
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\nComparaison des features importantes:")
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(comparison_df)

    # Visualisation
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(important_features))

    plt.bar(index, comparison_df['Erreurs Confiantes (moyenne)'], bar_width, label='Erreurs Confiantes')
    plt.bar(index + bar_width, comparison_df['Prédictions Correctes (moyenne)'], bar_width,
            label='Prédictions Correctes')

    plt.xlabel('Features')
    plt.ylabel('Valeur Moyenne')
    plt.title('Comparaison des features importantes: Erreurs vs Prédictions Correctes')
    plt.xticks(index + bar_width / 2, important_features, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, 'compare_errors_vs_correct.png'), dpi=300, bbox_inches='tight')
    plt.close()



# You already have CuPy imported as cp, so we will use it.
def analyze_predictions_by_range(X_test, y_pred_proba, shap_values_all, prob_min=0.90, prob_max=1.00,
                                 top_n_features=None,
                                 output_dir=None):
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Helper function to convert CuPy arrays to NumPy
    def convert_to_numpy(arr):
        # Check if cp is available and arr is a cp array
        return arr.get() if isinstance(arr, cp.ndarray) else arr

    # Convert y_pred_proba and shap_values_all to NumPy if necessary
    y_pred_proba = convert_to_numpy(y_pred_proba)
    shap_values_all = convert_to_numpy(shap_values_all)

    # 1. Identifier les échantillons dans la plage de probabilités spécifiée
    prob_mask = (y_pred_proba >= prob_min) & (y_pred_proba <= prob_max)
    selected_samples = X_test[prob_mask]
    selected_proba = y_pred_proba[prob_mask]

    # Vérifier s'il y a des échantillons dans la plage spécifiée
    if len(selected_samples) == 0:
        print(f"Aucun échantillon trouvé dans la plage de probabilités {prob_min:.2f} - {prob_max:.2f}")
        return

    print(f"Nombre d'échantillons dans la plage {prob_min:.2f} - {prob_max:.2f}: {len(selected_samples)}")

    # Calculer l'importance des features basée sur les valeurs SHAP
    feature_importance = np.abs(shap_values_all).mean(0)

    # Sélectionner les top features si spécifié
    if top_n_features is not None and top_n_features < len(X_test.columns):
        top_features_indices = np.argsort(feature_importance)[-top_n_features:]
        top_features = X_test.columns[top_features_indices]
        selected_samples = selected_samples[top_features]
        X_test_top = X_test[top_features]
        shap_values_selected = shap_values_all[prob_mask][:, top_features_indices]
    else:
        top_features = X_test.columns
        X_test_top = X_test
        shap_values_selected = shap_values_all[prob_mask]

    # 2. Examiner ces échantillons
    with open(os.path.join(output_dir, 'selected_samples_details.txt'), 'w',encoding='utf-8') as f:
        f.write(f"Échantillons avec probabilités entre {prob_min:.2f} et {prob_max:.2f}:\n")
        for i, (idx, row) in enumerate(selected_samples.iterrows()):
            f.write(f"\nÉchantillon {i + 1} (Probabilité: {selected_proba[i]:.4f}):\n")
            for feature, value in row.items():
                f.write(f"  {feature}: {value}\n")

    # 3. Analyser les statistiques de ces échantillons
    stats = selected_samples.describe()
    stats.to_csv(os.path.join(output_dir, 'selected_samples_statistics.csv'), index=False, sep=';')

    # 4. Comparer avec les statistiques globales
    comparison_list = []
    for feature in top_features:
        global_mean = X_test_top[feature].mean()
        selected_mean = selected_samples[feature].mean()
        diff = selected_mean - global_mean
        comparison_list.append({
            'Feature': feature,
            'Global_Mean': global_mean,
            'Selected_Mean': selected_mean,
            'Difference': diff
        })
    comparison = pd.DataFrame(comparison_list)
    comparison.to_csv(os.path.join(output_dir, 'global_vs_selected_comparison.csv'), index=False, sep=';')

    # 5. Visualiser les distributions
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(X_test_top[feature], kde=True, label='Global')
        sns.histplot(selected_samples[feature], kde=True, label=f'Selected ({prob_min:.2f} - {prob_max:.2f})')
        plt.title(f"Distribution de {feature}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'distribution_{feature}.png'))
        plt.close()

    # 6. Analyse des corrélations entre les features pour ces échantillons
    correlation_matrix = selected_samples.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f"Matrice de corrélation pour les échantillons avec probabilités entre {prob_min:.2f} et {prob_max:.2f}")
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    # 7. Comparer les valeurs SHAP pour ces échantillons
    shap.summary_plot(shap_values_selected, selected_samples, plot_type="bar", show=False)
    plt.title(
        f"Importance des features SHAP pour les échantillons avec probabilités entre {prob_min:.2f} et {prob_max:.2f}")
    plt.savefig(os.path.join(output_dir, 'shap_importance.png'))
    plt.close()

    print(f"Analyse terminée. Les résultats ont été sauvegardés dans le dossier '{output_dir}'.")


def init_dataSet(df=None, nanvalue_to_newval=None, selected_columns=None):
    # Gestion des valeurs NaN
    if nanvalue_to_newval is not None:
        df = df.fillna(nanvalue_to_newval)
        nan_value = nanvalue_to_newval
    else:
        nan_value = np.nan

    print("Division des données en ensembles d'entraînement et de test...")
    try:
        train_df, nb_SessionTrain, test_df, nb_SessionTest = split_sessions(df, test_size=0.2, min_train_sessions=2,
                                                                            min_test_sessions=2)
    except ValueError as e:
        print(f"Erreur lors de la division des sessions : {e}")
        sys.exit(1)

    print(f"Nombre de session dans x_train  {nb_SessionTrain}  ")
    print(f"Nombre de session dans x_test  {nb_SessionTest}  ")

    # Garder une copie de train_df avec son index original
    X_train_full = train_df.copy()

    # Préparation des features et de la cible
    X_train = train_df[selected_columns]
    y_train_label = train_df['class_binaire']
    X_test = test_df[selected_columns]
    y_test_label = test_df['class_binaire']

    # Suppression des échantillons avec la classe 99 en préservant l'index
    mask_train = y_train_label != 99
    X_train = X_train[mask_train]
    y_train_label = y_train_label[mask_train]

    mask_test = y_test_label != 99
    X_test = X_test[mask_test]
    y_test_label = y_test_label[mask_test]

    # Sauvegarder les copies avec catégories avant suppression
    X_train_withCat = X_train.copy()
    y_train_label_withCat = y_train_label.copy()
    X_test_withCat = X_test.copy()
    y_test_label_withCat = y_test_label.copy()

    # Colonnes à supprimer
    columns_to_drop = ['class_binaire', 'date', 'trade_category', 'SessionStartEnd']

    # Supprimer les colonnes en préservant l'index
    X_train = X_train.drop(columns=columns_to_drop, errors='ignore')
    X_test = X_test.drop(columns=columns_to_drop, errors='ignore')

    return (X_train_full, X_train, y_train_label, X_test, y_test_label,
            nb_SessionTrain, nb_SessionTest, nan_value)


import csv
import os


def save_correlations_to_csv(high_corr_pairs, results_directory, threshold=0.7):
    # Création du nom du fichier
    csv_filename = f"correlations_above_{threshold}.csv"
    csv_path = os.path.join(results_directory, csv_filename)

    # Initialisation
    seen_pairs = set()
    csv_data = []

    # Traitement des paires de corrélation
    for feature_i, feature_j, corr_value in high_corr_pairs:
        # Créer une paire triée pour éviter les doublons
        pair = tuple(sorted([feature_i, feature_j]))

        # Si la paire n'est pas encore vue et ce n'est pas une auto-corrélation
        if pair not in seen_pairs and feature_i != feature_j:
            # Affichage dans la console
            print(f"{feature_i} <-> {feature_j}: {corr_value:.4f}")

            # Préparation données CSV
            interaction = f"{feature_i} <-> {feature_j}"
            formatted_value = f"{corr_value:.4f}"
            csv_data.append([interaction, formatted_value])

            seen_pairs.add(pair)

    # Sauvegarde du CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Correlation_Pair', 'Correlation_Value'])
        writer.writerows(csv_data)

    print(f"\nFichier CSV sauvegardé: {csv_path}")
    return csv_path


# Utilisation
"""
# Si high_corr_pairs est une liste de tuples (feature_i, feature_j, corr_value)
save_correlations_to_csv(
    high_corr_pairs=high_corr_pairs,
    results_directory=results_directory,
    threshold=0.7  # Seuil de corrélation utilisé
)
"""

def train_finalModel_analyse(xgb=None,X_train=None,X_test=None,y_train_label=None,y_test_label=None,dtrain=None,dtest=None,
                             nb_SessionTest=None,nan_value=None,feature_names=None,best_params=None,config=None,weight_param=None,
                             user_input=None):
    results_directory = config.get('results_directory', None)

    if results_directory == None:
        exit(25)

    # Réduire X_train à seulement les colonnes sélectionnées
    X_train = X_train[feature_names]
    X_test = X_test[feature_names]


    # Créer les DMatrix pour l'entraînement
    sample_weights_train = compute_sample_weight('balanced', y=y_train_label)
    dtrain = xgb.DMatrix(X_train, label=y_train_label, weight=sample_weights_train)

    # Créer les DMatrix pour le test
    sample_weights_test = compute_sample_weight('balanced', y=y_test_label)
    dtest = xgb.DMatrix(X_test, label=y_test_label, weight=sample_weights_test)

    # best_params = study_xgb.best_params.copy()
    best_params['tree_method'] = 'hist'
    best_params['device'] = config.get('device_', 'cuda')
    early_stopping_rounds = config.get('early_stopping_rounds', 13)
    if(early_stopping_rounds==13):
        print("early_stopping_rounds n'a pas été position dans config")
        exit(1)


    optimal_threshold = best_params['threshold']
    print(f"## Seuil utilisé : {optimal_threshold:.4f}")
    num_boost_round = best_params.pop('num_boost_round', None)
    print(
        f"Num Boost : {num_boost_round}")



    optuna_options_method=config.get('optuna_options_method', None)
    print(optuna_options_method)
    if (optuna_options_method==None):
        exit(13)

    # Configurer custom_metric et obj_function si nécessaire
    if optuna_options_method == optuna_options.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED:
        metric_dict = {
            'profit_per_tp': best_params['profit_per_tp'],
            'loss_per_fp': best_params['loss_per_fp'],
            'penalty_per_fn': best_params['penalty_per_fn'],
            'threshold': optimal_threshold
        }
        if config['device_'] == 'cuda':
            custom_metric = lambda preds, dtrain: custom_metric_ProfitBased_gpu(preds, dtrain, metric_dict)
            obj_function = create_weighted_logistic_obj_gpu(best_params['w_p'], best_params['w_n'])
        else:
            custom_metric = lambda preds, dtrain: custom_metric_ProfitBased_cpu(preds, dtrain, metric_dict)
            obj_function = create_weighted_logistic_obj_cpu(best_params['w_p'], best_params['w_n'])

        best_params['disable_default_eval_metric'] = 1
    else:
        custom_metric = None
        obj_function = None
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = ['aucpr', 'logloss']
        print("not USE_OPTIMA_CUSTOM_METRIC_PROFITBASED")
        exit(66)

    print(f"Seuil optimal: {optimal_threshold}")
    # Supprimer les paramètres non utilisés par XGBoost mais uniquement dans l'optimisation
    parameters_to_removetoAvoidXgboostError = ['loss_per_fp', 'penalty_per_fn', 'profit_per_tp', 'threshold', 'w_p',
                                               'w_n','nb_split_weight','std_penalty_factor','weight_split']
    for param in parameters_to_removetoAvoidXgboostError:
        best_params.pop(param, None)  # None est la valeur par défaut si la clé n'existe pas

    print(f"best_params dans les parametres d'optimisations  filtrés dans parametre non compatibles xgboost: \n{best_params}")



    # Entraîner le modèle final
    evals_result = {}  # Créez un dictionnaire vide pour stocker les résultats
    try:
        final_model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            obj=obj_function,
            custom_metric=custom_metric,
            early_stopping_rounds=early_stopping_rounds,
            maximize=True,
            verbose_eval=20,
            evals_result=evals_result
        )

    except xgb.core.XGBoostError as e:
        print(f"\nXGBoost Error: {str(e)}")
        raise

    except ValueError as e:
        print(f"\nValue Error: {str(e)}")
        raise

    except Exception as e:
        print(f"\nUnexpected Error: {str(e)}")
        raise

    # Affichage des résultats clés
    print(f"Meilleur nombre d'itérations : {final_model.best_iteration}")
    print(f"Meilleur score : {final_model.best_score}")

    # Création du nom de fichier avec timestamp
    current_time = datetime.now()
    timestamp = current_time.strftime("%y%m%d_%H_%M_%S")  # Format: YYMMDD_HH_MM_SS

    # Création du répertoire si nécessaire
    save_dir = os.path.join(results_directory, 'optuna_results')
    os.makedirs(save_dir, exist_ok=True)

    # Sauvegarde du modèle
    model_file = os.path.join(save_dir, f"final_training_model_{timestamp}.json")
    final_model.save_model(model_file)
    print(f"\nModèle sauvegardé: {model_file}")

    def add_early_stopping_zone(ax, best_iteration, color='orange', alpha=0.2):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.axvspan(best_iteration, xmax, facecolor=color, alpha=alpha)
        ax.text(best_iteration + (xmax - best_iteration) / 2, ymax, 'Zone post early stopping',
                horizontalalignment='center', verticalalignment='top', fontsize=12, color='orange')

    def plot_custom_metric_evolution_with_trade_info(model, evals_result, metric_name='custom_metric_ProfitBased',
                                                     n_train_trades=None, n_test_trades=None, results_directory=None,user_input=None):
        if not evals_result or 'train' not in evals_result or 'test' not in evals_result:
            print("Résultats d'évaluation incomplets ou non disponibles.")
            return

        train_metric = evals_result['train'][metric_name]
        test_metric = evals_result['test'][metric_name]

        iterations = list(range(1, len(train_metric) + 1))
        best_test_iteration = np.argmax(test_metric)

        fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(24, 14))
        fig.suptitle(f'Entraînement du modèle final avec les paramètres optimaux (Optuna) :\n'
                     f'Évaluation du score {metric_name} sur l\'ensemble d\'entraînement (X_train) '
                     f'et un nouvel ensemble de test indépendant (X_test)', fontsize=12)

        def add_vertical_line_and_annotations(ax, is_train, is_normalized=False):
            ax.axvline(x=best_test_iteration, color='green', linestyle='--')
            y_pos = ax.get_ylim()[1] if is_train else ax.get_ylim()[0]
            score = train_metric[best_test_iteration] if is_train else test_metric[best_test_iteration]
            if is_normalized:
                score = (score - min(train_metric if is_train else test_metric)) / (
                        max(train_metric if is_train else test_metric) - min(
                    train_metric if is_train else test_metric))
            ax.annotate(f'{"Train" if is_train else "Test"} Score: {score:.2f}',
                        (best_test_iteration, y_pos), xytext=(5, 5 if is_train else -5),
                        textcoords='offset points', ha='left', va='bottom' if is_train else 'top',
                        fontsize=8, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        # X_train (Non Normalisé)
        ax1.plot(iterations, train_metric, label='Train', color='blue')
        ax1.set_title(f'X_train (Non Normalized)', fontsize=14)
        ax1.set_xlabel('Number of Iterations', fontsize=12)
        ax1.set_ylabel(f'{metric_name} Score', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.text(0.5, 0.1, f'Profit cumulés réalisé sur {n_train_trades} trades',
                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12,
                 color='blue')
        add_vertical_line_and_annotations(ax1, is_train=True)
        add_early_stopping_zone(ax1, best_test_iteration)

        # X_test (Non Normalisé)
        ax2.plot(iterations, test_metric, label='Test', color='red')
        ax2.set_title(f'X_test (Non Normalized)', fontsize=14)
        ax2.set_xlabel('Number of Iterations', fontsize=12)
        ax2.set_ylabel(f'{metric_name} Score', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.text(0.5, 0.1, f'Profit cumulés réalisé sur {n_test_trades} trades',
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=12,
                 color='red')
        add_vertical_line_and_annotations(ax2, is_train=False)
        add_early_stopping_zone(ax2, best_test_iteration)

        # Normalisation
        train_min, train_max = min(train_metric), max(train_metric)
        test_min, test_max = min(test_metric), max(test_metric)
        train_normalized = [(val - train_min) / (train_max - train_min) for val in train_metric]
        test_normalized = [(val - test_min) / (test_max - test_min) for val in test_metric]

        # X_train (Normalisé)
        ax3.plot(iterations, train_normalized, label='Train (Normalized)', color='blue')
        ax3.set_title(f'X_train (Normalized)', fontsize=14)
        ax3.set_xlabel('Number of Iterations', fontsize=12)
        ax3.set_ylabel(f'Normalized {metric_name} Score', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.text(0.5, 0.1,
                 f'Profit cumulés réalisé sur {n_train_trades} trades\nNorm Ratio: [{train_min:.4f}, {train_max:.4f}]',
                 horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize=12,
                 color='blue')
        add_vertical_line_and_annotations(ax3, is_train=True, is_normalized=True)
        add_early_stopping_zone(ax3, best_test_iteration)

        # X_test (Normalisé)
        ax4.plot(iterations, test_normalized, label='Test (Normalized)', color='red')
        ax4.set_title(f'X_test (Normalized)', fontsize=14)
        ax4.set_xlabel('Number of Iterations', fontsize=12)
        ax4.set_ylabel(f'Normalized {metric_name} Score', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.text(0.5, 0.1,
                 f'Profit cumulés réalisé sur {n_test_trades} trades\nNorm Ratio: [{test_min:.4f}, {test_max:.4f}]',
                 horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes, fontsize=12,
                 color='red')
        add_vertical_line_and_annotations(ax4, is_train=False, is_normalized=True)
        add_early_stopping_zone(ax4, best_test_iteration)

        # X_test (Non Normalisé) jusqu'à best_test_iteration
        ax5.plot(iterations[:best_test_iteration + 1], test_metric[:best_test_iteration + 1], label='Test', color='red')
        ax5.set_title(f'X_test (Non Normalized) until best Test Score', fontsize=14)
        ax5.set_xlabel('Number of Iterations', fontsize=12)
        ax5.set_ylabel(f'{metric_name} Score', fontsize=12)
        ax5.legend(fontsize=10)
        ax5.grid(True, linestyle='--', alpha=0.7)
        add_vertical_line_and_annotations(ax5, is_train=False)

        # X_test (Normalisé) jusqu'à best_test_iteration
        ax6.plot(iterations[:best_test_iteration + 1], test_normalized[:best_test_iteration + 1],
                 label='Test (Normalized)', color='red')
        ax6.set_title(f'X_test (Normalized) until best Test Score', fontsize=14)
        ax6.set_xlabel('Number of Iterations', fontsize=12)
        ax6.set_ylabel(f'Normalized {metric_name} Score', fontsize=12)
        ax6.legend(fontsize=10)
        ax6.grid(True, linestyle='--', alpha=0.7)
        add_vertical_line_and_annotations(ax6, is_train=False, is_normalized=True)
        plt.savefig(os.path.join(results_directory, f'Evolution of {metric_name} Score with Trade Information'),
                    dpi=300,
                    bbox_inches='tight')
        plt.tight_layout()
        if user_input.lower() == 'd':
            plt.show()
        plt.close()

    # Utilisation de la fonction
    plot_custom_metric_evolution_with_trade_info(final_model, evals_result, n_train_trades=len(X_train),
                                                 n_test_trades=len(X_test), results_directory=results_directory,user_input=user_input)



    print_notification('###### FIN: ENTRAINEMENT MODELE FINAL ##########', color="blue")

    print_notification('###### DEBUT: ANALYSE DES DEPENDENCES SHAP DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")
    importance_df, shap_comparison, shap_values_train, shap_values_test = main_shap_analysis(
        final_model, X_train, y_train_label, X_test, y_test_label,
        save_dir=os.path.join(results_directory, 'shap_dependencies_results'))
    print_notification('###### FIN: ANALYSE DES DEPENDENCES SHAP DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")

    print_notification('###### DEBUT: ANALYSE DE L\'IMPACT DES VALEURS NaN DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")

    # Appeler la fonction d'analyse
    analyze_nan_impact(model=final_model, X_train=X_train, feature_names=feature_names,
                       shap_values=shap_values_train, nan_value=nan_value,
                       save_dir=os.path.join(results_directory, 'nan_analysis_results'))

    print_notification('###### FIN: ANALYSE DE L\'IMPACT DES VALEURS NaN DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")

    # Prédiction et évaluation
    print_notification('###### DEBUT: GENERATION PREDICTION AVEC MOBEL FINAL (TEST) ##########', color="blue")

    # Obtenir les probabilités prédites pour la classe positive
    y_test_predProba = final_model.predict(dtest)

    # Convertir les prédictions en CuPy avant transformation
    y_test_predProba = cp.asarray(y_test_predProba)
    y_test_predProba = sigmoidCustom(y_test_predProba)  # Appliquer la transformation sigmoïde sur CuPy

    # Vérification des prédictions après transformation
    min_val = cp.min(y_test_predProba).item()
    max_val = cp.max(y_test_predProba).item()
    # print(f"Plage de valeurs après transformation sigmoïde : [{min_val:.4f}, {max_val:.4f}]")

    print(f"Plage de valeurs : [{min_val:.4f}, {max_val:.4f}]")

    # Vérifier si les valeurs sont dans l'intervalle [0, 1]
    if min_val < 0 or max_val > 1:
        print("ERREUR : Les prédictions ne sont pas dans l'intervalle [0, 1] attendu pour une classification binaire.")
        print("Vous devez appliquer une transformation (comme sigmoid) aux prédictions.")
        print("Exemple : y_test_predProba = sigmoidCustom(final_model.predict(dtest))")
        exit(11)
    else:
        print("Les prédictions sont dans l'intervalle [0, 1] attendu pour une classification binaire.")

    # Appliquer un seuil optimal pour convertir les probabilités en classes
    y_test_pred_threshold = (y_test_predProba > optimal_threshold).astype(int)

    print_notification('###### FIN: GENERATION PREDICTION AVEC MOBEL FINAL (TEST) ##########', color="blue")

    print_notification('###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur (XTEST) ##########',
                       color="blue")

    ###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur XTEST ##########

    # Pour la courbe de calibration et l'histogramme
    plot_calibrationCurve_distrib(y_test_label, y_test_predProba, optimal_threshold=optimal_threshold,
                                  user_input=user_input,
                                  num_sessions=nb_SessionTest, results_directory=results_directory)

    # Pour le graphique des taux FP/TP par feature

    import warnings

    if 'deltaTimestampOpeningSection1min' in X_test.columns:
        plot_fp_tp_rates(X_test, y_test_label, y_test_predProba, 'deltaTimestampOpeningSection1min',
                         optimal_threshold, user_input=user_input, index_size=5, results_directory=results_directory)
    else:
        warnings.warn(
            "La colonne 'deltaTimestampOpeningSection1min' n'est pas présente dans le jeu de test - Graphique non généré",
            UserWarning)

    print("\nDistribution des probabilités prédites sur XTest:")
    print(f"seuil: {optimal_threshold}")
    print(f"Min : {y_test_predProba.min():.4f}")
    print(f"Max : {y_test_predProba.max():.4f}")
    print(f"Moyenne : {y_test_predProba.mean():.4f}")
    print(f"Médiane : {np.median(y_test_predProba):.4f}")

    # Compter le nombre de prédictions dans différentes plages de probabilité
    # Définir le pas pour les intervalles en dessous de optimal_threshold
    step_below = 0.1  # Vous pouvez ajuster ce pas selon vos besoins

    # Créer les intervalles en dessous de optimal_threshold
    ranges_below = np.arange(0, optimal_threshold, step_below)
    ranges_below = np.append(ranges_below, optimal_threshold)

    # Définir le pas pour les intervalles au-dessus de optimal_threshold
    step_above = 0.02  # Taille des intervalles souhaitée au-dessus du seuil

    # Calculer le prochain multiple de step_above au-dessus de optimal_threshold
    next_multiple = np.ceil(optimal_threshold / step_above) * step_above

    # Créer les intervalles au-dessus de optimal_threshold
    ranges_above = np.arange(next_multiple, 1.0001, step_above)

    # Combiner les intervalles
    ranges = np.concatenate((ranges_below, ranges_above))
    ranges = np.unique(ranges)  # Supprimer les doublons et trier

    # Maintenant, vous pouvez utiliser ces ranges pour votre histogramme
    hist, _ = np.histogram(y_test_predProba, bins=ranges)

    # Convertir les tableaux CuPy en NumPy si nécessaire
    y_test_predProba_np = cp.asnumpy(y_test_predProba) if isinstance(y_test_predProba, cp.ndarray) else y_test_predProba
    y_test_label_np = cp.asnumpy(y_test_label) if isinstance(y_test_label, cp.ndarray) else y_test_label

    print("\nDistribution des probabilités prédites avec TP et FP sur XTest:")
    cum_tp = 0
    cum_fp = 0
    for i in range(len(ranges) - 1):
        mask = (y_test_predProba_np >= ranges[i]) & (y_test_predProba_np < ranges[i + 1])
        predictions_in_range = y_test_predProba_np[mask]
        true_values_in_range = y_test_label_np[mask]
        tp = np.sum((predictions_in_range >= optimal_threshold) & (true_values_in_range == 1))
        fp = np.sum((predictions_in_range >= optimal_threshold) & (true_values_in_range == 0))
        total_trades_val = tp + fp
        win_rate = tp / total_trades_val * 100 if total_trades_val > 0 else 0
        cum_tp = cum_tp + tp
        cum_fp = cum_fp + fp
        print(
            f"Probabilité {ranges[i]:.2f} - {ranges[i + 1]:.2f} : {hist[i]} prédictions, TP: {tp}, FP: {fp}, Winrate: {win_rate:.2f}%")

    total_trades_cum = cum_tp + cum_fp
    Winrate = cum_tp / total_trades_cum * 100 if total_trades_cum > 0 else 0
    print(f"=> Test final: X_test avec model final optimisé : TP: {cum_tp}, FP: {cum_fp}, Winrate: {Winrate:.2f}%")
    # On prend la valeur 'max' pour chaque paramètre
    profit_value = float(weight_param['profit_per_tp']['max'])
    loss_value = float(weight_param['loss_per_fp']['max'])

    print(f"  - PNL : {cum_tp * profit_value + cum_fp * loss_value:.2f}")

    print("Statistiques de y_pred_proba:")
    print(f"Nombre d'éléments: {len(y_test_predProba)}")
    print(f"Min: {np.min(y_test_predProba)}")
    print(f"Max: {np.max(y_test_predProba)}")
    print(f"Valeurs uniques: {np.unique(y_test_predProba)}")
    print(f"Y a-t-il des NaN?: {np.isnan(y_test_predProba).any()}")

    # Définissez min_precision si vous voulez l'utiliser, sinon laissez-le à None
    min_precision = None  # ou une valeur comme 0.7 si vous voulez l'utiliser

    # Création de la figure avec trois sous-graphiques côte à côte
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

    # Sous-graphique 1 : Courbe ROC

    # Convertir les tableaux CuPy en NumPy
    y_test_label_np = cp.asnumpy(y_test_label) if isinstance(y_test_label, cp.ndarray) else y_test_label
    y_test_predProba_np = cp.asnumpy(y_test_predProba) if isinstance(y_test_predProba, cp.ndarray) else y_test_predProba

    # Calculer la courbe ROC et le score AUC
    fpr, tpr, _ = roc_curve(y_test_label_np, y_test_predProba_np)
    auc_score = roc_auc_score(y_test_label_np, y_test_predProba_np)

    ax1.plot(fpr, tpr, color='blue', linestyle='-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.grid(True)
    ax1.legend(loc='lower right', fontsize=10)

    # Sous-graphique 2 : Distribution des probabilités prédites
    bins = np.linspace(y_test_predProba.min(), y_test_predProba.max(), 100)

    # Assurez-vous que y_test_predProba est en NumPy

    # Conversion de y_test_predProba en NumPy
    y_test_predProba_np = cp.asnumpy(y_test_predProba) if isinstance(y_test_predProba, cp.ndarray) else y_test_predProba

    # Assurez-vous que optimal_threshold est un scalaire Python
    optimal_threshold = float(optimal_threshold)

    # Créez les masques pour les valeurs au-dessus et en dessous du seuil
    mask_below = y_test_predProba_np <= optimal_threshold
    mask_above = y_test_predProba_np > optimal_threshold

    # Créez bins comme un tableau NumPy
    bins = np.linspace(np.min(y_test_predProba_np), np.max(y_test_predProba_np), 100)

    # Utilisez ces masques avec y_test_predProba_np pour l'histogramme
    ax2.hist(y_test_predProba_np[mask_below], bins=bins, color='orange',
             label=f'Prédictions ≤ {optimal_threshold:.4f}', alpha=0.7)
    ax2.hist(y_test_predProba_np[mask_above], bins=bins, color='blue',
             label=f'Prédictions > {optimal_threshold:.4f}', alpha=0.7)

    ax2.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Seuil de décision ({optimal_threshold:.4f})')
    ax2.set_title('Proportion de prédictions négatives (fonction du choix du seuil) sur XTest', fontsize=14,
                  fontweight='bold')
    ax2.set_xlabel('Proportion de prédictions négatives (fonction du choix du seuil)', fontsize=12)
    ax2.set_ylabel('Nombre de prédictions', fontsize=12)

    # Ajout des annotations pour les comptes

    # Convertir y_test_predProba en NumPy si c'est un tableau CuPy
    y_test_predProba_np = cp.asnumpy(y_test_predProba) if isinstance(y_test_predProba, cp.ndarray) else y_test_predProba

    # Utiliser la version NumPy pour les calculs
    num_below = np.sum(y_test_predProba_np <= optimal_threshold)
    num_above = np.sum(y_test_predProba_np > optimal_threshold)

    ax2.text(0.05, 0.95, f'Count ≤ {optimal_threshold:.4f}: {num_below}', color='orange', transform=ax2.transAxes,
             va='top')
    ax2.text(0.05, 0.90, f'Count > {optimal_threshold:.4f}: {num_above}', color='blue', transform=ax2.transAxes,
             va='top')

    ax2.legend(fontsize=10)

    def to_numpy(arr):
        if isinstance(arr, cp.ndarray):
            return arr.get()
        elif isinstance(arr, np.ndarray):
            return arr
        else:
            return np.array(arr)

    # Convertir y_test_label et y_test_predProba en tableaux NumPy
    y_test_label_np = to_numpy(y_test_label)
    y_test_predProba_np = to_numpy(y_test_predProba)

    # Sous-graphique 3 : Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test_label_np, y_test_predProba_np)
    ax3.plot(recall, precision, color='green', marker='.', linestyle='-', linewidth=2)
    ax3.set_title('Courbe Precision-Recall', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Recall (Taux de TP)', fontsize=12)
    ax3.set_ylabel('Precision (1 - Taux de FP)', fontsize=12)
    ax3.grid(True)

    # Ajout de la ligne de précision minimale si définie
    if min_precision is not None:
        ax3.axhline(y=min_precision, color='r', linestyle='--', label=f'Précision minimale ({min_precision:.2f})')
        ax3.legend(fontsize=10)

    # Ajustement de la mise en page
    plt.tight_layout()

    # Sauvegarde et affichage du graphique
    plt.savefig(os.path.join(results_directory, 'roc_distribution_precision_recall_combined.png'), dpi=300,
                bbox_inches='tight')

    # Afficher ou fermer la figure selon l'entrée de l'utilisateur
    if user_input.lower() == 'd':
        plt.show()  # Afficher les graphiques
    plt.close()  # Fermer après l'affichage ou sans affichage

    print_notification('###### FIN: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur (XTEST) ##########',
                       color="blue")

    ###### DEBUT: ANALYSE SHAP ##########
    print_notification('###### DEBUT: ANALYSE SHAP ##########', color="blue")

    def analyze_shap_feature_importance(shap_values, X, ensembleType='train', save_dir='./shap_feature_importance/'):
        """
        Analyse l'importance des features basée sur les valeurs SHAP et génère des visualisations.

        Parameters:
        -----------
        shap_values : np.array
            Les valeurs SHAP calculées pour l'ensemble de données.
        X : pd.DataFrame
            Les features de l'ensemble de données.
        ensembleType : str, optional
            Type d'ensemble de données ('train' ou 'test'). Détermine le suffixe des fichiers sauvegardés.
        save_dir : str, optional
            Le répertoire où sauvegarder les graphiques générés (par défaut './shap_feature_importance/').

        Returns:
        --------
        dict
            Un dictionnaire contenant les résultats clés de l'analyse.
        """
        if ensembleType not in ['train', 'test']:
            raise ValueError("ensembleType doit être 'train' ou 'test'")

        os.makedirs(save_dir, exist_ok=True)
        suffix = f"_{ensembleType}"

        # Calcul des valeurs SHAP moyennes pour chaque feature
        shap_mean = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': shap_mean,
            'effect': np.mean(shap_values, axis=0)  # Effet moyen (positif ou négatif)
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        # Graphique des 20 features les plus importantes
        top_20_features = feature_importance.head(20)
        plt.figure(figsize=(12, 10))
        colors = ['#FF9999', '#66B2FF']  # Rouge clair pour négatif, bleu clair pour positif
        bars = plt.barh(top_20_features['feature'], top_20_features['importance'],
                        color=[colors[1] if x > 0 else colors[0] for x in top_20_features['effect']])

        plt.title(f"Feature Importance Determined By SHAP Values ({ensembleType.capitalize()} Set)", fontsize=16)
        plt.xlabel('Mean |SHAP Value| (Average Impact on Model Output Magnitude)', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.legend([plt.Rectangle((0, 0), 1, 1, fc=colors[0]), plt.Rectangle((0, 0), 1, 1, fc=colors[1])],
                   ['Diminue la probabilité de succès', 'Augmente la probabilité de succès'],
                   loc='lower right', fontsize=10)
        plt.text(0.5, 1.05, "La longueur de la barre indique l'importance globale de la feature.\n"
                            "La couleur indique si la feature tend à augmenter (bleu) ou diminuer (rouge) la probabilité de succès du trade.",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_importance_binary_trade{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Visualisation des 40 valeurs SHAP moyennes absolues les plus importantes
        plt.figure(figsize=(24, 9))
        plt.bar(feature_importance['feature'][:40], feature_importance['importance'][:40])
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Top 40 Features par Importance SHAP (valeurs moyennes absolues) - {ensembleType.capitalize()} Set")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_importance_mean_abs{suffix}.png'))
        plt.close()

        # Analyse supplémentaire : pourcentage cumulatif de l'importance
        feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum() / feature_importance[
            'importance'].sum()

        # Top 10 features
        top_10_features = feature_importance['feature'].head(40).tolist()

        # Nombre de features nécessaires pour expliquer 80% de l'importance
        features_for_80_percent = feature_importance[feature_importance['cumulative_importance'] <= 0.8].shape[0]

        results = {
            'feature_importance': feature_importance,
            'top_10_features': top_10_features,
            'features_for_80_percent': features_for_80_percent
        }

        print(f"Graphiques SHAP pour l'ensemble {ensembleType} sauvegardés sous:")
        print(f"- {os.path.join(save_dir, f'shap_importance_binary_trade{suffix}.png')}")
        print(f"- {os.path.join(save_dir, f'shap_importance_mean_abs{suffix}.png')}")
        print(f"\nTop 10 features basées sur l'analyse SHAP ({ensembleType}):")
        print(top_10_features)
        print(
            f"\nNombre de features nécessaires pour expliquer 80% de l'importance ({ensembleType}) : {features_for_80_percent}")

        return results

    resulat_train_shap_feature_importance = analyze_shap_feature_importance(shap_values_train, X_train,
                                                                            ensembleType='train',
                                                                            save_dir=os.path.join(results_directory,
                                                                                                  'shap_feature_importance'))
    resulat_test_shap_feature_importance = analyze_shap_feature_importance(shap_values_test, X_test,
                                                                           ensembleType='test',
                                                                           save_dir=os.path.join(results_directory,
                                                                                                 'shap_feature_importance'))
    ###### FIN: ANALYSE SHAP ##########

    ###### DEBUT: ANALYSE DES ERREURS ##########
    print_notification('###### DEBUT: ANALYSE DES ERREURS ##########', color="blue")
    # Analyse des erreurs

    results_df, error_df = analyze_errors(X_test, y_test_label, y_test_pred_threshold, y_test_predProba, feature_names,
                                          save_dir=os.path.join(results_directory, 'analyse_error'),
                                          top_features=resulat_test_shap_feature_importance['top_10_features'])

    print_notification('###### FIN: ANALYSE DES ERREURS ##########', color="blue")
    ###### FIN: ANALYSE DES ERREURS ##########

    ###### DEBUT: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########
    print_notification('###### DEBUT: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########', color="blue")

    # Exemple d'utilisation :
    analyze_predictions_by_range(X_test, y_test_predProba, shap_values_test, prob_min=0.5, prob_max=1.00,
                                 top_n_features=20,output_dir=results_directory)

    feature_importance = np.abs(shap_values_test).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': feature_importance
    })

    # 1. Identifier les erreurs les plus confiantes
    errors = results_df[results_df['true_label'] != results_df['predicted_label']]
    confident_errors = errors.sort_values('prediction_probability', ascending=False)

    # 2. Récupérer les features importantes à partir de l'analyse SHAP
    important_features = feature_importance_df['feature'].head(10).tolist()

    print("Visualisation des erreurs confiantes:")
    plot_confident_errors(
        shap_values_test,
        confident_errors=confident_errors,
        X_test=X_test,
        feature_names=feature_names,
        n=5,results_directory=results_directory)
    # plot_confident_errors(xgb_classifier, confident_errors, X_test, X_test.columns,explainer_Test)

    # Exécution des analyses
    print("\nAnalyse des erreurs confiantes:")

    analyze_confident_errors(shap_values_test, confident_errors=confident_errors, X_test=X_test,
                             feature_names=feature_names, important_features=important_features, n=5)
    correct_predictions = results_df[results_df['true_label'] == results_df['predicted_label']]
    print("\nComparaison des erreurs vs prédictions correctes:")
    compare_errors_vs_correct(confident_errors.head(30), correct_predictions, X_test, important_features,
                              results_directory)
    print("\nAnalyse SHAP terminée. Les visualisations ont été sauvegardées.")
    print("\nAnalyse terminée. Les visualisations ont été sauvegardées.")
    print_notification('###### FIN: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########', color="blue")
    ###### FIN: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########

    ###### DEBUT: CALCUL DES CORRELACTION ##########


    def analyze_and_save_feature_correlations(X_train, results_directory, threshold=0.8):
        """
        Analyzes feature correlations in a given DataFrame, visualizes the correlation matrix,
        identifies pairs of highly correlated features, and saves the heatmap to a file.

        Parameters:
        - X_train (pd.DataFrame): The feature DataFrame.
        - results_directory (str): The directory where the correlation heatmap will be saved.
        - threshold (float): The correlation threshold for identifying highly correlated feature pairs.

        Returns:
        - high_corr_pairs (list of tuples): Sorted list of tuples with highly correlated feature pairs and their correlation values.
        """
        # Calculate the correlation matrix
        corr_matrix = X_train.corr()

        # Visualize the correlation matrix with smaller font size and two decimal precision
        plt.figure(figsize=(20, 20))
        sns.heatmap(corr_matrix, annot=True, fmt='.1f', cmap='coolwarm', annot_kws={"size": 4})
        plt.title('Feature Correlation Matrix', fontsize=10)

        # Ensure the results directory exists
        os.makedirs(results_directory, exist_ok=True)

        # Save the heatmap
        heatmap_path = os.path.join(results_directory, 'feature_correlation_matrix.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Identify pairs of highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    feature_i = corr_matrix.columns[i]
                    feature_j = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]
                    high_corr_pairs.append((feature_i, feature_j, corr_value))

        # Sort the results by the absolute correlation value in descending order
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        # Display the sorted results in the console
        print(f"Highly correlated pairs (threshold={threshold}):")
        for feature_i, feature_j, corr_value in high_corr_pairs:
            print(f"{feature_i} <-> {feature_j}: {corr_value:.2f}")

        return high_corr_pairs

    threshold = 0.75  # Set correlation threshold

    # Call the function
    high_corr_pairs = analyze_and_save_feature_correlations(X_train, results_directory, threshold)
    # Si high_corr_pairs est une liste de tuples (feature_i, feature_j, corr_value)
    save_correlations_to_csv(
        high_corr_pairs=high_corr_pairs,
        results_directory=results_directory,
        threshold=0.75  # Seuil de corrélation utilisé
    )
    ###### FIN: CALCUL DES CORRELACTION ##########


    ###### DEBUT: CALCUL DES VALEURS D'INTERACTION SHAP ##########
    print_notification("###### DEBUT: CALCUL DES VALEURS D'INTERACTION SHAP ##########", color="blue")
    # Calcul des valeurs d'interaction SHAP
    shap_interaction_values = final_model.predict(xgb.DMatrix(X_test), pred_interactions=True)
    # Exclure le biais en supprimant la dernière ligne et la dernière colonne
    shap_interaction_values = shap_interaction_values[:, :-1, :-1]

    # Vérification de la compatibilité des dimensions
    print("Shape of shap_interaction_values:", shap_interaction_values.shape)
    print("Number of features in X_test:", len(X_test.columns))

    if shap_interaction_values.shape[1:] != (len(X_test.columns), len(X_test.columns)):
        print("Erreur : Incompatibilité entre les dimensions des valeurs d'interaction SHAP et le nombre de features.")
        print(f"Dimensions des valeurs d'interaction SHAP : {shap_interaction_values.shape}")
        print(f"Nombre de features dans X_test : {len(X_test.columns)}")

        # Afficher les features de X_test
        print("Features de X_test:")
        print(list(X_test.columns))

        # Tenter d'accéder aux features du modèle
        try:
            model_features = final_model.feature_names
            print("Features du modèle:")
            print(model_features)

            # Comparer les features
            x_test_features = set(X_test.columns)
            model_features_set = set(model_features)

            missing_features = x_test_features - model_features_set
            extra_features = model_features_set - x_test_features

            if missing_features:
                print("Features manquantes dans le modèle:", missing_features)
            if extra_features:
                print("Features supplémentaires dans le modèle:", extra_features)
        except AttributeError:
            print("Impossible d'accéder aux noms des features du modèle.")
            print("Type du modèle:", type(final_model))
            print("Attributs disponibles:", dir(final_model))

        print("Le calcul des interactions SHAP est abandonné.")
        return  # ou sys.exit(1) si vous voulez quitter le programme entièrement

    # Si les dimensions sont compatibles, continuez avec le reste du code
    interaction_matrix = np.abs(shap_interaction_values).sum(axis=0)
    feature_names = X_test.columns
    interaction_df = pd.DataFrame(interaction_matrix, index=feature_names, columns=feature_names)

    # Masquer la diagonale (interactions d'une feature avec elle-même)
    np.fill_diagonal(interaction_df.values, 0)

    # Sélection des top N interactions (par exemple, top 10)
    N = 80
    top_interactions = interaction_df.unstack().sort_values(ascending=False).head(N)

    # Visualisation des top interactions
    plt.figure(figsize=(24, 16))
    top_interactions.plot(kind='bar')
    plt.title(f"Top {N} Feature Interactions (SHAP Interaction Values)", fontsize=16)
    plt.xlabel("Feature Pairs", fontsize=12)
    plt.ylabel("Total Interaction Strength", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, 'top_feature_interactions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Initialisation
    seen_pairs = set()
    csv_filename = f"Top_{N // 2}_interaction.csv"
    csv_path = os.path.join(results_directory, csv_filename)

    # Affichage du titre et préparation des données CSV
    print(f"Top {N // 2} Feature Interactions:")
    csv_data = []

    # Première boucle pour l'affichage console et préparation CSV
    for (f1, f2), value in top_interactions.items():
        # Créer une paire triée pour garantir que (A,B) et (B,A) sont considérées comme identiques
        pair = tuple(sorted([f1, f2]))

        # Si la paire n'a pas encore été vue et ce n'est pas une interaction avec soi-même
        if pair not in seen_pairs and f1 != f2:
            # Affichage dans la console
            print(f"{f1} <-> {f2}: {value:.4f}")

            # Préparation données CSV
            interaction = f"{f1} <-> {f2}"
            formatted_value = f"{value:.4f}"
            csv_data.append([interaction, formatted_value])

            seen_pairs.add(pair)

    # Sauvegarde du CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Interaction', 'Value'])
        writer.writerows(csv_data)

    print(f"\nCSV file saved at: {csv_path}")

    # Création de la heatmap
    top_features = interaction_df.sum().sort_values(ascending=False).head(N).index
    plt.figure(figsize=(26, 20))  # Augmenter la taille de la figure

    # Créer la heatmap avec des paramètres ajustés
    sns.heatmap(interaction_df.loc[top_features, top_features].round(0).astype(int),
                annot=True,
                cmap='coolwarm',
                fmt='d',  # Afficher les valeurs comme des entiers
                annot_kws={'size': 7},  # Réduire la taille de la police des annotations
                square=True,  # Assurer que les cellules sont carrées
                linewidths=0.5,  # Ajouter des lignes entre les cellules
                cbar_kws={'shrink': .8})  # Ajuster la taille de la barre de couleur

    plt.title(f"SHAP Interaction Values for Top {N} Features", fontsize=16)
    plt.tight_layout()
    plt.xticks(rotation=90, ha='center')  # Rotation verticale des labels de l'axe x
    plt.yticks(rotation=0)  # S'assurer que les labels de l'axe y sont horizontaux

    # Sauvegarde de la heatmap
    plt.savefig(
        os.path.join(results_directory, 'feature_interaction_heatmap.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

    print("Graphique d'interaction sauvegardé sous 'feature_interaction_heatmap.png'")
    print_notification("###### FIN: CALCUL DES VALEURS D'INTERACTION SHAP ##########", color="blue")
    ###### FIN: CALCUL DES VALEURS D'INTERACTION SHAP ##########

    return True


from sklearn.metrics import make_scorer
import numpy as np


def custom_profit_scorer(y_true, y_pred_proba, metric_dict=None, normalize=False):
    """
    Adaptation de custom_metric_Profit pour sklearn

    Args:
        y_true: vraies valeurs
        y_pred_proba: probabilités prédites
        metric_dict: dictionnaire des paramètres
        normalize: normalisation du profit
    """
    if metric_dict is None:
        metric_dict = {}

    CHECK_THRESHOLD = 0.55555555
    threshold = metric_dict.get('threshold', CHECK_THRESHOLD)
    if threshold == CHECK_THRESHOLD:
        raise ValueError("Invalid threshold value detected")
    # Suppression de l'application de la sigmoïde
    # y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
    # y_pred_proba = np.clip(y_pred_proba, 0.0, 1.0)

    # Vérification des valeurs
    min_val = np.min(y_pred_proba)
    max_val = np.max(y_pred_proba)
    if min_val < 0 or max_val > 1:
        #return float('-inf')  # Retourne une valeur très basse en cas de problème
        raise ValueError(f"Probabilities out of bounds: min={min_val}, max={max_val}")

    # Conversion en prédictions binaires
    y_pred = (y_pred_proba > threshold).astype(int)

    # Calcul des métriques
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Calcul du profit
    profit_per_tp = metric_dict.get('profit_per_tp', 1.0)
    loss_per_fp = metric_dict.get('loss_per_fp', -1.1)
    penalty_per_fn = metric_dict.get('penalty_per_fn', -0.1)

    total_profit = (tp * profit_per_tp +
                    fp * loss_per_fp +
                    fn * penalty_per_fn)

    if normalize:
        total_trades = tp + fp
        if total_trades > 0:
            return total_profit / total_trades
        return 0.0

    return total_profit


import numpy as np
from numba import jit


@jit(nopython=True)
def create_mask_numba(timestamps, periods_starts, periods_ends):
    """
    Crée un masque avec Numba pour les périodes sélectionnées
    """
    mask = np.zeros(len(timestamps), dtype=np.bool_)
    for i in range(len(timestamps)):
        for j in range(len(periods_starts)):
            if periods_starts[j] <= timestamps[i] < periods_ends[j]:
                mask[i] = True
                break
    return mask


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

"""
ANALYSE DES DONNÉES DE TRADING
Format timestamp spécifique : 0-1380
0-239.999 : 22h-23h59.999
240-1380 : 00h-20h59.999
"""

def analyze_trade_distribution(df_original, df_filtered, time_periods_dict):
    """
    Analyse améliorée de la distribution temporelle des trades
    """
    # Configuration du style
    plt.style.use('default')

    # Création de la figure
    fig = plt.figure(figsize=(20, 12), facecolor='white')
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    def prepare_data(data_df):
        df_copy = data_df.copy()
        # Regroupement par intervalles de 10 minutes
        df_copy['time_bin'] = (df_copy['deltaTimestampOpeningSection1min'] // 10).astype(int) * 10
        df_grouped = df_copy.groupby('time_bin')['class_binaire'].value_counts().unstack(fill_value=0)

        # S'assurer que l'index est complet de 0 à 1380 par pas de 10
        full_index = pd.Index(range(0, 1381, 10))
        df_grouped = df_grouped.reindex(full_index, fill_value=0)

        # Extraction des succès et échecs
        successes = df_grouped.get(1, pd.Series(0, index=df_grouped.index))
        failures = df_grouped.get(0, pd.Series(0, index=df_grouped.index))

        return df_grouped, successes, failures

    def format_time_label(minutes):
        """Convertit les minutes depuis 22:00 en format heure:minute"""
        total_minutes = (minutes + 22 * 60) % (24 * 60)
        hour = total_minutes // 60
        minute = total_minutes % 60
        return f"{int(hour):02d}:{int(minute):02d}"

    # Préparation des données pour les deux graphiques
    df_trades_original = df_original[df_original['class_binaire'].isin([0, 1])].copy()
    df_grouped_original, successes_original, failures_original = prepare_data(df_trades_original)

    df_trades_filtered = df_filtered[df_filtered['class_binaire'].isin([0, 1])].copy()
    df_grouped_filtered, successes_filtered, failures_filtered = prepare_data(df_trades_filtered)

    # Positions x pour les barres
    x_positions = range(len(df_grouped_original))
    width = 0.8

    # 1. Premier graphique - Tous les trades
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(x_positions, successes_original, width, color='#2ecc71', alpha=0.7)  # Supprimé label='Succès'
    ax1.bar(x_positions, failures_original, width, bottom=successes_original, color='#e74c3c',
            alpha=0.7)  # Supprimé label='Échecs'
    ax1.set_title('Distribution horaire des trades', fontsize=14, pad=15)
    ax1.set_ylabel('Nombre de trades', fontsize=12)
    # Supprimé : ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=12)

    # 2. Deuxième graphique - Trades filtrés
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(x_positions, successes_filtered, width, color='#2ecc71', alpha=0.7)  # Supprimé label='Succès'
    ax2.bar(x_positions, failures_filtered, width, bottom=successes_filtered, color='#e74c3c',
            alpha=0.7)  # Supprimé label='Échecs'
    ax2.set_title('Distribution des trades - Périodes sélectionnées uniquement', fontsize=14, pad=15)
    ax2.set_ylabel('Nombre de trades', fontsize=12)
    # Supprimé : ax2.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=12)

    # Configuration commune des axes
    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.5)

        # Labels d'axe toutes les 30 minutes (3 bins de 10 minutes)
        xticks_pos = list(range(0, len(df_grouped_original), 3))
        xticks_labels = [format_time_label(df_grouped_original.index[pos]) for pos in xticks_pos]
        ax.set_xticks(xticks_pos)
        ax.set_xticklabels(xticks_labels, rotation=90, fontsize=8)
        ax.set_xlim(-0.5, len(df_grouped_original) - 0.5)

        # Ligne de minuit
        #midnight_idx = 120 // 10  # 120 minutes après 22h, divisé par la taille du bin (10 minutes)
        #ax.axvline(x=midnight_idx, color='red', linestyle='--', alpha=0.7)

        ymin, ymax = ax.get_ylim()

        # Coloration des zones (toutes les périodes, qu'elles soient sélectionnées ou non)
        for name, info in time_periods_dict.items():
            # Les indices correspondent aux bins de 10 minutes
            start_idx = info['start'] // 10
            end_idx = info['end'] // 10

            color = f"#{format(hash(name) % 0xffffff, '06x')}"
            alpha = 0.45 if info['selected'] else 0.25  # Alpha différent selon 'selected'

            # Coloration de la zone
            ax.axvspan(start_idx, end_idx, alpha=alpha, color=color)

            # Label centré sur la zone
            center = start_idx + (end_idx - start_idx) / 2
            ax.text(center, ymax * 0.9,
                    f"{name}\n{format_time_label(info['start'])}-{format_time_label(info['end'])}",
                    rotation=90, va='center', ha='center', fontsize=8, color='black')

    # Calcul des statistiques
    total_success = successes_filtered.sum()
    total_failure = failures_filtered.sum()
    total_all = total_success + total_failure

    # Texte des statistiques
    if total_all > 0:
        stats = []
        global_win_rate = (total_success / total_all * 100)

        for name, info in time_periods_dict.items():
            if info['selected']:
                start_idx = info['start'] // 10
                end_idx = info['end'] // 10

                period_successes = successes_filtered.iloc[start_idx:end_idx].sum()
                period_failures = failures_filtered.iloc[start_idx:end_idx].sum()
                period_total = period_successes + period_failures

                if period_total > 0:
                    period_win_rate = (period_successes / period_total * 100)
                    stats.append(
                        f"{name} ({format_time_label(info['start'])}-{format_time_label(info['end'])}):\n"
                        f"  Trades: {period_total} ({period_total / total_all * 100:.1f}%)\n"
                        f"  Win Rate: {period_win_rate:.1f}%"
                    )

        stats_text = (
                f"Statistiques Globales:\n"
                f"Total Trades: {total_all}\n"
                f"Succès: {total_success} ({total_success / total_all * 100:.1f}%)\n"
                f"Échecs: {total_failure} ({total_failure / total_all * 100:.1f}%)\n"
                f"Win Rate Global: {global_win_rate:.1f}%\n\n"
                + "\n\n".join(stats)
        )
    else:
        stats_text = "Aucun trade dans la période sélectionnée"

    # Affichage des statistiques
    plt.figtext(1.02, 0.6, stats_text, fontsize=12,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, pad=10))

    plt.suptitle('Analyse de la Distribution des Trades sur 24H', fontsize=16, y=0.97)
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.2, hspace=0.4)

    return {
        'total_trades': total_all,
        'successes': total_success,
        'failures': total_failure,
        'win_rate': global_win_rate,
        'hourly_stats': {
            'timestamps': df_grouped_filtered.index.tolist(),
            'volumes': (successes_filtered + failures_filtered).tolist()
        }
    }




def sessions_selection(df, selected_sessions=None, custom_sections=None, time_periods_dict=None):
    """
    Met class_binaire à 99 pour les sessions NON sélectionnées et analyse la distribution.
    """
    df_filtered = df.copy()
    timestamps = df['deltaTimestampOpening'].values

    if time_periods_dict is not None:
        # Sélection des périodes activées
        selected_periods = [
            (info['start'], info['end'])
            for info in time_periods_dict.values()
            if info['selected']
        ]

        periods_starts = np.array([start for start, end in selected_periods], dtype=np.float64)
        periods_ends = np.array([end for start, end in selected_periods], dtype=np.float64)

        # Logging et calcul des statistiques par période
        cum_total_count = 0
        print("\nUtilisation des périodes du dictionnaire:")

        for name, info in time_periods_dict.items():
            status = "activée" if info['selected'] else "désactivée"
            start_time = format_time(info['start'])
            end_time = format_time(info['end'])
            print(f"- {name}: {info['start']} à {info['end']} ({start_time} - {end_time}) ({status})")

            if info['selected']:
                # Calcul des statistiques pour cette période
                mask_period = (timestamps >= info['start']) & (timestamps < info['end'])
                period_data = df[mask_period]

                success_count = len(period_data[period_data['class_binaire'] == 1])
                failure_count = len(period_data[period_data['class_binaire'] == 0])
                total_count = success_count + failure_count
                win_rate = (success_count / total_count * 100) if total_count > 0 else 0
                cum_total_count += total_count

                print(f"  Statistiques pour {name}:")
                print(f"  - Minutes depuis 22h: {info['start']} - {info['end']}")
                print(f"  - Heures: {start_time} - {end_time}")
                print(f"  - Trades réussis: {success_count}")
                print(f"  - Trades échoués: {failure_count}")
                print(f"  - Total trades: {total_count}")
                print(f"  - Win Rate: {win_rate:.2f}%\n")

    elif custom_sections is not None:
        valid_sections = [s for s in custom_sections if s['name'] in selected_sessions]
        periods_starts = np.array([s['start'] for s in valid_sections], dtype=np.float64)
        periods_ends = np.array([s['end'] for s in valid_sections], dtype=np.float64)

        print("\nPériodes sélectionnées:")
        for section in valid_sections:
            start_time = format_time(section['start'])
            end_time = format_time(section['end'])
            print(f"- {section['name']}: {section['start']} à {section['end']} ({start_time} - {end_time})")
    else:
        return df_filtered

    print(f"\ncum_total_count des sessions = {cum_total_count}")

    # Créer le masque avec Numba et appliquer le filtrage
    mask_selected = create_mask_numba(timestamps, periods_starts, periods_ends)
    df_filtered.loc[~mask_selected, 'class_binaire'] = 99

    # Comptage des trades dans df_filtered
    success_filtered = len(df_filtered[df_filtered['class_binaire'] == 1])
    failure_filtered = len(df_filtered[df_filtered['class_binaire'] == 0])
    filtered_99 = len(df_filtered[df_filtered['class_binaire'] == 99])
    total_filtered = success_filtered + failure_filtered

    print("\nNombre de trades dans df_filtered:")
    print(f"df_filtered-> Trades réussis (1): {success_filtered}")
    print(f"df_filtered-> Trades échoués (0): {failure_filtered}")
    print(f"df_filtered-> Trades filtrés (99): {filtered_99}")
    print(f"df_filtered-> Total trades: {total_filtered}")

    # Vérification de cohérence
    if total_filtered != cum_total_count:
        print("\nATTENTION: Incohérence détectée!")
        print(f"Total des trades filtrés ({total_filtered}) != ")
        print(f"Somme des trades par période ({cum_total_count})")

    # Analyser et visualiser la distribution
    stats = analyze_trade_distribution(df, df_filtered, time_periods_dict)

    print("\nRésumé des statistiques:")
    print(f"Total trades analysés: {stats['total_trades']}")
    print(f"Win Rate global: {stats['win_rate']:.1f}%")

    plt.show()
    exit(0)
    return df_filtered


def format_time(minutes):
    """Convertit les minutes depuis 22:00 en format heure:minute"""
    total_minutes = (minutes + 22 * 60) % (24 * 60)
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{int(hour):02d}:{int(minute):02d}"


def calculate_normalized_objectives(
        tp_train_list, fp_train_list, tp_val_list, fp_val_list,
        scores_train_list, scores_val_list, fold_stats,
        scale_objectives=False,
        use_imbalance_penalty=True
):
    # Vérification que toutes les listes ont la même longueur
    lengths = [len(tp_train_list), len(fp_train_list),
               len(tp_val_list), len(fp_val_list),
               len(scores_train_list), len(scores_val_list)]

    if not all(length == lengths[0] for length in lengths):
        print("ERREUR: Les listes n'ont pas toutes la même longueur!")
        return {
            'pnl_norm_objective': float('-inf'),
            'winrate_diff_norm_objective': float('inf')
        }

    if not lengths[0]:  # Si les listes sont vides
        print("ERREUR: Les listes sont vides!")
        return {
            'pnl_norm_objective': float('-inf'),
            'winrate_diff_norm_objective': float('inf')
        }

    fold_metrics = []

    # Calcul des métriques pour chaque fold
    for i in range(len(scores_train_list)):
        train_trades = tp_train_list[i] + fp_train_list[i]
        val_trades = tp_val_list[i] + fp_val_list[i]

        # Protection contre division par zéro
        train_trades = max(train_trades, 1e-8)
        val_trades = max(val_trades, 1e-8)

        # Winrate pour chaque fold
        train_winrate = tp_train_list[i] / train_trades
        val_winrate = tp_val_list[i] / val_trades

        fold_metrics.append({
            'winrate_diff': abs(train_winrate - val_winrate),
            'n_trades': val_trades,
            'val_pnl': scores_val_list[i]
        })

    # Calcul de la pénalité de déséquilibre
    max_trades = max(m['n_trades'] for m in fold_metrics)
    min_trades = max(min(m['n_trades'] for m in fold_metrics), 1e-8)
    imbalance_penalty = 1 + math.log(max_trades / min_trades)

    print(f"\nImbalance statistics:")
    print(f"Max trades: {max_trades}")
    print(f"Min trades: {min_trades}")
    print(f"Imbalance penalty: {imbalance_penalty:.4f}")

    # Calcul pondéré des objectifs
    total_pnl = 0
    total_winrate_diff = 0
    total_weight = 0

    for metrics in fold_metrics:
        weight = 1  # Poids égal pour chaque fold
        total_pnl += metrics['val_pnl'] * weight
        total_winrate_diff += metrics['winrate_diff'] * weight
        total_weight += weight

    # Normalisation des objectifs
    avg_pnl = total_pnl / total_weight if total_weight > 0 else float('-inf')
    avg_winrate_diff = total_winrate_diff / total_weight if total_weight > 0 else float('inf')

    # Application de la pénalité si activée
    if use_imbalance_penalty:
        pnl_norm_objective = avg_pnl * (1 / imbalance_penalty)
        winrate_diff_norm_objective = avg_winrate_diff * imbalance_penalty
    else:
        pnl_norm_objective = avg_pnl
        winrate_diff_norm_objective = avg_winrate_diff

    # Scale objectives si demandé
    if scale_objectives:
        # Définir les plages de normalisation
        pnl_range = (-100, 100)  # À ajuster selon vos besoins
        winrate_diff_range = (0, 1)

        pnl_norm_objective = normalize_to_range(
            pnl_norm_objective,
            old_min=pnl_range[0],
            old_max=pnl_range[1]
        )

        winrate_diff_norm_objective = normalize_to_range(
            winrate_diff_norm_objective,
            old_min=winrate_diff_range[0],
            old_max=winrate_diff_range[1]
        )



    return {
        'pnl_norm_objective': pnl_norm_objective,
        'winrate_diff_norm_objective': winrate_diff_norm_objective,
        'raw_metrics': {
            'avg_pnl': avg_pnl,
            'avg_winrate_diff': avg_winrate_diff,
            'imbalance_penalty': imbalance_penalty
        }
    }


def normalize_to_range(value, old_min, old_max, new_min=0, new_max=1):
    """
    Normalise une valeur d'une plage à une autre
    """
    if old_max == old_min:
        return new_min

    value = max(min(value, old_max), old_min)
    normalized = (value - old_min) / (old_max - old_min)
    scaled = normalized * (new_max - new_min) + new_min

    return scaled



def process_RFE_filteringg(params=None, trial=None, weight_param=None, metric_dict=None,selected_columns=None, n_features_to_select=None, X_train=None,y_train_label=None):
    # Configuration de RFECV
    # Configuration de RFECV
    params4RFECV = params.copy()
    params4RFECV['device'] = 'cpu'
    params4RFECV['tree_method'] = 'auto'
    # Ajouter early_stopping_rounds dans les paramètres
    params4RFECV['enable_categorical'] = False
    # Configuration des paramètres (reste inchangé)
    w_p = trial.suggest_float('w_p', weight_param['w_p']['min'], weight_param['w_p']['max'])
    w_n = trial.suggest_float('w_n', weight_param['w_n']['min'], weight_param['w_n']['max'])

    def custom_profit_scorer(y_true, y_pred_proba, metric_dict, normalize=False):
        """
        Calcule le profit en fonction des prédictions de probabilités.

        Args:
            y_true: Labels réels
            y_pred_proba: Probabilités prédites (utilise la colonne 1 pour les prédictions positives)
            metric_dict: Dictionnaire contenant threshold, profit_per_tp, etc.
            normalize: Si True, normalise le profit par le nombre d'échantillons
        """
        # S'assurer que y_pred_proba est un tableau 2D
        if len(y_pred_proba.shape) == 1:
            y_pred_proba = y_pred_proba.reshape(-1, 1)
            y_pred_proba = np.column_stack((1 - y_pred_proba, y_pred_proba))

        # Prendre les probabilités de la classe positive
        positive_probs = y_pred_proba[:, 1]

        # Convertir en prédictions binaires selon le seuil
        y_pred = (positive_probs >= metric_dict['threshold']).astype(int)

        # Calculer les vrais positifs et faux positifs
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        # Calculer le profit
        profit = (tp * metric_dict['profit_per_tp']) - (fp * metric_dict['loss_per_fp'])

        if normalize:
            profit = profit / len(y_true)

        return profit
        # Vérifier que metric_dict est défini

    if metric_dict is None:
        raise ValueError("metric_dict ne peut pas être None")

    required_keys = ['profit_per_tp', 'loss_per_fp', 'threshold']
    missing_keys = [key for key in required_keys if key not in metric_dict]
    if missing_keys:
        raise ValueError(f"metric_dict doit contenir les clés suivantes: {missing_keys}")

    # Define the custom gradient and hessian functions
    def weighted_logistic_gradient_cpu_4classifier(predt: np.ndarray, y: np.ndarray, w_p: float,
                                                   w_n: float) -> np.ndarray:
        """Calculate the gradient for weighted logistic loss (CPU)."""
        predt = 1.0 / (1.0 + np.exp(-predt))  # Sigmoid function
        weights = np.where(y == 1, w_p, w_n)
        grad = weights * (predt - y)
        return grad

    def weighted_logistic_hessian_cpu_4classifier(predt: np.ndarray, y: np.ndarray, w_p: float,
                                                  w_n: float) -> np.ndarray:
        """Calculate the hessian for weighted logistic loss (CPU)."""
        predt = 1.0 / (1.0 + np.exp(-predt))
        weights = np.where(y == 1, w_p, w_n)
        hess = weights * predt * (1.0 - predt)
        return hess

    def weighted_logistic_hessian_cpu_4classifier(predt: np.ndarray, y: np.ndarray, w_p: float,
                                                  w_n: float) -> np.ndarray:
        """Calculate the hessian for weighted logistic loss (CPU)."""
        predt = 1.0 / (1.0 + np.exp(-predt))  # Sigmoid function
        weights = np.where(y == 1, w_p, w_n)
        hess = weights * predt * (1.0 - predt)
        return hess

    def weighted_logistic_obj_cpu_4classifier(y_true: np.ndarray, y_pred: np.ndarray, w_p: float, w_n: float) -> \
            Tuple[
                np.ndarray, np.ndarray]:
        """Custom objective function for weighted logistic loss."""
        predt = y_pred
        y = y_true
        grad = weighted_logistic_gradient_cpu_4classifier(predt, y, w_p, w_n)
        hess = weighted_logistic_hessian_cpu_4classifier(predt, y, w_p, w_n)
        return grad, hess

    # Create a partial function with fixed weights
    obj_function = partial(
        weighted_logistic_obj_cpu_4classifier,
        w_p=w_p,
        w_n=w_n
    )

    params4RFECV['objective'] = obj_function
    params4RFECV['disable_default_eval_metric'] = 1
    # Création du scorer personnalisé
    custom_scorer_partial = partial(
        custom_profit_scorer,
        metric_dict=metric_dict,
        normalize=False
    )

    custom_scorer = make_scorer(
        custom_scorer_partial,
        response_method="predict_proba",
        greater_is_better=True
    )

    # XGBoost wrapper class
    class XGBWrapper(xgb.XGBClassifier):
        """Wrapper for XGBoost with early stopping integrated into the constructor."""

        def fit(self, X, y, **kwargs):
            eval_set = kwargs.pop('eval_set', None)
            return super().fit(
                X, y,
                eval_set=eval_set,
                verbose=False,
                **kwargs
            )

    # Configuration et exécution de RFECV
    model4RFECV = XGBWrapper(**params4RFECV, n_jobs=1)
    cv_inner = KFold(n_splits=5, shuffle=False)

    rfecv = RFECV(
        estimator=model4RFECV,
        cv=cv_inner,
        step=1,
        scoring=custom_scorer,
        n_jobs=-1,  # Utilisation de tous les cœurs pour accélérer le processus
        min_features_to_select=5  # Nombre minimal de caractéristiques à sélectionner
    )

    def get_optimal_n_features(cv_results):
        """Retourne le nombre optimal de caractéristiques basé sur les scores moyens et ajustés."""
        mean_scores = cv_results['mean_test_score']
        std_scores = cv_results['std_test_score']

        # Calculer le score ajusté (mean - std)
        adjusted_scores = mean_scores - 2 * std_scores

        # Option 1 : Choisir le nombre de caractéristiques avec le score moyen maximal
        optimal_n_features_mean = np.argmax(mean_scores) + 1
        print(f"Nombre optimal de features basé sur le meilleur score moyen : {optimal_n_features_mean}")
        print(f"Score moyen optimal : {mean_scores.max():.2f}")

        # Option 2 : Choisir le nombre de caractéristiques avec le meilleur score ajusté (mean - std)
        optimal_n_features_adjusted = np.argmax(adjusted_scores) + 1
        print(f"Nombre optimal de features basé sur mean-std : {optimal_n_features_adjusted}")
        print(f"Score ajusté optimal : {adjusted_scores.max():.2f}")

        # Créer un DataFrame pour visualiser tous les scores
        scores_df = pd.DataFrame({
            'n_features': range(1, len(mean_scores) + 1),
            'mean_score': mean_scores,
            'std_score': std_scores,
            'adjusted_score': adjusted_scores
        })

        # Afficher le tableau des scores
        print("\nScores détaillés :")
        print(scores_df)

        # Retourner le nombre optimal basé sur le score ajusté
        return optimal_n_features_adjusted

    # Fit RFECV
    rfecv.fit(X_train, y_train_label)

    print('model4RF2CV done')

    # Obtenir le nombre optimal de caractéristiques
    optimal_n_features = get_optimal_n_features(rfecv.cv_results_)
    if (optimal_n_features < n_features_to_select):
        try:
            def get_top_n_features(selected_columns, n_features_to_select, X_train):
                if len(selected_columns) < n_features_to_select:
                    raise ValueError(
                        f"La liste selected_columns contient seulement {len(selected_columns)} features, "
                        f"alors que {n_features_to_select} sont demandées.")

                # Créer un masque pour les features sélectionnées
                feature_mask = X_train.columns.isin(selected_columns[:n_features_to_select])

                # Obtenir les indices triés
                sorted_indices = np.where(feature_mask)[0]

                # Obtenir les noms des caractéristiques correspondantes
                selected_feature_names = X_train.columns[sorted_indices]

                return selected_feature_names

            selected_feature_names = get_top_n_features(selected_columns, n_features_to_select, X_train)
        except ValueError as e:
            print(f"Erreur: {str(e)}")
    else:
        # Obtenir les rangs des caractéristiques
        feature_rankings = rfecv.ranking_

        # Obtenir les indices des caractéristiques triés par rang
        sorted_indices = np.argsort(feature_rankings)

        # Sélectionner les indices des 'optimal_n_features' meilleures caractéristiques
        optimal_indices = sorted_indices[:optimal_n_features]

        # Obtenir les noms des caractéristiques correspondantes
        selected_feature_names = X_train.columns[optimal_indices]

    # Afficher les caractéristiques sélectionnées
    print(f"Caractéristiques correspondant à optimal_n_features ({selected_feature_names}):")
    print(selected_feature_names.tolist())

    print(f"Nombre de features sélectionnées: {len(selected_feature_names)}")
    print("\nFeatures sélectionnées:")
    for i, feature in enumerate(selected_feature_names, 1):
        print(f"{i}. {feature}")

    # Affichage des scores de cross-validation
    print("\nScores de cross-validation par nombre de features:")
    """
    cv_scores = pd.DataFrame({
        'n_features': range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
        'mean_score': rfecv.cv_results_['mean_test_score'],
        'std_score': rfecv.cv_results_['std_test_score']
    })
    print(cv_scores)
    """
    # Log des résultats
    return selected_feature_names;


def calculate_fold_stats(labels, set_name):
    # Calcul des décisions (trades)
    decisions = (labels != 99).sum()
    success = (labels == 1).sum()
    failures = (labels == 0).sum()
    success_rate = success / decisions if decisions > 0 else 0

    return {
        f"{set_name}_n_trades": decisions,
        f"{set_name}_n_class_1": success,
        f"{set_name}_n_class_0": failures,
        f"{set_name}_class_ratio": success_rate,
        f"{set_name}_success_rate": success_rate
    }



