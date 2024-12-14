import calendar
import seaborn as sns
from termcolor import colored
import cupy as cp
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, \
    average_precision_score, matthews_corrcoef,precision_recall_curve, precision_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

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
import logging
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import RFECV
import csv
import math
from functools import partial
from sklearn.model_selection import KFold, TimeSeriesSplit
import json
import shutil
from parameters import *
from definition import *


CUSTOM_SESSIONS = {
        "Opening": {
            "start": 0,
            "end": 120,
            "session_type_index": 0,
            "selected": False,
            "description": "22:00-23:05",
        },
        "Asie": {
            "start": 120,
            "end": 535,
            "session_type_index": 1,
            "selected": True,
            "description": "23:05-6:55",
        },
        "OpenUk": {
            "start": 535,
            "end": 545,
            "session_type_index": 2,
            "selected": False,
            "description": "6:55-07:05",
        },
        "preOpenEurope": {
            "start": 545,
            "end": 595,
            "session_type_index": 2,
            "selected": True,
            "description": "07:05-07:55",
        },
        "OpenEurope": {  # 8h = 480 minutes -> 475-485
            "start": 595,
            "end": 605,
            "session_type_index": 2,
            "selected": False,
            "description": "07:55-08:05",
        },
        "MorningEurope": {
            "start": 605,
            "end": 865,
            "session_type_index": 2,
            "selected": True,
            "description": "08:05-12:25",
        },
        "OpenpreOpenUS": {  # 12h30 = 750 minutes -> 745-755
            "start": 865,
            "end": 875,
            "session_type_index": 3,
            "selected": False,
            "description": "12:25-12:35",
        },
        "preOpenUS": {
            "start": 875,
            "end": 925,
            "session_type_index": 3,
            "selected": False,
            "description": "12:35-13:25",
        },
        "OpenMoringUS": {  # 13h30 = 810 minutes -> 805-815
            "start": 925,
            "end": 935,
            "session_type_index": 3,
            "selected": False,
            "description": "13:25-13:35",
        },
        "MoringUS": {
            "start": 935,
            "end": 1065,
            "session_type_index": 3,
            "selected": False,
            "description": "13:35-15:45",
        },
        "AfternonUS": {
            "start": 1065,
            "end": 1195,
            "session_type_index": 4,
            "selected": False,
            "description": "15:45-17:55",
        },
        "Evening": {
            "start": 1195,
            "end": 1280,
            "session_type_index": 5,
            "selected": False,
            "description": "17:55-19:20",
        },
        "Close": {
            "start": 1280,
            "end": 1380,
            "session_type_index": 6,
            "selected": False,
            "description": "19:20-21:00",
        }
    }

from functools import reduce
from numba import njit, prange
from dateutil.relativedelta import relativedelta
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Variable globale pour suivre si la fonction a déjà été appelée
_first_call_save_r_trialesults = True

def detect_environment():
    """
    Détecte l'environnement d'exécution
    Returns: 'colab', 'pycharm', ou 'other'
    """
    # Vérification pour Google Colab
    try:
        from google.colab import drive
        return 'colab'
    except ImportError:
        # Vérification pour PyCharm
        if 'PYCHARM_HOSTED' in os.environ or 'PYCHARM_MATPLOTLIB_PORT' in os.environ:
            return 'pycharm'
        return 'other'

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


def get_val_cv_time_range(X_full, X, index_val=None):
    """
    Obtient la plage temporelle en utilisant les index originaux.

    Args:
        X_full (pd.DataFrame): DataFrame complet avec toutes les colonnes incluant 'timeStampOpening'
        X (pd.DataFrame): DataFrame potentiellement réduit mais avec les mêmes indices que X_full
        index_val (array-like, optional): Indices de validation. Si None, utilise tout le DataFrame

    Returns:
        tuple: (start_time, end_time, num_sessions)
            - start_time (int): Timestamp du début de la période
            - end_time (int): Timestamp de la fin de la période
            - num_sessions (int): Nombre de sessions dans cette période

    Raises:
        ValueError: Si les DataFrames sont vides ou si 'timeStampOpening' n'existe pas
        KeyError: Si les index ne correspondent pas entre X_full et X
    """
    # Vérifications préliminaires
    if X_full.empty or X.empty:
        raise ValueError("Les DataFrames ne peuvent pas être vides")

    if 'timeStampOpening' not in X_full.columns:
        raise ValueError("La colonne 'timeStampOpening' doit être présente dans X_full")

    try:
        # Si index_val est None, utiliser tous les indices de X
        if index_val is None:
            original_indices = X.index
        else:
            # Vérifier que index_val est valide
            if len(index_val) == 0:
                raise ValueError("index_val ne peut pas être vide")
            # Récupérer les index originaux correspondant aux indices de validation
            original_indices = X.index[index_val]

        # Vérifier que les indices existent dans X_full
        if not all(idx in X_full.index for idx in [original_indices[0], original_indices[-1]]):
            raise KeyError("Certains indices ne sont pas présents dans X_full")

        # Utiliser ces index originaux pour accéder à X_full
        start_time = X_full.loc[original_indices[0], 'timeStampOpening']
        end_time = X_full.loc[original_indices[-1], 'timeStampOpening']

        # Vérifier l'ordre chronologique
        if end_time < start_time:
            start_time, end_time = end_time, start_time

        # Extraire les données en utilisant les index originaux
        df_extracted = X_full.loc[original_indices[0]:original_indices[-1]]

        # Vérifier que df_extracted n'est pas vide
        if df_extracted.empty:
            raise ValueError("Aucune donnée n'a été extraite pour la période spécifiée")

        num_sessions = calculate_and_display_sessions(df_extracted)[0]

        return start_time, end_time, num_sessions

    except Exception as e:
        raise ValueError(f"Erreur lors du traitement des données temporelles: {str(e)}")

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
                                  optimal_threshold=None, show_histogram=True, num_sessions=25,results_directory=None):
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


    plt.close()


from datetime import datetime, timedelta


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cupy as cp  # Assurez-vous que cupy est installé si vous utilisez des arrays CuPy

def to_numpy(array):
    return cp.asnumpy(array) if isinstance(array, cp.ndarray) else array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# import cupy as cp  # Décommentez si vous utilisez des tableaux CuPy

def to_numpy(array):
    """Convertit les tableaux CuPy en tableaux NumPy, si applicable."""
    try:
        import cupy as cp
        if isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
    except ImportError:
        pass
    return np.asarray(array)

def plot_fp_tp_rates(X_test, y_test_label, y_test_predProba, feature_deltaTime_name,
                     optimal_threshold, dataset_name, results_directory=None):
    """
    Trace les taux de FP et TP empilés en fonction du temps sur une période de 22h à 21h.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), facecolor='white')

    def format_time_label(minutes):
        total_minutes = minutes + (22 * 60)  # Ajouter 22h en minutes
        total_minutes = total_minutes % (24 * 60)  # Modulo pour revenir à 0 après 24h
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{int(hours):02d}:{int(minutes):02d}"

    def plot_rates(ax, n_bins):
        bins = np.linspace(0, 1380, n_bins + 1)

        # Convertir explicitement les tableaux en NumPy
        feature_values_np = to_numpy(X_test[feature_deltaTime_name].values)
        y_test_label_np = to_numpy(y_test_label)
        y_test_predProba_np = to_numpy(y_test_predProba)

        df = pd.DataFrame({
            'feature': feature_values_np,
            'y_true': y_test_label_np,
            'y_pred': y_test_predProba_np >= optimal_threshold
        })

        # Création des bins et calcul des taux
        df['bin'] = pd.cut(df['feature'], bins=bins, include_lowest=True)
        grouped = df.groupby('bin')

        rates = grouped.apply(lambda x: pd.Series({
            'FP_rate': ((x['y_pred'] == 1) & (x['y_true'] == 0)).sum() / len(x) if len(x) > 0 else 0,
            'TP_rate': ((x['y_pred'] == 1) & (x['y_true'] == 1)).sum() / len(x) if len(x) > 0 else 0
        }))

        # Récupérer les bords gauches des bins pour positionner les barres
        bin_edges = rates.index.categories.left
        bin_widths = rates.index.categories.right - rates.index.categories.left

        # Ajuster la largeur des barres pour un léger espacement
        bar_widths = bin_widths * 0.98

        # Tracer les barres empilées
        ax.bar(bin_edges, rates['TP_rate'], width=bar_widths, align='edge',
               color='green', label='Taux de Vrais Positifs', alpha=0.7)
        ax.bar(bin_edges, rates['FP_rate'], width=bar_widths, align='edge',
               bottom=rates['TP_rate'], color='red', label='Taux de Faux Positifs', alpha=0.7)

        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, 0.30)  # Limite de l'axe Y à 30%

        # Configuration des labels de l'axe X
        hour_marks = np.arange(0, 1381, 60)
        hour_labels = [format_time_label(m) for m in hour_marks]

        ax.set_xticks(hour_marks)
        ax.set_xticklabels(hour_labels, rotation=45, ha='right', fontsize=10)

        ax.set_ylabel('Taux', fontsize=12)
        ax.set_title(f'Taux de FP et TP par {feature_deltaTime_name} (bins={n_bins})', fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

    # Tracer les deux graphiques avec les différents nombres de bins
    plot_rates(ax1, 25)
    plot_rates(ax2, 75)

    # Ajustement de la mise en page
    plt.tight_layout()

    if results_directory:
        file_path = os.path.join(results_directory, f'fp_tp_rates_{dataset_name}_by_{feature_deltaTime_name}.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé sous : {file_path}")


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
    return tn, fp, fn, tp


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
    Tuple[numpy.ndarray, shap.Explanation, dict]
        - Les valeurs SHAP calculées
        - L'objet d'explication SHAP
        - Un dictionnaire contenant les résultats clés de l'analyse
    """
    # Création du répertoire principal de sauvegarde
    os.makedirs(save_dir, exist_ok=True)

    # Création du sous-répertoire pour les graphiques de dépendance
    dependence_dir = os.path.join(save_dir, 'dependence_plot')
    os.makedirs(dependence_dir, exist_ok=True)

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
        # Calcul de la moyenne des valeurs absolues des valeurs SHAP (importance globale)
        feature_importance = np.abs(shap_values).mean(axis=0)

        # Calcul de la moyenne des valeurs SHAP (effet moyen signé)
        feature_effect = np.mean(shap_values, axis=0)

        # Appliquer le signe de l'effet moyen à l'importance
        signed_feature_importance = feature_importance * np.sign(feature_effect)

        # Création du DataFrame avec les importances
        shap_df = pd.DataFrame({
            'feature': X.columns,
            'importance': signed_feature_importance,
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
        shap_df_save = shap_df.drop('abs_importance', axis=1)

        # Sauvegarde du CSV
        csv_path = os.path.join(save_dir, f'shap_values_{dataset_name}.csv')
        shap_df_save.to_csv(csv_path, index=False, sep=';')
        print(f"\nValeurs SHAP sauvegardées dans: {csv_path}")

        # Création du dictionnaire results avant la création des graphiques de dépendance
        results = {
            'feature_importance': shap_df_save,
            'top_10_features': shap_df['feature'].head(40).tolist(),
            'features_for_80_percent': shap_df[shap_df['cumulative_importance_percentage'] <= 80].shape[0]
        }

        print(f"\nTop 10 features basées sur l'analyse SHAP:")
        print(results['top_10_features'][:10])  # N'affiche que les 10 premiers
        print(
            f"\nNombre de features nécessaires pour expliquer 80% de l'importance : {results['features_for_80_percent']}")

    except Exception as e:
        print(f"Erreur lors du calcul des importances: {str(e)}")
        traceback.print_exc()
        raise

        # Création des graphiques de dépendance si demandé
    if create_dependence_plots:
        print("\nCréation des graphiques de dépendance...")
        most_important_features = shap_df['feature'].head(max_dependence_plots)

        for idx, feature in enumerate(most_important_features):
            try:
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(feature, shap_values, X, show=False)
                plt.title(f"SHAP Dependence Plot - {feature} - {dataset_name}")
                plt.tight_layout()

                # Ajout du préfixe numérique au nom du fichier
                dep_plot_path = os.path.join(dependence_dir, f'{idx}_shap_dependence_{feature}_{dataset_name}.png')
                plt.savefig(dep_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Graphique de dépendance créé pour {feature}")
            except Exception as e:
                print(f"Erreur lors de la création du graphique de dépendance pour {feature}: {str(e)}")
                traceback.print_exc()

    print(f"\nAnalyse SHAP terminée - Tous les résultats sont sauvegardés dans: {save_dir}")

    return shap_values, shap_values_explanation, results




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
    # Créer le répertoire principal de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    # Créer le sous-répertoire pour les analyses combinées
    combined_dir = os.path.join(save_dir, 'shap_combined')
    os.makedirs(combined_dir, exist_ok=True)

    # Vérifier que les colonnes sont identiques dans X_train et X_test
    if not all(X_train.columns == X_test.columns):
        raise ValueError("Les colonnes de X_train et X_test doivent être identiques.")

    # Calculer l'importance des features basée sur les valeurs SHAP absolues moyennes
    feature_importance = np.abs(shap_values_train).mean(0)
    top_features = X_train.columns[np.argsort(feature_importance)[-top_n:]]

    # Pour chaque feature importante
    for idx, feature in enumerate(reversed(top_features)):
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

        # 2. Distribution par classe sur train
        feature_idx = X_train.columns.get_loc(feature)

        # Calcul des statistiques
        mean_class_0 = shap_values_train[y_train_label == 0, feature_idx].mean()
        mean_class_1 = shap_values_train[y_train_label == 1, feature_idx].mean()
        count_class_0 = sum(y_train_label == 0)
        count_class_1 = sum(y_train_label == 1)

        # Histogramme à la place du KDE
        ax2.hist(shap_values_train[y_train_label == 0, feature_idx],
                 bins=50, alpha=0.5, color='blue', label='Classe 0', density=True)
        ax2.hist(shap_values_train[y_train_label == 1, feature_idx],
                 bins=50, alpha=0.5, color='red', label='Classe 1', density=True)

        # Ajout des statistiques sur le graphique
        stats_text = f"Classe 0 (n={count_class_0}):\nMoyenne SHAP = {mean_class_0:.6f}\n\n" \
                     f"Classe 1 (n={count_class_1}):\nMoyenne SHAP = {mean_class_1:.6f}"

        # Positionnement du texte dans le coin supérieur droit
        ax2.text(0.95, 0.95, stats_text,
                 transform=ax2.transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round,pad=0.5',
                           facecolor='white',
                           alpha=0.8,
                           edgecolor='gray'))

        ax2.set_title("Distribution par Classe suur x_train")
        ax2.set_xlabel("SHAP Value")
        ax2.set_ylabel("Density")
        ax2.legend()

        # 3. Scatter plot SHAP
        feature_idx = list(X_train.columns).index(feature)
        shap.plots.scatter(shap_explanation_train[:, feature_idx], ax=ax3, show=False)
        ax3.set_title("SHAP Scatter Plot sur x_train")

        # Ajout du titre global
        plt.suptitle(f"SHAP Analysis - {feature}", fontsize=16, y=1.05)
        plt.tight_layout()

        # Sauvegarde de la figure dans le sous-répertoire shap_combined
        save_path = os.path.join(combined_dir, f'{idx}_shap_combined_analysis_{feature}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Analyse combinée créée pour la feature: {feature} (rang {idx})")

    print(f"Les graphiques de distribution SHAP ont été sauvegardés dans {combined_dir}")

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
    shap_values_train,shap_explanation_train,resulat_train_shap_feature_importance = analyze_shap_values(final_model, X_train, y_train_label, "Training_Set",
                                            create_dependence_plots=True,
                                            max_dependence_plots=40, save_dir=save_dir)
    shap_values_test,shap_explanation_test, resulat_test_shap_feature_importance= analyze_shap_values(final_model, X_test, y_test_label, "Test_Set", create_dependence_plots=True,
                                           max_dependence_plots=40, save_dir=save_dir)

    # Comparaison des importances de features et des distributions SHAP
    importance_df = compare_feature_importance(shap_values_train, shap_values_test, X_train, X_test, save_dir=save_dir)
    compare_shap_distributions(
        shap_values_train=shap_values_train,shap_explanation_train=shap_explanation_train, shap_values_test=shap_values_test,shap_explanation_test=shap_explanation_test,
        X_train=X_train,y_train_label=y_train_label, X_test=X_test,y_test_label=y_test_label ,
        top_n=40, save_dir=save_dir)

    # Comparaison des valeurs SHAP moyennes
    shap_comparison = compare_mean_shap_values(
        shap_values_train=shap_values_train, shap_values_test=shap_values_test, X_train=X_train, save_dir=save_dir)

    return importance_df, shap_comparison, shap_values_train, shap_values_test,resulat_test_shap_feature_importance


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
    # Créer le sous-dossier 'distribution_features' dans output_dir
    distribution_dir = os.path.join(output_dir, 'distribution_features')
    os.makedirs(distribution_dir, exist_ok=True)

    # Générer et sauvegarder les graphiques dans le sous-dossier
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(X_test_top[feature], kde=True, label='Global')
        sns.histplot(selected_samples[feature], kde=True, label=f'Selected ({prob_min:.2f} - {prob_max:.2f})')
        plt.title(f"Distribution de {feature}")
        plt.legend()

        # Enregistrer dans le sous-dossier
        plt.savefig(os.path.join(distribution_dir, f'distribution_{feature}.png'))
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


def init_dataSet(df_init=None, nanvalue_to_newval=None, config=None, CUSTOM_SESSIONS_=None, results_directory=None):
    # Gestion des valeurs NaN
    selected_columns = config.get('selected_columns', None)

    df_filtered = sessions_selection(df_init, CUSTOM_SESSIONS_=CUSTOM_SESSIONS_,
                                   results_directory=results_directory)

    if nanvalue_to_newval is not None:
        df_filtered = df_filtered.fillna(nanvalue_to_newval).infer_objects(copy=False)
        nan_value = nanvalue_to_newval
    else:
        nan_value = np.nan

    print("Division des données en ensembles d'entraînement et de test...")
    try:
        test_size_ratio = config.get('test_size_ratio', 0.8)
        train_df, nb_SessionTrain, test_df, nb_SessionTest = split_sessions(df_filtered, test_size=test_size_ratio, min_train_sessions=2,
                                                                          min_test_sessions=2)
    except ValueError as e:
        print(f"Erreur lors de la division des sessions : {e}")
        sys.exit(1)

    print(f"Nombre de session dans x_train  {nb_SessionTrain}  ")
    print(f"Nombre de session dans x_test  {nb_SessionTest}  ")

    # On crée une copie de selected_columns pour ne pas modifier la liste originale
    columns_to_use = selected_columns.copy()

    # Création des versions "full" avec les colonnes sélectionnées
    X_train_full = train_df[:]
    y_train_full_label = train_df['class_binaire']
    X_test_full = test_df[:]
    y_test_full_label = test_df['class_binaire']

    # Création des versions filtrées (sans classe 99)
    mask_train = y_train_full_label != 99
    X_train = X_train_full[mask_train]
    X_train =X_train[columns_to_use]
    y_train_label = y_train_full_label[mask_train]

    mask_test = y_test_full_label != 99
    X_test = X_test_full[mask_test]
    X_test = X_test[columns_to_use]
    y_test_label = y_test_full_label[mask_test]

    # ===== Gestion des colonnes temporelles pour le logging et la validation croisée =====
    # Ces colonnes sont nécessaires pour certains modes de validation croisée
    # et pour le logging temporel des prédictions, mais peuvent être absentes de X_train_full
    # si elles ne sont pas utilisées comme features

    # Liste des colonnes temporelles nécessaires
    temporal_columns = ['timeStampOpening', 'SessionStartEnd', 'deltaTimestampOpening']

    for col in temporal_columns:
        if col not in X_train_full.columns:
            # Vérification de la présence de la colonne dans le DataFrame initial
            # df_init contient toutes les colonnes d'origine avant la sélection des features
            if col not in df_init.columns:
                raise ValueError(
                    f"La colonne '{col}' n'est pas présente dans df_init. Cette colonne est nécessaire pour les logs temporels et la validation croisée.")

            # Création d'une copie de X_train_full lors de la première modification
            # pour éviter les modifications involontaires sur le DataFrame d'origine
            if isinstance(X_train_full, pd.DataFrame) and not X_train_full.columns.empty:
                X_train_full = X_train_full.copy()

            # Ajout de la colonne en respectant l'indexation de X_train_full
            # On utilise .loc pour garantir un alignement correct des indices
            # entre df_init et X_train_full
            X_train_full[col] = df_init.loc[X_train_full.index, col]
            print(f"Colonne '{col}' ajoutée à X_train_full pour les logs temporels et la validation croisée")

    # ===== Vérification de la cohérence des distributions de classes =====
    # Compare les proportions de classes entre les différents ensembles de données
    # pour s'assurer qu'il n'y a pas d'incohérence dans la séparation des données

    def get_class_counts(y):
        """Retourne un tuple (count_0, count_1) pour un array/series donné"""
        if isinstance(y, pd.Series):
            return (y == 0).sum(), (y == 1).sum()
        return (y == 0).sum(), (y == 1).sum()

    # Vérification des ensembles complets
    counts_train_full = get_class_counts(y_train_full_label)
    counts_test_full = get_class_counts(y_test_full_label)

    # Vérification des ensembles réduits
    counts_train = get_class_counts(y_train_label)
    counts_test = get_class_counts(y_test_label)

    # Vérification de la cohérence des proportions
    if counts_train_full != counts_train or counts_test_full != counts_test:
        error_msg = f"""
    Incohérence détectée dans la distribution des classes:
    Train full    : {counts_train_full[0]} classe 0, {counts_train_full[1]} classe 1
    Train réduit  : {counts_train[0]} classe 0, {counts_train[1]} classe 1
    Test full     : {counts_test_full[0]} classe 0, {counts_test_full[1]} classe 1
    Test réduit   : {counts_test[0]} classe 0, {counts_test[1]} classe 1
    """
        raise ValueError(error_msg)

    return (X_train_full, y_train_full_label, X_test_full, y_test_full_label,
            X_train, y_train_label, X_test, y_test_label,
            nb_SessionTrain, nb_SessionTest, nan_value)

def add_early_stopping_zone(ax, best_iteration, color='orange', alpha=0.2):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.axvspan(best_iteration, xmax, facecolor=color, alpha=alpha)
    ax.text(best_iteration + (xmax - best_iteration) / 2, ymax, 'Zone post early stopping',
            horizontalalignment='center', verticalalignment='top', fontsize=12, color='orange')

def plot_custom_metric_evolution_with_trade_info(model, evals_result, metric_name='custom_metric_ProfitBased',
                                                 n_train_trades=None, n_test_trades=None, results_directory=None):
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

    plt.close()

def save_correlations_to_csv(corr_matrix, results_directory):
    csv_filename = "all_correlations.csv"
    csv_path = os.path.join(results_directory, csv_filename)

    # Initialisation
    seen_pairs = set()
    csv_data = []

    # Traitement de toutes les corrélations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            feature_i = corr_matrix.columns[i]
            feature_j = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]

            # Créer une paire triée pour éviter les doublons
            pair = tuple(sorted([feature_i, feature_j]))

            # Si la paire n'est pas encore vue et ce n'est pas une auto-corrélation
            if pair not in seen_pairs and feature_i != feature_j:
                # Préparation données CSV
                interaction = f"{feature_i} <-> {feature_j}"
                absolute_corr = abs(corr_value)
                csv_data.append([
                    interaction,
                    f"{corr_value:.4f}",
                    f"{absolute_corr:.4f}"
                ])
                seen_pairs.add(pair)

    # Tri par corrélation absolue décroissante
    csv_data.sort(key=lambda x: float(x[2]), reverse=True)

    # Sauvegarde du CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Correlation_Pair', 'Correlation_Value', 'Absolute_Correlation'])
        writer.writerows(csv_data)

    print(f"\nFichier CSV sauvegardé: {csv_path}")
    return csv_path
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

    csv_path = save_correlations_to_csv(corr_matrix, results_directory)

    return high_corr_pairs




def analyze_shap_interactions(final_model, X_test, results_directory, dataset_name="test"):
    """
    Analyse les interactions SHAP et génère des graphiques ainsi qu'un CSV.

    Parameters:
    -----------
    final_model : xgboost.Booster
        Modèle XGBoost entraîné.
    X_test : pandas.DataFrame
        Données pour lesquelles les interactions SHAP doivent être calculées.
    results_directory : str
        Répertoire où sauvegarder les résultats.
    dataset_name : str, optional (default="test")
        Nom du dataset (par exemple, "train" ou "test") pour personnaliser les fichiers générés.

    Returns:
    --------
    None
    """

    # Calcul des valeurs d'interaction SHAP
    shap_interaction_values = final_model.predict(xgb.DMatrix(X_test), pred_interactions=True)
    # Exclure le biais en supprimant la dernière ligne et la dernière colonne
    shap_interaction_values = shap_interaction_values[:, :-1, :-1]

    # Vérification de la compatibilité des dimensions
    print(f"Shape of shap_interaction_values ({dataset_name}):", shap_interaction_values.shape)
    print(f"Number of features in X_test ({dataset_name}):", len(X_test.columns))

    if shap_interaction_values.shape[1:] != (len(X_test.columns), len(X_test.columns)):
        print(
            "Erreur : Incompatibilité entre les dimensions des valeurs d'interaction SHAP et le nombre de features."
        )
        print(f"Dimensions des valeurs d'interaction SHAP : {shap_interaction_values.shape}")
        print(f"Nombre de features dans X_test ({dataset_name}): {len(X_test.columns)}")
        return

    # Calcul de la matrice d'interactions
    interaction_matrix = np.abs(shap_interaction_values).sum(axis=0)
    feature_names = X_test.columns
    interaction_df = pd.DataFrame(interaction_matrix, index=feature_names, columns=feature_names)

    # Masquer la diagonale (interactions d'une feature avec elle-même)
    np.fill_diagonal(interaction_df.values, 0)

    # Récupérer toutes les interactions non nulles
    all_interactions = interaction_df.unstack().sort_values(ascending=False)

    # Pour le graphique, on garde les top N
    N = 80
    top_interactions = all_interactions.head(N)

    # Visualisation des top interactions
    plt.figure(figsize=(24, 16))
    top_interactions.plot(kind='bar')
    plt.title(f"Top {N} Feature Interactions ({dataset_name})", fontsize=16)
    plt.xlabel("Feature Pairs", fontsize=12)
    plt.ylabel("Total Interaction Strength", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(
        os.path.join(results_directory, f'top_feature_interactions_{dataset_name}.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()

    # Calculer la somme totale des interactions pour la normalisation
    total_interaction_value = sum(value for (f1, f2), value in all_interactions.items() if f1 != f2)

    # Initialisation
    seen_pairs = set()
    csv_filename = f"all_interactions_{dataset_name}.csv"
    csv_path = os.path.join(results_directory, csv_filename)

    # Préparation des données CSV avec normalisation
    csv_data = []

    for (f1, f2), value in all_interactions.items():
        pair = tuple(sorted([f1, f2]))
        if f1 != f2 and pair not in seen_pairs:
            percentage = 2 * (value / total_interaction_value) * 100
            if len(seen_pairs) < 40:
                print(f"{f1} <-> {f2}: {value:.4f} ({percentage:.2f}%)")
            interaction = f"{f1} <-> {f2}"
            csv_data.append([interaction, f"{value:.4f}", f"{percentage:.4f}"])
            seen_pairs.add(pair)

    csv_data.sort(key=lambda x: float(x[1]), reverse=True)

    # Sauvegarde du CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['Interaction', 'Value', 'Percentage_Contribution'])
        writer.writerows(csv_data)

    print(f"\nLes interactions ({dataset_name}) ont été sauvegardées dans : {csv_path}")
    print(f"Nombre total d'interactions sauvegardées ({dataset_name}) : {len(csv_data)}")

    # Création de la heatmap
    top_features = interaction_df.sum().sort_values(ascending=False).head(N).index
    plt.figure(figsize=(26, 20))
    sns.heatmap(
        interaction_df.loc[top_features, top_features].round(0).astype(int),
        annot=True, cmap='coolwarm', fmt='d',
        annot_kws={'size': 7}, square=True, linewidths=0.5,
        cbar_kws={'shrink': .8}
    )
    plt.title(f"SHAP Interaction Values Heatmap ({dataset_name})", fontsize=16)
    plt.tight_layout()
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0)
    plt.savefig(
        os.path.join(results_directory, f'feature_interaction_heatmap_{dataset_name}.png'),
        dpi=300, bbox_inches='tight'
    )
    plt.close()

    print(f"Heatmap des interactions sauvegardée sous 'feature_interaction_heatmap_{dataset_name}.png'")


def train_finalModel_analyse(xgb=None,
                             X_train=None, X_train_full=None,  X_test=None, X_test_full=None,
                             y_train_label=None,y_test_label=None,dtrain=None,dtest=None,
                             nb_SessionTest=None,nan_value=None,feature_names=None,best_params=None,config=None,weight_param=None):
    results_directory = config.get('results_directory', None)

    if results_directory == None:
        exit(25)

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



    xgb_metric_custom=config.get('xgb_metric_custom', None)
    print(xgb_metric_custom)
    if (xgb_metric_custom==None):
        exit(13)

    # Configurer custom_metric et obj_function si nécessaire
    if xgb_metric_custom == xgb_metric.XGB_METRIC_CUSTOM_METRIC_PROFITBASED:
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
        print("not XGB_METRIC_CUSTOM_METRIC_PROFITBASED")
        exit(66)

    print(f"Seuil optimal: {optimal_threshold}")
    # Supprimer les paramètres non utilisés par XGBoost mais uniquement dans l'optimisation
    parameters_to_removetoAvoidXgboostError = ['loss_per_fp', 'penalty_per_fn', 'profit_per_tp', 'threshold', 'w_p',
                                               'w_n','nb_split_weight','std_penalty_factor','weight_split']
    for param in parameters_to_removetoAvoidXgboostError:
        best_params.pop(param, None)  # None est la valeur par défaut si la clé n'existe pas

    print(f"best_params dans les parametres d'optimisations  filtrés dans parametre non compatibles xgboost: \n{best_params}")

    start_time_train, end_time_train, num_sessions_train = get_val_cv_time_range(X_train_full, X_train)
    print(f"\nPériode d'entraînement : du {timestamp_to_date_utc(start_time_train)} au {timestamp_to_date_utc(end_time_train)}")
    print(f"Nombre de sessions entraînement : {num_sessions_train}")

    start_time_test, end_time_test, num_sessions_test = get_val_cv_time_range(X_test_full, X_test)
    print(f"Période de test : du {timestamp_to_date_utc(start_time_test)} au {timestamp_to_date_utc(end_time_test)}")
    print(f"Nombre de sessions test : {num_sessions_test}\n")

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


    # Utilisation de la fonction
    plot_custom_metric_evolution_with_trade_info(final_model, evals_result, n_train_trades=len(X_train),
                                                 n_test_trades=len(X_test), results_directory=results_directory)

    print_notification('###### FIN: ENTRAINEMENT MODELE FINAL ##########', color="blue")

    print_notification('###### DEBUT: ANALYSE DES DEPENDENCES SHAP DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")
    importance_df, shap_comparison, shap_values_train, shap_values_test,resulat_test_shap_feature_importance = main_shap_analysis(
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
    y_train_predProba = final_model.predict(dtrain)


    # Convertir les prédictions en CuPy avant transformation

    y_train_predProba = cp.asarray(y_train_predProba)
    y_train_predProba = sigmoidCustom(y_train_predProba)

    y_test_predProba = cp.asarray(y_test_predProba)
    y_test_predProba = sigmoidCustom(y_test_predProba)


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
    y_train_pred_threshold = (y_train_predProba > optimal_threshold).astype(int)


    print_notification('###### FIN: GENERATION PREDICTION AVEC MOBEL FINAL (TEST) ##########', color="blue")

    print_notification('###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur (XTEST) ##########',
                       color="blue")

    ###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur XTEST ##########

    # Pour la courbe de calibration et l'histogramme
    plot_calibrationCurve_distrib(y_test_label, y_test_predProba, optimal_threshold=optimal_threshold,
                                  num_sessions=nb_SessionTest, results_directory=results_directory)

    # Pour le graphique des taux FP/TP par feature

    import warnings

    if 'deltaTimestampOpeningSession1min' in X_test.columns:
        plot_fp_tp_rates(X_test=X_train, y_test_label=y_train_label, y_test_predProba=y_train_predProba,
                         feature_deltaTime_name='deltaTimestampOpeningSession1min', optimal_threshold=optimal_threshold,
                         dataset_name="train",  results_directory=results_directory)
        plot_fp_tp_rates(X_test=X_test, y_test_label=y_test_label, y_test_predProba=y_test_predProba,
                         feature_deltaTime_name='deltaTimestampOpeningSession1min',optimal_threshold=optimal_threshold,
                         dataset_name="test", results_directory=results_directory)


    else:
        warnings.warn(
            "La colonne 'deltaTimestampOpeningSession1min' n'est pas présente dans le jeu de test - Graphique non généré",
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

    plt.close()  # Fermer après l'affichage ou sans affichage

    print_notification('###### FIN: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur (XTEST) ##########',
                       color="blue")


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
    important_features = feature_importance_df['feature'].head(20).tolist()

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

    # Call the function
    high_corr_pairs = analyze_and_save_feature_correlations(X_train, results_directory, threshold=0.75)
    # Si high_corr_pairs est une liste de tuples (feature_i, feature_j, corr_value)
    ###### FIN: CALCUL DES CORRELACTION ##########


    ###### DEBUT: CALCUL DES VALEURS D'INTERACTION SHAP ##########
    print_notification("###### DEBUT: CALCUL DES VALEURS D'INTERACTION SHAP ##########", color="blue")
    analyze_shap_interactions(final_model, X_train, results_directory,dataset_name='X_train') ## utile pour filtrer les données
    analyze_shap_interactions(final_model, X_test, results_directory,dataset_name='X_test') ## utile pour verfiier la pertinance, stabilité et donc l'overfitting (variable qui bouge etc)

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
    penalty_per_fn = metric_dict.get('penalty_per_fn', 0)

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
        df_copy['time_bin'] = (df_copy['deltaTimestampOpeningSession1min'] // 10).astype(int) * 10
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
    # Calcul des statistiques globales (avant sélection)
    total_success_original = successes_original.sum()
    total_failure_original = failures_original.sum()
    total_all_original = total_success_original + total_failure_original
    global_win_rate_original = (total_success_original / total_all_original * 100) if total_all_original > 0 else 0

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
                f"Statistiques Globales avant sélection:\n"
                f" Total Trades: {total_all_original}\n"
                f" Succès: {total_success_original} ({total_success_original / total_all_original * 100:.1f}%)\n"
                f" Échecs: {total_failure_original} ({total_failure_original / total_all_original * 100:.1f}%)\n"
                f" Win Rate Global: {global_win_rate_original:.1f}%\n\n"
                f"Selection : statistiques Globales:\n"
                f" Total Trades: {total_all}\n"
                f" Succès: {total_success} ({total_success / total_all * 100:.1f}%)\n"
                f" Échecs: {total_failure} ({total_failure / total_all * 100:.1f}%)\n"
                f" Win Rate Global: {global_win_rate:.1f}%\n\n"
                + "\n\n".join(stats)
        )
    else:
        stats_text = "Aucun trade dans la période sélectionnée"

    # Affichage des statistiques
    plt.figtext(1.02, 0.6, stats_text, fontsize=12,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.9, pad=10))

    plt.suptitle('Analyse de la Distribution des Trades sur 24H', fontsize=16, y=0.97)
    plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.2, hspace=0.4)

    # Retourner à la fois la figure et les statistiques
    stats = {
        'total_trades': total_all,
        'successes': total_success,
        'failures': total_failure,
        'win_rate': global_win_rate,
        'hourly_stats': {
            'timestamps': df_grouped_filtered.index.tolist(),
            'volumes': (successes_filtered + failures_filtered).tolist()
        }
    }

    return fig, stats



import warnings
def print_warning(message):
    print(f"\033[91m/!\ Warning: {message}\033[0m")
def sessions_selection(df, selected_sessions=None, CUSTOM_SESSIONS_=None
                       ,results_directory=None):
    """
    Met class_binaire à 99 pour les sessions NON sélectionnées et analyse la distribution.
    """

    if results_directory is None:
        print_warning("Le répertoire 'results_directory' est None")


    df_filtered = df.copy()
    timestamps = df['deltaTimestampOpening'].values
    print(df_filtered.shape)
    if CUSTOM_SESSIONS_ is not None:
        # Sélection des périodes activées
        selected_periods = [
            (info['start'], info['end'])
            for info in CUSTOM_SESSIONS_.values()
            if info['selected']
        ]

        periods_starts = np.array([start for start, end in selected_periods], dtype=np.float64)
        periods_ends = np.array([end for start, end in selected_periods], dtype=np.float64)

        # Logging et calcul des statistiques par période
        cum_total_count = 0
        print("\nUtilisation des périodes du dictionnaire:")

        for name, info in CUSTOM_SESSIONS_.items():
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

    elif CUSTOM_SESSIONS_ is not None:
        valid_sections = [s for s in CUSTOM_SESSIONS_ if s['name'] in selected_sessions]
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
    fig, stats = analyze_trade_distribution(df, df_filtered, CUSTOM_SESSIONS_)

    print("\nRésumé des statistiques:")
    print(f"Total trades analysés: {stats['total_trades']}")
    print(f"Win Rate global: {stats['win_rate']:.1f}%")
    if results_directory is not None:
        plt.savefig(os.path.join(results_directory, 'initial_dataset_and_select_sessions.png'), dpi=300, bbox_inches='tight')

    plt.show(block=False)
    plt.pause(2)  # Pause de 2 secondes
    print(df_filtered.shape)
    return df_filtered


def format_time(minutes):
    """Convertit les minutes depuis 22:00 en format heure:minute"""
    total_minutes = (minutes + 22 * 60) % (24 * 60)
    hour = total_minutes // 60
    minute = total_minutes % 60
    return f"{int(hour):02d}:{int(minute):02d}"


def calculate_global_ecart(
        tp_train_list, fp_train_list,
        tp_val_list, fp_val_list,
        use_trade_weighting=False
):
    """
    Calcule l'écart global entre les ratios de winrate (validation / train) sur plusieurs folds.

    Args:
        tp_train_list (list): Liste des true positives (TP) pour les données d'entraînement sur chaque fold.
        fp_train_list (list): Liste des false positives (FP) pour les données d'entraînement sur chaque fold.
        tp_val_list (list): Liste des true positives (TP) pour les données de validation sur chaque fold.
        fp_val_list (list): Liste des false positives (FP) pour les données de validation sur chaque fold.
        use_trade_weighting (bool): Si True, pondère l'écart global par le nombre de trades dans chaque fold.

    Returns:
        float: L'écart global entre les ratios validation/train, pondéré ou non.
    """
    ecarts = []  # Liste des écarts par fold
    poids = []  # Liste des poids associés à chaque fold (nombre de trades)

    # Calcul des écarts et des poids pour chaque fold
    for i in range(len(tp_train_list)):
        # Calcul du nombre total de trades (train et validation)
        train_trades = tp_train_list[i] + fp_train_list[i]
        val_trades = tp_val_list[i] + fp_val_list[i]

        # Éviter la division par zéro en remplaçant par une très petite valeur
        train_trades = max(train_trades, 1e-8)
        val_trades = max(val_trades, 1e-8)

        # Calcul des winrates
        train_winrate = tp_train_list[i] / train_trades
        val_winrate = tp_val_list[i] / val_trades

        # Calcul du ratio validation/train et de l'écart par rapport à 1
        ratio = val_winrate / max(train_winrate, 1e-8)
        ecart = abs(ratio - 1.0)

        # Ajouter l'écart et le poids (nombre de trades en validation) aux listes
        ecarts.append(ecart)
        poids.append(val_trades)  # On peut utiliser une autre définition de poids si nécessaire

    # Si aucun fold n'est présent, retourner un écart infini
    if len(ecarts) == 0:
        return float('inf')

    if use_trade_weighting:
        # Moyenne pondérée par le nombre de trades
        poids_total = sum(poids)
        if poids_total > 0:
            ecart_global = sum(e * p for e, p in zip(ecarts, poids)) / poids_total
        else:
            # Si tous les poids sont nuls, retourner un écart infini
            ecart_global = float('inf')
    else:
        # Moyenne simple des écarts
        ecart_global = sum(ecarts) / len(ecarts)

    return ecart_global

def calculate_normalized_objectives(
        tp_train_list, fp_train_list, tp_val_list, fp_val_list,
        scores_train_list, scores_val_list, fold_stats,
        scale_objectives=False,
        use_imbalance_penalty=True,
        use_trade_weighting_for_ecart=True  # Activer ou non la pondération
):
    # Vérification que toutes les listes ont la même longueur
    lengths = [len(tp_train_list), len(fp_train_list),
               len(tp_val_list), len(fp_val_list),
               len(scores_train_list), len(scores_val_list)]

    if not all(length == lengths[0] for length in lengths):
        print("ERREUR: Les listes n'ont pas toutes la même longueur!")
        return {
            'pnl_norm_objective': float('-inf'),
            'ecart_train_val': float('inf')
        }

    if not lengths[0]:  # Si les listes sont vides
        print("ERREUR: Les listes sont vides!")
        return {
            'pnl_norm_objective': float('-inf'),
            'ecart_train_val': float('inf')
        }

    # Calcul des métriques pour chaque fold
    fold_metrics = []
    for i in range(len(scores_train_list)):
        train_trades = tp_train_list[i] + fp_train_list[i]
        val_trades = tp_val_list[i] + fp_val_list[i]

        train_trades = max(train_trades, 1e-8)  # Éviter la division par zéro
        val_trades = max(val_trades, 1e-8)

        train_winrate = tp_train_list[i] / train_trades
        val_winrate = tp_val_list[i] / val_trades

        fold_metrics.append({
            'winrate_diff': abs(train_winrate - val_winrate),
            'n_trades': val_trades,
            'val_pnl': scores_val_list[i]
        })

    # Calcul du PnL et de l'écart global (en remplaçant la logique existante)
    total_pnl = sum(m['val_pnl'] for m in fold_metrics)
    avg_pnl = total_pnl / len(fold_metrics) if fold_metrics else float('-inf')

    ecart_global = calculate_global_ecart(
        tp_train_list, fp_train_list, tp_val_list, fp_val_list,
        use_trade_weighting=use_trade_weighting_for_ecart
    )

    # Pénalité d'imbalance si activée
    max_trades = max(m['n_trades'] for m in fold_metrics)
    min_trades = max(min(m['n_trades'] for m in fold_metrics), 1e-8)
    imbalance_penalty = 1 + math.log(max_trades / min_trades)

    if use_imbalance_penalty:
        pnl_norm_objective = avg_pnl * (1 / imbalance_penalty)
        ecart_train_val = ecart_global * imbalance_penalty
    else:
        pnl_norm_objective = avg_pnl
        ecart_train_val = ecart_global

    # Normalisation des objectifs si activée
    if scale_objectives:
        pnl_range = (-100, 100)  # Plage ajustable pour le PnL
        winrate_diff_range = (0, 1)

        pnl_norm_objective = normalize_to_range(
            pnl_norm_objective,
            old_min=pnl_range[0],
            old_max=pnl_range[1]
        )

        ecart_train_val = normalize_to_range(
            ecart_train_val,
            old_min=winrate_diff_range[0],
            old_max=winrate_diff_range[1]
        )

    return {
        'pnl_norm_objective': pnl_norm_objective,
        'ecart_train_val': ecart_train_val,
        'raw_metrics': {
            'avg_pnl': avg_pnl,
            'ecart_global': ecart_global,
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

def calculate_profitBased_gpu(y_true_gpu, y_pred_threshold_gpu, metric_dict):
    """
    Calcule les métriques de profit directement sur GPU sans conversions inutiles

    Args:
        y_true_gpu: cp.ndarray - Labels déjà sur GPU
        y_pred_threshold_gpu: cp.ndarray - Prédictions déjà sur GPU
        metric_dict: dict - Dictionnaire des paramètres de métrique
    """
    # Vérification que les entrées sont bien sur GPU
    if not isinstance(y_true_gpu, cp.ndarray):
        raise TypeError("y_true_gpu doit être un tableau CuPy")
    if not isinstance(y_pred_threshold_gpu, cp.ndarray):
        raise TypeError("y_pred_threshold_gpu doit être un tableau CuPy")

    # Calcul des métriques de base
    tp = cp.sum((y_true_gpu == 1) & (y_pred_threshold_gpu == 1))
    fp = cp.sum((y_true_gpu == 0) & (y_pred_threshold_gpu == 1))
    fn = cp.sum((y_true_gpu == 1) & (y_pred_threshold_gpu == 0))

    # Récupération des paramètres de profit/perte
    profit_per_tp = metric_dict.get('profit_per_tp', 1.0)
    loss_per_fp = metric_dict.get('loss_per_fp', -1.1)
    penalty_per_fn = metric_dict.get('penalty_per_fn', -0.1)

    # Calcul du profit total incluant les pénalités FN
    total_profit = (tp * profit_per_tp) + (fp * loss_per_fp) + (fn * penalty_per_fn)

    return float(total_profit), int(tp), int(fp)


def custom_metric_Profit(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict, normalize: bool = False) -> Tuple[
    str, float]:
    """Fonction commune pour calculer les métriques de profit"""
    # Conversion des données en GPU une seule fois
    y_true_gpu = cp.asarray(dtrain.get_label())
    predt_gpu = cp.asarray(predt)

    # Application de la sigmoid et normalisation
    predt_gpu = sigmoidCustom(predt_gpu)
    predt_gpu = cp.clip(predt_gpu, 0.0, 1.0)

    # Vérification des prédictions
    mean_pred = cp.mean(predt_gpu).item()
    std_pred = cp.std(predt_gpu).item()
    min_val = cp.min(predt_gpu).item()
    max_val = cp.max(predt_gpu).item()

    if min_val < 0 or max_val > 1:
        logging.warning(f"Prédictions hors intervalle [0, 1]: [{min_val:.4f}, {max_val:.4f}]")
        return "custom_metric_ProfitBased", float('-inf')

    # Application du seuil directement sur GPU
    threshold = metric_dict.get('threshold', 0.55555555)
    y_pred_threshold_gpu = (predt_gpu > threshold).astype(cp.int32)

    # Calcul du profit et des métriques
    total_profit, tp, fp = calculate_profitBased_gpu(y_true_gpu, y_pred_threshold_gpu, metric_dict)

    # Normalisation si demandée
    if normalize:
        total_trades = tp + fp
        final_profit = total_profit / total_trades if total_trades > 0 else total_profit
        metric_name = 'custom_metric_ProfitBased_norm'
    else:
        final_profit = total_profit
        metric_name = 'custom_metric_ProfitBased'

    return metric_name, float(final_profit)

# Création des deux fonctions spécifiques à partir de la fonction commune
def custom_metric_ProfitBased_gpu(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict) -> Tuple[str, float]:
    """Version GPU de la métrique de profit"""
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

def add_session_id(df, time_periods_dict):
    """
    Ajoute une colonne session_ID basée sur deltaTimestampOpeningSession1min
    """
    # Créer une colonne session_ID avec valeur par défaut None
    df['session_type_index'] = -1

    # Pour chaque période dans le dictionnaire
    for session_name, info in time_periods_dict.items():
        # Créer un masque pour les timestamps dans cette période
        mask = (df['deltaTimestampOpening'] >= info['start']) & \
               (df['deltaTimestampOpening'] < info['end'])
        # Assigner le session_type_index aux lignes correspondantes
        df.loc[mask, 'session_type_index'] = info['session_type_index']

    return df


def manage_rfe_selection(X_train, y_train_label, config, trial, params, weight_param, metric_dict):
    """
    Gère la sélection des features avec RFE selon la configuration.

    Args:
        X_train (pd.DataFrame): DataFrame contenant les features d'entraînement
        y_train_label: Labels d'entraînement
        config (dict): Configuration contenant les paramètres
        trial (optuna.Trial): Trial Optuna pour l'optimisation
        params (dict): Paramètres du modèle
        weight_param: Paramètres de pondération
        metric_dict (dict): Dictionnaire des métriques

    Returns:
        tuple: (X_train modifié, liste des noms des features sélectionnées)
    """
    # Récupération du paramètre RFE de la config avec valeur par défaut NO_RFE
    use_of_rfe_in_optuna = config.get('use_of_rfe_in_optuna', rfe_param.NO_RFE)

    # Liste qui contiendra les noms des features sélectionnées
    selected_feature_names = []

    if use_of_rfe_in_optuna != rfe_param.NO_RFE:
        # Log de l'activation du RFE
        print(f"\n------- RFE activé:")

        # Application du RFE pour la sélection des features
        X_train, selected_feature_names = select_features_ifRfe(
            X_train=X_train,
            y_train_label=y_train_label,
            trial=trial,
            config=config,
            params=params,
            weight_param=weight_param,
            metric_dict=metric_dict,
            use_of_rfe_in_optuna=use_of_rfe_in_optuna
        )

        # Log du nombre de features sélectionnées
        print(f"              - Features sélectionnées avec rfe: {len(selected_feature_names)}")
    else:
        # Si RFE non activé, on garde toutes les features
        selected_feature_names = X_train.columns.tolist()

    return X_train, selected_feature_names
def select_features_ifRfe(X_train, y_train_label, trial, config, params, weight_param, metric_dict,    use_of_rfe_in_optuna=None,
                          selected_columns=None):
    """Sélection des features avec RFE"""
    if use_of_rfe_in_optuna == rfe_param.RFE_WITH_OPTUNA:
        n_features_to_select = trial.suggest_int("n_features_to_select", 1, X_train.shape[1])
    elif use_of_rfe_in_optuna == rfe_param.RFE_AUTO:
        n_features_to_select = config.get('min_features_if_RFE_AUTO', 5)

    selected_feature_names = process_RFE_filteringg(params, trial, weight_param, metric_dict,
                                                    selected_columns, n_features_to_select,
                                                    X_train, y_train_label)
    X_train_selected = X_train[selected_feature_names]

    return X_train_selected, selected_feature_names
def manage_rfe_selection(X_train, y_train_label, config, trial, params, weight_param, metric_dict):
    """
    Gère la sélection des features avec RFE selon la configuration.

    Args:
        X_train (pd.DataFrame): DataFrame contenant les features d'entraînement
        y_train_label: Labels d'entraînement
        config (dict): Configuration contenant les paramètres
        trial (optuna.Trial): Trial Optuna pour l'optimisation
        params (dict): Paramètres du modèle
        weight_param: Paramètres de pondération
        metric_dict (dict): Dictionnaire des métriques

    Returns:
        tuple: (X_train modifié, liste des noms des features sélectionnées)
    """
    # Récupération du paramètre RFE de la config avec valeur par défaut NO_RFE
    use_of_rfe_in_optuna = config.get('use_of_rfe_in_optuna', rfe_param.NO_RFE)

    # Liste qui contiendra les noms des features sélectionnées
    selected_feature_names = []

    if use_of_rfe_in_optuna != rfe_param.NO_RFE:
        # Log de l'activation du RFE
        print(f"\n------- RFE activé:")

        # Application du RFE pour la sélection des features
        X_train, selected_feature_names = select_features_ifRfe(
            X_train=X_train,
            y_train_label=y_train_label,
            trial=trial,
            config=config,
            params=params,
            weight_param=weight_param,
            metric_dict=metric_dict,
            use_of_rfe_in_optuna=use_of_rfe_in_optuna
        )

        # Log du nombre de features sélectionnées
        print(f"              - Features sélectionnées avec rfe: {len(selected_feature_names)}")
    else:
        # Si RFE non activé, on garde toutes les features
        selected_feature_names = X_train.columns.tolist()

    return X_train, selected_feature_names


import cupy as cp
import numpy as np
import xgboost as xgb
from typing import Dict, List, Any, Tuple


def convert_metrics_to_numpy_safe(metrics):
    """Convertit sûrement les métriques GPU en numpy"""
    if isinstance(metrics, cp.ndarray):
        return float(cp.asnumpy(metrics))
    return float(metrics)

def calculate_time_difference(start_date_str, end_date_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)
    diff = relativedelta(end_date, start_date)
    return diff
def compute_balanced_weights_gpu(y):
    """Version GPU de compute_sample_weight('balanced')"""
    unique_classes, class_counts = cp.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(unique_classes)

    # Calcul des poids comme sklearn
    weights = n_samples / (n_classes * class_counts)
    samples_weights = weights[y.astype(cp.int32)]

    return samples_weights


def compute_confusion_matrix_cupy(y_true_gpu, y_pred_gpu):
    """Calcule la matrice de confusion sur GPU"""
    # Conversion en CuPy si nécessaire
    if not isinstance(y_true_gpu, cp.ndarray):
        y_true_gpu = cp.asarray(y_true_gpu, dtype=cp.int32)
    if not isinstance(y_pred_gpu, cp.ndarray):
        y_pred_gpu = cp.asarray(y_pred_gpu, dtype=cp.int32)

    # Calculs sur GPU
    tp = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 1))
    fp = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 1))
    tn = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 0))
    fn = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 0))

    return tn, fp, fn, tp


def predict_and_process(pred_proba, threshold):
    # Supposons que pred_proba est déjà un tableau CuPy
    pred_proba = sigmoidCustom(pred_proba)
    pred_proba = cp.clip(pred_proba, 0.0, 1.0)
    pred = (pred_proba > threshold).astype(cp.int32)
    return pred_proba, pred


def calculate_fold_stats_gpu(labels_gpu, set_name):
    """Calcule les statistiques du fold en gardant tout sur GPU"""
    if not isinstance(labels_gpu, cp.ndarray):
        labels_gpu = cp.asarray(labels_gpu)

    decisions = cp.sum(labels_gpu != 99)
    success = cp.sum(labels_gpu == 1)
    failures = cp.sum(labels_gpu == 0)
    success_rate = cp.where(decisions > 0, success / decisions, cp.float32(0.0))

    return {
        f"{set_name}_n_trades": decisions,
        f"{set_name}_n_class_1": success,
        f"{set_name}_n_class_0": failures,
        f"{set_name}_class_ratio": success_rate,
        f"{set_name}_success_rate": success_rate
    }

def calculate_winrate_gpu(tp_val, fp_val):
    """Calcule le winrate de manière sûre sur GPU"""
    if not isinstance(tp_val, cp.ndarray):
        tp_val = cp.asarray(tp_val, dtype=cp.float32)
    if not isinstance(fp_val, cp.ndarray):
        fp_val = cp.asarray(fp_val, dtype=cp.float32)

    # Calcul sur GPU
    tp_fp_sum = tp_val + fp_val
    mask = tp_fp_sum != 0
    winrate = cp.where(mask, tp_val / tp_fp_sum, cp.float32(0.0))

    return winrate




def setup_metric_dict(trial, weight_param, optima_score,metric_dict):
    """Configure le dictionnaire des métriques en fonction des paramètres Optuna"""
    # Configuration du seuil
    threshold_value = trial.suggest_float('threshold',
                                          weight_param['threshold']['min'],
                                          weight_param['threshold']['max'])


    # Initialisation du dictionnaire des métriques
    if optima_score == xgb_metric.XGB_METRIC_CUSTOM_METRIC_TP_FP:
        # Suggérer les poids pour la métrique combinée
        profit_ratio_weight = trial.suggest_float('profit_ratio_weight',
                                                  weight_param['profit_ratio_weight']['min'],
                                                  weight_param['profit_ratio_weight']['max'])

        win_rate_weight = trial.suggest_float('win_rate_weight',
                                              weight_param['win_rate_weight']['min'],
                                              weight_param['win_rate_weight']['max'])

        selectivity_weight = trial.suggest_float('selectivity_weight',
                                                 weight_param['selectivity_weight']['min'],
                                                 weight_param['selectivity_weight']['max'])

        # Normalisation des poids
        total_weight = profit_ratio_weight + win_rate_weight + selectivity_weight
        metric_dict = {
            'profit_ratio_weight': profit_ratio_weight / total_weight,
            'win_rate_weight': win_rate_weight / total_weight,
            'selectivity_weight': selectivity_weight / total_weight,
        }

    elif optima_score == xgb_metric.XGB_METRIC_CUSTOM_METRIC_PROFITBASED:
        metric_dict = {
            'profit_per_tp': trial.suggest_float('profit_per_tp',
                                                 weight_param['profit_per_tp']['min'],
                                                 weight_param['profit_per_tp']['max']),
            'loss_per_fp': trial.suggest_float('loss_per_fp',
                                               weight_param['loss_per_fp']['min'],
                                               weight_param['loss_per_fp']['max']),
            'penalty_per_fn': trial.suggest_float('penalty_per_fn',
                                                  weight_param['penalty_per_fn']['min'],
                                                  weight_param['penalty_per_fn']['max'])
        }
    metric_dict['threshold'] = threshold_value

    return metric_dict




def setup_model_params(trial, model_type, random_state_seed_, device):
    """Configure les paramètres du modèle pour l'entraînement"""
    # Initialisation des variables
    params = None
    num_boost_round = None

    modele_param_optuna_range = get_model_param_range(model_type)

    if model_type == modeleType.XGB:
        params = {
            'max_depth': trial.suggest_int('max_depth',
                                           modele_param_optuna_range['max_depth']['min'],
                                           modele_param_optuna_range['max_depth']['max']),
            'learning_rate': trial.suggest_float('learning_rate',
                                                 modele_param_optuna_range['learning_rate']['min'],
                                                 modele_param_optuna_range['learning_rate']['max'],
                                                 log=modele_param_optuna_range['learning_rate'].get('log', False)),
            'min_child_weight': trial.suggest_int('min_child_weight',
                                                  modele_param_optuna_range['min_child_weight']['min'],
                                                  modele_param_optuna_range['min_child_weight']['max']),
            'subsample': trial.suggest_float('subsample',
                                             modele_param_optuna_range['subsample']['min'],
                                             modele_param_optuna_range['subsample']['max']),
            'colsample_bytree': trial.suggest_float('colsample_bytree',
                                                    modele_param_optuna_range['colsample_bytree']['min'],
                                                    modele_param_optuna_range['colsample_bytree']['max']),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel',
                                                     modele_param_optuna_range['colsample_bylevel']['min'],
                                                     modele_param_optuna_range['colsample_bylevel']['max']),
            'colsample_bynode': trial.suggest_float('colsample_bynode',
                                                    modele_param_optuna_range['colsample_bynode']['min'],
                                                    modele_param_optuna_range['colsample_bynode']['max']),
            'gamma': trial.suggest_float('gamma',
                                         modele_param_optuna_range['gamma']['min'],
                                         modele_param_optuna_range['gamma']['max']),
            'reg_alpha': trial.suggest_float('reg_alpha',
                                             modele_param_optuna_range['reg_alpha']['min'],
                                             modele_param_optuna_range['reg_alpha']['max'],
                                             log=modele_param_optuna_range['reg_alpha'].get('log', False)),
            'reg_lambda': trial.suggest_float('reg_lambda',
                                              modele_param_optuna_range['reg_lambda']['min'],
                                              modele_param_optuna_range['reg_lambda']['max'],
                                              log=modele_param_optuna_range['reg_lambda'].get('log', False)),
            'random_state': random_state_seed_,
            'tree_method': 'hist',
            'device': device,
        }
        num_boost_round = trial.suggest_int('num_boost_round',
                                            modele_param_optuna_range['num_boost_round']['min'],
                                            modele_param_optuna_range['num_boost_round']['max'])

    elif model_type == modeleType.CATBOOT:
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'verbose': False,
            'early_stopping_rounds': 50,
            'task_type': 'GPU' if device == 'cuda' else 'CPU',
            'boosting_type': 'Ordered',
            'has_time': True,
            'random_seed': random_state_seed_,
            'thread_count': -1,
            'fold_permutation_block_size': 10
        }

        # Ajout des paramètres optimisés
        for param_name, param_range in modele_param_optuna_range.items():
            if 'values' in param_range:
                params[param_name] = trial.suggest_categorical(param_name, param_range['values'])
            elif 'log' in param_range:
                params[param_name] = trial.suggest_float(param_name, param_range['min'],
                                                         param_range['max'], log=True)
            else:
                if isinstance(param_range['min'], int) and isinstance(param_range['max'], int):
                    params[param_name] = trial.suggest_int(param_name, param_range['min'],
                                                           param_range['max'])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_range['min'],
                                                             param_range['max'])

        num_boost_round = params.pop('iterations', None)  # Pour CatBoost, iterations = num_boost_round

    # Vérification finale
    if params is None or num_boost_round is None:
        raise ValueError(f"Type de modèle non supporté ou paramètres non initialisés: {model_type}")

    return params, num_boost_round



def compute_winrate_safe(tp, total_trades):
    """Calcul sécurisé du winrate sur GPU"""
    mask = total_trades != 0
    return cp.where(mask, tp / total_trades, cp.float32(0.0))


def validate_fold_indices(train_pos, val_pos):
    """
    Valide les indices des folds pour assurer leur intégrité.

    Args:
        train_pos (np.array): Indices d'entraînement
        val_pos (np.array): Indices de validation

    Raises:
        ValueError: Si les indices ne sont pas valides
    """
    # Conversion en entiers
    train_pos = train_pos.astype(int)
    val_pos = val_pos.astype(int)

    # Vérification des ensembles vides
    if len(train_pos) == 0 or len(val_pos) == 0:
        raise ValueError("⚠️ Ensemble vide détecté")

    # Vérifier que les indices sont strictement croissants
    if not all(train_pos[i] < train_pos[i + 1] for i in range(len(train_pos) - 1)):
        raise ValueError("Les indices d'entraînement ne sont pas strictement croissants")

    if not all(val_pos[i] < val_pos[i + 1] for i in range(len(val_pos) - 1)):
        raise ValueError("Les indices de validation ne sont pas strictement croissants")

    # Vérifier que validation vient après entraînement
    if not max(train_pos) < min(val_pos):
        raise ValueError("Les indices de validation doivent être postérieurs aux indices d'entraînement")


def prepare_fold_data(X_train, y_train_label, train_pos, val_pos):
    """
    Prépare les données pour un fold spécifique.

    Args:
        X_train (pd.DataFrame): Données d'entraînement complètes
        y_train_label (np.array/pd.Series): Labels
        train_pos (np.array): Indices d'entraînement
        val_pos (np.array): Indices de validation

    Returns:
        dict: Données préparées pour le fold
    """
    # Préparation des labels selon le type
    y_train_fold = y_train_label.iloc[train_pos] if isinstance(y_train_label, pd.Series) \
        else y_train_label[train_pos]
    y_val_fold = y_train_label.iloc[val_pos] if isinstance(y_train_label, pd.Series) \
        else y_train_label[val_pos]

    fold_data = {
        'train_indices': train_pos,
        'val_indices': val_pos,
        'y_train': y_train_fold,
        'y_val': y_val_fold,
        'X_train': X_train.iloc[train_pos] if isinstance(X_train, pd.DataFrame) else X_train[train_pos],
        'X_val': X_train.iloc[val_pos] if isinstance(X_train, pd.DataFrame) else X_train[val_pos]
    }

    # Calculer les statistiques de distribution
    train_dist = np.unique(y_train_fold, return_counts=True)
    val_dist = np.unique(y_val_fold, return_counts=True)

    fold_data['distributions'] = {
        'train': dict(zip(train_dist[0], train_dist[1])),
        'val': dict(zip(val_dist[0], val_dist[1]))
    }

    return fold_data


def log_fold_info(fold_num, nb_split_tscv, X_train_full, fold_data):
    """
    Affiche les informations détaillées sur le fold courant.

    Args:
        fold_num (int): Numéro du fold actuel
        nb_split_tscv (int): Nombre total de folds
        X_train_full (pd.DataFrame): Données d'entraînement complètes
        fold_data (dict): Données du fold préparées
    """
    print(f"\nFold {fold_num + 1}/{nb_split_tscv}")
    print(f"X_train_full index : {X_train_full.index.min()}-{X_train_full.index.max()}")

    # Affichage des ranges d'indices
    train_pos = fold_data['train_indices']
    val_pos = fold_data['val_indices']
    print(f"Train pos range: {min(train_pos)}-{max(train_pos)}")
    print(f"Val pos range: {min(val_pos)}-{max(val_pos)}")

    # Affichage des distributions
    print("\nDistribution des classes:")
    print("Train:", {k: int(v) for k, v in fold_data['distributions']['train'].items()})
    print("Val:", {k: int(v) for k, v in fold_data['distributions']['val'].items()})

    # Calcul et affichage des ratios
    train_dist = fold_data['distributions']['train']
    val_dist = fold_data['distributions']['val']

    if 1 in train_dist and 0 in train_dist:
        train_ratio = train_dist[1] / (train_dist[0] + train_dist[1])
        print(f"Ratio positif train: {train_ratio:.2%}")

    if 1 in val_dist and 0 in val_dist:
        val_ratio = val_dist[1] / (val_dist[0] + val_dist[1])
        print(f"Ratio positif validation: {val_ratio:.2%}")


def calculate_final_results(metrics_dict, arrays, all_fold_stats, nb_split_tscv):
    """
    Calcule les résultats finaux de la validation croisée.

    Args:
        metrics_dict (dict): Métriques accumulées
        arrays (dict): Arrays GPU des métriques par fold
        all_fold_stats (dict): Statistiques par fold
        nb_split_tscv (int): Nombre de folds
    """
    try:
        # Conversion des métriques en numpy
        final_metrics = {key: convert_metrics_to_numpy_safe(value)
                         for key, value in metrics_dict.items()
                         if key.startswith('total_')}

        # Calcul des statistiques globales
        mean_val_score = float(cp.mean(arrays['scores_val']).get())
        std_val_score = float(cp.std(arrays['scores_val']).get())

        results = {
            'metrics': final_metrics,
            'fold_stats': all_fold_stats,
            'winrates_by_fold': arrays['winrates'],
            'nb_trades_by_fold': arrays['nb_trades'],
            'scores_train_by_fold': arrays['scores_train'],
            'tp_train_by_fold': arrays['tp_train'],
            'fp_train_by_fold': arrays['fp_train'],
            'tp_val_by_fold': arrays['tp_val'],
            'fp_val_by_fold': arrays['fp_val'],
            'scores_val_by_fold': arrays['scores_val'],
            'mean_val_score': mean_val_score,
            'std_val_score': std_val_score
        }

        return results

    except Exception as e:
        print(f"Erreur dans calculate_final_results: {str(e)}")
        raise


def handle_exception(e):
    """
    Gère les exceptions de manière uniforme.

    Args:
        e (Exception): L'exception à traiter
    """
    print(f"\nErreur détaillée dans la validation croisée:")
    print(f"Type d'erreur: {type(e).__name__}")
    print(f"Message: {str(e)}")

    import traceback
    traceback.print_exc()


def cleanup_gpu_memory(data_gpu):
    """
    Nettoie la mémoire GPU.

    Args:
        data_gpu (dict): Dictionnaire contenant les données GPU à nettoyer
    """
    print("\nNettoyage de la mémoire GPU")

    # Nettoyage des données GPU
    if isinstance(data_gpu, dict):
        for key in data_gpu:
            if isinstance(data_gpu[key], cp.ndarray):
                data_gpu[key] = None

    # Libération des blocs de mémoire
    cp.get_default_memory_pool().free_all_blocks()


def convert_metrics_to_numpy_safe(value):
    """
    Convertit en toute sécurité les métriques GPU en numpy.

    Args:
        value: Valeur à convertir (peut être CuPy array ou autre)
    Returns:
        La valeur convertie en format numpy
    """
    try:
        if isinstance(value, cp.ndarray):
            return value.get()
        if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], cp.ndarray):
            return [v.get() if isinstance(v, cp.ndarray) else v for v in value]
        return value
    except Exception as e:
        print(f"Erreur lors de la conversion: {str(e)}")
        return value


def update_metrics_and_arrays(metrics_dict, arrays, fold_results, fold_num, all_fold_stats):
    """
    Met à jour les métriques et les arrays GPU avec les résultats du fold courant.

    Args:
        metrics_dict (dict): Dictionnaire des métriques à mettre à jour
        arrays (dict): Dictionnaire des arrays GPU
        fold_results (dict): Résultats du fold courant
        fold_num (int): Numéro du fold
        all_fold_stats (dict): Statistiques de tous les folds
    """
    try:
        # Mise à jour des métriques du dictionnaire
        metrics_dict = update_fold_metrics(
            metrics_dict,
            fold_results['val_metrics'],
            fold_results['train_metrics'],
            fold_num
        )

        # Stockage des statistiques du fold
        all_fold_stats[fold_num] = fold_results['fold_stats']

        # Mise à jour des arrays GPU
        arrays['winrates'][fold_num] = fold_results['fold_stats']['val_winrate']
        arrays['nb_trades'][fold_num] = fold_results['fold_stats']['val_trades']
        arrays['tp_train'][fold_num] = fold_results['train_metrics']['tp']
        arrays['fp_train'][fold_num] = fold_results['train_metrics']['fp']
        arrays['tp_val'][fold_num] = fold_results['val_metrics']['tp']
        arrays['fp_val'][fold_num] = fold_results['val_metrics']['fp']
        arrays['scores_val'][fold_num] = fold_results['val_metrics']['score']
        arrays['scores_train'][fold_num] = fold_results['train_metrics']['score']

        # Vérification optionnelle des mises à jour
        verify_updates = False  # Mettre à True pour le débogage
        if verify_updates:
            arrays_to_check = {
                'winrates': arrays['winrates'],
                'nb_trades': arrays['nb_trades'],
                'tp_val': arrays['tp_val'],
                'scores_val': arrays['scores_val']
            }
            for name, arr in arrays_to_check.items():
                print(f"{name}: sum={cp.sum(arr)}, mean={cp.mean(arr)}")

        return metrics_dict

    except Exception as e:
        print(f"Erreur dans update_metrics_and_arrays: {str(e)}")
        raise


def update_fold_metrics(metrics_dict, val_metrics, train_metrics, fold_num):
    """
    Met à jour les métriques du dictionnaire avec les résultats d'un nouveau fold.

    Args:
        metrics_dict (dict): Dictionnaire des métriques à mettre à jour
        val_metrics (dict): Métriques de validation
        train_metrics (dict): Métriques d'entraînement
        fold_num (int): Numéro du fold

    Returns:
        dict: Dictionnaire des métriques mis à jour
    """
    try:
        # Mise à jour des totaux de validation
        metrics_dict['total_tp_val'] = metrics_dict.get('total_tp_val', 0) + val_metrics['tp']
        metrics_dict['total_fp_val'] = metrics_dict.get('total_fp_val', 0) + val_metrics['fp']
        metrics_dict['total_tn_val'] = metrics_dict.get('total_tn_val', 0) + val_metrics.get('tn', 0)
        metrics_dict['total_fn_val'] = metrics_dict.get('total_fn_val', 0) + val_metrics.get('fn', 0)

        # Mise à jour des totaux d'entraînement
        metrics_dict['total_tp_train'] = metrics_dict.get('total_tp_train', 0) + train_metrics['tp']
        metrics_dict['total_fp_train'] = metrics_dict.get('total_fp_train', 0) + train_metrics['fp']
        metrics_dict['total_tn_train'] = metrics_dict.get('total_tn_train', 0) + train_metrics.get('tn', 0)
        metrics_dict['total_fn_train'] = metrics_dict.get('total_fn_train', 0) + train_metrics.get('fn', 0)

        return metrics_dict

    except Exception as e:
        print(f"Erreur dans update_fold_metrics: {str(e)}")
        raise
def run_cross_validation(X_train, X_train_full, y_train_label, trial, params,
                         metric_dict, cv, nb_split_tscv, weight_param,
                         framework='xgboost', is_log_enabled=False, **kwargs):
    """
    Validation croisée unifiée pour XGBoost et CatBoost.

    Args:
        framework (str): 'xgboost' ou 'catboost'
        kwargs: Paramètres spécifiques au framework (num_boost_round pour XGBoost, etc.)
    """
    try:
        print_notification(f"\n=== Début nouvelle validation croisée {framework.upper()} ===")
        print(f"Nombre de features: {len(X_train.columns)}\n:: {list(X_train.columns)}")

        # Vérifications communes
        validate_inputs(X_train, y_train_label)

        # Initialisation des métriques et arrays
        metrics_dict = initialize_metrics_dict(nb_split_tscv)
        arrays = initialize_gpu_arrays(nb_split_tscv)
        all_fold_stats = {}

        # Préparation données GPU - Interface commune
        data_gpu = prepare_gpu_data(X_train, y_train_label)

        # Sélection du processor de fold selon le framework
        fold_processor = select_fold_processor(framework)

        # Boucle CV
        for fold_num, (train_pos, val_pos) in enumerate(cv.split(X_train)):
            # Validation des indices
            validate_fold_indices(train_pos, val_pos)

            # Préparation des données du fold
            fold_data = prepare_fold_data(X_train, y_train_label, train_pos, val_pos)

            # Log des informations du fold
            if is_log_enabled:
                log_fold_info(fold_num, nb_split_tscv, X_train_full, fold_data)

            # Traitement du fold avec le processor approprié
            fold_results = fold_processor(
                X_train=X_train,
                X_train_full=X_train_full,
                train_pos=train_pos,  # Ajouté
                val_pos=val_pos,  # Ajouté
                params=params,
                data_gpu=data_gpu,
                metric_dict=metric_dict,
                trial=trial,
                weight_param=weight_param,
                is_log_enabled=is_log_enabled,
                **kwargs
            )

            # Mise à jour des métriques et statistiques
            update_metrics_and_arrays(metrics_dict, arrays, fold_results, fold_num, all_fold_stats)

        # Calcul des résultats finaux
        results = calculate_final_results(metrics_dict, arrays, all_fold_stats, nb_split_tscv)

        print(f"\nMémoire GPU finale: {cp.get_default_memory_pool().used_bytes() / 1024 ** 2:.2f} MB")
        return results

    except Exception as e:
        handle_exception(e)
        raise
    finally:
        cleanup_gpu_memory(data_gpu)


def prepare_gpu_data(X_train, y_train_label):
    """Interface commune pour la préparation des données GPU"""
    return {
        'X_train_gpu_full': cp.asarray(X_train.values, dtype=cp.float32),
        'y_train_gpu_full': cp.asarray(y_train_label.values if isinstance(y_train_label, pd.Series)
                        else y_train_label, dtype=cp.int32)
    }


def select_fold_processor(framework):
    """Sélectionne le processor approprié selon le framework"""
    processors = {
        'xgboost': process_cv_fold_xgboost,
        'catboost': process_cv_fold_catboost
    }
    if framework not in processors:
        raise ValueError(f"Framework non supporté: {framework}")
    return processors[framework]



def process_cv_fold_xgboost(X_train=None, X_train_full=None, fold_num=None, train_pos=None, val_pos=None, params=None, num_boost_round=None,
                    data_gpu=None, metric_dict=None, trial=None, weight_param=None, optima_score=None, xgb_metric=xgb_metric, is_log_enabled = False):
    try:
        # Debug initial
        # print(f"\n=== Debug Fold {fold_num} ===")

        # Extraction et vérification des données
        X_train_gpu = data_gpu['X_train_gpu_full'][train_pos].reshape(len(train_pos), -1)
        y_train_gpu = data_gpu['y_train_gpu_full'][train_pos].reshape(-1)
        X_val_gpu = data_gpu['X_train_gpu_full'][val_pos].reshape(len(val_pos), -1)
        y_val_gpu = data_gpu['y_train_gpu_full'][val_pos].reshape(-1)

        # Calcul des statistiques du fold
        fold_stats_current = {
            **calculate_fold_stats_gpu(y_train_gpu, "train"),
            **calculate_fold_stats_gpu(y_val_gpu, "val")
        }

        # Calcul et vérification des poids
        sample_weights_gpu = compute_balanced_weights_gpu(y_train_gpu)

        # Création des DMatrix avec vérification
        dtrain = xgb.DMatrix(X_train_gpu, label=y_train_gpu, weight=sample_weights_gpu)
        dval = xgb.DMatrix(X_val_gpu, label=y_val_gpu)
        evals_result = {}
        train_result = {}


        if optima_score == xgb_metric.XGB_METRIC_CUSTOM_METRIC_PROFITBASED:
            w_p = trial.suggest_float('w_p', weight_param['w_p']['min'], weight_param['w_p']['max'])
            w_n = trial.suggest_float('w_n', weight_param['w_n']['min'], weight_param['w_n']['max'])
            custom_metric = lambda predtTrain, dtrain: custom_metric_ProfitBased_gpu(predtTrain, dtrain, metric_dict)
            obj_function = create_weighted_logistic_obj_gpu(w_p, w_n)
            params['disable_default_eval_metric'] = 1

        else:
            custom_metric = None
            obj_function = None
            params.update({
                'objective': 'binary:logistic',
                'eval_metric': ['aucpr', 'logloss'],
                'disable_default_eval_metric': 0,
                'nthread': -1
            })

        # Entraînement avec monitoring
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            obj=obj_function,
            custom_metric=custom_metric,
            early_stopping_rounds=params.get('early_stopping_rounds', 100),
            verbose_eval=False,
            evals_result=evals_result,
            maximize=True
        )

        if optima_score == xgb_metric.XGB_METRIC_CUSTOM_METRIC_PROFITBASED:
            eval_scores = evals_result['eval']['custom_metric_ProfitBased']
            train_scores = evals_result['train']['custom_metric_ProfitBased']

        else:
            eval_scores = evals_result['eval']['aucpr']
            train_scores = evals_result['train']['aucpr']

        # Vérification des scores
        val_score_best = max(eval_scores)
        val_score_bestIdx = eval_scores.index(val_score_best)
        best_iteration = val_score_bestIdx + 1

        # Prédictions et vérifications validation
        val_pred_proba = model.predict(dval, iteration_range=(0, best_iteration))
        val_pred_proba = cp.asarray(val_pred_proba, dtype=cp.float32)
        val_pred_proba, val_pred = predict_and_process(val_pred_proba, metric_dict['threshold'])

        # Métriques validation
        tn_val, fp_val, fn_val, tp_val = compute_confusion_matrix_cupy(y_val_gpu, val_pred)

        val_metrics = {
            'tp': tp_val,
            'fp': fp_val,
            'tn': tn_val,
            'fn': fn_val,
            'total_samples': len(y_val_gpu),
            'score': val_score_best,
            'best_iteration': best_iteration
        }

        # Même processus pour l'entraînement
        train_pred_proba = model.predict(dtrain, iteration_range=(0, best_iteration))
        train_pred_proba = cp.asarray(train_pred_proba, dtype=cp.float32)
        train_pred_proba, train_pred = predict_and_process(train_pred_proba, metric_dict['threshold'])

        tn_train, fp_train, fn_train, tp_train = compute_confusion_matrix_cupy(y_train_gpu, train_pred)

        train_metrics = {
            'tp': tp_train,
            'fp': fp_train,
            'tn': tn_train,
            'fn': fn_train,
            'total_samples': len(y_train_gpu),
            'score': train_scores[val_score_bestIdx]
        }

        # Calculs finaux et statistiques
        tp_fp_sum_val = tp_val + fp_val
        tp_fp_sum_train = tp_train + fp_train

        if is_log_enabled:
            try:
                def timestamp_to_date_utc_(timestamp):
                    date_format = "%Y-%m-%d %H:%M:%S"
                    if isinstance(timestamp, pd.Series):
                        return timestamp.apply(lambda x: time.strftime(date_format, time.gmtime(x)))
                    else:
                        return time.strftime(date_format, time.gmtime(timestamp))
                # Traitement des informations temporelles
                start_time, end_time, val_sessions = get_val_cv_time_range(X_train_full, X_train, val_pos)
                time_diff = calculate_time_difference(timestamp_to_date_utc_(start_time),
                                                      timestamp_to_date_utc_(end_time))

                # S'assurer que les données sont en format CuPy
                n_train = y_train_gpu.size if isinstance(y_train_gpu, cp.ndarray) else cp.asarray(y_train_gpu).size
                n_val = y_val_gpu.size if isinstance(y_val_gpu, cp.ndarray) else cp.asarray(y_val_gpu).size

                # Calcul des trades réussis et échoués
                trades_reussis = cp.sum(y_val_gpu == 1).item()  # Nombre de 1 dans y_val_gpu
                trades_echoues = cp.sum(y_val_gpu == 0).item()  # Nombre de 0 dans y_val_gpu
                total_trades = trades_reussis + trades_echoues

                # Calcul du winrate
                winrate = (trades_reussis / total_trades * 100) if total_trades > 0 else 0

                # Calcul du ratio échoués/réussis
                ratio_reussite_echec = trades_reussis/trades_echoues if trades_echoues > 0 else float('inf')

                # Mise à jour du print
                print(
                    f"Période de validation - Du {timestamp_to_date_utc_(start_time)} ({start_time}) au {timestamp_to_date_utc_(end_time)} ({end_time}) "
                    f"(Durée: {time_diff.days} jours, {time_diff.months} mois, {time_diff.years} ans)\n"
                    f"Avant optimisation - nombre de trades réussis {trades_reussis} | échoués {trades_echoues} | total {total_trades} | winrate {winrate:.2f}% | réussis/ratio échoués {ratio_reussite_echec:.2f}\n"
                    f"Trades - Train: {n_train:,d}, Val: {n_val:,d}\n"
                    f"Train Metrics - TP: {tp_train:4d} | FP: {fp_train:4d} | TN: {tn_train:4d} | FN: {fn_train:4d}\n"
                    f"Val Metrics   - TP: {tp_val:4d} | FP: {fp_val:4d} | TN: {tn_val:4d} | FN: {fn_val:4d}")

            except Exception as e:
                print(f"Erreur 'process_cv_fold'->erreur lors de l'affichage des métriques: {str(e)}")

        fold_stats = {
            'val_winrate': compute_winrate_safe(tp_val, tp_fp_sum_val),
            'train_winrate': compute_winrate_safe(tp_train, tp_fp_sum_train),
            'val_trades': tp_fp_sum_val,
            'train_trades': tp_fp_sum_train,
            'fold_num': fold_num,
            'best_iteration': best_iteration,
            'val_score': val_score_best,
            'train_score': train_metrics['score'],
            'train_size': len(train_pos),
            'val_size': len(val_pos),
            **fold_stats_current  # Merge fold stats
        }

        return {
            'val_metrics': val_metrics,
            'train_metrics': train_metrics,
            'fold_stats': fold_stats,
            'evals_result': evals_result,
            'best_iteration': best_iteration,
            'val_score_best': val_score_best,
            'val_score_bestIdx': val_score_bestIdx,
            'debug_info': {
                'threshold_used': metric_dict['threshold'],
                'pred_proba_ranges': {
                    'val': {'min': float(cp.min(val_pred_proba)), 'max': float(cp.max(val_pred_proba))},
                    'train': {'min': float(cp.min(train_pred_proba)), 'max': float(cp.max(train_pred_proba))}
                }
            }
        }

    except Exception as e:
        print(f"\nErreur dans process_cv_fold:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        raise


""
def process_cv_fold_catboost(fold_data, params, data_gpu, metric_dict, **kwargs):
    """Processor spécifique à CatBoost"""
    """
    Contrairement à XGBoost, CatBoost ne propose pas (à l’heure actuelle) une intégration directe avec des tableaux Cupy pour passer directement les données sur GPU. L’approche recommandée pour CatBoost est plutôt de :

    Créer un Pool (CatBoostPool) à partir de données CPU (numpy arrays, pandas DataFrame, etc.).
    Spécifier l’option task_type="GPU" dans les paramètres du modèle CatBoost.
    """
    # Code CatBoost adapté
    fold_results=0
    return fold_results


def validate_inputs(X_train, y_train_label):
    """Validation commune des entrées"""
    print(f"Shape X_train: {X_train.shape}")
    print(f"Shape y_train_label: {y_train_label.shape}")
    print(f"Index y_train_label: {y_train_label.index.min()} à {y_train_label.index.max()}")


def initialize_gpu_arrays(nb_split_tscv):
    """Initialisation commune des arrays GPU"""
    return {
        'winrates': cp.zeros(nb_split_tscv, dtype=cp.float32),
        'nb_trades': cp.zeros(nb_split_tscv, dtype=cp.int32),
        'scores_train': cp.zeros(nb_split_tscv, dtype=cp.float32),
        'tp_train': cp.zeros(nb_split_tscv, dtype=cp.float32),
        'fp_train': cp.zeros(nb_split_tscv, dtype=cp.float32),
        'tp_val': cp.zeros(nb_split_tscv, dtype=cp.float32),
        'fp_val': cp.zeros(nb_split_tscv, dtype=cp.float32),
        'scores_val': cp.zeros(nb_split_tscv, dtype=cp.float32)
    }


def initialize_metrics_dict(nb_folds):
    """Initialise toutes les métriques nécessaires sur GPU"""
    return {
        # Métriques par fold - Validation
        'tp_val_by_fold': cp.zeros(nb_folds, dtype=cp.float32),
        'fp_val_by_fold': cp.zeros(nb_folds, dtype=cp.float32),
        'winrates_by_fold': cp.zeros(nb_folds, dtype=cp.float32),
        'nb_trades_by_fold': cp.zeros(nb_folds, dtype=cp.float32),

        # Métriques par fold - Entraînement
        'tp_train_by_fold': cp.zeros(nb_folds, dtype=cp.float32),
        'fp_train_by_fold': cp.zeros(nb_folds, dtype=cp.float32),

        # Totaux validation sur GPU
        'total_tp_val': cp.array(0, dtype=cp.float32),
        'total_fp_val': cp.array(0, dtype=cp.float32),
        'total_tn_val': cp.array(0, dtype=cp.float32),
        'total_fn_val': cp.array(0, dtype=cp.float32),

        # Totaux entraînement sur GPU
        'total_tp_train': cp.array(0, dtype=cp.float32),
        'total_fp_train': cp.array(0, dtype=cp.float32),
        'total_tn_train': cp.array(0, dtype=cp.float32),
        'total_fn_train': cp.array(0, dtype=cp.float32)
    }


def update_fold_metrics(metrics_dict, val_metrics, train_metrics, fold_idx):
    """Met à jour toutes les métriques pour un fold"""
    # S'assurer que les métriques sont sur GPU
    val_tp = cp.asarray(val_metrics['tp'], dtype=cp.float32)
    val_fp = cp.asarray(val_metrics['fp'], dtype=cp.float32)

    # Mise à jour des métriques par fold
    metrics_dict['tp_val_by_fold'][fold_idx] = val_tp
    metrics_dict['fp_val_by_fold'][fold_idx] = val_fp

    # Calcul du winrate sur GPU
    total_trades = val_tp + val_fp
    winrate = cp.where(total_trades > 0, val_tp / total_trades, cp.float32(0.0))

    # Mise à jour des winrates et trades
    metrics_dict['winrates_by_fold'][fold_idx] = winrate
    metrics_dict['nb_trades_by_fold'][fold_idx] = total_trades

    # Mise à jour des totaux
    metrics_dict['total_tp_val'] += val_tp
    metrics_dict['total_fp_val'] += val_fp
    metrics_dict['total_tn_val'] += val_metrics['tn']
    metrics_dict['total_fn_val'] += val_metrics['fn']

    # Même chose pour l'entraînement
    train_tp = cp.asarray(train_metrics['tp'], dtype=cp.float32)
    train_fp = cp.asarray(train_metrics['fp'], dtype=cp.float32)

    metrics_dict['tp_train_by_fold'][fold_idx] = train_tp
    metrics_dict['fp_train_by_fold'][fold_idx] = train_fp

    metrics_dict['total_tp_train'] += train_tp
    metrics_dict['total_fp_train'] += train_fp
    metrics_dict['total_tn_train'] += train_metrics['tn']
    metrics_dict['total_fn_train'] += train_metrics['fn']

    return metrics_dict


def report_trial_optuna(trial, best_trial_with_2_obj,rfe_param,modele_param_optuna_range,selected_columns,results_directory,config):
    # Récupération des valeurs depuis trial.user_attrs
    total_tp_val = trial.user_attrs['total_tp_val']
    total_fp_val = trial.user_attrs['total_fp_val']
    total_tn_val = trial.user_attrs['total_tn_val']
    total_fn_val = trial.user_attrs['total_fn_val']
    weight_param = trial.user_attrs['weight_param']
    nb_split_tscv = trial.user_attrs['nb_split_tscv']
    mean_cv_score = trial.user_attrs['mean_cv_score']
    std_dev_score = trial.user_attrs['std_dev_score']
    std_penalty_factor = trial.user_attrs['std_penalty_factor']
    score_adjustedStd_val = trial.user_attrs['score_adjustedStd_val']
    train_pnl_perTrades = trial.user_attrs['train_pnl_perTrades']
    val_pnl_perTrades = trial.user_attrs['val_pnl_perTrades']
    pnl_perTrade_diff = trial.user_attrs['pnl_perTrade_diff']
    total_samples_val = trial.user_attrs['total_samples_val']
    n_trials_optuna = trial.user_attrs['n_trials_optuna']
    total_samples_val = trial.user_attrs['total_samples_val']
    cummulative_pnl_val = trial.user_attrs['cummulative_pnl_val']
    scores_ens_val_list = trial.user_attrs['scores_ens_val_list']
    tp_fp_diff_val = trial.user_attrs['tp_fp_diff_val']
    tp_percentage = trial.user_attrs['tp_percentage']
    win_rate = trial.user_attrs['win_rate']
    selected_feature_names =trial.user_attrs['selected_feature_names']
    rfe_param_value = trial.user_attrs['use_of_rfe_in_optuna']
    profit_per_tp = trial.user_attrs['profit_per_tp']
    penalty_per_fn = trial.user_attrs['penalty_per_fn']

    trial.set_user_attr('win_rate', win_rate)
    weight_split = trial.user_attrs['weight_split']
    nb_split_weight = trial.user_attrs['nb_split_weight']
    pnl_norm_objective = trial.user_attrs['pnl_norm_objective']
    ecart_train_val = trial.user_attrs['ecart_train_val']
    print(f"   ##Essai actuel: ")
    print(
        f"    =>Objective 1, pnl_norm_objective : {pnl_norm_objective} avec weight_split {weight_split} nb_split_weight {nb_split_weight}")

    print(
        f"     -score_adjustedStd_val : {score_adjustedStd_val:.2f} avec Moyenne des pnl des {nb_split_tscv} iterations : {mean_cv_score:.2f}, std_dev_score : {std_dev_score}, std_penalty_factor={std_penalty_factor}")
    print(f"    =>Objective 2, pnl per trade: train {train_pnl_perTrades} // Val {val_pnl_perTrades} "
          f"donc diff val-train PNL per trade {pnl_perTrade_diff}\n"
          f"     ecart_train_val:{ecart_train_val}")

    if (rfe_param_value != rfe_param.NO_RFE):
        print(
            f"    =>Nombre de features sélectionnées par RFECVCV: {len(selected_feature_names)}, Noms des features: {', '.join(selected_feature_names)}")
    print(f"    =>Principal métrique pour l'essai en cours :")
    print(f"     -Nombre de: TP (True Positives) : {total_tp_val}, FP (False Positives) : {total_fp_val}, "
          f"TN (True Negative) : {total_tn_val}, FN (False Negative) : {total_fn_val},")
    print(f"     -Pourcentage Winrate           : {win_rate:.2f}%")
    print(f"     -Pourcentage de TP             : {tp_percentage:.2f}%")
    print(f"     -Différence (TP - FP)          : {tp_fp_diff_val}")
    print(
        f"     -PNL                           : {cummulative_pnl_val}, original: (scores_ens_val_list: {scores_ens_val_list})")
    print(f"     -Nombre de d'échantillons      : {total_tp_val + total_fp_val + total_tn_val + total_fn_val} dont {total_tp_val + total_fp_val} trade pris")


    if total_samples_val > 0:
        tp_percentage = (total_tp_val / total_samples_val) * 100
    else:
        tp_percentage = 0
    total_trades_val = total_tp_val + total_fp_val
    win_rate = total_tp_val / total_trades_val * 100 if total_trades_val > 0 else 0
    result_dict_trialOptuna = {
        "cummulative_pnl": cummulative_pnl_val,
        "win_rate_percentage": round(win_rate, 2),
        "scores_ens_val_list": scores_ens_val_list,
        "score_adjustedStd_val": score_adjustedStd_val,
        "std_dev_score": std_dev_score,
        "tp_fp_diff_val": tp_fp_diff_val,
        "total_trades_val": total_trades_val,
        "tp_percentage": round(tp_percentage, 3),
        "total_tp_val": total_tp_val,
        "total_fp_val": total_fp_val,
        "total_tn_val": total_tn_val,
        "total_fn_val": total_fn_val,
        "current_trial_number": trial.number + 1,
        "best_trial_with_2_Obj": best_trial_with_2_obj
    }

    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def convert_to_serializable_config(obj):
        """Convert non-serializable objects to a format suitable for JSON."""
        if isinstance(obj, xgb_metric):
            return str(obj)  # or obj.name or obj.value depending on the enum or custom class
        try:
            json.dumps(obj)  # Try to serialize it
            return obj  # If no error, return the object itself
        except (TypeError, ValueError):
            return str(obj)  # If it's not serializable, convert it to string

    import time
    from contextlib import contextmanager

    def safe_file_replace(temp_filename, target_filename, max_retries=5, delay=1):
        """
        Tente de remplacer un fichier de manière sécurisée avec plusieurs essais.
        """
        for attempt in range(max_retries):
            try:
                # Tenter de supprimer le fichier cible s'il existe
                if os.path.exists(target_filename):
                    os.remove(target_filename)
                    time.sleep(0.1)  # Petit délai après la suppression

                # Tenter le remplacement
                os.rename(temp_filename, target_filename)
                return True
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    print(f"Tentative {attempt + 1} échouée: {str(e)}")
                    time.sleep(delay)
                else:
                    print(f"Échec final après {max_retries} tentatives: {str(e)}")
                    # Tenter une copie comme solution de repli
                    try:
                        shutil.copy2(temp_filename, target_filename)
                        os.remove(temp_filename)
                        return True
                    except Exception as copy_error:
                        print(f"Échec de la copie de secours: {str(copy_error)}")
                        return False

    def save_trial_results(trial, result_dict_trialOptuna, config=None,
                           modele_param_optuna_range=None, weight_param=None, selected_columns=None,
                           save_dir="optuna_results",
                           result_file="optuna_results.json"):
        global _first_call_save_r_trialesults

        try:
            # Création du répertoire si nécessaire
            os.makedirs(save_dir, exist_ok=True)

            # Nettoyage au premier appel
            if _first_call_save_r_trialesults:
                if os.path.exists(save_dir):
                    for filename in os.listdir(save_dir):
                        file_path = os.path.join(save_dir, filename)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print(f'Attention: Échec suppression {file_path}: {e}')
                _first_call_save_r_trialesults = False

            result_file_path = os.path.join(save_dir, result_file)

            # Chargement des résultats existants
            results_data = {}
            if os.path.exists(result_file_path) and os.path.getsize(result_file_path) > 0:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        with open(result_file_path, 'r') as f:
                            results_data = json.load(f)
                        break
                    except json.JSONDecodeError as e:
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        print(f"Erreur lecture résultats: {e}")
                        backup_file = f"{result_file_path}.bak_{int(time.time())}"
                        shutil.copy2(result_file_path, backup_file)
                        print(f"Backup créé: {backup_file}")
                    except Exception as e:
                        print(f"Erreur inattendue: {e}")
                        break

            # Mise à jour des données
            if 'selected_columns' not in results_data:
                results_data['selected_columns'] = selected_columns
            if 'config' not in results_data:
                results_data['config'] = {k: convert_to_serializable_config(v) for k, v in config.items()}
            if 'weight_param' not in results_data:
                results_data['weight_param'] = weight_param
            if 'modele_param_optuna_range' not in results_data:
                results_data['modele_param_optuna_range'] = modele_param_optuna_range

            # Ajout des résultats du nouveau trial
            results_data[f"trial_{trial.number + 1}"] = {
                "best_result": {k: convert_to_serializable(v) for k, v in result_dict_trialOptuna.items()},
                "params": {k: convert_to_serializable(v) for k, v in trial.params.items()}
            }

            # Écriture sécurisée avec nom unique
            temp_filename = os.path.join(save_dir, f'temp_results_{int(time.time())}_{os.getpid()}.json')
            try:
                with open(temp_filename, 'w') as tf:
                    json.dump(results_data, tf, indent=4)

                # Remplacement sécurisé
                if not safe_file_replace(temp_filename, result_file_path):
                    raise Exception("Échec du remplacement du fichier")

            except Exception as e:
                print(f"Erreur lors de la sauvegarde: {e}")
                # Tentative de sauvegarde de secours
                backup_path = os.path.join(save_dir, f'backup_results_{int(time.time())}.json')
                try:
                    with open(backup_path, 'w') as bf:
                        json.dump(results_data, bf, indent=4)
                    print(f"Sauvegarde de secours créée: {backup_path}")
                except Exception as backup_error:
                    print(f"Échec de la sauvegarde de secours: {backup_error}")

            finally:
                # Nettoyage des fichiers temporaires
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except:
                        pass

        except Exception as e:
            print(f"Erreur globale dans save_trial_results: {e}")
            raise

    # print(f"   Config: {config}")

    # Appel de la fonction save_trial_results
    save_trial_results(
        trial,
        result_dict_trialOptuna,
        config=config,
        modele_param_optuna_range=modele_param_optuna_range, selected_columns=selected_columns, weight_param=weight_param,
        save_dir=os.path.join(results_directory, 'optuna_results'),  # 'optuna_results' should be a string
        result_file="optuna_results.json"
    )
    print(f"   {trial.number + 1}/{n_trials_optuna} Optuna results and model saved successfully.")

def callback_optuna(study, trial,optuna,study_xgb,rfe_param,config,results_directory):
    """
    Callback function for Optuna to monitor progress and log the best trial.
    """
    current_trial = trial.number
    print(f"\n {Fore.CYAN}Optuna Callback Current Trial {current_trial + 1}:{Style.RESET_ALL}")
    optuna_objective_type_value = trial.user_attrs.get('optuna_objective_type', optuna_doubleMetrics.DISABLE)
    cv_method = trial.user_attrs.get('cv_method', None)

    weightPareto_pnl_val = config.get('weightPareto_pnl_val', 0.6)
    weightPareto_pnl_diff = config.get('weightPareto_pnl_diff', 0.4)

    # Return early if the trial is not complete
    if trial.state != optuna.trial.TrialState.COMPLETE:
        return

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        study.set_user_attr('bestResult_dict', {
            "best_optunaTrial_number": None,
            "best_pnl_val": None,
            "best_pnl_perTrade_diff": None,
            "best_params": None
        })
        print("No completed trials found.")
        return
    contraints_reached=True
    # Check if the optimization is single-objective or multi-objective
    if optuna_objective_type_value != optuna_doubleMetrics.DISABLE:
        # Multi-objective optimization: custom selection of the best trial
        pareto_trials = study.best_trials
        weights = np.array([weightPareto_pnl_val, weightPareto_pnl_diff])

        if (optuna_objective_type_value == optuna_doubleMetrics.USE_DIST_TO_IDEAL):
            # Calculate distances for each Pareto front trial
            values_0 = np.array([t.values[0] for t in pareto_trials])
            values_1 = np.array([t.values[1] for t in pareto_trials])

            min_pnl, max_pnl = values_0.min(), values_0.max()
            min_pnl_diff, max_pnl_diff = values_1.min(), values_1.max()

            pnl_range = max_pnl - min_pnl or 1
            pnl_diff_range = max_pnl_diff - min_pnl_diff or 1

            pnl_normalized = (max_pnl - values_0) / pnl_range
            pnl_diff_normalized = (values_1 - min_pnl_diff) / pnl_diff_range

            distances = np.sqrt(
                (weightPareto_pnl_val * pnl_normalized) ** 2 +
                (weightPareto_pnl_diff * pnl_diff_normalized) ** 2
            )
            best_trial = pareto_trials[np.argmin(distances)]
        elif (optuna_objective_type_value == optuna_doubleMetrics.USE_WEIGHTED_AVG):
            # Weighted average calculation for custom selection
            df = pd.DataFrame({
                'PnL_val': [-t.values[0] for t in completed_trials],
                'pnl_perTrade_diff': [t.values[1] for t in completed_trials]
            })

            for col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                range_val = max_val - min_val or 1
                df[f'{col}_normalized'] = (df[col] - min_val) / range_val

            weighted_avg = df[['PnL_val_normalized', 'pnl_perTrade_diff_normalized']].dot(weights)
            best_trial = completed_trials[weighted_avg.idxmin()]
    else:  # (optuna_doubleMetrics ) nous utilisons le best trial d'optuna directement ici
        # Single-objective optimization: use best_trial directly
        try:
            best_trial = study.best_trial
        except ValueError as e:
            if "No feasible trials are completed yet" in str(e):
                print("Aucun essai respectant les contraintes (ecart_train_val <= 0.05) n'a été trouvé pour le moment")
                # Utiliser le trial actuel pour l'affichage
                best_trial = trial
                contraints_reached=False
            else:
                raise e

    # Create the result dictionary
    bestResult_dict = {
        "best_optunaTrial_number": best_trial.number + 1,
        "best_pnl_val": best_trial.user_attrs.get('best_pnl_val', None),
        "best_pnl_perTrade_diff": best_trial.user_attrs.get('pnl_perTrade_diff', None),
        "best_params": best_trial.params,
        'selected_feature_names': best_trial.user_attrs.get('selected_feature_names', None),
        'use_of_rfe_in_optuna': best_trial.user_attrs.get('use_of_rfe_in_optuna', None),

        'pnl_norm_objective': best_trial.values[0] if len(best_trial.values) > 0 else None,
        'ecart_train_val': best_trial.values[1] if len(best_trial.values) > 1 else None
    }
    study.set_user_attr('bestResult_dict', bestResult_dict)

    # Log additional metrics if available
    metrics = {
        'total_tp_val': best_trial.user_attrs.get('total_tp_val', None),
        'total_fp_val': best_trial.user_attrs.get('total_fp_val', None),
        'total_tn_val': best_trial.user_attrs.get('total_tn_val', None),
        'total_fn_val': best_trial.user_attrs.get('total_fn_val', None),
        'cummulative_pnl_val': best_trial.user_attrs.get('cummulative_pnl_val', None),
        'tp_fp_diff_val': best_trial.user_attrs.get('tp_fp_diff_val', None),
        'tp_percentage': best_trial.user_attrs.get('tp_percentage', None),
        'win_rate': best_trial.user_attrs.get('win_rate', None),
        'scores_ens_val_list': best_trial.user_attrs.get('scores_ens_val_list', None),
        'weight_split': best_trial.user_attrs.get('weight_split', None),
        'nb_split_weight': best_trial.user_attrs.get('nb_split_weight', None),

    }
    winrates_by_fold = best_trial.user_attrs.get('winrates_by_fold', None)
    nb_trades_by_fold = best_trial.user_attrs.get('nb_trades_by_fold', None),

    # Calcul des métriques
    selected_feature_names=best_trial.user_attrs.get('selected_feature_names', None)
    # Rapport
    modele_param_optuna_range = get_model_param_range(config['model_type'])

    report_trial_optuna(trial, best_trial.number + 1,rfe_param,modele_param_optuna_range,selected_feature_names,results_directory,config)

    method_names_pareto = {
        optuna_doubleMetrics.DISABLE: "DISABLE",
        optuna_doubleMetrics.USE_DIST_TO_IDEAL: "double avec USE_DIST_TO_IDEAL",
        optuna_doubleMetrics.USE_WEIGHTED_AVG: "double avec USE_WEIGHTED_AVG"
    }

    method_name_pareto = method_names_pareto.get(optuna_objective_type_value, "UNKNOWN")

    method_names_cv = {
        cv_config.TIME_SERIE_SPLIT: "TIME_SERIE_SPLIT",
        cv_config.TIME_SERIE_SPLIT_NON_ANCHORED: "TIME_SERIE_SPLIT_NON_ANCHORED USE_DIST_TO_IDEAL",
        cv_config.TIMESERIES_SPLIT_BY_ID:'TIMESERIES_SPLIT_BY_ID',
        cv_config.K_FOLD: "K_FOLD",
        cv_config.K_FOLD_SHUFFLE: "K_FOLD_SHUFFLE"

    }

    method_names_xgb_metric = {
        xgb_metric.XGB_METRIC_ROCAUC:"XGB_METRIC_ROCAUC",
        xgb_metric.XGB_METRIC_AUCPR:"XGB_METRIC_AUCPR",
        xgb_metric.XGB_METRIC_F1:"XGB_METRIC_F1",
        xgb_metric.XGB_METRIC_PRECISION:"XGB_METRIC_PRECISION",
        xgb_metric.XGB_METRIC_RECALL:"XGB_METRIC_RECALL",
        xgb_metric.XGB_METRIC_RECALL:"XGB_METRIC_MCC",
        xgb_metric.XGB_METRIC_YOUDEN_J:"XGB_METRIC_YOUDEN_J",
        xgb_metric.XGB_METRIC_SHARPE_RATIO:"XGB_METRIC_SHARPE_RATIO",
        xgb_metric.XGB_METRIC_CUSTOM_METRIC_PROFITBASED:"XGB_METRIC_CUSTOM_METRIC_PROFITBASED",
        xgb_metric.XGB_METRIC_CUSTOM_METRIC_TP_FP:"XGB_METRIC_CUSTOM_METRIC_TP_FP"
    }
    scaler_choice= config.get('scaler_choice', 0)
    winrates_formatted = [f"{x:.2f}" for x in winrates_by_fold]
    scores_ens_val_list_formatted = [f"{x:.2f}" for x in metrics['scores_ens_val_list']]
    method_names_xgb_metric=method_names_xgb_metric.get(config.get("xgb_metric_custom",None))
    method_name_cv = method_names_cv.get(cv_method, "UNKNOWN")
    use_imbalance_penalty = best_trial.user_attrs.get('use_imbalance_penalty', False)
    print(f"\n   {Fore.BLUE}##Meilleur essai jusqu'à present: {bestResult_dict['best_optunaTrial_number']}, "
          f"Methode=> Optuna: '{method_name_pareto} | CV: '{method_name_cv} | XGB metric: {method_names_xgb_metric} |\n"
          f"use_imbalance_penalty: {'Activé' if use_imbalance_penalty else 'Désactivé'}' |"
          f"scaler_choice: {scaler_choice}\n##{Style.RESET_ALL}")
    if  config['use_optuna_constraints_func']==True:
        ecart_train_val = trial.user_attrs.get('ecart_train_val', float('inf'))
        nb_trades_by_fold_list = trial.user_attrs.get('nb_trades_by_fold', [float('inf')])
        winrates_by_fold = trial.user_attrs.get('winrates_by_fold', None)
        contraints_list = calculate_constraints_optuna(trial=trial, config=config)
        #constraints_reached_check = all(c == 0 for c in contraints_list)
        #print("contraints_list: ",contraints_list)
        #if constraints_reached_check != contraints_reached:
         #   print(contraints_list)
          #  raise ValueError(
           #     f"Incohérence détectée : constraints_reached_check = {constraints_reached_check}, contraints_reached = {contraints_reached}")
        if contraints_reached:
            print(Fore.GREEN + f"\u2713 All constraints respected:\n"
                               f"    - ecart_train_val <= {config['constraint_ecart_train_val']} (val: {best_trial.user_attrs.get('ecart_train_val', None)})\n"
                               f"    - min_trades >= {config['constraint_min_trades_threshold_by_Fold']} (val: {min(best_trial.user_attrs.get('nb_trades_by_fold', []))})\n"
                               f"    - winrate >= {config['constraint_winrates_by_fold']}  (val: {min(best_trial.user_attrs.get('winrates_by_fold', []))})" + Style.RESET_ALL)
        else:
            print(Fore.RED + "\u2717 Some constraints not respected:" + Style.RESET_ALL)
            if contraints_list[0] > 0:
                print(
                    Fore.RED + f"    - ecart_train_val > {config['constraint_ecart_train_val']} (actual: {trial.user_attrs.get('ecart_train_val', None)})" + Style.RESET_ALL)
            else:
                print(
                    Fore.GREEN + f"    \u2713 ecart_train_val <= {config['constraint_ecart_train_val']}" + Style.RESET_ALL)

            if contraints_list[1] > 0:
                print(
                    Fore.RED + f"    - min_trades < {config['constraint_min_trades_threshold_by_Fold']} (actual: {min(trial.user_attrs.get('nb_trades_by_fold', []))})" + Style.RESET_ALL)
            else:
                print(
                    Fore.GREEN + f"    \u2713 min_trades >= {config['constraint_min_trades_threshold_by_Fold']}" + Style.RESET_ALL)

            if contraints_list[2] > 0:
                print(
                    Fore.RED + f"    - winrate < {config['constraint_winrates_by_fold']} (actual: {min(trial.user_attrs.get('winrates_by_fold', []))})" + Style.RESET_ALL)
            else:
                print(Fore.GREEN + f"    \u2713 winrate >= {config['constraint_winrates_by_fold']}" + Style.RESET_ALL)

    print(f"    =>Objective 1: pnl_norm_objective -> {bestResult_dict['pnl_norm_objective']:.4f} "
          f"avec weight_split: {metrics['weight_split']} nb_split_weight {metrics['nb_split_weight']}")
    if optuna_objective_type_value != optuna_doubleMetrics.DISABLE:
        print(
            f"    =>Objective 2: score différence par trade (train - val) -> {bestResult_dict['best_pnl_perTrade_diff']:.4f}\n"
            f"      score ecart_train_val   -> {bestResult_dict['ecart_train_val']}")
    if (bestResult_dict['use_of_rfe_in_optuna'] != rfe_param.NO_RFE):
        print(
            f"    =>Nombre de features sélectionnées par RFECVCV: {len(bestResult_dict['selected_feature_names'])}, Noms des features: {', '.join(bestResult_dict['selected_feature_names'])}")

    print(f"    Principal métrique pour le meilleur essai:")
    print(f"     -Nombre de: TP: {metrics['total_tp_val']}, FP: {metrics['total_fp_val']}, "
          f"TN: {metrics['total_tn_val']}, FN: {metrics['total_fn_val']}")
    print(f"     -Pourcentage Winrate           : {metrics['win_rate']:.2f}%")
    print(f"     -Pourcentage de TP             : {metrics['tp_percentage']:.2f}%")
    print(f"     -Différence (TP - FP)          : {metrics['tp_fp_diff_val']}")
    print(
        f"     -PNL                           : {metrics['cummulative_pnl_val']}, original: {scores_ens_val_list_formatted}")
    print(
        f"                                                        winrate : {winrates_formatted}")
    print(
        f"                                                        Nb trade :{nb_trades_by_fold}")

    print(
        f"     -Nombre d'échantillons           : {sum([metrics['total_tp_val'], metrics['total_fp_val'], metrics['total_tn_val'], metrics['total_fn_val']])} "
        f"dont {metrics['total_tp_val']+metrics['total_fp_val']} trades pris")
    print(f"    =>Hyperparamètres du meilleur score trouvé à date: {bestResult_dict['best_params']}")

    study_xgb.set_user_attr('bestResult_dict', bestResult_dict)

    if sum( metrics['scores_ens_val_list']) != 0 and metrics['cummulative_pnl_val'] == 0:
        raise ValueError(
            f"Erreur: La somme des métriques ({sum(metrics['scores_ens_val_list'])}) est différente de 0, mais cumulative_pnl_val est égal à 0.")
    else:
        print(
            f"Vérification réussie: Somme des métriques = {metrics['scores_ens_val_list']}, cumulative_pnl_val = {metrics['scores_ens_val_list']}")


@njit
def detect_naked_pocs_per_session(
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        pocs: np.ndarray,
        delta_timestamps: np.ndarray,
        session_start_idx: int,
        session_end_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    n = session_end_idx - session_start_idx
    dist_above = np.full(n, np.nan, dtype=np.float64)
    dist_below = np.full(n, np.nan, dtype=np.float64)

    for idx in range(n):
        current_idx = session_start_idx + idx
        current_close = closes[current_idx]
        closest_above = np.inf
        closest_below = -np.inf
        has_above = False
        has_below = False

        for past_idx in range(current_idx - 1, session_start_idx - 1, -1):
            past_poc = pocs[past_idx]
            is_naked = True

            for check_idx in range(past_idx + 1, current_idx + 1):
                if lows[check_idx] <= past_poc <= highs[check_idx]:
                    is_naked = False
                    break

            if is_naked:
                if past_poc > current_close:
                    dist = past_poc - current_close
                    if dist < abs(closest_above - current_close):
                        closest_above = past_poc
                        has_above = True
                elif past_poc < current_close:
                    dist = current_close - past_poc
                    if dist < abs(current_close - closest_below):
                        closest_below = past_poc
                        has_below = True

        if has_above:
            dist_above[idx] = closest_above - current_close
        if has_below:
            dist_below[idx] = current_close - closest_below

    return dist_above, dist_below

@njit(parallel=True)
def detect_naked_pocs_parallel(
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        pocs: np.ndarray,
        delta_timestamps: np.ndarray,
        session_starts: np.ndarray,
        session_ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = len(closes)
    dist_above = np.full(n, np.nan, dtype=np.float64)
    dist_below = np.full(n, np.nan, dtype=np.float64)

    num_sessions = len(session_starts)

    for s in prange(num_sessions):
        session_start_idx = session_starts[s]
        session_end_idx = session_ends[s]

        # Appeler la fonction pour traiter une session
        session_dist_above, session_dist_below = detect_naked_pocs_per_session(
            closes, highs, lows, pocs, delta_timestamps,
            session_start_idx, session_end_idx
        )

        # Copier les résultats dans les tableaux principaux
        dist_above[session_start_idx:session_end_idx] = session_dist_above
        dist_below[session_start_idx:session_end_idx] = session_dist_below

    return dist_above, dist_below

def calculate_naked_poc_distances(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    start_time = time.time()
    print(f"Start calculate_naked_poc_distances")

    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    pocs = df['pocPrice'].values.astype(np.float64)
    delta_timestamps = df['deltaTimestampOpening'].values.astype(np.float64)

    # Identifier les indices de début et de fin de chaque session
    session_start_idxs = df[df['SessionStartEnd'] == 10].index.values.astype(np.int64)
    session_end_idxs = np.append(session_start_idxs[1:], len(df))

    if len(session_start_idxs) == 0:
        raise ValueError("No sessions found with SessionStartEnd=10")

    dist_above, dist_below = detect_naked_pocs_parallel(
        closes, highs, lows, pocs, delta_timestamps,
        session_start_idxs, session_end_idxs
    )

    dist_above = -np.abs(dist_above)
    total_time = time.time() - start_time
    print(f"Total time (calculate_naked_poc_distances): {total_time:.2f} seconds")
    return pd.Series(dist_above, index=df.index), pd.Series(dist_below, index=df.index)

def calculate_vif(df):
    """
    Calcule le Variance Inflation Factor (VIF) pour chaque feature d'un DataFrame.
    Utilisé comme première méthode de filtrage des features.

    Args:
        df (pd.DataFrame): DataFrame contenant uniquement les colonnes numériques à analyser

    Returns:
        pd.DataFrame: DataFrame contenant les features et leurs valeurs VIF correspondantes
    """
    df_numeric = df.select_dtypes(include=[np.number])

    # Supprimer les colonnes constantes qui causeraient une division par zéro dans le calcul VIF
    df_numeric = df_numeric.loc[:, df_numeric.apply(pd.Series.nunique) != 1]

    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_numeric.columns

    vif_values = []
    for i in range(df_numeric.shape[1]):
        try:
            vif = variance_inflation_factor(df_numeric.values, i)
        except (np.linalg.LinAlgError, ValueError):
            vif = np.nan
        vif_values.append(vif)

    vif_data["VIF"] = vif_values
    return vif_data


def compute_vif_and_remove_multicollinearity(df, threshold):
    """
    Processus itératif de filtrage des features basé sur le VIF.
    Supprime une à une les features ayant le VIF le plus élevé jusqu'à ce que
    toutes les features restantes aient un VIF inférieur au seuil.

    Args:
        df (pd.DataFrame): DataFrame d'entrée
        threshold (float): Seuil VIF au-delà duquel une feature est considérée comme trop colinéaire

    Returns:
        pd.DataFrame: DataFrame contenant l'historique des valeurs VIF et le statut final des features
    """
    X_train_bis = df.select_dtypes(include=[np.number]).copy()
    vif_dfs = []
    iteration = 1
    removed_features = []
    retained_features = X_train_bis.columns.tolist()

    # Boucle itérative de suppression des features avec VIF élevé
    while True:
        vif_df = calculate_vif(X_train_bis)
        vif_df.columns = ['Feature', f'VIF{iteration}']
        vif_df[f'VIF{iteration}'] = pd.to_numeric(vif_df[f'VIF{iteration}'], errors='coerce')
        vif_dfs.append(vif_df)

        max_vif = vif_df[f'VIF{iteration}'].max()
        if pd.isna(max_vif) or max_vif <= threshold:
            break
        else:
            # Suppression de la feature avec le VIF le plus élevé
            max_feature = vif_df.loc[vif_df[f'VIF{iteration}'].idxmax(), 'Feature']
            X_train_bis = X_train_bis.drop(columns=[max_feature])
            removed_features.append(max_feature)
            retained_features.remove(max_feature)
            iteration += 1

    # Construction du DataFrame de résultat avec l'historique VIF
    vif_full = reduce(lambda left, right: pd.merge(left, right, on='Feature', how='outer'), vif_dfs)
    vif_full['Status'] = vif_full['Feature'].apply(lambda x: 'Conservé' if x in retained_features else 'Non conservé')

    # Gestion des valeurs manquantes dans l'historique VIF
    vif_columns = [col for col in vif_full.columns if col.startswith('VIF')]
    for col in vif_columns:
        vif_full[col] = vif_full[col].astype(object)

    for feature in vif_full['Feature']:
        feature_mask = vif_full['Feature'] == feature
        for idx, col in enumerate(vif_columns):
            if pd.isna(vif_full.loc[feature_mask, col]).values[0]:
                vif_full.loc[feature_mask, col] = 'n.a'

    # Réorganisation des colonnes
    vif_columns_sorted = sorted(vif_columns, key=lambda x: int(x[3:]), reverse=True)
    vif_full = vif_full[['Feature'] + vif_columns_sorted + ['Status']]

    return vif_full


def compute_and_keep_correled(merged_X_Y):
    """
    Compute linear (Pearson) and nonlinear (Spearman) correlation with the target variable.

    Parameters:
    -----------
    merged_X_Y : pandas.DataFrame
        Input DataFrame containing features and the target variable.

    Returns:
    --------
    corr_df : pandas.DataFrame
        DataFrame with columns 'Feature', 'lr_target', 'nlr_target' containing
        the linear and nonlinear correlation with the target variable.
    """
    # Compute linear and nonlinear correlations
    linear_correlation_target = merged_X_Y.corr()["target"] * 100
    linear_correlation_target.name = "lr_target"

    nonlinear_correlation_target = merged_X_Y.corr(method="spearman")["target"] * 100
    nonlinear_correlation_target.name = "nlr_target"

    # Combine into a DataFrame
    corr_df = pd.DataFrame({
        'Feature': linear_correlation_target.index,
        'lr_target': linear_correlation_target.values,
        'nlr_target': nonlinear_correlation_target.values
    })

    # Exclude the target row
    corr_df = corr_df[corr_df['Feature'] != 'target']

    return corr_df
def displaytNan_vifMiCorrFiltering(X=None, Y=None, selected_columns=None, name="name dataset",
                                   config=None, enable_vif_corr_mi=False):
    """
    Analyse complète des features d'un DataFrame avec options de filtrage.

    Cette fonction combine:
    1. L'analyse des valeurs manquantes et nulles
    2. Le filtrage optionnel des features basé sur:
       - VIF (multicolinéarité)
       - Corrélation
       - Information mutuelle

    Args:
        X (pd.DataFrame): DataFrame des features
        Y (pd.Series): Série des cibles
        selected_columns (list): Liste des colonnes à considérer
        name (str): Nom du dataset pour l'affichage
        config (dict): Configuration des seuils et colonnes à exclure
        enable_vif_corr_mi (bool): Active/désactive le filtrage des features

    Returns:
        list: Liste des colonnes conservées après filtrage si enable_vif_corr_mi=True
        None: Si enable_vif_corr_mi=False
    """
    # Initialisation des seuils
    vif_threshold = 0
    corr_threshold = 0
    mi_threshold = 0

    if config is not None:
        vif_threshold = config.get('vif_threshold', 0)
        corr_threshold = config.get('corr_threshold', 0)
        mi_threshold = config.get('mi_threshold', 0)

    # Création du DataFrame d'analyse
    analysis_df = pd.DataFrame()
    analysis_df['Feature'] = X.columns

    # Calcul des statistiques sur les NaN et zéros
    analysis_df['NaN Count'] = X.isna().sum().values
    analysis_df['NaN%'] = (X.isna().mean() * 100).values
    analysis_df['Zeros%'] = ((X == 0).mean() * 100).values
    analysis_df['NaN+Zeros%'] = analysis_df['NaN%'] + analysis_df['Zeros%']
    analysis_df['non NaN'] = len(X) - analysis_df['NaN Count']
    analysis_df['non NaN%'] = 100 - analysis_df['NaN%']

    # Liste pour stocker les colonnes conservées
    retained_columns = selected_columns.copy()

    if enable_vif_corr_mi:
        # Application du filtrage VIF (itératif)
        vif_full = compute_vif_and_remove_multicollinearity(X[selected_columns], vif_threshold)
        analysis_df = pd.merge(analysis_df, vif_full, on='Feature', how='left')

        # Mise à jour des colonnes conservées après VIF
        retained_columns = vif_full[vif_full['Status'] == 'Conservé']['Feature'].tolist()

        X_afterVIF = X[retained_columns]

        # Filtrage par corrélation
        merged_dfVIF_target = pd.concat([X_afterVIF, Y], axis=1)
        merged_dfVIF_target.columns = list(X_afterVIF.columns) + ['target']  # Pour être sûr
        corr_df = compute_and_keep_correled(merged_dfVIF_target)

        # Fusion des corrélations avec le DataFrame d'analyse
        analysis_df = pd.merge(analysis_df, corr_df, on='Feature', how='left')

        # Calcul des scores d'information mutuelle
        if Y.nunique() <= 20 or Y.dtype == 'object' or Y.dtype == 'category' or np.issubdtype(Y.dtype, np.integer):
            mi_scores = mutual_info_classif(X_afterVIF, Y)
        else:
            mi_scores = mutual_info_regression(X_afterVIF, Y)

        # Créer un DataFrame pour les scores MI
        mi_df = pd.DataFrame({'Feature': X_afterVIF.columns, 'mi': mi_scores})

        # Fusionner les scores MI avec analysis_df
        analysis_df = pd.merge(analysis_df, mi_df, on='Feature', how='left')

        # Récupération des colonnes VIF
        vif_columns = [col for col in analysis_df.columns if col.startswith('VIF')]
        vif_columns_sorted = sorted(vif_columns, key=lambda x: int(x[3:]), reverse=True)  # Tri décroissant

        # Identifier la dernière colonne VIF (la plus récente)
        if vif_columns_sorted:
            last_vif_column = vif_columns_sorted[0]
        else:
            last_vif_column = None

        # Mise à jour du 'Status' pour qu'il soit True ou False
        def compute_status(row):
            vif_value = row.get(last_vif_column, 'n.a')
            if vif_value != 'n.a' and not pd.isna(vif_value):
                vif_ok = float(vif_value) <= vif_threshold
            else:
                vif_ok = False

            corr_ok = (
                (abs(row.get('lr_target', 0)) > corr_threshold) or
                (abs(row.get('nlr_target', 0)) > corr_threshold)
            )

            mi_ok = row.get('mi', 0) > mi_threshold

            status = vif_ok and (corr_ok or mi_ok)
            return status

        analysis_df['Status'] = analysis_df.apply(compute_status, axis=1)

        # Mise à jour des colonnes conservées après application de toutes les conditions
        retained_columns = analysis_df[analysis_df['Status']]['Feature'].tolist()

        has_status = True
    else:
        vif_columns_sorted = []
        has_status = False
        # **Ajout de la colonne 'Status' avec une valeur par défaut**
        analysis_df['Status'] = True  # Ou False, selon ce qui est logique pour votre cas

    # Organisation des colonnes pour l'affichage
    base_columns = ['Feature', 'NaN Count', 'NaN%', 'Zeros%', 'NaN+Zeros%', 'non NaN', 'non NaN%']
    if has_status:
        base_columns.append('Status')

    correlation_columns = []
    if 'lr_target' in analysis_df.columns and 'nlr_target' in analysis_df.columns:
        correlation_columns = ['lr_target', 'nlr_target']
    mi_column = []
    if 'mi' in analysis_df.columns:
        mi_column = ['mi']

    if has_status:
        analysis_df_columns = base_columns + correlation_columns + mi_column + vif_columns_sorted
    else:
        analysis_df_columns = base_columns + correlation_columns + mi_column

    # Ajustement pour éviter les KeyError
    analysis_df_columns = [col for col in analysis_df_columns if col in analysis_df.columns]
    analysis_df = analysis_df[analysis_df_columns]

    # Formatage de l'affichage
    base_format = [
        "{:<48}",  # Feature
        "{:>10}",  # NaN Count
        "{:>8}",   # NaN%
        "{:>8}",   # Zeros%
        "{:>12}",  # NaN+Zeros%
        "{:>12}",  # Val non NaN
        "{:>12}",  # Val non NaN%
    ]
    if has_status:
        base_format.append("{:>8}")  # Status

    # Check if correlation columns exist
    correlation_format = ["{:>12}", "{:>12}"] if correlation_columns else []

    # Check if mi column exists
    mi_format = ["{:>12}"] if mi_column else []

    vif_formats = ["{:>12}"] * len(vif_columns_sorted)

    full_format = base_format + correlation_format + mi_format + vif_formats

    headers = [
        "Feature",
        "NaN Count",
        "NaN%",
        "Zeros%",
        "NaN+Zeros%",
        "Val non NaN",
        "Val non NaN%",
    ]
    if has_status:
        headers.append("Status")

    if correlation_columns:
        headers += correlation_columns

    if mi_column:
        headers += mi_column

    headers += vif_columns_sorted

    # Print header
    header_line = ''
    for header, fmt in zip(headers, full_format):
        header_line += fmt.format(header) + ' '
    print(header_line.strip())
    print("-" * len(header_line))

    # ANSI color codes
    RED = '\033[91m'    # Red for features not retained
    BLUE = '\033[94m'   # Blue for values exceeding thresholds
    ORANGE = '\033[93m' # Orange for excluded_columns_CorrCol
    YELLOW = '\033[33m' # Yellow for excluded_columns_category
    RESET = '\033[0m'   # Reset color

    # Color mapping for excluded columns
    if config is not None:
        excluded_columns_principal = config.get('excluded_columns_principal', [])
        excluded_columns_tradeDirection = config.get('excluded_columns_tradeDirection', [])
        excluded_columns_CorrCol = config.get('excluded_columns_CorrCol', [])
        excluded_columns_category = config.get('excluded_columns_category', [])
    else:
        excluded_columns_principal = []
        excluded_columns_tradeDirection = []
        excluded_columns_CorrCol = []
        excluded_columns_category = []

    for idx, row in analysis_df.iterrows():
        feature = row['Feature']

        # Get the basic stats
        nan_count = row['NaN Count']
        nan_percentage = row['NaN%']
        zeros_percentage = row['Zeros%']
        total_percentage = row['NaN+Zeros%']
        non_nan_count = row['non NaN']
        non_nan_percentage = row['non NaN%']

        # Initialize output values and colors
        output_values = [
            feature,
            int(nan_count),
            f"{nan_percentage:.2f}",
            f"{zeros_percentage:.2f}",
            f"{total_percentage:.2f}",
            int(non_nan_count),
            f"{non_nan_percentage:.2f}",
        ]

        output_colors = [''] * len(output_values)

        if has_status:
            status_value = 'True' if row['Status'] else 'False'
            output_values.append(status_value)
            output_colors.append('')
        else:
            status_value = 'n.a'  # Ou une autre valeur par défaut
            output_values.append(status_value)
            output_colors.append('')

        # Handle correlation columns if they exist
        if 'lr_target' in analysis_df.columns and 'nlr_target' in analysis_df.columns:
            lr_target = row.get('lr_target', 'n.a')
            nlr_target = row.get('nlr_target', 'n.a')

            # lr_target
            if lr_target != 'n.a' and not pd.isna(lr_target):
                lr_target_formatted = f"{lr_target:.2f}"
                output_values.append(lr_target_formatted)
                if abs(lr_target) > corr_threshold:
                    output_colors.append(BLUE)
                else:
                    output_colors.append('')
            else:
                output_values.append('n.a')
                output_colors.append('')

            # nlr_target
            if nlr_target != 'n.a' and not pd.isna(nlr_target):
                nlr_target_formatted = f"{nlr_target:.2f}"
                output_values.append(nlr_target_formatted)
                if abs(nlr_target) > corr_threshold:
                    output_colors.append(BLUE)
                else:
                    output_colors.append('')
            else:
                output_values.append('n.a')
                output_colors.append('')
        else:
            # If correlation columns do not exist
            output_values.extend(['n.a', 'n.a'])
            output_colors.extend(['', ''])

        # Handle mi column if it exists
        if 'mi' in analysis_df.columns:
            mi_value = row.get('mi', 'n.a')
            if mi_value != 'n.a' and not pd.isna(mi_value):
                mi_formatted = f"{mi_value:.4f}"
                output_values.append(mi_formatted)
                if mi_value > mi_threshold:
                    output_colors.append(BLUE)
                else:
                    output_colors.append('')
            else:
                output_values.append('n.a')
                output_colors.append('')
        else:
            # If mi column does not exist
            output_values.append('n.a')
            output_colors.append('')

        # Get vif_values
        vif_values = []
        for col in vif_columns_sorted:
            val = row.get(col, 'n.a')
            if isinstance(val, str) and val == 'n.a':
                vif_values.append('n.a')
            elif not pd.isna(val):
                vif_values.append(f"{val:.2f}")
            else:
                vif_values.append('n.a')

        # Append VIF values and their colors
        for idx_vif, vif_val in enumerate(vif_values):
            output_values.append(vif_val)
            # For coloring VIF values
            if idx_vif == 0 and vif_val != 'n.a':  # idx_vif == 0 corresponds to the last VIF column
                if float(vif_val) > vif_threshold:
                    # Color the last VIF column if value exceeds threshold
                    output_colors.append(BLUE)
                else:
                    output_colors.append('')
            else:
                output_colors.append('')

        # Color the entire row red if not retained
        if has_status and not row['Status']:
            row_color = RED
        else:
            row_color = ''

        # Color coding based on config
        if feature in excluded_columns_principal:
            row_color = RED
        elif feature in excluded_columns_CorrCol:
            row_color = ORANGE
        elif feature in excluded_columns_tradeDirection:
            row_color = BLUE
        elif feature in excluded_columns_category:
            row_color = YELLOW

        # Now, format and color each value
        formatted_values = []
        for val, fmt, color_code in zip(output_values, full_format, output_colors):
            formatted_value = fmt.format(val)
            if color_code:
                formatted_value = f"{color_code}{formatted_value}{RESET}"
            elif row_color:
                formatted_value = f"{row_color}{formatted_value}{RESET}"
            formatted_values.append(formatted_value)

        # Join the formatted values to form the output line
        output_line = ' '.join(formatted_values)
        print(output_line)

    if enable_vif_corr_mi:
        return retained_columns
    return None


def check_distribution_coherence(X_train, X_test, p_threshold=0.01):
    """
    Vérifie la cohérence des distributions entre X_train et X_test via un test de Kolmogorov-Smirnov.
    Pour chaque variable, si la p-value est très faible, cela indique que les distributions diffèrent significativement.

    Args:
        X_train (pd.DataFrame): Données d'entraînement.
        X_test (pd.DataFrame): Données de test.
        p_threshold (float): Seuil de p-value pour considérer qu'il y a un écart significatif.

    Returns:
        dict: dictionnaire contenant les features qui diffèrent significativement.
    """
    from scipy.stats import ks_2samp

    differing_features = {}
    for col in X_train.columns:
        # On ignore les colonnes non numériques
        if not np.issubdtype(X_train[col].dtype, np.number):
            continue

        # Test KS (Kolmogorov-Smirnov)
        statistic, p_value = ks_2samp(X_train[col], X_test[col])
        if p_value < p_threshold:
            differing_features[col] = {'statistic': statistic, 'p_value': p_value}
    return differing_features


def check_value_ranges(X_train, X_test):
    """
    Vérifie si X_test contient des valeurs hors des bornes (min, max) observées dans X_train.
    Retourne un dict avec la liste des features et le nombre de valeurs hors bornes.
    """
    out_of_bounds = {}
    for col in X_train.columns:
        if not np.issubdtype(X_train[col].dtype, np.number):
            continue
        train_min, train_max = X_train[col].min(), X_train[col].max()
        test_values = X_test[col]
        below_min_mask = test_values < train_min
        above_max_mask = test_values > train_max

        below_min = below_min_mask.sum()
        above_max = above_max_mask.sum()

        if below_min > 0 or above_max > 0:
            out_of_bounds[col] = {
                'below_min_count': int(below_min),
                'above_max_count': int(above_max),
                'train_min': float(train_min),
                'train_max': float(train_max),
                'below_min_values': test_values[below_min_mask].tolist(),  # Ajout des valeurs spécifiques
                'above_max_values': test_values[above_max_mask].tolist()  # Ajout des valeurs spécifiques
            }
    return out_of_bounds


def apply_scaling(X_train, X_test, save_path=None,chosen_scaler=None):
    """
    Applique la normalisation sur les données d'entraînement et de test.

    Args:
        X_train (pd.DataFrame): Données d'entraînement à normaliser
        X_test (pd.DataFrame): Données de test à normaliser
        config (dict): Configuration contenant le choix du scaler
        save_path: Chemin optionnel pour sauvegarder les paramètres du scaler

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler, scaler_params)
    """

    # Si le scaling est désactivé
    if chosen_scaler == scalerChoice.SCALER_DISABLE:
        return X_train, X_test, None, None

    # Création du scaler selon le choix
    if chosen_scaler == scalerChoice.SCALER_ROBUST:
        scaler = RobustScaler()
        scaler_name = "RobustScaler"
    elif chosen_scaler == scalerChoice.SCALER_STANDARD:
        scaler = StandardScaler()
        scaler_name = "StandardScaler"
    elif chosen_scaler == scalerChoice.SCALER_MINMAX:
        scaler = MinMaxScaler()
        scaler_name = "MinMaxScaler"
    elif chosen_scaler == scalerChoice.SCALER_MAXABS:
        scaler = MaxAbsScaler()
        scaler_name = "MaxAbsScaler"

    # Fit sur train et transform sur les deux
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Stockage des paramètres selon le type de scaler
    scaler_params = {
        'scaler_type': scaler_name,
        'features': X_train.columns.tolist()
    }

    if chosen_scaler == scalerChoice.SCALER_ROBUST:
        scaler_params.update({
            'center': dict(zip(X_train.columns, scaler.center_)),
            'scale': dict(zip(X_train.columns, scaler.scale_))
        })
        print("\nParamètres du RobustScaler:")
        print("Médianes:", scaler_params['center'])
        print("IQRs:", scaler_params['scale'])
    else:
        scaler_params.update({
            'mean': dict(zip(X_train.columns, scaler.mean_)),
            'scale': dict(zip(X_train.columns, scaler.scale_))
        })
        print("\nParamètres du StandardScaler:")
        print("Moyennes:", scaler_params['mean'])
        print("Écarts-types:", scaler_params['scale'])

    # Sauvegarde des paramètres si un chemin est fourni
    if save_path is not None:
        import os
        import json

        filename = f"{scaler_name.lower()}_params.json"
        full_path = os.path.join(save_path, filename)

        os.makedirs(save_path, exist_ok=True)


        with open(full_path, 'w') as f:
            json.dump(scaler_params, f, indent=4)
        print(f"\nParamètres sauvegardés dans: {full_path}")

    return X_train_scaled, X_test_scaled, scaler, scaler_params
def save_features_with_sessions(df, custom_sections, file_path):
    # Sauvegarder le DataFrame normalement
    df.to_csv(file_path, sep=';', index=False, encoding='iso-8859-1')

    # Ajouter CUSTOM_SECTIONS à la fin avec un séparateur spécial
    with open(file_path, 'a', encoding='iso-8859-1') as f:
        f.write('\n###CUSTOM_SECTIONS_START###\n')
        for section, values in custom_sections.items():
            f.write(f"{section};{values['start']};{values['end']};{values['session_type_index']};"
                    f"{values['selected']};{values['description']}\n")
        f.write('###CUSTOM_SECTIONS_END###\n')


from io import StringIO


def load_features_and_sections(file_path):
    try:
        print('0: Starting the function')

        # Vérifiez si le fichier existe
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        print(f"File exists: {file_path}")

        # Vérifier la taille du fichier
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size / (1024 * 1024):.2f} MB")

        # Lire le fichier par morceaux si volumineux
        lines = []
        with open(file_path, 'r', encoding='iso-8859-1') as f:
            for i, line in enumerate(f):
                lines.append(line.strip())
                if i % 100000 == 0:  # Afficher une progression tous les 100,000 lignes
                    print(f"{i} lines read...")
        print('1: File read successfully')

        start_marker = '###CUSTOM_SECTIONS_START###'
        end_marker = '###CUSTOM_SECTIONS_END###'
        print('2: Start and end markers defined')

        # Trouver les indices des marqueurs de début et de fin
        try:
            start_index = lines.index(start_marker)
            end_index = lines.index(end_marker, start_index)
            print(f'3: Markers found - start: {start_index}, end: {end_index}')
        except ValueError:
            print('3: Markers not found, reading the entire file as DataFrame')
            data = '\n'.join(lines)
            features_df = pd.read_csv(StringIO(data), sep=';', encoding='iso-8859-1', low_memory=False)
            custom_sections = {}
            print('4: Entire file read into DataFrame')
            return features_df, custom_sections

        # Extraire les lignes de données et les sections personnalisées
        data_lines = lines[:start_index]
        custom_section_lines = lines[start_index + 1:end_index]
        print(
            f'4: Data lines and custom section lines extracted - data lines: {len(data_lines)}, custom section lines: {len(custom_section_lines)}')

        # Convertir les lignes de données en DataFrame
        data = '\n'.join(data_lines)
        features_df = pd.read_csv(StringIO(data), sep=';', encoding='iso-8859-1', low_memory=False)
        print('5: Data lines converted to DataFrame')

        # Conversion sécurisée de 'deltaTimestampOpening' si elle existe
        if 'deltaTimestampOpening' in features_df.columns:
            print('6: Converting deltaTimestampOpening column')
            features_df['deltaTimestampOpening'] = pd.to_numeric(features_df['deltaTimestampOpening'],
                                                                 errors='coerce').fillna(0).astype(int)
            print('6: deltaTimestampOpening column converted')

        # Analyser les sections personnalisées
        custom_sections = {}
        for line in custom_section_lines:
            if line.strip():  # Vérifier si la ligne n'est pas vide
                parts = line.strip().split(';')
                if len(parts) >= 6:
                    section, start, end, type_idx, selected, description = parts[:6]
                    # Si la description contient des points-virgules, les regrouper
                    if len(parts) > 6:
                        description = ';'.join(parts[5:])
                    custom_sections[section] = {
                        'start': int(start),
                        'end': int(end),
                        'session_type_index': int(type_idx),
                        'selected': selected.lower() == 'true',
                        'description': description
                    }
        print(f'7: Custom sections parsed - {len(custom_sections)} sections found')

        return features_df, custom_sections

    except Exception as e:
        print(f"Error: {e}")
        raise e

def compare_dataframes_train_test(X_train, X_test):
    """
    Compare les colonnes entre X_train et X_test.

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training dataset déjà sélectionné
    X_test : pandas.DataFrame
        Test dataset déjà sélectionné

    Returns:
    --------
    tuple
        (bool, str) - (True si colonnes identiques, message avec détails)
    """
    try:
        # Obtention des listes de colonnes
        train_cols = list(X_train.columns)
        test_cols = list(X_test.columns)

        # Vérification de l'égalité des colonnes
        columns_match = set(train_cols) == set(test_cols)

        if columns_match:
            # Si les colonnes sont identiques
            message = f"Nombre de colonnes dans X_train et X_test = {len(train_cols)}\n"
            message += "Colonnes : \n" + ", ".join(train_cols)
        else:
            # Si les colonnes sont différentes
            message = f"Nombre de colonnes dans X_train = {len(train_cols)}\n"
            message += f"Nombre de colonnes dans X_test = {len(test_cols)}\n"
            message += "\nColonnes dans X_train : \n" + ", ".join(train_cols)
            message += "\nColonnes dans X_test : \n" + ", ".join(test_cols)

            # Détail des différences
            train_only = set(train_cols) - set(test_cols)
            test_only = set(test_cols) - set(train_cols)

            if train_only:
                message += f"\nColonnes uniquement dans X_train: {', '.join(train_only)}"
            if test_only:
                message += f"\nColonnes uniquement dans X_test: {', '.join(test_only)}"

        print(message)
        return columns_match, message

    except Exception as e:
        error_message = f"Erreur lors de la comparaison: {str(e)}"
        print(error_message)
        return False, error_message

def display_metrics(cv_results):
    """
    Affiche toutes les métriques de manière organisée
    """
    # Métriques par fold
    # print("\n=== Métriques par Fold ===")
    winrates_by_fold_cpu = cp.asnumpy(cv_results['winrates_by_fold'])
    nb_trades_by_fold_cpu = cp.asnumpy(cv_results['nb_trades_by_fold'])
    scores_train_by_fold_cpu = cp.asnumpy(cv_results['scores_train_by_fold'])
    tp_train_by_fold_cpu = cp.asnumpy(cv_results['tp_train_by_fold'])
    fp_train_by_fold_cpu = cp.asnumpy(cv_results['fp_train_by_fold'])
    tp_val_by_fold_cpu = cp.asnumpy(cv_results['tp_val_by_fold'])
    fp_val_by_fold_cpu = cp.asnumpy(cv_results['fp_val_by_fold'])
    scores_val_by_fold_cpu = cp.asnumpy(cv_results['scores_val_by_fold'])

    print("\nWinrates par fold      :", winrates_by_fold_cpu)
    print("Nombre trades par fold :", nb_trades_by_fold_cpu)
    print("\nScores Train par fold  :", scores_train_by_fold_cpu)
    print("TP Train par fold      :", tp_train_by_fold_cpu)
    print("FP Train par fold      :", fp_train_by_fold_cpu)
    print("\nTP Val par fold        :", tp_val_by_fold_cpu)
    print("FP Val par fold        :", fp_val_by_fold_cpu)
    print("Scores Val par fold    :", scores_val_by_fold_cpu)

    # Totaux validation
    print("\n=== Totaux Validation ===")
    total_tp_val = float(cv_results['metrics']['total_tp_val'])
    total_fp_val = float(cv_results['metrics']['total_fp_val'])
    total_tn_val = float(cv_results['metrics']['total_tn_val'])
    total_fn_val = float(cv_results['metrics']['total_fn_val'])

    print(f"Total TP : {total_tp_val}")
    print(f"Total FP : {total_fp_val}")
    print(f"Total TN : {total_tn_val}")
    print(f"Total FN : {total_fn_val}")

    # Totaux entraînement
    print("\n=== Totaux Entraînement ===")
    total_tp_train = float(cv_results['metrics']['total_tp_train'])
    total_fp_train = float(cv_results['metrics']['total_fp_train'])
    total_tn_train = float(cv_results['metrics']['total_tn_train'])
    total_fn_train = float(cv_results['metrics']['total_fn_train'])

    print(f"Total TP : {total_tp_train}")
    print(f"Total FP : {total_fp_train}")
    print(f"Total TN : {total_tn_train}")
    print(f"Total FN : {total_fn_train}")

    # Statistiques supplémentaires
    print("\n=== Statistiques Globales ===")
    print(f"Taux de succès validation : {(total_tp_val / (total_tp_val + total_fp_val)) * 100:.2f}%" if (
        total_tp_val + total_fp_val) > 0 else "N/A")
    print(f"Taux de succès entraînement : {(total_tp_train / (total_tp_train + total_fp_train)) * 100:.2f}%" if (
        total_tp_train + total_fp_train) > 0 else "N/A")



from sklearn.model_selection import BaseCrossValidator

class CustomSessionTimeSeriesSplit_byID(BaseCrossValidator):

    def __init__(self, session_ids, n_splits=5):
        self.session_ids = np.array(session_ids)
        self.n_splits = n_splits
        # Trier les sessions uniques pour garantir l'ordre chronologique
        self.unique_sessions = np.sort(np.unique(self.session_ids))

    def split(self, X, y=None, groups=None):
        n_sessions = len(self.unique_sessions)
        print("____CustomSessionTimeSeriesSplit_byID____")
        if self.n_splits >= n_sessions:
            raise ValueError(
                f"Le nombre de splits ({self.n_splits}) doit être inférieur au nombre de sessions uniques ({n_sessions}).")

        # Calculer la taille minimale pour chaque fold
        min_sessions_per_fold = n_sessions // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # Calculer les indices pour ce fold
            train_end = min_sessions_per_fold * (fold + 1)
            val_start = train_end
            val_end = val_start + min_sessions_per_fold

            # Sélectionner les sessions
            train_sessions = self.unique_sessions[:train_end]
            val_sessions = self.unique_sessions[val_start:val_end]

            # Obtenir les indices correspondants
            train_mask = np.isin(self.session_ids, train_sessions)
            val_mask = np.isin(self.session_ids, val_sessions)

            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            """
            # Vérifications et logs
            print(f"\nFold {fold + 1}:")
            print(f"Sessions train: {len(train_sessions)} ({train_sessions[0]} à {train_sessions[-1]})")
            print(f"Sessions val: {len(val_sessions)} ({val_sessions[0]} à {val_sessions[-1]})")
            print(f"Indices train: {len(train_indices)}")
            print(f"Indices val: {len(val_indices)}")
            """
            yield train_indices, val_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class NonAnchoredWalkForwardCV(BaseCrossValidator):
    """
    Validation croisée Non-Anchored Walk-Forward avec ratio stable.

    - Le ratio r = val_size / train_size est maintenu constant à partir du second fold.
    - Le leftover est ajouté au premier train pour utiliser au mieux toutes les données.
    - Chaque split est déterminé à l'avance en fonction du ratio, puis on avance dans les données.
    """

    def __init__(self, n_splits, r=1.0):
        self.n_splits = n_splits
        self.r = r

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        N = len(X)
        nb_split_tscv = self.n_splits
        r = self.r

        # Calcul du train_size maximum
        # On veut (train_size + val_size)*nb_split_tscv <= N
        # val_size = r * train_size => (1 + r)*train_size * nb_split_tscv <= N
        # train_size <= N / (nb_split_tscv*(1+r))
        train_size = int(N // (nb_split_tscv * (1 + r)))
        val_size = int(train_size * r)

        required_size = nb_split_tscv * (train_size + val_size)
        leftover = N - required_size

        # Ajout du leftover au premier train
        train_size_first = train_size + leftover

        # Fold 1
        current_index = 0
        train_indices = np.arange(current_index, current_index + train_size_first)
        current_index += train_size_first
        val_indices = np.arange(current_index, current_index + val_size)
        current_index += val_size

        yield train_indices, val_indices

        # Folds suivants
        for i in range(1, nb_split_tscv):
            train_indices = np.arange(current_index, current_index + train_size)
            current_index += train_size
            val_indices = np.arange(current_index, current_index + val_size)
            current_index += val_size

            yield train_indices, val_indices
def convert_metrics_to_numpy(metrics_dict):
    """
    Convertit de manière sûre les métriques GPU en arrays NumPy

    Args:
        metrics_dict: Dictionnaire contenant les métriques CuPy
    Returns:
        Dict avec les mêmes métriques en NumPy
    """
    numpy_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, cp.ndarray):
            numpy_metrics[key] = cp.asnumpy(value)
        else:
            numpy_metrics[key] = value
    return numpy_metrics


def calculate_time_difference(start_date_str, end_date_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)
    diff = relativedelta(end_date, start_date)
    return diff

def setup_cv_method(df_init=None,X_train=None,y_train_label=None,cv_method=None, nb_split_tscv=None, config=None):
    # D'abord, vérifier si la colonne existe dans df_init
    if 'timeStampOpening' not in df_init.columns:
        raise ValueError("La colonne 'timeStampOpening' est absente de df_init.")

    """Configure la méthode de validation croisée"""
    if cv_method == cv_config.TIME_SERIE_SPLIT:
        return TimeSeriesSplit(n_splits=nb_split_tscv)
    elif cv_method == cv_config.TIME_SERIE_SPLIT_NON_ANCHORED:
        r = config.get('non_acnhored_val_ratio', 1)

        cv=NonAnchoredWalkForwardCV(n_splits=nb_split_tscv, r=r)
        return cv

    elif cv_method == cv_config.TIMESERIES_SPLIT_BY_ID:
        # Ensuite, appliquer la logique en fonction de cv_method
        # On garde une trace des colonnes originales
        X_train_ = X_train.copy()

        original_columns = set(X_train_.columns)

        # Ajout des colonnes nécessaires si elles n'existent pas déjà
        if 'timeStampOpening' not in X_train_.columns:
            X_train_['timeStampOpening'] = df_init.loc[X_train_.index, 'timeStampOpening']

        if 'session_type_index' not in X_train_.columns:
            X_train_['session_type_index'] = df_init.loc[X_train_.index, 'session_type_index']

        # Vérification des valeurs
        columns_to_check = ['timeStampOpening']
        comparison_results = X_train_[columns_to_check].eq(df_init.loc[X_train_.index, columns_to_check])
        discrepancies = comparison_results[~comparison_results.all(axis=1)]

        if not discrepancies.empty:
            print("\nDétails des divergences :")
            print(discrepancies)
            raise ValueError("divergence df_init X_train_")


        # Création d'une colonne temporaire session_type_index_utc
        X_train_['session_type_index_utc'] = timestamp_to_date_utc(X_train_['timeStampOpening']).str[:10] + '_' + X_train_[
            'session_type_index'].astype(str)

        session_ids = X_train_['session_type_index_utc'].values

        # Suppression de la colonne temporaire

        cv = CustomSessionTimeSeriesSplit_byID(session_ids=session_ids, n_splits=nb_split_tscv)


        return cv
    elif cv_method == cv_config.K_FOLD:
        return KFold(n_splits=nb_split_tscv, shuffle=False)
    elif cv_method == cv_config.K_FOLD_SHUFFLE:
        return KFold(n_splits=nb_split_tscv, shuffle=True, random_state=42)
    else:
        raise ValueError(f"Unknown cv_method: {cv_method}")

def calculate_constraints_optuna(trial=None, config=None):
    """
    Calcule les contraintes pour un modèle d'optimisation en utilisant trial et config.

    Args:
        trial: Instance Optuna trial pour récupérer les user_attrs.
        config (dict): Configuration contenant les seuils de contraintes.

    Returns:
        list: Contraintes calculées.
    """
    # Récupération des valeurs depuis trial
    ecart_train_val = trial.user_attrs.get('ecart_train_val', float('inf'))
    nb_trades_by_fold_list = trial.user_attrs.get('nb_trades_by_fold', [float('inf')])
    winrates_by_fold = trial.user_attrs.get('winrates_by_fold', None)

    # Récupération des seuils depuis config
    constraint_min_trades_threshold_by_Fold = config.get('constraint_min_trades_threshold_by_Fold', float('inf'))
    constraint_ecart_train_val = config.get('constraint_ecart_train_val', float('inf'))
    constraint_winrates_by_fold = config.get('constraint_winrates_by_fold', None)

    # Calcul des valeurs minimales dans les listes, avec vérification
    min_trades = min(nb_trades_by_fold_list) if isinstance(nb_trades_by_fold_list, list) and nb_trades_by_fold_list else float('inf')
    min_winrate = min(winrates_by_fold) if isinstance(winrates_by_fold, list) and winrates_by_fold else float('inf')

    # Calcul des contraintes
    constraint_ecart = max(0, ecart_train_val - constraint_ecart_train_val)
    constraint_min_trades = max(0, constraint_min_trades_threshold_by_Fold - min_trades)
    constraint_winrates = max(0, constraint_winrates_by_fold - min_winrate)

    # Regroupement des contraintes
    constraints = [
        constraint_ecart,       # Contrainte sur l'écart train-val
        constraint_min_trades,  # Contrainte sur le nombre minimal de trades
        constraint_winrates     # Contrainte sur le winrate minimal
    ]

    return constraints
