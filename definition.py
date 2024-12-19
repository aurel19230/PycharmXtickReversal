from enum import Enum
import numpy as np
import cupy as cp
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import numba as nb
import time


class cv_config(Enum):
    TIME_SERIE_SPLIT = 0
    TIME_SERIE_SPLIT_NON_ANCHORED = 1
    TIMESERIES_SPLIT_BY_ID = 2
    K_FOLD = 3
    K_FOLD_SHUFFLE = 4

class modelType (Enum):
    XGB=0
    LGBM=1
    CATBOOST=2

class optuna_doubleMetrics(Enum):
    DISABLE = 0
    USE_DIST_TO_IDEAL = 1
    USE_WEIGHTED_AVG = 2

class rfe_param(Enum):
    NO_RFE = 0
    RFE_WITH_OPTUNA = 1
    RFE_AUTO = 2


class model_customMetric(Enum):
    LGB_CUSTOM_METRIC_PROFITBASED=0
    XGB_METRIC_ROCAUC = 1
    XGB_METRIC_AUCPR = 2
    XGB_METRIC_F1 = 4
    XGB_METRIC_PRECISION = 5
    XGB_METRIC_RECALL = 6
    XGB_METRIC_MCC = 7
    XGB_METRIC_YOUDEN_J = 8
    XGB_METRIC_SHARPE_RATIO = 9
    XGB_CUSTOM_METRIC_PROFITBASED = 10
    XGB_CUSTOM_METRIC_TP_FP = 11

class scalerChoice(Enum):
    SCALER_DISABLE = 0
    SCALER_STANDARD = 1
    SCALER_ROBUST = 2
    SCALER_MINMAX = 3  # Nouveau : échelle [0,1]
    SCALER_MAXABS = 4  # Nouveau : échelle [-1,1]


class ScalerMode(Enum):
    FIT_TRANSFORM = 0  # Pour l'entraînement : fit + transform
    TRANSFORM = 1
     # Pour le test : transform uniquement

import cupy as cp
import numpy as np


def sigmoidCustom(x):
# Supposons que x est déjà un tableau CuPy
  return 1 / (1 + cp.exp(-x))
def sigmoidCustom_cpu(x):
    """Custom sigmoid function."""
    return 1 / (1 + np.exp(-x))
def predict_and_process(pred_proba, threshold):
    # Supposons que pred_proba est déjà un tableau CuPy
    pred_proba = sigmoidCustom(pred_proba)
    pred_proba = cp.clip(pred_proba, 0.0, 1.0)
    pred = (pred_proba > threshold).astype(cp.int32)
    return pred_proba, pred


def compute_confusion_matrix_cupy(y_true_gpu, y_pred_gpu):
    tp = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 1))
    fp = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 1))
    tn = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 0))
    fn = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 0))
    return tn, fp, fn, tp


def log_cv_fold_metrics(X_train_full, X_train, val_pos, Y_train_cv, y_val_cv, tp_train, fp_train,
                        tn_train, fn_train, tp_val, fp_val, tn_val, fn_val):
    """
    Log les métriques pour chaque fold de validation croisée avec gestion d'erreurs.

    Args:
        X_train_full (pd.DataFrame): Données d'entraînement complètes
        X_train (pd.DataFrame): Données d'entraînement du fold
        val_pos (array-like): Positions des données de validation
        Y_train_cv (cp.ndarray): Labels d'entraînement sur GPU
        y_val_cv (cp.ndarray): Labels de validation sur GPU
        tp_train, fp_train, tn_train, fn_train (int): Métriques d'entraînement
        tp_val, fp_val, tn_val, fn_val (int): Métriques de validation

    Returns:
        dict: Dictionnaire contenant les métriques calculées
    """
    metrics = {}
    try:
        # 1. Conversion des timestamps en dates UTC
        def timestamp_to_date_utc_(timestamp):
            try:
                date_format = "%Y-%m-%d %H:%M:%S"
                if isinstance(timestamp, pd.Series):
                    return timestamp.apply(lambda x: time.strftime(date_format, time.gmtime(x)))
                return time.strftime(date_format, time.gmtime(timestamp))
            except Exception as e:
                print(f"Erreur lors de la conversion timestamp->date: {str(e)}")
                return None

        # 2. Récupération des informations temporelles
        try:
            start_time, end_time, val_sessions = get_val_cv_time_range(X_train_full, X_train, val_pos)
            time_diff = calculate_time_difference(timestamp_to_date_utc_(start_time),
                                                  timestamp_to_date_utc_(end_time))
            metrics['start_time'] = start_time
            metrics['end_time'] = end_time
            metrics['time_diff'] = time_diff
        except Exception as e:
            print(f"Erreur lors de la récupération des informations temporelles: {str(e)}")
            return None

        # 3. Calcul des tailles d'échantillons
        try:
            n_train = Y_train_cv.size if isinstance(Y_train_cv, cp.ndarray) else cp.asarray(
                Y_train_cv).size
            n_val = y_val_cv.size if isinstance(y_val_cv, cp.ndarray) else cp.asarray(y_val_cv).size
            metrics['n_train'] = n_train
            metrics['n_val'] = n_val
        except Exception as e:
            print(f"Erreur lors du calcul des tailles d'échantillons: {str(e)}")
            return None

        # 4. Calcul des trades avant optimisation
        try:
            trades_reussis = cp.sum(y_val_cv == 1).item()
            trades_echoues = cp.sum(y_val_cv == 0).item()
            total_trades = trades_reussis + trades_echoues

            winrate = (trades_reussis / total_trades * 100) if total_trades > 0 else 0
            ratio_reussite_echec = trades_reussis / trades_echoues if trades_echoues > 0 else float('inf')

            metrics.update({
                'trades_reussis': trades_reussis,
                'trades_echoues': trades_echoues,
                'total_trades': total_trades,
                'winrate': winrate,
                'ratio_reussite_echec': ratio_reussite_echec
            })
        except Exception as e:
            print(f"Erreur lors du calcul des trades: {str(e)}")
            return None

        # 5. Métriques de confusion
        try:
            metrics.update({
                'train_metrics': {
                    'tp': tp_train, 'fp': fp_train,
                    'tn': tn_train, 'fn': fn_train
                },
                'val_metrics': {
                    'tp': tp_val, 'fp': fp_val,
                    'tn': tn_val, 'fn': fn_val
                }
            })
        except Exception as e:
            print(f"Erreur lors de l'enregistrement des métriques de confusion: {str(e)}")
            return None

        # 6. Affichage formaté des résultats
        try:
            print(
                f"\nPériode de validation - Du {timestamp_to_date_utc_(start_time)} ({start_time}) "
                f"au {timestamp_to_date_utc_(end_time)} ({end_time})\n"
                f"Durée: {time_diff.days} jours, {time_diff.months} mois, {time_diff.years} ans\n"
                f"\nAvant optimisation:\n"
                f"Trades réussis: {trades_reussis} | échoués: {trades_echoues} | "
                f"total: {total_trades} | winrate: {winrate:.2f}% | "
                f"ratio réussis/échoués: {ratio_reussite_echec:.2f}\n"
                f"\nNombre de trades:\n"
                f"Train: {n_train:,d}, Validation: {n_val:,d}\n"
                f"\nMétriques Train:\n"
                f"TP: {tp_train:4d} | FP: {fp_train:4d} | TN: {tn_train:4d} | FN: {fn_train:4d}\n"
                f"\nMétriques Validation:\n"
                f"TP: {tp_val:4d} | FP: {fp_val:4d} | TN: {tn_val:4d} | FN: {fn_val:4d}\n"
            )

            return metrics

        except Exception as e:
            print(f"Erreur lors de l'affichage formaté: {str(e)}")
            return None

    except Exception as e:
        print(f"Erreur générale dans log_cv_fold_metrics: {str(e)}")
        return None

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

    # print(f"Nombre de sessions complètes : {complete_sessions}")
    # print(f"Minutes résiduelles : {residual_minutes:.2f}")
    # print(f"Équivalent en sessions des minutes résiduelles : {residual_sessions:.2f}")
    # print(f"Nombre total de sessions (complètes + résiduelles) : {total_sessions:.2f}")

    return total_sessions, date_startSection, date_endSection

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



def timestamp_to_date_utc(timestamp):
    date_format = "%Y-%m-%d %H:%M:%S"
    if isinstance(timestamp, pd.Series):
        return timestamp.apply(lambda x: time.strftime(date_format, time.gmtime(x)))
    else:
        return time.strftime(date_format, time.gmtime(timestamp))


def compute_winrate_safe(tp, total_trades):
    """Calcul sécurisé du winrate sur GPU"""
    mask = total_trades != 0
    return cp.where(mask, tp / total_trades, cp.float32(0.0))
