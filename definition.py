from enum import Enum
import numpy as np
import platform

if platform.system() != "Darwin":  # "Darwin" est le nom interne de macOS
    import cupy as cp
else:
    print("CuPy ne sera pas importé sur macOS.")

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

import numpy as np


def sigmoidCustom(x):
# Supposons que x est déjà un tableau CuPy
  return 1 / (1 + cp.exp(-x))
def sigmoidCustom_cpu(x):
    """Custom sigmoid function."""
    return 1 / (1 + np.exp(-x))


def predict_and_process(pred_proba, threshold, config):
    """Applique la sigmoid et le seuillage sur les prédictions.

    Args:
        pred_proba: array de probabilités brutes
        threshold: seuil de classification
        config: dict contenant la configuration avec la clé 'device_'

    Returns:
        tuple: (probabilités après sigmoid, prédictions binaires)
    """
    if config['device_'] == 'cpu':
        pred_proba = sigmoidCustom_cpu(pred_proba)
        pred_proba = np.clip(pred_proba, 0.0, 1.0)
        pred = (pred_proba > threshold).astype(np.int32)
    else:
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


def compute_confusion_matrix_cpu(y_true, y_pred, config):
    """Calcule la matrice de confusion.

    Args:
        y_true: Labels réels
        y_pred: Prédictions
        config: dict contenant la configuration avec la clé 'device_'

    Returns:
        tuple: (TN, FP, FN, TP) - éléments de la matrice de confusion
    """
    if config['device_'] == 'cpu':
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))
    else:
        TP = cp.sum((y_true == 1) & (y_pred == 1))
        TN = cp.sum((y_true == 0) & (y_pred == 0))
        FP = cp.sum((y_true == 0) & (y_pred == 1))
        FN = cp.sum((y_true == 1) & (y_pred == 0))

    return TN, FP, FN, TP


def predict_and_compute_metrics(model, X_data, y_true, best_iteration, threshold, config):
    """
    Effectue les prédictions et calcule les métriques de confusion pour un jeu de données.

    Args:
        model: Le modèle entraîné
        X_data: Les features d'entrée
        y_true: Les labels réels
        best_iteration: Le meilleur nombre d'itérations du modèle
        threshold: Le seuil de classification
        config: Dictionnaire de configuration

    Returns:
        tuple: (pred_proba, predictions, (tn, fp, fn, tp), y_true_converted)
    """
    # Prédictions
    pred_proba = model.predict(X_data, iteration_range=(0, best_iteration))

    # Conversion GPU si nécessaire
    if config['device_'] != 'cpu':
        import cupy as cp
        pred_proba = cp.asarray(pred_proba, dtype=cp.float32)

    # Application du seuil et processing
    pred_proba, predictions = predict_and_process(pred_proba, threshold, config)

    # Conversion des données pour la matrice de confusion
    if config['device_'] != 'cpu':
        import cupy as cp
        y_true_converted = cp.asarray(y_true) if isinstance(y_true, (np.ndarray, pd.Series)) else y_true
        predictions_converted = cp.asarray(predictions) if isinstance(predictions,
                                                                      (np.ndarray, pd.Series)) else predictions
    else:
        y_true_converted = y_true
        predictions_converted = predictions

    # Calcul de la matrice de confusion
    tn, fp, fn, tp = compute_confusion_matrix_cpu(y_true_converted, predictions_converted, config)

    return pred_proba, predictions_converted, (tn, fp, fn, tp), y_true_converted


def log_cv_fold_metrics(X_train_full, X_train_cv_pd, X_val_cv_pd, val_pos, Y_train_cv, y_val_cv, tp_train, fp_train,
                        tn_train, fn_train, tp_val, fp_val, tn_val, fn_val, config, fold_num, nb_split_tscv, fold_raw_data):
    """
    Log les métriques de validation croisée au format standardisé.

    Format:
    === Fold N/Total ===
    === Ranges ===
    X_train_full index | Train pos range | Val pos range
    === TRAIN ===
    Distribution (Avant optimisation): stats
    Métriques (après optimisation): métriques
    === VALIDATION ===
    Période: dates et durée
    Distribution (Avant optimisation): stats
    Métriques (après optimisation): métriques

    Args:
        X_train_full (pd.DataFrame): Données d'entraînement complètes
        X_train_cv (pd.DataFrame): Données d'entraînement du fold
        X_val_cv_pd (pd.DataFrame): Données de validation du fold
        val_pos (array-like): Positions des données de validation
        Y_train_cv (cp.ndarray): Labels d'entraînement
        y_val_cv (cp.ndarray): Labels de validation
        tp_train, fp_train, tn_train, fn_train (int): Métriques d'entraînement
        tp_val, fp_val, tn_val, fn_val (int): Métriques de validation
        config (dict): Configuration
        fold_num (int): Numéro du fold courant
        nb_split_tscv (int): Nombre total de folds
        fold_data (dict): Données du fold

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

        # === Fold et Ranges ===
        print(f"\n============ Fold {fold_num + 1}/{nb_split_tscv} ============")
        print("=== Ranges ===")

        train_pos = fold_raw_data['train_indices']
        val_pos = fold_raw_data['val_indices']
        print(f"X_train_full index: {X_train_full.index.min()}-{X_train_full.index.max()} | "
              f"Train pos range: {min(train_pos)}-{max(train_pos)} | "
              f"Val pos range: {min(val_pos)}-{max(val_pos)}")

        # === TRAIN ===
        print("=== TRAIN ===")
        # Période
        start_time, end_time, _ = get_val_cv_time_range(X_full=X_train_full, X=X_train_cv_pd)
        time_diff = calculate_time_difference(
            timestamp_to_date_utc_(start_time),
            timestamp_to_date_utc_(end_time)
        )
        print(f"Période: Du {timestamp_to_date_utc_(start_time)} au {timestamp_to_date_utc_(end_time)} "
              f"(Durée: {time_diff.months} mois, {time_diff.days} jours)")

        # Distribution avant optimisation
        train_dist = fold_raw_data['distributions']['train']
        total_train = sum(train_dist.values())
        trades_reussis_train = train_dist.get(1, 0)
        trades_echoues_train = train_dist.get(0, 0)
        winrate_train = (trades_reussis_train / total_train * 100) if total_train > 0 else 0
        ratio_train = trades_reussis_train / trades_echoues_train if trades_echoues_train > 0 else float('inf')

        print("Distribution (Avant optimisation):")
        print(
            f"- Total: {total_train} échantillons | Trades réussis: {trades_reussis_train} | Trades échoués: {trades_echoues_train}")
        print(f"- Winrate: {winrate_train:.2f}%")
        print(f"- Ratio réussis/échoués: {ratio_train:.2f}")

        # Métriques après optimisation
        print("Métriques (après optimisation):")
        total_trades_train = tp_train + fp_train
        winrate_opti_train = (tp_train / total_trades_train * 100) if total_trades_train > 0 else 0
        print(f"- TP: {tp_train} | FP: {fp_train} | TN: {tn_train} | FN: {fn_train}")
        print(
            f"- Total trades pris: {total_trades_train} ({(total_trades_train / total_train * 100):.2f}% des échantillons)")
        print(f"- Winrate: {winrate_opti_train:.2f}%")

        # === VALIDATION ===
        print("=== VALIDATION ===")

        # Période
        start_time, end_time, _ = get_val_cv_time_range(X_full=X_train_full, X=X_val_cv_pd)
        time_diff = calculate_time_difference(
            timestamp_to_date_utc_(start_time),
            timestamp_to_date_utc_(end_time)
        )
        print(f"Période: Du {timestamp_to_date_utc_(start_time)} au {timestamp_to_date_utc_(end_time)} "
              f"(Durée: {time_diff.months} mois, {time_diff.days} jours)")

        # Distribution avant optimisation
        val_dist = fold_raw_data['distributions']['val']
        total_val = sum(val_dist.values())
        trades_reussis_val = val_dist.get(1, 0)
        trades_echoues_val = val_dist.get(0, 0)
        winrate_val = (trades_reussis_val / total_val * 100) if total_val > 0 else 0
        ratio_val = trades_reussis_val / trades_echoues_val if trades_echoues_val > 0 else float('inf')

        print("Distribution (Avant optimisation):")
        print(
            f"- Total: {total_val} échantillons | Trades réussis: {trades_reussis_val} | Trades échoués: {trades_echoues_val}")
        print(f"- Winrate: {winrate_val:.2f}%")
        print(f"- Ratio réussis/échoués: {ratio_val:.2f}")

        # Métriques après optimisation
        print("Métriques (après optimisation):")
        total_trades_val = tp_val + fp_val
        winrate_opti_val = (tp_val / total_trades_val * 100) if total_trades_val > 0 else 0
        print(f"- TP: {tp_val} | FP: {fp_val} | TN: {tn_val} | FN: {fn_val}")
        print(f"- Total trades pris: {total_trades_val} ({(total_trades_val / total_val * 100):.2f}% des échantillons)")
        print(f"- Winrate: {winrate_opti_val:.2f}%")

        # Mise à jour du dictionnaire des métriques
        metrics.update({
            'start_time': start_time,
            'end_time': end_time,
            'time_diff': time_diff,
            'train_metrics': {
                'total': total_train,
                'trades_reussis': trades_reussis_train,
                'trades_echoues': trades_echoues_train,
                'winrate': winrate_train,
                'ratio': ratio_train,
                'tp': tp_train, 'fp': fp_train,
                'tn': tn_train, 'fn': fn_train,
                'total_trades_pris': total_trades_train,
                'winrate_opti': winrate_opti_train
            },
            'val_metrics': {
                'total': total_val,
                'trades_reussis': trades_reussis_val,
                'trades_echoues': trades_echoues_val,
                'winrate': winrate_val,
                'ratio': ratio_val,
                'tp': tp_val, 'fp': fp_val,
                'tn': tn_val, 'fn': fn_val,
                'total_trades_pris': total_trades_val,
                'winrate_opti': winrate_opti_val
            }
        })

        return metrics

    except Exception as e:
        print(f"Erreur dans log_cv_fold_metrics: {str(e)}")
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
    """
    if X_full.empty:
        raise ValueError("X_full ne peut pas etre vide")

    if X.empty:
        raise ValueError("X ne peut pas etre vide")

    """
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
            print(index_val)
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


def compute_winrate_safe(tp, total_trades, config):
    """Calcul sécurisé du winrate sur GPU/CPU"""
    # Détermine la bibliothèque à utiliser (CuPy ou NumPy)
    xp = cp if config['device_'] == 'cuda' else np

    # Utilise un masque pour éviter toute division par zéro
    mask = total_trades != 0
    result = xp.zeros_like(total_trades, dtype=xp.float32)  # Initialise avec 0.0
    result[mask] = tp[mask] / total_trades[mask]  # Effectue la division uniquement pour les indices valides

    return result



def log_fold_info(fold_num, nb_split_tscv, X_train_full, fold_data):
    """
    Affiche les informations détaillées sur le fold courant.

    Args:
        fold_num (int): Numéro du fold actuel
        nb_split_tscv (int): Nombre total de folds
        X_train_full (pd.DataFrame): Données d'entraînement complètes
        fold_data (dict): Données du fold préparées
    """
