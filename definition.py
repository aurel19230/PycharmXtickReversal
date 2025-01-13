from enum import Enum
import numpy as np
import platform
from termcolor import colored
from colorama import Fore, Style, init

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
    LGB_CUSTOM_METRIC_FOCALLOSS = 1
    XGB_METRIC_ROCAUC = 11
    XGB_METRIC_AUCPR = 12
    XGB_METRIC_F1 = 13
    XGB_METRIC_PRECISION = 14
    XGB_METRIC_RECALL = 15
    XGB_METRIC_MCC = 16
    XGB_METRIC_YOUDEN_J = 17
    XGB_METRIC_SHARPE_RATIO = 18
    XGB_CUSTOM_METRIC_PROFITBASED = 19
    XGB_CUSTOM_METRIC_TP_FP = 20

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
    x = np.array(x, dtype=np.float64)  # Conversion explicite
    return 1 / (1 + np.exp(-x))



def predict_and_process(pred_proba_raw, threshold, config):
    """Applique la sigmoid et le seuillage sur les prédictions.

    Args:
        pred_proba: array de probabilités brutes
        threshold: seuil de classification
        config: dict contenant la configuration avec la clé 'device_'

    Returns:
        tuple: (probabilités après sigmoid, prédictions binaires)
    """
    if config['device_'] == 'cpu':
        pred_proba_afterSig = sigmoidCustom_cpu(pred_proba_raw)
        pred_proba_afterSig = np.clip(pred_proba_afterSig, 0.0, 1.0)
        pred = (pred_proba_afterSig > threshold).astype(np.int32)
    else:
        pred_proba_afterSig = sigmoidCustom(pred_proba_raw)
        pred_proba_afterSig = cp.clip(pred_proba_afterSig, 0.0, 1.0)
        pred = (pred_proba_afterSig > threshold).astype(cp.int32)

    return pred_proba_afterSig, pred


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
    pred_proba_log_odds = model.predict(X_data, iteration_range=(0, best_iteration),raw_score=True)
    #print(f"pred_proba_log_oddsMin: {np.min(pred_proba_log_odds)}, Max: {np.max(pred_proba_log_odds)}")

    # Conversion GPU si nécessaire
    if config['device_'] != 'cpu':
        import cupy as cp
        pred_proba_log_odds = cp.asarray(pred_proba_log_odds, dtype=cp.float32)

    # Application du seuil et processing
    pred_proba_afterSig, predictions = predict_and_process(pred_proba_log_odds, threshold, config)

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

    return pred_proba_afterSig,pred_proba_log_odds, predictions_converted, (tn, fp, fn, tp), y_true_converted


def compute_raw_train_dist(X_train_full, X_train_cv_pd, X_val_cv_pd, tp_train, fp_train,
                        tn_train, fn_train, tp_val, fp_val, tn_val, fn_val, fold_num, nb_split_tscv, fold_raw_data
                           ,is_log_enabled):
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



        # Distribution avant optimisation
        train_dist = fold_raw_data['distributions']['train']
        total_samples_train = sum(train_dist.values())
        class1_train = train_dist.get(1, 0)
        class0_train = train_dist.get(0, 0)
        winrate_train = (class1_train / total_samples_train * 100) if total_samples_train > 0 else 0
        ratio_train = class1_train / class0_train if class0_train > 0 else float('inf')
        train_pos = fold_raw_data['train_indices']
        val_pos = fold_raw_data['val_indices']
        start_time_train, end_time_train, _ = get_val_cv_time_range(X_full=X_train_full, X=X_train_cv_pd)
        time_diff = calculate_time_difference(
            timestamp_to_date_utc_(start_time_train),
            timestamp_to_date_utc_(end_time_train)
        )
        total_trades_train = tp_train + fp_train
        winrate_opti_train = (tp_train / total_trades_train * 100) if total_trades_train > 0 else 0


        # Période
        start_time_val, end_time_val, _ = get_val_cv_time_range(X_full=X_train_full, X=X_val_cv_pd)
        time_diff = calculate_time_difference(
            timestamp_to_date_utc_(start_time_val),
            timestamp_to_date_utc_(end_time_val)
        )

        # Distribution avant optimisation
        val_dist = fold_raw_data['distributions']['val']
        total_samples_val = sum(val_dist.values())
        class1_val = val_dist.get(1, 0)
        class0_val = val_dist.get(0, 0)
        winrate_val = (class1_val / total_samples_val * 100) if total_samples_val > 0 else 0
        ratio_val = class1_val / class0_val if class0_val > 0 else float('inf')


        total_trades_val = tp_val + fp_val
        winrate_opti_val = (tp_val / total_trades_val * 100) if total_trades_val > 0 else 0


        # Mise à jour du dictionnaire des métriques
        raw_metrics={
            'train_metrics': {
                'total_samples_train': total_samples_train,
                'class1_train': class1_train,
                'class0_train': class0_train,
                'winrate': winrate_train,
                'ratio_train': ratio_train,
                'start_time_train': timestamp_to_date_utc_(start_time_train),
                'end_time_train': timestamp_to_date_utc_(end_time_train)
            },
            'val_metrics': {
                'total': total_samples_val,
                'class1_val': class1_val,
                'class0_val': class0_val,
                'winrate': winrate_val,
                'ratio_val': ratio_val,
                'start_time_val': timestamp_to_date_utc_(start_time_val),
                'end_time_val': timestamp_to_date_utc_(end_time_val)
            }
        }
        if is_log_enabled:
            # === Fold et Ranges ===
            print(f"\n============ Fold {fold_num + 1}/{nb_split_tscv} ============")
            print("=== Ranges ===")

            print(f"X_train_full index: {X_train_full.index.min()}-{X_train_full.index.max()} | "
                  f"Train pos range: {min(train_pos)}-{max(train_pos)} | "
                  f"Val pos range: {min(val_pos)}-{max(val_pos)}")

            # === TRAIN ===
            print("=== TRAIN ===")
            # Période

            print(f"Période: Du {timestamp_to_date_utc_(start_time_train)} au {timestamp_to_date_utc_(end_time_train)} "
                  f"(Durée: {time_diff.months} mois, {time_diff.days} jours)")
            print("Distribution (Avant optimisation):")
            print(
                f"- Total: {total_samples_train} échantillons | class1: {class1_train} | class0: {class0_train}")
            print(f"- Winrate: {winrate_train:.2f}%")
            print(f"- Ratio réussis/échoués: {ratio_train:.2f}")

            # Métriques après optimisation
            print("Métriques (après optimisation):")

            print(f"- TP: {tp_train} | FP: {fp_train} | TN: {tn_train} | FN: {fn_train}")
            print(
                f"- Total trades pris: {total_trades_train} ({(total_trades_train / total_samples_train * 100):.2f}% des échantillons)")
            print(f"- Winrate: {winrate_opti_train:.2f}%")

            # === VALIDATION ===
            print("=== VALIDATION ===")
            print(f"Période: Du {timestamp_to_date_utc_(start_time_val)} au {timestamp_to_date_utc_(end_time_val)} "
                  f"(Durée: {time_diff.months} mois, {time_diff.days} jours)")
            print("Distribution (Avant optimisation):")
            print(
                f"- Total: {total_samples_val} échantillons | Trades réussis: {class1_val} | Trades échoués: {class0_val}")
            print(f"- Winrate: {winrate_val:.2f}%")
            print(f"- Ratio réussis/échoués: {ratio_val:.2f}")

            # Métriques après optimisation
            print("Métriques (après optimisation):")
            print(f"- TP: {tp_val} | FP: {fp_val} | TN: {tn_val} | FN: {fn_val}")
            print(
                f"- Total trades pris: {total_trades_val} ({(total_trades_val / total_samples_val * 100):.2f}% des échantillons)")
            print(f"- Winrate: {winrate_opti_val:.2f}%")
        return raw_metrics
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
    """
    Calcule le winrate de manière sécurisée en pourcentage.

    Args:
        tp: Nombre de True Positives (trades gagnants)
        total_trades: Nombre total de trades
        config: Dictionnaire de configuration avec la clé 'device_'

    Returns:
        array: Winrate en pourcentage (0-100) avec gestion des divisions par zéro
    """
    # Sélectionne la bibliothèque appropriée selon le device
    xp = cp if config['device_'] == 'cuda' else np

    # Crée un masque pour identifier les positions où total_trades n'est pas zéro
    mask = total_trades != 0

    # Initialise le tableau de résultats avec des zéros
    result = xp.zeros_like(total_trades, dtype=xp.float32)

    # Calcule le winrate en pourcentage uniquement pour les positions valides
    # Multiplie par 100 pour convertir en pourcentage
    result[mask] = (tp[mask] / total_trades[mask]) * 100

    return result


def compute_winrate_ratio_difference(winrate_train, winrate_val, config):
    """
    Calcule l'écart relatif entre deux winrates sous forme de ratio en pourcentage.

    Args:
        winrate_train: Winrate de l'ensemble d'entraînement (déjà calculé)
        winrate_val: Winrate de l'ensemble de validation (déjà calculé)
        config: Configuration avec le type de device (CPU/GPU)

    Returns:
        array: Écart relatif en pourcentage avec gestion des cas spéciaux
    """
    # Sélectionne la bibliothèque selon le device
    xp = cp if config['device_'] == 'cuda' else np

    # Initialise le résultat avec des zéros
    result = xp.zeros_like(winrate_train, dtype=xp.float32)

    # Crée un masque pour les positions où winrate_train n'est pas zéro
    mask = winrate_train != 0

    # Calcule le ratio uniquement pour les positions valides
    result[mask] = (winrate_train[mask] - winrate_val[mask]) / winrate_train[mask] * 100

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


def calculate_profit_ratio(tp, fp, tp_fp_sum, profit_per_tp, loss_per_fp):
    """
    Calcule le ratio profit par trade avec gestion de la division par zéro

    Args:
        tp: Nombre de True Positives
        fp: Nombre de False Positives
        tp_fp_sum: Somme des trades (TP + FP)
        profit_per_tp: Profit par trade gagnant
        loss_per_fp: Perte par trade perdant

    Returns:
        float: Ratio de profit par trade, 0 si aucun trade
    """
    if tp_fp_sum == 0:
        return 0.0

    return (tp * profit_per_tp + fp * loss_per_fp) / tp_fp_sum


import numpy as np


def calculate_ratio_difference(ratio_train, ratio_val, config):
    """
    Calcule l'écart relatif entre les ratios train et validation
    avec gestion de la division par zéro et compatibilité GPU/CPU

    Args:
        ratio_train: Ratio sur l'ensemble d'entraînement (float)
        ratio_val: Ratio sur l'ensemble de validation (float)
        config: Dictionnaire de configuration contenant la clé 'device_'

    Returns:
        float: Écart relatif en pourcentage
    """
    # Sélection de la bibliothèque appropriée (CuPy ou NumPy)
    xp = cp if config['device_'] == 'cuda' else np

    # Conversion des valeurs en type compatible avec le device
    ratio_train = xp.array(ratio_train, dtype=xp.float32)
    ratio_val = xp.array(ratio_val, dtype=xp.float32)

    # Gestion du cas où ratio_train est 0
    if xp.equal(ratio_train, 0):
        return 0.0 if xp.equal(ratio_val, 0) else 100.0

    # Calcul de l'écart relatif
    result = ((ratio_train - ratio_val) / ratio_train) * 100

    # Arrondir à 2 décimales
    return float(xp.round(result, decimals=2))

def format_constraint_message(is_violated, config, trial, constraint_name, config_key, attribute_key, check_type):
    """
    Formate un message de contrainte de manière uniforme avec les valeurs minimales ou maximales.

    Args:
        is_violated (bool): Indique si la contrainte est violée.
        config (dict): Dictionnaire de configuration contenant les seuils.
        trial (optuna.Trial): Essai Optuna en cours.
        constraint_name (str): Nom de la contrainte pour l'affichage.
        config_key (str): Clé pour accéder au seuil dans config.
        attribute_key (str): Clé pour accéder aux valeurs dans trial.user_attrs.
        check_type (str): Type de comparaison, 'max' ou 'min'.
    """
    # Récupération des valeurs avec gestion des cas vides
    values = trial.user_attrs.get(attribute_key, [])
    threshold = config.get(config_key, 0)

    # Détermine la valeur à afficher (min ou max selon check_type)
    if check_type == 'max':
        value_to_compare = max(values) if values else 0
        comparison = ">" if is_violated else "<="
    elif check_type == 'min':
        value_to_compare = min(values) if values else 0
        comparison = "<" if is_violated else ">="
    else:
        raise ValueError("check_type must be 'max' or 'min'.")

    # Détermine le style du message
    color = Fore.RED if is_violated else Fore.GREEN
    symbol = "-" if is_violated else "\u2713"

    # Formate les valeurs de la liste pour l'affichage
    formatted_values = [f"{x:.2f}" for x in values] if values else []

    # Construction du message
    if is_violated:
        return (
            color +
            f"    {symbol} {constraint_name} {comparison} {threshold} " +
            f"({check_type}: {value_to_compare:.2f}, values: {formatted_values})" +
            Style.RESET_ALL
        )
    else:
        return (
            color +
            f"    {symbol} {constraint_name} {comparison} {threshold}" +
            Style.RESET_ALL
        )

from sklearn.linear_model import LinearRegression

def linear_regression_slope_market_trend(series):
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0][0]
    return slope

def apply_slope_with_session_check(data, window):
    result = pd.Series(index=data.index, dtype=float)

    for idx in result.index:
        historical_data = data.loc[:idx]
        last_session_start = historical_data[::-1]['SessionStartEnd'].eq(10).idxmax()
        bars_since_session_start = len(historical_data.loc[last_session_start:idx])

        if bars_since_session_start >= window:
            series = data.loc[:idx, 'close'].tail(window)
            result[idx] = linear_regression_slope_market_trend(series)
        else:
            result[idx] = np.nan

    return result