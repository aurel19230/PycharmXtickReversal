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
    TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVTRAIN = 1
    TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVVAL = 2
    TIMESERIES_SPLIT_BY_ID = 3
    K_FOLD = 4
    K_FOLD_SHUFFLE = 5

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


class model_custom_objective(Enum):
    LGB_CUSTOM_OBJECTIVE_PROFITBASED=0
    LGB_CUSTOM_OBJECTIVE_BINARY = 10
    LGB_CUSTOM_OBJECTIVE_CROSS_ENTROPY = 11
    LGB_CUSTOM_OBJECTIVE_CROSS_ENTROPY_LAMBDA = 12
    XGB_CUSTOM_OBJECTIVE_PROFITBASED = 20


class model_custom_metric(Enum):
    LGB_CUSTOM_METRIC_PNL=0
    XGB_METRIC_ROCAUC = 10
    XGB_METRIC_AUCPR = 11
    XGB_METRIC_F1 = 12
    XGB_METRIC_PRECISION = 13
    XGB_METRIC_RECALL = 14
    XGB_METRIC_MCC = 15
    XGB_METRIC_YOUDEN_J = 16
    XGB_METRIC_SHARPE_RATIO = 17
    XGB_CUSTOM_METRIC_PNL = 18

class scalerChoice(Enum):
    SCALER_DISABLE = 0
    SCALER_STANDARD = 1
    SCALER_ROBUST = 2
    SCALER_MINMAX = 3  # Nouveau : échelle [0,1]
    SCALER_MAXABS = 4  # Nouveau : échelle [-1,1]

class AutoFilteringOptions(Enum):
    DISPLAY_MODE_NOFILTERING = 0
    ENABLE_VIF_CORR_MI = 1
    ENABLE_MRMR = 2
    ENABLE_FISHER=3

###############################################################################
# Classe d'options de filtrage
###############################################################################



class ScalerMode(Enum):
    FIT_TRANSFORM = 0  # Pour l'entraînement : fit + transform
    TRANSFORM = 1
     # Pour le test : transform uniquement
def check_label_pnl_alignment(data):
    """
    Vérifie que dans data, pour chaque échantillon où y_train_no99_fullRange == 1,
    y_pnl_data_train_no99_fullRange > 0.
    """
    # Extraire les tableaux
    y_labels = data['y_train_no99_fullRange']
    y_pnl = data['y_pnl_data_train_no99_fullRange']

    # Vérifier la même longueur
    if len(y_labels) != len(y_pnl):
        raise ValueError(f"check_label_pnl_alignment Taille différente: y_labels={len(y_labels)}, y_pnl={len(y_pnl)}")

    # Créer un masque pour les labels == 1
    mask_positive = (y_labels == 1)

    # Vérifier que tous les PnL correspondants sont > 0
    if not np.all(y_pnl[mask_positive] > 0):
        # Récupérer les indices où la condition échoue
        bad_indices = np.where(y_pnl[mask_positive] <= 0)[0]
        raise ValueError(
            f"Pour y_train_no99_fullRange==1, certains PnL ne sont pas > 0. Indices relatifs (dans les positifs)={bad_indices}"
        )
    else:
        print("Vérification OK : tous les échantillons label=1 ont un PnL strictement > 0.")

import numpy as np
def prepare_dataSplit_cv_train_val(config, data, train_pos, val_pos):
    """
    Prépare les données d'entraînement et de validation pour la validation croisée,
    en gérant la conversion GPU/CPU selon la configuration.

    Parameters:
    -----------
    config : dict
        Dictionnaire de configuration contenant la clé 'device_' avec la valeur 'cpu' ou 'gpu'.
    data : dict
        Dictionnaire contenant les données nécessaires pour les folds.
    train_pos : np.ndarray
        Positions des échantillons pour l'entraînement.
    val_pos : np.ndarray
        Positions des échantillons pour la validation.

    Returns:les
    --------
    dict : Contient les données transformées.
    """
    #check_label_pnl_alignment(data)
    # Extract fold data
    X_train_cv = data['X_train_no99_fullRange'][train_pos].reshape(len(train_pos), -1)
    # X_train_cv_pd reste un DataFrame
    X_train_cv_pd = data['X_train_no99_fullRange_pd'].iloc[train_pos]
    Y_train_cv = data['y_train_no99_fullRange'][train_pos].reshape(-1)

    X_val_cv = data['X_train_no99_fullRange'][val_pos].reshape(len(val_pos), -1)
    # X_val_cv_pd reste un DataFrame
    X_val_cv_pd = data['X_train_no99_fullRange_pd'].iloc[val_pos]
    y_val_cv = data['y_train_no99_fullRange'][val_pos].reshape(-1)

    y_pnl_data_train_cv = data['y_pnl_data_train_no99_fullRange'][train_pos].reshape(-1)
    y_pnl_data_val_cv=data['y_pnl_data_train_no99_fullRange'][val_pos].reshape(-1)

    # Handle GPU/CPU conversion
    if config['device_'] != 'cpu':
        import cupy as cp

        # Conversion GPU -> CPU (CuPy -> NumPy)
        X_train_cv = cp.asnumpy(X_train_cv)
        X_val_cv = cp.asnumpy(X_val_cv)
        Y_train_cv = cp.asnumpy(Y_train_cv)
        y_val_cv = cp.asnumpy(y_val_cv)
        y_pnl_data_train_cv=cp.asnumpy(y_pnl_data_train_cv);
        y_pnl_data_val_cv=cp.asnumpy(y_pnl_data_val_cv);

    else:
        # Pas de conversion nécessaire si on est déjà sur CPU
        pass


    return X_train_cv,X_train_cv_pd,Y_train_cv,X_val_cv,X_val_cv_pd, y_val_cv,y_pnl_data_train_cv,y_pnl_data_val_cv

def sigmoidCustom(x):
# Supposons que x est déjà un tableau CuPy
  return 1 / (1 + cp.exp(-x))


def verify_alignment(y_true_class_binaire, y_pnl_data):
    """
    Vérifie l'alignement entre les classes et le signe des PnL théoriques:
    - Si y_true_class_binaire == 1, alors y_pnl_data doit être > 0
    - Si y_true_class_binaire == 0, alors y_pnl_data doit être < 0

    Lève une ValueError si la condition n'est pas respectée.
    """
    import numpy as np

    # Convertir en tableaux numpy si ce n'est pas déjà le cas
    y_true_array = np.array(y_true_class_binaire)
    y_pnl_array = np.array(y_pnl_data)

    if np.any(y_pnl_array == 0):
        zero_indices = np.where(y_pnl_array == 0)[0]
        raise ValueError(
            f"La variable y_pnl_array contient des valeurs exactement égales à 0 ce qui n'est pas possible {zero_indices}")

    # Vérifier l'alignement des classes positives (y_true == 1)
    misaligned_positives = np.where((y_true_array.astype(bool)) & (y_pnl_array < 0))[0]
    #print(y_true_array.shape)
    #print(y_pnl_array.shape)


    if len(misaligned_positives) > 0:
        raise ValueError(f"Désalignement détecté: y_true == 1 mais y_pnl_data < 0 aux indices {misaligned_positives}")

    # Vérifier l'alignement des classes négatives (y_true == 0)
    misaligned_negatives = np.where((~y_true_array.astype(bool)) & (y_pnl_array > 0))[0]

    if len(misaligned_negatives) > 0:
        raise ValueError(f"Désalignement détecté: y_true == 0 mais y_pnl_data > 0 aux indices {misaligned_negatives}")

    #print("Vérification réussie: Les classes 0 et 1 sont correctement alignées avec le signe des PnL théoriques.")
def calculate_profitBased(y_true_class_binaire, y_pred_threshold, y_pnl_data_train_cv=None,y_pnl_data_val_cv_OrTest=None,
                          metric_dict=None, config=None):
    """
    Calcule les métriques de profit en utilisant les valeurs PNL réelles des trades
    avec des opérations vectorisées pour plus d'efficacité
    """
    """
    unique_values, counts = np.unique(y_pnl_data_train_cv, return_counts=True)

    # Créer un DataFrame pour un affichage plus clair
    distribution_df = pd.DataFrame({
        'Valeur': unique_values,
        'Fréquence': counts,
        'Pourcentage': (counts / len(y_pnl_data_train_cv) * 100).round(2)
    })

    # Trier par fréquence décroissante
    distribution_df = distribution_df.sort_values(by='Fréquence', ascending=False)

    # Afficher le nombre total de valeurs uniques
    print(f"Nombre total de valeurs différentes: {len(unique_values)}")
    
    # Afficher la distribution
    print("\nDistribution des valeurs:")
    print(distribution_df)
    """
    # Conversion en numpy arrays pour un traitement efficace
    y_true_class_binaire = np.asarray(y_true_class_binaire)
    y_pred = np.asarray(y_pred_threshold)

    # Vérification de la configuration
    if config is None:
        raise ValueError("config ne peut pas être None dans calculate_profitBased")



    # Récupération des paramètres
    penalty_per_fn = metric_dict.get('penalty_per_fn', config.get('penalty_per_fn', 0))
    #y_pnl_data_train_cv = config.get('y_pnl_data_train_cv', None)
    #y_pnl_data_val_cv_OrTest = config.get('y_pnl_data_val_cv_OrTest', None)

    # Identification du jeu de données (train ou validation)
    if len(y_true_class_binaire) == len(y_pnl_data_train_cv):
        y_pnl_data = y_pnl_data_train_cv
    elif len(y_true_class_binaire) == len(y_pnl_data_val_cv_OrTest):
        y_pnl_data = y_pnl_data_val_cv_OrTest
    else:
        raise ValueError(f"Impossible d'identifier l'ensemble de données. Taille actuelle: {len(y_true_class_binaire)}, "
                         f"Taille train: {len(y_pnl_data_train_cv)}, "
                         f"Taille val pour val de val croisée ou ensemble test pour entrainement final y_pnl_data_val_cv_OrTest: {len(y_pnl_data_val_cv_OrTest)}")

    #print(f"Données conforme. Taille actuelle, y_true_class_binaire: {len(y_true_class_binaire)}, "
        #        f"Nombre de 0 y_true_class_binaire y_true_class_binaire: {np.sum(y_true_class_binaire == 0)}, "
        # f"Taille train, y_pnl_data_train_cv: {len(y_pnl_data_train_cv)}, "
     #f"Taille val pour val de val croisée ou ensemble test pour entrainement final,y_pnl_data_val_cv_OrTest: {len(y_pnl_data_val_cv_OrTest)}")

    # Vérification de l'alignement des données
    if not (len(y_true_class_binaire) == len(y_pred) == len(y_pnl_data)):
        raise ValueError(
            f"Erreur d'alignement: y_true ({len(y_true_class_binaire)} lignes), y_pred ({len(y_pred)} lignes), y_pnl_data ({len(y_pnl_data)} lignes)")

    # Vérification de l'absence de zéros dans y_true
    if 0 in y_pnl_data_val_cv_OrTest:
        nb_zeros = np.sum(y_pnl_data_val_cv_OrTest == 0)
        raise ValueError(f"La variable y_true contient des zéros, ce qui n'est pas autorisé pour SL et TP. NB 0:{nb_zeros}")

    try:
        verify_alignment(y_true_class_binaire, y_pnl_data)  # y_true et y_pnl_data sont des séries ou tableaux
        # Continuer avec le reste du traitement si la vérification est réussie
    except ValueError as e:
        print(f"Erreur: {e}")
        # Gérer l'erreur en la relançant pour arrêter l'exécution
        raise ValueError(f"Vérification échouée: {e}")

    # Création de masques booléens pour chaque cas (opérations vectorisées)
    tp_mask = (y_true_class_binaire == 1) & (y_pred == 1)
    fp_mask = (y_true_class_binaire == 0) & (y_pred == 1)
    fn_mask = (y_true_class_binaire == 1) & (y_pred == 0)

    # Comptage des événements
    tp = np.sum(tp_mask)
    fp = np.sum(fp_mask)
    fn = np.sum(fn_mask)

    # Calcul vectorisé des profits/pertes
    # Pour les profits (TP), on prend les valeurs positives aux indices tp_mask
    pnl_values = y_pnl_data.values if hasattr(y_pnl_data, 'values') else y_pnl_data

    # Calcul du profit pour les vrais positifs (prend uniquement les valeurs positives)
    tp_profits = np.sum(np.maximum(0, pnl_values[tp_mask])) if tp > 0 else 0

    # Calcul des pertes pour les faux positifs (prend uniquement les valeurs négatives ou nulles)
    fp_losses = np.sum(np.minimum(0, pnl_values[fp_mask])) if fp > 0 else 0

    # Calcul de la pénalité pour les faux négatifs
    fn_penalty = fn * penalty_per_fn

    # Calcul du profit total
    total_profit = tp_profits + fp_losses + fn_penalty

    return float(total_profit), int(tp), int(fp)

# Fonction sigmoid pour XGBoost
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
    model_type = config['model_type']
    if model_type == modelType.XGB:
        import xgboost as xgb
       # dData = xgb.DMatrix(X_data)
        pred_proba_log_odds = model.predict(X_data, iteration_range=(0, best_iteration), output_margin=True)
        #print("pred_proba_log_odds ",pred_proba_log_odds)
    elif model_type == modelType.LGBM:
        pred_proba_log_odds = model.predict(X_data, num_iteration=best_iteration, raw_score=True)
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


def compute_raw_train_dist(X_train_full, X_train_cv_pd, X_val_cv_pd, fold_num, nb_split_tscv, fold_raw_data
                           ,is_log_enabled,df_init_candles):
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
        time_diff_train = calculate_time_difference(
            timestamp_to_date_utc_(start_time_train),
            timestamp_to_date_utc_(end_time_train)
        )



        # Période
        start_time_val, end_time_val, _ = get_val_cv_time_range(X_full=X_train_full, X=X_val_cv_pd)
        time_diff_val = calculate_time_difference(
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


        # indice consécutif pour les données d'entraînement sur le fold
        train_start = X_train_cv_pd.index[0]  # Premier indice
        train_end = X_train_cv_pd.index[-1]  # Dernier indice
        close_cv_train = df_init_candles['close'].loc[train_start:train_end]
        high_cv_train = df_init_candles['high'].loc[train_start:train_end]
        low_cv_train = df_init_candles['low'].loc[train_start:train_end]

        #slope_cv_train = linear_regression_slope_market_trend(close_cv_train)
        slope_cv_train,r2_slope_cv_train,counter_moves_train=calculate_trend_strength(close_cv_train,high_cv_train,low_cv_train)

        # indice consécutif pour les données de validation sur le fold
        val_start = X_val_cv_pd.index[0]
        val_end = X_val_cv_pd.index[-1]
        close_cv_val = df_init_candles['close'].loc[val_start:val_end]
        high_cv_val = df_init_candles['high'].loc[val_start:val_end]
        low_cv_val = df_init_candles['low'].loc[val_start:val_end]

        #slope_cv_val = linear_regression_slope_market_trend(close_cv_val)
        slope_cv_val,r2_slope_cv_val,counter_moves_val=calculate_trend_strength(close_cv_val,high_cv_val,low_cv_val)


        # Mise à jour du dictionnaire des métriques
        raw_metrics={
            'train_metrics': {
                'total_samples_train': total_samples_train,
                'class1_train': class1_train,
                'class0_train': class0_train,
                'winrate': winrate_train,
                'ratio_train': ratio_train,
                'start_time_train': timestamp_to_date_utc_(start_time_train),
                'end_time_train': timestamp_to_date_utc_(end_time_train),
                'slope_cv_train':slope_cv_train,
                'r2_slope_cv_train': r2_slope_cv_train,
                'counter_moves_train': counter_moves_train,
            },
            'val_metrics': {
                'total': total_samples_val,
                'class1_val': class1_val,
                'class0_val': class0_val,
                'winrate': winrate_val,
                'ratio_val': ratio_val,
                'start_time_val': timestamp_to_date_utc_(start_time_val),
                'end_time_val': timestamp_to_date_utc_(end_time_val),
                'slope_cv_val': slope_cv_val,
                'r2_slope_cv_val':r2_slope_cv_val,
                'counter_moves_val': counter_moves_val
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
                  f"(Durée: {time_diff_train.months} mois, {time_diff_train.days} jours)")



            print("Indices d'entraînement :")
            print(f"De {train_start} à {train_end}")
            print("Valeurs des closes correspondantes de df_init_candles pour l'entraînement :")
            vals_close_train = close_cv_train.tolist()
            print(f"[{', '.join(map(str, vals_close_train[:10]))} ... {', '.join(map(str, vals_close_train[-10:]))}]")
            print(f"Pente de la régression linéaire pour l'entraînement: {slope_cv_train}\n")

            print("Distribution (Avant optimisation):")
            print(
                f"- Total: {total_samples_train} échantillons | class1: {class1_train} | class0: {class0_train}")
            print(f"- Winrate: {winrate_train:.2f}%")
            print(f"- Ratio réussis/échoués: {ratio_train:.2f}")

            # Métriques après optimisation
            print("Métriques (après optimisation):")


            # === VALIDATION ===
            print("\n=== VALIDATION ===")
            print(f"Période: Du {timestamp_to_date_utc_(start_time_val)} au {timestamp_to_date_utc_(end_time_val)} "
                  f"(Durée: {time_diff_val.months} mois, {time_diff_val.days} jours)")
            # Pour les données de validation

            print("Indices de validation :")
            print(f"De {val_start} à {val_end}")
            print("Valeurs des closes correspondantes de df_init_candles pour la validation :")
            vals_close_val = close_cv_val.tolist()
            print(f"[{', '.join(map(str, vals_close_val[:10]))} ... {', '.join(map(str, vals_close_val[-10:]))}]")
            print(f"Pente de la régression linéaire pour l'entraînement: {slope_cv_val}\n")

            print("Distribution (Avant optimisation):")
            print(
                f"- Total: {total_samples_val} échantillons | Trades réussis: {class1_val} | Trades échoués: {class0_val}")
            print(f"- Winrate: {winrate_val:.2f}%")
            print(f"- Ratio réussis/échoués: {ratio_val:.2f}")

        return raw_metrics
    except Exception as e:
        print(f"#########################################")
        print(f"Erreur dans compute_raw_train_dist: {str(e)}")
        print(f"#########################################")
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

import ta
def calculate_trend_strength(close=None, high=None, low=None):
    """
    Calcule un score de force de tendance entre 0 et 100
    en combinant plusieurs métriques
    """
    try:
        # 1. Régression linéaire pour la direction et la régularité
        X = np.arange(len(close)).reshape(-1, 1)
        y = close.values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)

        # Force de la pente (normalisée)
        slope = reg.coef_[0][0]
        r2 = reg.score(X, y)  # R² pour mesurer la linéarité
        # 3. Analyse améliorée des contre-tendances avec high/low
        if slope > 0:  # Tendance haussière
            # Un counter move significatif est quand :
            # - Le low descend sous le low précédent
            # - ET/OU le high n'arrive pas à dépasser le high précédent
            low_breaks = (low+1.5 < low.shift(1)).sum()
            high_fails = (high < high.shift(1)).sum()
            counter_moves = (low_breaks ) / ( len(close))
        else:  # Tendance baissière
            # Un counter move significatif est quand :
            # - Le high monte au-dessus du high précédent
            # - ET/OU le low n'arrive pas à descendre sous le low précédent
            high_breaks = (high-1.5 > high.shift(1)).sum()
            low_fails = (low > low.shift(1)).sum()
            counter_moves = (high_breaks ) / ( len(close))

        # On pourrait aussi ajouter l'amplitude
        if slope > 0:
            counter_amplitude = abs((low - low.shift(1)) / low.shift(1))[low < low.shift(1)].mean()
        else:
            counter_amplitude = abs((high - high.shift(1)) / high.shift(1))[high > high.shift(1)].mean()




        """
        # 2. Calcul des drawdowns pour évaluer les pullbacks
        price_series = pd.Series(series)
        rolling_max = price_series.expanding().max()
        drawdowns = (price_series - rolling_max) / rolling_max * 100
        max_drawdown = abs(drawdowns.min())
    
        
    
        # 4. Calcul du score composite
        trend_score = (
                abs(slope) * 40 +  # Force de la direction (40% du score)
                r2 * 30 +  # Régularité (30% du score)
                (1 - counter_moves) * 20 +  # Continuité (20% du score)
                (1 - max_drawdown / 100) * 10  # Résistance aux pullbacks (10% du score)
        )
    
        return {
            'trend_score': trend_score,
            'slope': slope,
            'r2': r2,
            'max_drawdown': max_drawdown,
            'counter_moves_ratio': counter_moves
        }
        """



        return slope,r2,counter_moves
    except Exception as e:
        print(f"Error in calculate_trend_strength: {str(e)}")
        raise e  # On relance l'exception capturée


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


def compute_sample_weights(Y_train, y_val):
    """Calculer les poids pour l'entraînement et la validation."""
    N0 = np.sum(Y_train == 0)
    N1 = np.sum(Y_train == 1)
    N = len(Y_train)
    w_0 = N / N0
    w_1 = N / N1
    sample_weights_train = np.where(Y_train == 1, w_1, w_0)
    sample_weights_val = np.ones(len(y_val))
    return sample_weights_train, sample_weights_val


def get_best_iteration(evals_result, custom_metric_eval):
    """
    Extraction de la meilleure itération en fonction du custom metric.
    Retourne : (best_iteration, best_idx, val_best, train_best)
    """
    if custom_metric_eval in (model_custom_metric.XGB_CUSTOM_METRIC_PNL,
                              model_custom_metric.LGB_CUSTOM_METRIC_PNL):
        eval_scores = evals_result['eval']['custom_metric_PNL']
        train_scores = evals_result['train']['custom_metric_PNL']
    else:
        eval_scores = evals_result['eval']['auc']
        train_scores = evals_result['train']['auc']
        raise ValueError("Fonction metric non reconnue, impossible de continuer.")

    val_best = max(eval_scores)
    best_idx = eval_scores.index(val_best)
    best_iteration = best_idx + 1
    train_best = train_scores[best_idx]
    return best_iteration, best_idx, val_best, train_best


def compile_fold_stats(fold_num, best_iteration, train_pred_proba_log_odds, train_metrics, winrate_train,
                       tp_fp_sum_train, tp_fp_tn_fn_sum_train, train_best, train_pos, train_trades_samples_perct,
                       val_pred_proba_log_odds, val_metrics, winrate_val, tp_fp_sum_val, tp_fp_tn_fn_sum_val,
                       val_best, val_pos, val_trades_samples_perct, ratio_difference,
                       perctDiff_ratioTradeSample_train_val):
    """Compilation des statistiques du fold dans un dictionnaire commun."""
    return {
        'fold_num': fold_num,
        'best_iteration': best_iteration,
        'train_pred_proba_log_odds': train_pred_proba_log_odds,
        'train_metrics': train_metrics,
        'train_winrate': winrate_train,
        'train_trades': tp_fp_sum_train,
        'train_samples': tp_fp_tn_fn_sum_train,
        'train_bestIdx_custom_metric_pnl': train_best,
        'train_size': len(train_pos) if train_pos is not None else None,
        'train_trades_samples_perct': train_trades_samples_perct,
        'val_pred_proba_log_odds': val_pred_proba_log_odds,
        'val_metrics': val_metrics,
        'val_winrate': winrate_val,
        'val_trades': tp_fp_sum_val,
        'val_samples': tp_fp_tn_fn_sum_val,
        'val_bestIdx_custom_metric_pnl': val_best,
        'val_size': len(val_pos) if val_pos is not None else None,
        'val_trades_samples_perct': val_trades_samples_perct,
        'perctDiff_winrateRatio_train_val': ratio_difference,
        'perctDiff_ratioTradeSample_train_val': perctDiff_ratioTradeSample_train_val
    }


def compile_debug_info(model_weight_optuna, config, val_pred_proba, y_train_predProba):
    """Construction du dictionnaire de debug."""
    xp = np if config['device_'] != 'cuda' else cp
    return {
        'threshold_used': model_weight_optuna['threshold'],
        'pred_proba_ranges': {
            'val': {
                'min': float(xp.min(val_pred_proba)),
                'max': float(xp.max(val_pred_proba))
            },
            'train': {
                'min': float(xp.min(y_train_predProba)),
                'max': float(xp.max(y_train_predProba))
            }
        }
    }

