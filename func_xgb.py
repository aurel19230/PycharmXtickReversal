import cupy as cp
import numpy as np
import xgboost as xgb
import warnings
from definition import *
from typing import Tuple
import logging

# Fonctions GPU mises à jour

def xgb_weighted_logistic_gradient_Cupygpu(predt, dtrain, w_p, w_n):
    predt_gpu = cp.asarray(predt)
    y_gpu = cp.asarray(dtrain.get_label())

    predt_sigmoid = sigmoidCustom(predt_gpu)
    grad = predt_sigmoid - y_gpu
    # Appliquer les poids après le calcul initial du gradient
    weights = cp.where(y_gpu == 1, w_p, w_n)
    grad *= weights

    return grad  # Retourner directement le tableau CuPy

def xgb_weighted_logistic_hessian_Cupygpu(predt, dtrain, w_p, w_n):
    predt_gpu = cp.asarray(predt)
    y_gpu = cp.asarray(dtrain.get_label())

    predt_sigmoid = sigmoidCustom(predt_gpu)
    hess = predt_sigmoid * (1 - predt_sigmoid)
    # Appliquer les poids après le calcul initial de la hessienne
    weights = cp.where(y_gpu == 1, w_p, w_n)
    hess *= weights

    return hess  # Retourner directement le tableau CuPy

def xgb_create_weighted_logistic_obj_gpu(w_p: float, w_n: float):
    def weighted_logistic_obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        grad = xgb_weighted_logistic_gradient_Cupygpu(predt, dtrain, w_p, w_n)
        hess = xgb_weighted_logistic_hessian_Cupygpu(predt, dtrain, w_p, w_n)
        return grad, hess
    return weighted_logistic_obj

# Fonction pour vérifier la disponibilité du GPU

def xgb_calculate_profitBased_gpu(y_true_gpu, y_pred_threshold_gpu, metric_dict):
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


def xgb_custom_metric_Profit(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict, normalize: bool = False) -> Tuple[
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
    total_profit, tp, fp = xgb_calculate_profitBased_gpu(y_true_gpu, y_pred_threshold_gpu, metric_dict)

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
def xgb_custom_metric_ProfitBased_gpu(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict) -> Tuple[str, float]:
    """Version GPU de la métrique de profit"""
    return xgb_custom_metric_Profit(predt, dtrain, metric_dict, normalize=False)


def xgb_calculate_profitBased_cpu(y_true, y_pred_threshold, metric_dict):
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

def xgb_custom_metric_Profit_cpu(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict, normalize: bool = False) -> Tuple[str, float]:
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

    total_profit, tp, fp = xgb_calculate_profitBased_cpu(y_true, y_pred_threshold, metric_dict)

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

def xgb_custom_metric_ProfitBased_cpu(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict) -> Tuple[str, float]:
    return xgb_custom_metric_Profit_cpu(predt, dtrain, metric_dict, normalize=False)

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

def xgb_create_weighted_logistic_obj_cpu(w_p: float, w_n: float):
    def weighted_logistic_obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        grad = weighted_logistic_gradient_cpu(predt, dtrain, w_p, w_n)
        hess = weighted_logistic_hessian_cpu(predt, dtrain, w_p, w_n)
        return grad, hess
    return weighted_logistic_obj
