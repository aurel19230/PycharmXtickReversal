import numpy as np
import xgboost as xgb
import warnings
from definition import *
from typing import Tuple
import logging

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.expand_frame_repr', False)
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

""""""
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


# Métrique personnalisée pour XGBoost
# Métrique personnalisée pour XGBoost
def xgb_custom_metric_PNL(y_pnl_data_train_cv=None, y_pnl_data_val_cv_OrTest=None,
        metric_dict=None, config=None):
    #print(xgb_custom_metric_PNL)
    def profit_metric(preds, y_true_class_binaire):
        """
        XGBoost custom metric pour le profit
        """
        y_true_class_binaire = y_true_class_binaire.get_label()

        # Application de la sigmoid
        preds = sigmoidCustom_cpu(preds)
        preds = np.clip(preds, 0.0, 1.0)

        # Vérification des prédictions
        min_val = np.min(preds)
        max_val = np.max(preds)
        if min_val < 0 or max_val > 1:
            logging.warning(f"Les prédictions sont hors de l'intervalle [0, 1]: [{min_val:.4f}, {max_val:.4f}]")
            return 'custom_metric_ProfitBased', float('-inf')

        # Application du seuil
        threshold = metric_dict.get('threshold', 0.55555555)
        #print(threshold)
        y_pred_threshold = (preds > threshold).astype(int)

        # Calcul du profit et des métriques
        total_profit, tp, fp = calculate_profitBased(
            y_true_class_binaire=y_true_class_binaire,
            y_pred_threshold=y_pred_threshold,
            y_pnl_data_train_cv=y_pnl_data_train_cv,
            y_pnl_data_val_cv_OrTest=y_pnl_data_val_cv_OrTest,
            metric_dict=metric_dict,
            config=config
        )
        #print(f"[Iter] Profit: {total_profit}, TP: {tp}, FP: {fp}, Threshold: {threshold}")

        # Normalisation éventuelle du profit
        normalize = metric_dict.get('normalize', False)
        if normalize:
            total_trades_val = tp + fp
            if total_trades_val > 0:
                final_profit = total_profit / total_trades_val
            else:
                final_profit = 0.0
            metric_name = 'custom_metric_PNL_norm'
        else:
            final_profit = total_profit
            metric_name = 'custom_metric_PNL'

        return metric_name, float(final_profit)

    return profit_metric

def weighted_logistic_gradient_cpu(predt: np.ndarray, dtrain: xgb.DMatrix, w_p: float, w_n: float) -> np.ndarray:
    """Calcule le gradient pour la perte logistique pondérée (CPU)."""
    print(weighted_logistic_gradient_cpu)
    y = dtrain.get_label()
    predt = 1.0 / (1.0 + np.exp(-predt))  # Fonction sigmoïde
    weights = np.where(y == 1, w_p, w_n)
    grad = weights * (predt - y)
    return grad


# Fonction objective XGBoost - Version ajustée
def xgb_create_weighted_logistic_obj_cpu(w_p: float, w_n: float):

    def weighted_logistic_obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        y = dtrain.get_label()
        #print(f"weighted_logistic_obj y:{y} \ predt:{predt} \ w_p:{w_p} \ w_n:{w_n}")
        predt = sigmoidCustom_cpu(predt)

        # Calcul du gradient et de la hessienne avec les poids
        weights = 1#np.where(y == 1, w_p, w_n)
        grad = weights * (predt - y)
        hess = weights * predt * (1.0 - predt)

        return grad, hess

    return weighted_logistic_obj


def train_and_evaluate_xgb_model(
        X_train_cv=None,
        X_val_cv=None,
        Y_train_cv=None,
        y_val_cv=None,
        y_pnl_data_train_cv=None,
        y_pnl_data_val_cv_OrTest=None,
        params=None,
        other_params=None,
        config=None,
        fold_num=0,
        fold_raw_data=None,
        fold_stats_current=None,
        train_pos=None,
        val_pos=None,
        log_evaluation=0,
):
    # Calcul des poids
    sample_weights_train, sample_weights_val = compute_sample_weights(Y_train_cv, y_val_cv)

    # Création des DMatrix XGBoost avec les poids
    dtrain = xgb.DMatrix(X_train_cv, label=Y_train_cv, weight=sample_weights_train)
    dval = xgb.DMatrix(X_val_cv, label=y_val_cv, weight=sample_weights_val)

    # Configuration des paramètres selon l'objectif
    custom_objective_lossFct = config.get('custom_objective_lossFct', 13)
    if custom_objective_lossFct == model_custom_objective.XGB_CUSTOM_OBJECTIVE_PROFITBASED:
        obj_function = xgb_create_weighted_logistic_obj_cpu(other_params['w_p'], other_params['w_n'])
        params['disable_default_eval_metric'] = 1
    # elif custom_objective_lossFct == model_custom_objective.XGB_CUSTOM_OBJECTIVE_BINARY:
    #     obj_function = None
    #     params.update({'objective': 'binary:logistic'})
    # elif custom_objective_lossFct == model_custom_objective.XGB_CUSTOM_OBJECTIVE_CROSS_ENTROPY:
    #     obj_function = None
    #     params.update({'objective': 'binary:logistic'})  # XGBoost equivalent for cross_entropy
    # elif custom_objective_lossFct == model_custom_objective.XGB_CUSTOM_OBJECTIVE_CROSS_ENTROPY_LAMBDA:
    #     obj_function = None
    #     params.update({'objective': 'binary:logitraw'})  # XGBoost closest equivalent
    # else:
    #     raise ValueError("Choisir une fonction objective / Fonction objective non reconnue")

    # Configuration de la métrique d'évaluation personnalisée
    if config.get('custom_metric_eval', 13) == model_custom_metric.XGB_CUSTOM_METRIC_PNL:
        custom_metric = xgb_custom_metric_PNL(y_pnl_data_train_cv=y_pnl_data_train_cv,y_pnl_data_val_cv_OrTest=y_pnl_data_val_cv_OrTest,
                                              metric_dict=other_params, config=config)
    else:
        custom_metric = None
        params.update({'eval_metric': ['auc', 'logloss']})

    evals_result = {}

    # Entraînement du modèle
    current_model = xgb.train(
        params,
        dtrain,
        num_boost_round=other_params['num_boost_round'],
        evals=[(dtrain, 'train'), (dval, 'eval')],
        obj=obj_function,
        custom_metric=custom_metric,
        early_stopping_rounds=config.get('early_stopping_rounds', 13),
        verbose_eval=log_evaluation,
        evals_result=evals_result,
        maximize=True
    )

    # Extraction de la meilleure itération
    best_iteration, best_idx, val_best, train_best = get_best_iteration(evals_result,
                                                                        config.get('custom_metric_eval', 13))

    # Prédictions et métriques
    val_pred_proba, val_pred_proba_log_odds, val_pred, (tn_val, fp_val, fn_val, tp_val), y_val_cv = \
        predict_and_compute_metrics(model=current_model, X_data=dval, y_true=y_val_cv,
                                    best_iteration=best_iteration, threshold=other_params['threshold'],
                                    config=config)

    y_train_predProba, train_pred_proba_log_odds, train_pred, (tn_train, fp_train, fn_train, tp_train), Y_train_cv = \
        predict_and_compute_metrics(model=current_model, X_data=dtrain, y_true=Y_train_cv,
                                    best_iteration=best_iteration, threshold=other_params['threshold'],
                                    config=config)

    # Construction des métriques
    val_metrics = {
        'tp': tp_val,
        'fp': fp_val,
        'tn': tn_val,
        'fn': fn_val,
        'total_samples': len(y_val_cv),
        'val_bestVal_custom_metric_pnl': val_best,
        'best_iteration': best_iteration
    }
    train_metrics = {
        'tp': tp_train,
        'fp': fp_train,
        'tn': tn_train,
        'fn': fn_train,
        'total_samples': len(Y_train_cv),
        'train_bestVal_custom_metric_pnl': train_best
    }

    # Calculs complémentaires (winrate, trades, pourcentages, etc.)
    tp_fp_tn_fn_sum_val = tp_val + fp_val + tn_val + fn_val
    tp_fp_tn_fn_sum_train = tp_train + fp_train + tn_train + fn_train
    tp_fp_sum_val = tp_val + fp_val
    tp_fp_sum_train = tp_train + fp_train

    train_trades_samples_perct = round(tp_fp_sum_train / tp_fp_tn_fn_sum_train * 100,
                                       2) if tp_fp_tn_fn_sum_train else 0.00
    val_trades_samples_perct = round(tp_fp_sum_val / tp_fp_tn_fn_sum_val * 100, 2) if tp_fp_tn_fn_sum_val else 0.00
    perctDiff_ratioTradeSample_train_val = abs(
        calculate_ratio_difference(train_trades_samples_perct, val_trades_samples_perct, config))
    winrate_train = compute_winrate_safe(tp_train, tp_fp_sum_train, config)
    winrate_val = compute_winrate_safe(tp_val, tp_fp_sum_val, config)
    ratio_difference = abs(calculate_ratio_difference(winrate_train, winrate_val, config))

    # Compilation des statistiques du fold
    fold_stats = compile_fold_stats(
        fold_num, best_iteration, train_pred_proba_log_odds, train_metrics, winrate_train,
        tp_fp_sum_train, tp_fp_tn_fn_sum_train, train_best, train_pos, train_trades_samples_perct,
        val_pred_proba_log_odds, val_metrics, winrate_val, tp_fp_sum_val, tp_fp_tn_fn_sum_val,
        val_best, val_pos, val_trades_samples_perct, ratio_difference, perctDiff_ratioTradeSample_train_val
    )
    if fold_stats_current is not None:
        fold_stats.update(fold_stats_current)

    debug_info = compile_debug_info(other_params, config, val_pred_proba, y_train_predProba)

    # Retour de l'ensemble
    return {
        'current_model': current_model,
        'fold_raw_data': fold_raw_data,
        'y_train_predProba': y_train_predProba,
        'eval_metrics': val_metrics,
        'train_metrics': train_metrics,
        'fold_stats': fold_stats,
        'evals_result': evals_result,
        'best_iteration': best_iteration,
        'val_score_bestIdx': best_idx,
        'debug_info': debug_info,
    }