import lightgbm as lgb

from definition import *

# Fonction objective personnalisée pour LightGBM
def lgb_weighted_logistic_objective(w_p: float, w_n: float):
    def weighted_logistic_obj(preds, train_data):
        y_true = train_data.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))  # Application de la sigmoid

        # Calcul du gradient et de la hessienne avec les poids
        weights = np.where(y_true == 1, w_p, w_n)
        grad = (preds - y_true) * weights
        hess = preds * (1.0 - preds) * weights

        return grad, hess

    return weighted_logistic_obj


# Fonction auxiliaire pour le calcul du profit (équivalent à xgb_calculate_profitBased_gpu)
def lgb_calculate_profitBased(y_true=None, y_pred_threshold=None, metric_dict=None,config=None):
    """
    Calcule les métriques de profit pour LightGBM
    """
    # Calcul des métriques de base
    tp = np.sum((y_true == 1) & (y_pred_threshold == 1))
    fp = np.sum((y_true == 0) & (y_pred_threshold == 1))
    fn = np.sum((y_true == 1) & (y_pred_threshold == 0))

    # Récupération des paramètres de profit/perte
    profit_per_tp = config.get('profit_per_tp', 11111)
    loss_per_fp = config.get('loss_per_fp', 11111)
    penalty_per_fn = metric_dict.get('penalty_per_fn', 11111)
    if (penalty_per_fn == 11111 or profit_per_tp == 11111 or loss_per_fp == 11111):
        raise ValueError(
            f"Paramètres invalides : penalty_per_fn={penalty_per_fn}, profit_per_tp={profit_per_tp}, loss_per_fp={loss_per_fp}")

    # Calcul du profit total incluant les pénalités FN
    total_profit = (tp * profit_per_tp) + (fp * loss_per_fp) + (fn * penalty_per_fn)

    return float(total_profit), int(tp), int(fp)


# Métrique personnalisée pour LightGBM
def lgb_custom_metric_PNL(metric_dict=None,config=None):
    def profit_metric(preds,train_data):
        """
        LightGBM custom metric pour le profit
        """
        y_true = train_data.get_label()

        # Application de la sigmoid
        preds = 1.0 / (1.0 + np.exp(-preds))
        preds = np.clip(preds, 0.0, 1.0)

        # Vérification des prédictions
        if np.any(preds < 0) or np.any(preds > 1):
            return 'profit_based', float('-inf'), False

        # Application du seuil
        threshold = metric_dict.get('threshold', 0.55555555)
        y_pred_threshold = (preds > threshold).astype(np.int32)

        # Calcul du profit et des métriques
        total_profit, tp, fp = lgb_calculate_profitBased(y_true=y_true, y_pred_threshold=y_pred_threshold, metric_dict=metric_dict,config=config)

        # Le troisième paramètre (True) indique qu'une valeur plus élevée est meilleure
        return 'custom_metric_PNL', float(total_profit), True

    return profit_metric


def train_and_evaluate_lightgbm_model(
        X_train_cv=None,
        X_val_cv=None,
        X_train_cv_pd=None,
        X_val_cv_pd=None,
        Y_train_cv=None,
        y_val_cv=None,
        data=None,
        params=None,
        model_weight_optuna=None,
        config=None,
        fold_num=0,
        fold_raw_data=None,
        fold_stats_current=None,
        train_pos=None,
        val_pos=None,
        X_train_full=None,
        df_init_candles=None,
        is_log_enabled=False,
        log_evaluation=0,
        nb_split_tscv=0
    ):
    """
    Train and evaluate a LightGBM model with custom metrics and objectives,
    and integrate predictions, metrics computation, logging, and fold statistics.
    """

    w_p = model_weight_optuna['w_p']
    w_n = model_weight_optuna['w_n']
    num_boost_round = model_weight_optuna['num_boost_round']
    custom_objective_lossFct = config.get('custom_objective_lossFct', 13)
    custom_metric_eval=config.get('custom_metric_eval', 13)
    evals_result = {}


    # Calcul des poids pour l'ensemble d'entraînement
    N0 = np.sum(Y_train_cv == 0)
    N1 = np.sum(Y_train_cv == 1)
    N = len(Y_train_cv)

    w_0 = N / N0
    w_1 = N / N1

    sample_weights_train = np.where(Y_train_cv == 1, w_1, w_0)
    sample_weights_val = np.ones(len(y_val_cv))  # Pas pondéré pour la validation

    # Création des datasets LightGBM avec les poids
    ltrain = lgb.Dataset(X_train_cv, label=Y_train_cv, weight=sample_weights_train)
    lval = lgb.Dataset(X_val_cv, label=y_val_cv, weight=sample_weights_val)

    # Configuration des paramètres
    if custom_objective_lossFct == model_custom_objective.LGB_CUSTOM_OBJECTIVE_PROFITBASED:
        params.update({
            'objective': lgb_weighted_logistic_objective(w_p, w_n),
            'metric': None,
            'device_type': 'cpu'
        })

    elif custom_objective_lossFct == model_custom_objective.LGB_CUSTOM_OBJECTIVE_BINARY:
        # Adapter la métrique selon vos besoins
        params.update({
            'objective': 'binary',
            'metric': None,
            'device_type': 'cpu'

        })
    elif custom_objective_lossFct == model_custom_objective.LGB_CUSTOM_OBJECTIVE_CROSS_ENTROPY:
        # Adapter la métrique selon vos besoins
        params.update({
            'objective': 'cross_entropy',
            'metric': None,
            'device_type': 'cpu'

        })
    elif custom_objective_lossFct == model_custom_objective.LGB_CUSTOM_OBJECTIVE_CROSS_ENTROPY_LAMBDA:
        # Adapter la métrique selon vos besoins
        params.update({
            'objective': 'cross_entropy_lambda',
            'metric': None,
            'device_type': 'cpu'

        })
    else:
        raise ValueError("Choisir une fonction objective / Fonction objective non reconnue")

    if (custom_metric_eval==model_custom_metric.LGB_CUSTOM_METRIC_PNL):
        custom_metric = lgb_custom_metric_PNL(metric_dict=model_weight_optuna,config=config)
    else:
        params.update({ 'metric': ['auc', 'binary_logloss']})



    params['early_stopping_rounds'] = config.get('early_stopping_rounds', 13)
    params['verbose'] = -1

    # Entraînement du modèle
    current_model = lgb.train(
        params=params,
        train_set=ltrain,
        num_boost_round=num_boost_round,
        valid_sets=[ltrain, lval],
        valid_names=['train', 'eval'],
        feval=custom_metric,
        callbacks=[lgb.record_evaluation(evals_result),
        lgb.log_evaluation(period=log_evaluation)]  # Affiche les métriques toutes les 10 itérations
    )

    # Extraction des scores
    if custom_metric_eval == model_custom_metric.LGB_CUSTOM_METRIC_PNL:
        eval_scores = evals_result['eval']['custom_metric_PNL']
        train_scores = evals_result['train']['custom_metric_PNL']
    else:
        eval_scores = evals_result['eval']['auc']
        train_scores = evals_result['train']['auc']

    # Détermination de la meilleure itération
    val_score_best = max(eval_scores)
    val_score_bestIdx = eval_scores.index(val_score_best)
    best_iteration = val_score_bestIdx + 1
    train_score = train_scores[val_score_bestIdx]

    val_pred_proba, val_pred_proba_log_odds,val_pred, (tn_val, fp_val, fn_val, tp_val), y_val_cv = predict_and_compute_metrics(
        model=current_model,
        X_data=X_val_cv,
        y_true=y_val_cv,
        best_iteration=best_iteration,
        threshold=model_weight_optuna['threshold'],
        config=config
    )

    # Pour l'entraînement
    y_train_predProba, train_pred_proba_log_odds,train_pred, (tn_train, fp_train, fn_train, tp_train), Y_train_cv = predict_and_compute_metrics(
        model=current_model,
        X_data=X_train_cv,
        y_true=Y_train_cv,
        best_iteration=best_iteration,
        threshold=model_weight_optuna['threshold'],
        config=config
    )
    # Métriques val & train
    val_metrics = {
        'tp': tp_val,
        'fp': fp_val,
        'tn': tn_val,
        'fn': fn_val,
        'total_samples': len(y_val_cv),
        'score': val_score_best,
        'best_iteration': best_iteration
    }

    train_metrics = {
        'tp': tp_train,
        'fp': fp_train,
        'tn': tn_train,
        'fn': fn_train,
        'total_samples': len(Y_train_cv),
        'score': train_score
    }

    # Calcul winrate et trades
    tp_fp_tn_fn_sum_val = tp_val + fp_val + tn_val + fn_val
    tp_fp_tn_fn_sum_train = tp_train + fp_train + tn_train + fn_train
    tp_fp_sum_val = tp_val + fp_val
    tp_fp_sum_train = tp_train + fp_train

    ratio_profitPerTrade_val = calculate_profit_ratio(
        tp_val,
        fp_val,
        tp_fp_sum_val,
        config['profit_per_tp'],
        config['loss_per_fp']
    )
    # Calcul des ratios
    ratio_profitPerTrade_train = calculate_profit_ratio(
        tp_train,
        fp_train,
        tp_fp_sum_train,
        config['profit_per_tp'],
        config['loss_per_fp']
    )
    # Calcul de l'écart en pourcentage de la distribution train vs sample
    train_trades_samples_perct=round(tp_fp_sum_train / tp_fp_tn_fn_sum_train * 100, 2) if tp_fp_tn_fn_sum_train != 0 else 0.00
    val_trades_samples_perct= round(tp_fp_sum_val / tp_fp_tn_fn_sum_val * 100,2) if tp_fp_tn_fn_sum_val != 0 else 0.00
    perctDiff_ratioTradeSample_train_val = abs(calculate_ratio_difference(train_trades_samples_perct,val_trades_samples_perct,config))

    winrate_train=compute_winrate_safe(tp_train, tp_fp_sum_train, config)
    winrate_val=compute_winrate_safe(tp_val, tp_fp_sum_val, config)
    ratio_difference = abs(calculate_ratio_difference(winrate_train, winrate_val,config))

    # Compilation des stats du fold
    fold_stats = {
        'fold_num': fold_num,
        'best_iteration': best_iteration,
        'train_pred_proba_log_odds': train_pred_proba_log_odds,
        'train_metrics': train_metrics,
        'train_winrate': winrate_train,
        'train_trades': tp_fp_sum_train,
        'train_samples': tp_fp_tn_fn_sum_train,
        'train_score': train_metrics['score'],
        'train_size': len(train_pos) if train_pos is not None else None,
        'train_trades_samples_perct':train_trades_samples_perct ,

        'val_pred_proba_log_odds': val_pred_proba_log_odds,
        'val_metrics': val_metrics,
        'val_winrate': winrate_val,
        'val_trades': tp_fp_sum_val,
        'val_samples': tp_fp_tn_fn_sum_val,
        'val_score': val_score_best,
        'val_size': len(val_pos) if val_pos is not None else None,
         'val_trades_samples_perct': val_trades_samples_perct,

        'perctDiff_winrateRatio_train_val':ratio_difference,
        'perctDiff_ratioTradeSample_train_val':perctDiff_ratioTradeSample_train_val
    }

    if fold_stats_current is not None:
        fold_stats.update(fold_stats_current)

    # Logging des métriques si souhaité


    # Au début du script, définir xp en fonction du device
    xp = np if config['device_'] != 'cuda' else cp  # cp pour cupy, np pour numpy

    # Ensuite le code peut être simplifié en :
    debug_info = {
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


    # Retour exact comme spécifié
    return {
        'current_model': current_model,
        'fold_raw_data':fold_raw_data,
        'y_train_predProba':y_train_predProba,
        'eval_metrics': val_metrics,
        'train_metrics': train_metrics,
        'fold_stats': fold_stats,
        'evals_result': evals_result,
        'best_iteration': best_iteration,
        'val_score_best': val_score_best,
        'val_score_bestIdx': val_score_bestIdx,
        'debug_info': debug_info,
    }
