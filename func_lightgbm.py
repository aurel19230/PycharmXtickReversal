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

def lgb_focal_loss_objective(alpha=0.25, gamma=2.0, w_p=1.0, w_n=1.0, eps=1e-9):
    if not (0 < alpha < 1):
        raise ValueError(f"alpha doit être dans (0,1)")
    if gamma <= 0:
        raise ValueError(f"gamma doit être >0")
    if w_p <= 0 or w_n <= 0:
        raise ValueError(f"w_p et w_n doivent être >0")
    if not (0 < eps < 1):
        raise ValueError(f"eps doit être dans (0,1)")

    print(f"[FocalLoss] alpha={alpha}, gamma={gamma}, w_p={w_p}, w_n={w_n}, eps={eps}")

    def focal_loss_grad_hess(preds, train_data):
        #print(preds)
        y_true = train_data.get_label()
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, eps, 1 - eps)

        grad = np.zeros_like(p)
        hess = np.zeros_like(p)

        idx1 = (y_true == 1)
        idx0 = (y_true == 0)

        p1 = p[idx1]
        p0 = p[idx0]

        # gradient y=1 (version "mathématiquement correcte")
        grad_y1 = alpha * (
            gamma * (1 - p1)**(gamma - 1) * np.log(p1)
            - (1 - p1)**gamma / p1
        ) * p1 * (1 - p1)
        grad[idx1] = w_p * grad_y1

        # gradient y=0
        grad_y0 = (1 - alpha) * (
            gamma * (p0**(gamma - 1)) * np.log(1 - p0)
            - (p0**gamma)/(1 - p0)
        ) * p0*(1 - p0)
        grad[idx0] = w_n * grad_y0

        # Hessienne approx
        hess_approx = p*(1 - p)
        hess[idx1] = w_p * hess_approx[idx1]
        hess[idx0] = w_n * hess_approx[idx0]

        # ***** INVERSION DU SIGNE *****
        # LightGBM attend le gradient de la fonction qu'on MINIMISE.
        # Or FL = - alpha*(1-p)^gamma log(p), le "moins" fait parfois confusion.
        grad = -grad

        return grad, hess

    return focal_loss_grad_hess



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
    if penalty_per_fn == 11111 or profit_per_tp == 11111 or loss_per_fp == 11111:
        print(f"penalty_per_fn: {penalty_per_fn}, profit_per_tp: {profit_per_tp}, loss_per_fp: {loss_per_fp}")
        exit(101)

    # Calcul du profit total incluant les pénalités FN
    total_profit = (tp * profit_per_tp) + (fp * loss_per_fp) + (fn * penalty_per_fn)

    return float(total_profit), int(tp), int(fp)


# Métrique personnalisée pour LightGBM
def lgb_custom_metric_ProfitBased(metric_dict=None,config=None):
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
        return 'custom_metric_ProfitBased', float(total_profit), True

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
    evals_result = {}


    if config['device_'] != 'cpu':
        import cupy as cp
        X_train_cv = cp.asnumpy(X_train_cv.get())
        X_val_cv = cp.asnumpy(X_val_cv.get())
        Y_train_cv = cp.asnumpy(Y_train_cv.get())
        y_val_cv = cp.asnumpy(y_val_cv.get())
    else:
        X_train_cv = X_train_cv
        X_val_cv = X_val_cv
        Y_train_cv = Y_train_cv
        y_val_cv = y_val_cv

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
    if custom_objective_lossFct == model_customMetric.LGB_CUSTOM_METRIC_PROFITBASED:
        params.update({
            'objective': lgb_weighted_logistic_objective(w_p, w_n),
            'metric': None,
            'device_type': 'cpu'
        })
        custom_metric = lgb_custom_metric_ProfitBased(metric_dict=model_weight_optuna,config=config)
    elif custom_objective_lossFct == model_customMetric.LGB_CUSTOM_METRIC_FOCALLOSS:
        params.update({
            'objective': lgb_focal_loss_objective(alpha=0.25, gamma=2.0,w_p=w_p, w_n=w_n),
            #'objective': lgb_weighted_logistic_objective(w_p, w_n),
            'metric': None,
            'device_type': 'cpu'
        })
        custom_metric = lgb_custom_metric_ProfitBased(metric_dict=model_weight_optuna,config=config)
    else:
        custom_metric = None
        # Adapter la métrique selon vos besoins
        params.update({
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'num_threads': -1
        })

    params['early_stopping_rounds'] = config.get('early_stopping_rounds', 13)
    params['verbose'] = -1

    # Entraînement du modèle
    model = lgb.train(
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
    if custom_objective_lossFct == model_customMetric.LGB_CUSTOM_METRIC_PROFITBASED or \
            custom_objective_lossFct == model_customMetric.LGB_CUSTOM_METRIC_FOCALLOSS:
        eval_scores = evals_result['eval']['custom_metric_ProfitBased']
        train_scores = evals_result['train']['custom_metric_ProfitBased']
    else:
        eval_scores = evals_result['eval']['auc']
        train_scores = evals_result['train']['auc']

    # Détermination de la meilleure itération
    val_score_best = max(eval_scores)
    val_score_bestIdx = eval_scores.index(val_score_best)
    best_iteration = val_score_bestIdx + 1
    train_score = train_scores[val_score_bestIdx]

    val_pred_proba, val_pred_proba_log_odds,val_pred, (tn_val, fp_val, fn_val, tp_val), y_val_cv = predict_and_compute_metrics(
        model=model,
        X_data=X_val_cv,
        y_true=y_val_cv,
        best_iteration=best_iteration,
        threshold=model_weight_optuna['threshold'],
        config=config
    )

    # Pour l'entraînement
    y_train_predProba, train_pred_proba_log_odds,train_pred, (tn_train, fp_train, fn_train, tp_train), Y_train_cv = predict_and_compute_metrics(
        model=model,
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
    perctDiff_ratioTradeSample_train_val = calculate_ratio_difference(train_trades_samples_perct,val_trades_samples_perct,config)

    winrate_train=compute_winrate_safe(tp_train, tp_fp_sum_train, config)
    winrate_val=compute_winrate_safe(tp_val, tp_fp_sum_val, config)
    ratio_difference = calculate_ratio_difference(winrate_train, winrate_val,config)

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


    raw_metrics=compute_raw_train_dist(
                X_train_full, X_train_cv_pd,X_val_cv_pd,
                tp_train, fp_train, tn_train, fn_train,
                tp_val, fp_val, tn_val, fn_val,fold_num, nb_split_tscv, fold_raw_data,
            is_log_enabled)

    # Retour exact comme spécifié
    return {
        'model': model,
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
    },raw_metrics
