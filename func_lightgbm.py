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
def lgb_calculate_profitBased(y_true, y_pred_threshold, metric_dict):
    """
    Calcule les métriques de profit pour LightGBM
    """
    # Calcul des métriques de base
    tp = np.sum((y_true == 1) & (y_pred_threshold == 1))
    fp = np.sum((y_true == 0) & (y_pred_threshold == 1))
    fn = np.sum((y_true == 1) & (y_pred_threshold == 0))

    # Récupération des paramètres de profit/perte
    profit_per_tp = metric_dict.get('profit_per_tp', 1.0)
    loss_per_fp = metric_dict.get('loss_per_fp', -1.1)
    penalty_per_fn = metric_dict.get('penalty_per_fn', -0.1)

    # Calcul du profit total incluant les pénalités FN
    total_profit = (tp * profit_per_tp) + (fp * loss_per_fp) + (fn * penalty_per_fn)

    return float(total_profit), int(tp), int(fp)


# Métrique personnalisée pour LightGBM
def lgb_custom_metric_ProfitBased(metric_dict):
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
        total_profit, tp, fp = lgb_calculate_profitBased(y_true, y_pred_threshold, metric_dict)

        # Le troisième paramètre (True) indique qu'une valeur plus élevée est meilleure
        return 'custom_metric_ProfitBased', float(total_profit), True

    return profit_metric


# Exemple d'utilisation
def train_lightgbm_model(X_train, y_train, X_valid, y_valid, metric_dict, w_p=1.0, w_n=1.0):
    # Création des datasets LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    # Paramètres du modèle
    params = {
        'objective': lgb_weighted_logistic_objective(w_p, w_n),
        'metric': 'None',  # Désactiver les métriques par défaut
        'verbose': -1,
        'device_type': 'gpu'  # Utilisation du GPU
    }

    # Création de la métrique personnalisée
    profit_metric = lgb_custom_metric_ProfitBased(metric_dict)

    # Entraînement avec callback pour enregistrer les résultats
    evals_result = {}
    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['training', 'validation'],
        num_boost_round=100,
        feval=profit_metric,
        callbacks=[lgb.record_evaluation(evals_result)]
    )

    return model, evals_result

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

    # Création des datasets LightGBM
    ltrain = lgb.Dataset(X_train_cv, label=Y_train_cv)
    lval = lgb.Dataset(X_val_cv, label=y_val_cv)

    # Configuration des paramètres
    if custom_objective_lossFct == model_customMetric.LGB_CUSTOM_METRIC_PROFITBASED:
        params.update({
            'objective': lgb_weighted_logistic_objective(w_p, w_n),
            'metric': None,
            'device_type': 'cpu'
        })
        custom_metric = lgb_custom_metric_ProfitBased(model_weight_optuna)
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
    if custom_objective_lossFct == model_customMetric.LGB_CUSTOM_METRIC_PROFITBASED:
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

    val_pred_proba, val_pred, (tn_val, fp_val, fn_val, tp_val), y_val_cv = predict_and_compute_metrics(
        model=model,
        X_data=X_val_cv,
        y_true=y_val_cv,
        best_iteration=best_iteration,
        threshold=model_weight_optuna['threshold'],
        config=config
    )

    # Pour l'entraînement
    y_train_predProba, train_pred, (tn_train, fp_train, fn_train, tp_train), Y_train_cv = predict_and_compute_metrics(
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
    tp_fp_sum_val = tp_val + fp_val
    tp_fp_sum_train = tp_train + fp_train
    tp_fp_tn_fn_sum_val = tp_val + fp_val+tn_val + fn_val
    tp_fp_tn_fn_sum_train = tp_train + fp_train + tn_train + fn_train

    # Compilation des stats du fold
    fold_stats = {
        'eval_metrics': val_metrics,
        'train_metrics': train_metrics,
        'val_winrate': compute_winrate_safe(tp_val, tp_fp_sum_val,config),
        'train_winrate': compute_winrate_safe(tp_train, tp_fp_sum_train,config),
        'val_trades': tp_fp_sum_val,
        'val_samples': tp_fp_tn_fn_sum_val,
        'train_trades': tp_fp_sum_train,
        'train_samples': tp_fp_tn_fn_sum_train,
        'fold_num': fold_num,
        'best_iteration': best_iteration,
        'val_score': val_score_best,
        'train_score': train_metrics['score'],
        'train_size': len(train_pos) if train_pos is not None else None,
        'val_size': len(val_pos) if val_pos is not None else None
    }

    if fold_stats_current is not None:
        fold_stats.update(fold_stats_current)

    # Logging des métriques si souhaité
    if is_log_enabled:
        log_cv_fold_metrics(
                X_train_full, X_train_cv_pd,X_val_cv_pd ,val_pos,
                Y_train_cv, y_val_cv,
                tp_train, fp_train, tn_train, fn_train,
                tp_val, fp_val, tn_val, fn_val,config,fold_num, nb_split_tscv, fold_raw_data
            )

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
        'debug_info': debug_info
    }
