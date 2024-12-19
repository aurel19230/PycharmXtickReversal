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
    def profit_metric(preds, train_data):
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
        Y_train_cv=None,
        y_val_cv=None,
        params=None,
        model_weight_optuna=None,
        config=None,
        fold_num=None,
        fold_stats_current=None,
        train_pos=None,
        val_pos=None,
        X_train_full=None,
        is_log_enabled=False,
        log_evaluation=0
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

    # ---- Calcul des prédictions et métriques ----
    # Prédictions validation
    val_pred_proba = model.predict(X_val_cv, iteration_range=(0, best_iteration))
    if config['device_'] != 'cpu':
        val_pred_proba = cp.asarray(val_pred_proba, dtype=cp.float32)
    val_pred_proba, val_pred = predict_and_process(val_pred_proba, model_weight_optuna['threshold'],config)

    # Conversion pour le calcul de la matrice de confusion
    if config['device_'] != 'cpu':
        y_val_cv = cp.asarray(y_val_cv) if isinstance(y_val_cv, (np.ndarray, pd.Series)) else y_val_cv
        val_pred = cp.asarray(val_pred) if isinstance(val_pred, (np.ndarray, pd.Series)) else val_pred
    tn_val, fp_val, fn_val, tp_val = compute_confusion_matrix_cpu(y_val_cv, val_pred,config)

    # Prédictions entraînement
    train_pred_proba = model.predict(X_train_cv, iteration_range=(0, best_iteration))
    if config['device_'] != 'cpu':
        train_pred_proba = cp.asarray(train_pred_proba, dtype=cp.float32)
    train_pred_proba, train_pred = predict_and_process(train_pred_proba, model_weight_optuna['threshold'],config)

    # Conversion pour le calcul de la matrice de confusion
    if config['device_'] != 'cpu':
        Y_train_cv = cp.asarray(Y_train_cv) if isinstance(Y_train_cv, (np.ndarray, pd.Series)) else Y_train_cv
        train_pred = cp.asarray(train_pred) if isinstance(train_pred, (np.ndarray, pd.Series)) else train_pred
    tn_train, fp_train, fn_train, tp_train = compute_confusion_matrix_cpu(Y_train_cv, train_pred,config)

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

    # Logging des métriques si souhaité
    if is_log_enabled:
        log_cv_fold_metrics(
            X_train_full, X_train_cv, val_pos,
            Y_train_cv, y_val_cv,
            tp_train, fp_train, tn_train, fn_train,
            tp_val, fp_val, tn_val, fn_val
        )

    # Compilation des stats du fold
    fold_stats = {
        'eval_metrics': val_metrics,
        'train_metrics': train_metrics,
        'val_winrate': compute_winrate_safe(tp_val, tp_fp_sum_val),
        'train_winrate': compute_winrate_safe(tp_train, tp_fp_sum_train),
        'val_trades': tp_fp_sum_val,
        'train_trades': tp_fp_sum_train,
        'fold_num': fold_num,
        'best_iteration': best_iteration,
        'val_score': val_score_best,
        'train_score': train_metrics['score'],
        'train_size': len(train_pos) if train_pos is not None else None,
        'val_size': len(val_pos) if val_pos is not None else None
    }

    if fold_stats_current is not None:
        fold_stats.update(fold_stats_current)

    # debug_info tel que demandé
    debug_info = {
        'threshold_used': model_weight_optuna['threshold'],
        'pred_proba_ranges': {
            'val': {
                'min': float(cp.min(val_pred_proba)),
                'max': float(cp.max(val_pred_proba))
            },
            'train': {
                'min': float(cp.min(train_pred_proba)),
                'max': float(cp.max(train_pred_proba))
            }
        }
    }

    # Retour exact comme spécifié
    return {
        'model': model,
        'eval_metrics': val_metrics,
        'train_metrics': train_metrics,
        'fold_stats': fold_stats,
        'evals_result': evals_result,
        'best_iteration': best_iteration,
        'val_score_best': val_score_best,
        'val_score_bestIdx': val_score_bestIdx,
        'debug_info': debug_info
    }
