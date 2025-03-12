import lightgbm as lgb

from definition import *
def train_and_evaluate_randomforest_model(
        X_train_cv=None,
        X_val_cv=None,
        y_train_cv=None,
        y_val_cv=None,
        y_pnl_data_train_cv=None,
        y_pnl_data_val_cv_OrTest=None,
        params=None,
        model_weight_optuna=None,
        config=None,
        fold_num=0,
        fold_raw_data=None,
        fold_stats_current=None,
        train_pos=None,
        val_pos=None,
):
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np

    # Calcul des poids
    sample_weights_train, sample_weights_val = compute_sample_weights(y_train_cv, y_val_cv)

    # Configuration des paramètres de base pour RandomForest
    rf_params = {
        'n_estimators': model_weight_optuna.get('n_estimators', 100),
        'max_depth': model_weight_optuna.get('max_depth', None),
        'min_samples_split': model_weight_optuna.get('min_samples_split', 2),
        'min_samples_leaf': model_weight_optuna.get('min_samples_leaf', 1),
        'max_features': model_weight_optuna.get('max_features', 'sqrt'),
        'bootstrap': model_weight_optuna.get('bootstrap', True),
        'random_state': model_weight_optuna.get('random_state', 42),
        'n_jobs': model_weight_optuna.get('n_jobs', -1),
        'class_weight': model_weight_optuna.get('class_weight', None)
    }

    # Mettre à jour avec les paramètres personnalisés
    if params:
        rf_params.update(params)

    # Entraînement du modèle RandomForest
    current_model = RandomForestClassifier(**rf_params)
    current_model.fit(X_train_cv, y_train_cv, sample_weight=sample_weights_train)

    # Prédictions et métriques pour la validation
    val_pred_proba = current_model.predict_proba(X_val_cv)[:, 1]
    val_pred_proba_log_odds = np.log(val_pred_proba / (1 - val_pred_proba + 1e-10))
    threshold = model_weight_optuna.get('threshold', 0.5)
    val_pred = (val_pred_proba >= threshold).astype(int)

    # Calcul des métriques pour la validation
    from sklearn.metrics import confusion_matrix
    tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val_cv, val_pred).ravel()

    # Prédictions et métriques pour l'entraînement
    y_train_predProba = current_model.predict_proba(X_train_cv)[:, 1]
    train_pred_proba_log_odds = np.log(y_train_predProba / (1 - y_train_predProba + 1e-10))
    train_pred = (y_train_predProba >= threshold).astype(int)

    # Calcul des métriques pour l'entraînement
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train_cv, train_pred).ravel()

    # Calcul des métriques PnL personnalisées si nécessaire
    # if config.get('custom_metric_eval', 13) == model_custom_metric.FOREST_CUSTOM_METRIC_PNL:
    #     # Version simplifiée d'une métrique PnL pour RandomForest
    #     val_best = calculate_custom_pnl_metric(val_pred, y_val_cv, y_pnl_data_val_cv_OrTest, model_weight_optuna,
    #                                            config)
    #     train_best = calculate_custom_pnl_metric(train_pred, y_train_cv, y_pnl_data_train_cv, model_weight_optuna,
    #                                              config)
    # else:
    #     # Utilisation des métriques standards comme AUC
    #     from sklearn.metrics import roc_auc_score
    #     val_best = roc_auc_score(y_val_cv, val_pred_proba)
    #     train_best = roc_auc_score(y_train_cv, y_train_predProba)
    val_best=0
    train_best=0
    # Best iteration n'a pas de sens pour RandomForest, mais on le garde pour la cohérence
    best_iteration = None
    best_idx = None

    # Construction des métriques
    val_metrics = {
        'tp': tp_val,
        'fp': fp_val,
        'tn': tn_val,
        'fn': fn_val,
        'total_samples': len(y_val_cv),
        'val_bestIdx_custom_metric_pnl': val_best,
        'best_iteration': best_iteration
    }

    train_metrics = {
        'tp': tp_train,
        'fp': fp_train,
        'tn': tn_train,
        'fn': fn_train,
        'total_samples': len(y_train_cv),
        'train_bestIdx_custom_metric_pnl': train_best
    }

    # Calculs additionnels pour les statistiques
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

    # Informations de debug
    debug_info = {
        'model_params': rf_params,
        'threshold': threshold,
        'feature_importances': dict(zip(range(X_train_cv.shape[1]), current_model.feature_importances_))
    }

    return {
        'current_model': current_model,
        'fold_raw_data': fold_raw_data,
        'y_train_predProba': y_train_predProba,
        'eval_metrics': val_metrics,
        'train_metrics': train_metrics,
        'fold_stats': fold_stats,
        'evals_result': None,  # RandomForest n'a pas d'evals_result comme LightGBM
        'best_iteration': best_iteration,
        'val_score_bestIdx': best_idx,
        'debug_info': debug_info,
    }