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
        other_params=None,
        config=None,
        fold_num=0,
        fold_raw_data=None,
        fold_stats_current=None,
        train_pos=None,
        val_pos=None,
        log_evaluation=0,
):
    """
    Entraîne et évalue soit un RandomForest sklearn, soit XGBRFClassifier (XGBoost),
    selon use_xgbrf=True/False.
    """
    import numpy as np
    import time
    model_type=config['model_type']

    # -- 0) Fonction interne pour charger le bon classifieur ------------------
    if model_type==modelType.XGBRF:
        from xgboost import XGBRFClassifier
        # par défaut, pour simuler un "RF" XGBoost, on s'assure learning_rate=1.0
        # et on peut utiliser tree_method='gpu_hist' si on veut le GPU
        # cf. doc : https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRFClassifier
        ModelClass = XGBRFClassifier
    else:
        from sklearn.ensemble import RandomForestClassifier
        ModelClass = RandomForestClassifier

    # -- 1) Mesure du temps total de la fonction -----------------------------
    start_time_total = time.time()

    # -- 2) Calcul des poids, si besoin --------------------------------------
    start_time_weights = time.time()
    sample_weights_train, sample_weights_val = compute_sample_weights(y_train_cv, y_val_cv)
    weights_time = time.time() - start_time_weights

    # -- 3) Entraînement du modèle -------------------------------------------
    start_time_train = time.time()
    current_model = ModelClass(**params)
    # Vérifier que sample_weight est supporté (XGBRFClassifier le supporte aussi)
    current_model.fit(X_train_cv, y_train_cv, sample_weight=sample_weights_train)
    train_time = time.time() - start_time_train

    # -- 4) Prédictions + log-odds sur la Validation -------------------------
    start_time_val_pred = time.time()
    val_pred_proba = current_model.predict_proba(X_val_cv)[:, 1]
    val_pred_proba_log_odds = np.log(val_pred_proba / (1 - val_pred_proba + 1e-10))

    threshold = other_params.get('threshold', 0.5)
    val_pred = (val_pred_proba >= threshold).astype(int)
    val_pred_time = time.time() - start_time_val_pred

    # -- 5) Métriques Val ----------------------------------------------------
    start_time_val_metrics = time.time()
    from sklearn.metrics import confusion_matrix
    tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val_cv, val_pred).ravel()
    val_metrics_time = time.time() - start_time_val_metrics

    # -- 6) Prédictions + métriques sur Train -------------------------------
    start_time_train_pred = time.time()
    y_train_predProba = current_model.predict_proba(X_train_cv)[:, 1]
    train_pred_proba_log_odds = np.log(y_train_predProba / (1 - y_train_predProba + 1e-10))
    train_pred = (y_train_predProba >= threshold).astype(int)
    train_pred_time = time.time() - start_time_train_pred

    start_time_train_metrics = time.time()
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train_cv, train_pred).ravel()
    train_metrics_time = time.time() - start_time_train_metrics

    # -- 7) Calcul PNL sur Validation ---------------------------------------
    start_time_val_pnl = time.time()
    val_pnl = 0
    if y_pnl_data_val_cv_OrTest is not None:
        tp_mask_val = (y_val_cv == 1) & (val_pred == 1)
        fp_mask_val = (y_val_cv == 0) & (val_pred == 1)
        val_pnl = np.sum(y_pnl_data_val_cv_OrTest[tp_mask_val]) + np.sum(y_pnl_data_val_cv_OrTest[fp_mask_val])
    val_pnl_time = time.time() - start_time_val_pnl

    # -- 8) Calcul PNL sur Train --------------------------------------------
    start_time_train_pnl = time.time()
    train_pnl = 0
    if y_pnl_data_train_cv is not None:
        tp_mask_train = (y_train_cv == 1) & (train_pred == 1)
        fp_mask_train = (y_train_cv == 0) & (train_pred == 1)
        train_pnl = np.sum(y_pnl_data_train_cv[tp_mask_train]) + np.sum(y_pnl_data_train_cv[fp_mask_train])
    train_pnl_time = time.time() - start_time_train_pnl

    # -- 9) Best iteration (pas vraiment utile pour RF, mais conservé) ------
    best_iteration = None
    best_idx = None

    # -- 10) Construit les dictionnaires de métriques -----------------------
    start_time_metrics_build = time.time()
    val_metrics = {
        'tp': tp_val,
        'fp': fp_val,
        'tn': tn_val,
        'fn': fn_val,
        'total_samples': len(y_val_cv),
        'val_bestVal_custom_metric_pnl': val_pnl,
        'best_iteration': best_iteration
    }

    train_metrics = {
        'tp': tp_train,
        'fp': fp_train,
        'tn': tn_train,
        'fn': fn_train,
        'total_samples': len(y_train_cv),
        'train_bestVal_custom_metric_pnl': train_pnl
    }

    # Quelques stats globales
    tp_fp_tn_fn_sum_val = tp_val + fp_val + tn_val + fn_val
    tp_fp_tn_fn_sum_train = tp_train + fp_train + tn_train + fn_train
    tp_fp_sum_val = tp_val + fp_val
    tp_fp_sum_train = tp_train + fp_train

    train_trades_samples_perct = round(tp_fp_sum_train / tp_fp_tn_fn_sum_train * 100, 2) if tp_fp_tn_fn_sum_train else 0.00
    val_trades_samples_perct = round(tp_fp_sum_val / tp_fp_tn_fn_sum_val * 100, 2) if tp_fp_tn_fn_sum_val else 0.00

    perctDiff_ratioTradeSample_train_val = abs(
        calculate_ratio_difference(train_trades_samples_perct, val_trades_samples_perct, config)
    )
    winrate_train = compute_winrate_safe(tp_train, tp_fp_sum_train, config)
    winrate_val = compute_winrate_safe(tp_val, tp_fp_sum_val, config)
    ratio_difference = abs(calculate_ratio_difference(winrate_train, winrate_val, config))

    fold_stats = compile_fold_stats(
        fold_num, best_iteration, train_pred_proba_log_odds, train_metrics, winrate_train,
        tp_fp_sum_train, tp_fp_tn_fn_sum_train, train_pnl, train_pos, train_trades_samples_perct,
        val_pred_proba_log_odds, val_metrics, winrate_val, tp_fp_sum_val, tp_fp_tn_fn_sum_val,
        val_pnl, val_pos, val_trades_samples_perct, ratio_difference, perctDiff_ratioTradeSample_train_val
    )

    if fold_stats_current is not None:
        fold_stats.update(fold_stats_current)

    metrics_build_time = time.time() - start_time_metrics_build

    # -- 11) Temps total ----------------------------------------------------
    total_time = time.time() - start_time_total

    # -- 12) Feature importances -------------------------------------------
    # (XGBRFClassifier possède .feature_importances_, tout comme sklearn)
    importances = current_model.feature_importances_
    feature_importances_dict = dict(zip(range(X_train_cv.shape[1]), importances))

    # -- 13) Debug info ----------------------------------------------------
    debug_info = {
        'model_params': params,
        'threshold': threshold,
        'feature_importances': feature_importances_dict,
        'execution_times': {
            'total': total_time,
            'weights_calculation': weights_time,
            'training': train_time,
            'validation_prediction': val_pred_time,
            'validation_metrics': val_metrics_time,
            'training_prediction': train_pred_time,
            'training_metrics': train_metrics_time,
            'validation_pnl': val_pnl_time,
            'training_pnl': train_pnl_time,
            'metrics_building': metrics_build_time
        }
    }

    return {
        'current_model': current_model,
        'fold_raw_data': fold_raw_data,
        'y_train_predProba': y_train_predProba,
        'eval_metrics': val_metrics,
        'train_metrics': train_metrics,
        'fold_stats': fold_stats,
        'evals_result': None,
        'best_iteration': best_iteration,
        'val_score_bestIdx': best_idx,
        'debug_info': debug_info,
    }
