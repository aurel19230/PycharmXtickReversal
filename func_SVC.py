
from definition import *

def train_and_evaluate_svc_model(
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
        log_evaluation=0,
):
    from sklearn.svm import SVC
    import numpy as np
    import time

    # Mesure du temps total de la fonction
    start_time_total = time.time()

    # Calcul des poids
    start_time_weights = time.time()
    sample_weights_train, sample_weights_val = compute_sample_weights(y_train_cv, y_val_cv)
    weights_time = time.time() - start_time_weights
    #print(f"Temps de calcul des poids: {weights_time:.4f} secondes")

    # Assurer que les paramètres par défaut sont définis
    if params is None:
        params = {}

    # Récupérer l'option de probabilité depuis la configuration
    svc_probability = config.get('svc_probability', False)
    params['probability'] = svc_probability

    # Récupérer le type de noyau depuis la configuration
    svc_kernel = config.get('svc_kernel', 'rbf')
    params['kernel'] = svc_kernel

    # Mesure du temps d'entraînement
    start_time_train = time.time()
    # Entraînement du modèle SVC
    current_model = SVC(**params)
    current_model.fit(X_train_cv, y_train_cv, sample_weight=sample_weights_train)
    train_time = time.time() - start_time_train
    print(f"Temps d'entraînement du modèle: {train_time:.4f} secondes")

    # Prédictions et métriques pour la validation
    start_time_val_pred = time.time()

    # Selon l'option de probabilité, récupérer les prédictions différemment
    if svc_probability:
        val_pred_proba = current_model.predict_proba(X_val_cv)[:, 1]
        threshold = model_weight_optuna.get('threshold', 0.5)
        val_pred = (val_pred_proba >= threshold).astype(int)
        val_pred_proba_log_odds = np.log(val_pred_proba / (1 - val_pred_proba + 1e-10))
    else:
        val_pred = current_model.predict(X_val_cv)
        # Utiliser la distance à l'hyperplan (decision_function) comme proxy pour la probabilité
        decision_values = current_model.decision_function(X_val_cv)
        # Normaliser les valeurs de décision pour une approximation de probabilité
        val_pred_proba = 1 / (1 + np.exp(-decision_values))
        val_pred_proba_log_odds = decision_values

    val_pred_time = time.time() - start_time_val_pred
    #print(f"Temps de prédiction sur validation: {val_pred_time:.4f} secondes")

    # Calcul des métriques pour la validation
    start_time_val_metrics = time.time()
    from sklearn.metrics import confusion_matrix
    tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val_cv, val_pred).ravel()
    val_metrics_time = time.time() - start_time_val_metrics
    #print(f"Temps de calcul des métriques de validation: {val_metrics_time:.4f} secondes")

    # Mesure du temps pour les prédictions sur l'ensemble d'entraînement
    start_time_train_pred = time.time()

    # Selon l'option de probabilité, récupérer les prédictions différemment
    if svc_probability:
        y_train_predProba = current_model.predict_proba(X_train_cv)[:, 1]
        threshold = model_weight_optuna.get('threshold', 0.5)
        train_pred = (y_train_predProba >= threshold).astype(int)
        train_pred_proba_log_odds = np.log(y_train_predProba / (1 - y_train_predProba + 1e-10))
    else:
        train_pred = current_model.predict(X_train_cv)
        # Utiliser la distance à l'hyperplan (decision_function) comme proxy pour la probabilité
        decision_values = current_model.decision_function(X_train_cv)
        # Normaliser les valeurs de décision pour une approximation de probabilité
        y_train_predProba = 1 / (1 + np.exp(-decision_values))
        train_pred_proba_log_odds = decision_values

    train_pred_time = time.time() - start_time_train_pred
    #print(f"Temps de prédiction sur entraînement: {train_pred_time:.4f} secondes")

    # Calcul des métriques pour l'entraînement
    start_time_train_metrics = time.time()
    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train_cv, train_pred).ravel()
    train_metrics_time = time.time() - start_time_train_metrics
    #print(f"Temps de calcul des métriques d'entraînement: {train_metrics_time:.4f} secondes")

    # Initialisation des valeurs de PNL
    val_pnl = 0
    train_pnl = 0

    # Calcul du PNL pour les données de validation
    start_time_val_pnl = time.time()
    if y_pnl_data_val_cv_OrTest is not None:
        # Créer des masques pour les vrais positifs et faux positifs
        tp_mask_val = (y_val_cv == 1) & (val_pred == 1)  # Vrais positifs
        fp_mask_val = (y_val_cv == 0) & (val_pred == 1)  # Faux positifs

        # Vérification de cohérence (optionnelle)
        tp_count = np.sum(tp_mask_val)
        if tp_count != tp_val:
            print(f"Attention: Nombre de TP calculé ({tp_count}) différent de tp_val ({tp_val})")

        fp_count = np.sum(fp_mask_val)
        if fp_count != fp_val:
            print(f"Attention: Nombre de FP calculé ({fp_count}) différent de fp_val ({fp_val})")

        # Calculer PNL total pour validation
        val_pnl = np.sum(y_pnl_data_val_cv_OrTest[tp_mask_val]) + np.sum(y_pnl_data_val_cv_OrTest[fp_mask_val])

        # Vérification de la cohérence entre prédictions et PNL réels
        fp_pnls = y_pnl_data_val_cv_OrTest[fp_mask_val]
        positive_fp_pnls = fp_pnls[fp_pnls > 0]

        if len(positive_fp_pnls) > 0:
            print(f"Attention: {len(positive_fp_pnls)} faux positifs ont un PNL > 0, ce qui semble incohérent")
            print(f"Valeurs PNL incohérentes pour les FP: {positive_fp_pnls}")

        tp_pnls = y_pnl_data_val_cv_OrTest[tp_mask_val]
        negative_tp_pnls = tp_pnls[tp_pnls <= 0]

        if len(negative_tp_pnls) > 0:
            print(f"Attention: {len(negative_tp_pnls)} vrais positifs ont un PNL ≤ 0, ce qui semble incohérent")
            print(f"Valeurs PNL incohérentes pour les TP: {negative_tp_pnls}")
    val_pnl_time = time.time() - start_time_val_pnl
    #print(f"Temps de calcul du PNL de validation: {val_pnl_time:.4f} secondes")

    # Calcul du PNL pour les données d'entraînement
    start_time_train_pnl = time.time()
    if y_pnl_data_train_cv is not None:
        # Créer des masques pour les vrais positifs et faux positifs
        tp_mask_train = (y_train_cv == 1) & (train_pred == 1)  # Vrais positifs
        fp_mask_train = (y_train_cv == 0) & (train_pred == 1)  # Faux positifs

        # Vérification de cohérence (optionnelle)
        tp_count = np.sum(tp_mask_train)
        if tp_count != tp_train:
            print(f"Attention: Nombre de TP calculé ({tp_count}) différent de tp_train ({tp_train})")

        fp_count = np.sum(fp_mask_train)
        if fp_count != fp_train:
            print(f"Attention: Nombre de FP calculé ({fp_count}) différent de fp_train ({fp_train})")

        # Calculer PNL total pour entraînement
        train_pnl = np.sum(y_pnl_data_train_cv[tp_mask_train]) + np.sum(y_pnl_data_train_cv[fp_mask_train])

        # Vérification de la cohérence entre prédictions et PNL réels pour les données d'entraînement
        fp_pnls_train = y_pnl_data_train_cv[fp_mask_train]
        positive_fp_pnls_train = fp_pnls_train[fp_pnls_train > 0]

        if len(positive_fp_pnls_train) > 0:
            print(
                f"Attention: {len(positive_fp_pnls_train)} faux positifs dans les données d'entraînement ont un PNL > 0, ce qui semble incohérent")
            print(f"Valeurs PNL incohérentes pour les FP: {positive_fp_pnls_train}")

        # Vérifier les vrais positifs (devraient avoir PNL > 0)
        tp_pnls_train = y_pnl_data_train_cv[tp_mask_train]
        negative_tp_pnls_train = tp_pnls_train[tp_pnls_train <= 0]

        if len(negative_tp_pnls_train) > 0:
            print(
                f"Attention: {len(negative_tp_pnls_train)} vrais positifs dans les données d'entraînement ont un PNL ≤ 0, ce qui semble incohérent")
            print(f"Valeurs PNL incohérentes pour les TP: {negative_tp_pnls_train}")
    train_pnl_time = time.time() - start_time_train_pnl
    #print(f"Temps de calcul du PNL d'entraînement: {train_pnl_time:.4f} secondes")

    # Best iteration n'a pas de sens pour SVC, mais on le garde pour la cohérence
    best_iteration = None
    best_idx = None

    # Construction des métriques
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
        tp_fp_sum_train, tp_fp_tn_fn_sum_train, train_pnl, train_pos, train_trades_samples_perct,
        val_pred_proba_log_odds, val_metrics, winrate_val, tp_fp_sum_val, tp_fp_tn_fn_sum_val,
        val_pnl, val_pos, val_trades_samples_perct, ratio_difference, perctDiff_ratioTradeSample_train_val
    )

    if fold_stats_current is not None:
        fold_stats.update(fold_stats_current)
    metrics_build_time = time.time() - start_time_metrics_build
    #print(f"Temps de construction des métriques: {metrics_build_time:.4f} secondes")

    # Mesure du temps total
    total_time = time.time() - start_time_total
    #print(f"Temps total d'exécution de la fonction: {total_time:.4f} secondes")

    # Extraire les vecteurs de support (spécifique à SVC)
    n_support = current_model.n_support_
    support_vectors_info = {
        'n_support_per_class': n_support,
        'total_support_vectors': sum(n_support),
        'support_vector_ratio': sum(n_support) / len(X_train_cv)
    }

    # Informations de debug avec les temps d'exécution
    debug_info = {
        'model_params': params,
        'kernel': svc_kernel,
        'threshold': model_weight_optuna.get('threshold', 0.5) if svc_probability else None,
        'using_probabilities': svc_probability,
        'support_vectors_info': support_vectors_info,
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