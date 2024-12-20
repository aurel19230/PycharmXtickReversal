from numba import njit
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
from functools import partial
from typing import Dict, List, Tuple
from datetime import datetime
import optuna
from colorama import Fore, Style, init
from func_standard import detect_environment
from definition import *
import time

from sklearn.utils.class_weight import compute_sample_weight
import os
from numba import njit
from sklearn.metrics import precision_recall_curve, log_loss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, \
    average_precision_score, matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


from sklearn.model_selection import KFold, TimeSeriesSplit
import sys

import tempfile

import shap
import pandas as pd
import numpy as np

import pandas as pd
# Define the custom_metric class using Enum
from enum import Enum

# Configuration pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.expand_frame_repr', False)


def install_and_import_packages(packages, ENV=None):
    """
    Installe et importe les packages nécessaires
    """
    if ENV == 'colab':
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                return True
        except Exception as e:
            print(f"Erreur lors de l'installation des packages: {str(e)}")
            return False
    return False


# Import des packages communs


# Utilisation
ENV = detect_environment()

# Import des fonctions selon l'environnement

from func_standard import (print_notification,
                           check_gpu_availability,
                           optuna_doubleMetrics,
                           callback_optuna,
                           calculate_weighted_adjusted_score_custom,
                           model_customMetric, scalerChoice,
                           train_finalModel_analyse, init_dataSet,
                           calculate_normalized_objectives,
                           run_cross_validation,
                           setup_model_params_optuna, setup_model_weight_optuna,cv_config, displaytNan_vifMiCorrFiltering,
                           load_features_and_sections, apply_scaling, manage_rfe_selection, display_metrics,
                           check_distribution_coherence, check_value_ranges, setup_cv_method,
                           calculate_constraints_optuna,remove_nan_inf,add_session_id)
if ENV == 'pycharm':
    import keyboard




# Après le redémarrage, on peut importer les packages
from colorama import Fore, Style, init


# 3. Optimisation des splits


########################################
#########    FUNCTION DEF      #########
########################################


# Fonction optimisée avec Numba pour vérifier et remplacer les valeurs
@njit
def process_values(data, new_val):
    count_replacements = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if value > 90000:
                integer_part = int(value)
                decimal_part = value - integer_part
                decimal_digits = int(decimal_part * 100000)
                if decimal_digits == 54789:
                    # Remplacer la valeur par new_val
                    data[i, j] = new_val
                    count_replacements += 1
    return data, count_replacements


def plot_learning_curve(learning_curve_data, title='Courbe d\'apprentissage', filename='learning_curve.png'):
    if learning_curve_data is None:
        print("Pas de données de courbe d'apprentissage à tracer.")
        return
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Taille de l'ensemble d'entraînement")
    plt.ylabel("Score")
    plt.grid()

    # Conversion des listes en arrays NumPy pour effectuer des opérations mathématiques
    train_sizes = np.array(learning_curve_data['train_sizes'])
    train_scores_mean = np.array(learning_curve_data['train_scores_mean'])
    val_scores_mean = np.array(learning_curve_data['val_scores_mean'])

    # Tracé des courbes sans les bandes représentant l'écart-type
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Score de validation")

    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()


def average_learning_curves(learning_curve_data_list):
    if not learning_curve_data_list:
        return None

    # Extraire toutes les tailles d'entraînement uniques
    all_train_sizes = sorted(set(size for data in learning_curve_data_list for size in data['train_sizes']))

    # Initialiser les listes pour stocker les scores moyens
    avg_train_scores = []
    avg_val_scores = []

    for size in all_train_sizes:
        train_scores = []
        val_scores = []
        for data in learning_curve_data_list:
            if size in data['train_sizes']:
                index = data['train_sizes'].index(size)
                train_scores.append(data['train_scores_mean'][index])
                val_scores.append(data['val_scores_mean'][index])

        if train_scores and val_scores:
            avg_train_scores.append(np.mean(train_scores))
            avg_val_scores.append(np.mean(val_scores))

    return {
        'train_sizes': all_train_sizes,
        'train_scores_mean': avg_train_scores,
        'val_scores_mean': avg_val_scores
    }


from sklearn.model_selection import train_test_split

# Ajoutez cette variable globale au début de votre script
global lastBest_score
lastBest_score = float('-inf')


def timestamp_to_date_utc_(timestamp):
    date_format = "%Y-%m-%d %H:%M:%S"
    if isinstance(timestamp, pd.Series):
        return timestamp.apply(lambda x: time.strftime(date_format, time.gmtime(x)))
    else:
        return time.strftime(date_format, time.gmtime(timestamp))


# Et dans get_val_cv_time_range, utiliser les index correctement :


from dateutil.relativedelta import relativedelta

from sklearn.model_selection import BaseCrossValidator

from sklearn.model_selection._split import BaseCrossValidator
import numpy as np
from typing import Iterator, Optional, Tuple, Union
from numbers import Integral


def objective_optuna(df_init=None, trial=None, study=None, X_train=None, X_test=None, y_train_label=None,
                     X_train_full=None,
                     device=None, modele_param_optuna_range=None, config=None, nb_split_tscv=None,
                     model_weight_optuna=None, weight_param=None,
                     random_state_seed_=None, is_log_enabled=None, cv_method=cv_config.K_FOLD,
                     selected_columns=None,model=None,
                     ):
    try:
        # État de l'itération
        if not hasattr(objective_optuna, 'iteration_counter'):
            objective_optuna.iteration_counter = 0
        objective_optuna.iteration_counter += 1

        print(f"\n{'=' * 50}")
        print_notification(f"Début itération Optuna #{objective_optuna.iteration_counter}")
        print(f"{'=' * 50}")

        np.random.seed(random_state_seed_)
        n_trials_optuna = config.get('n_trials_optuna', 4)


        params_optuna = setup_model_params_optuna(trial, config['model_type'], random_state_seed_, device)

        model_weight_optuna = setup_model_weight_optuna(trial, weight_param, config)

        cv = setup_cv_method(df_init=df_init,X_train=X_train, y_train_label=y_train_label,cv_method=cv_method,
                             nb_split_tscv=nb_split_tscv,config=config)

        X_train, selected_feature_names = manage_rfe_selection(X_train=X_train,
            y_train_label=y_train_label,
            config=config,
            trial=trial,
            params=params_optuna,
            model_weight_optuna=model_weight_optuna
        )

        cv_results = run_cross_validation(X_train=X_train, X_train_full=X_train_full,
            y_train_label=y_train_label,
            trial=trial,
            params=params_optuna,
            model_weight_optuna=model_weight_optuna,
            cv=cv,
            nb_split_tscv=nb_split_tscv,
            is_log_enabled=is_log_enabled,
            model=model,
            config=config
        )
        """
        # 8. Sauvegarde de l'état
        state = {
            'iteration': objective_optuna.iteration_counter,
            'trial_number': trial.number,
            'params': params,
            'threshold': model_weight_optuna['threshold'],
            'mean_score': cv_results['mean_val_score'],
            'std_score': cv_results['std_val_score'],
            'metrics': {k: float(v) for k, v in cv_results['metrics'].items()},
            'memory_usage': cp.get_default_memory_pool().used_bytes() / 1024 ** 2
        }
        """
        """
        print(f"\n{'=' * 50}")
        print(f"Fin itération Optuna #{objective_optuna.iteration_counter}")
        print(f"{'=' * 50}")
         """

    except Exception as e:
        print(f"\n{'!' * 50}")
        print(f"Erreur dans l'itération #{objective_optuna.iteration_counter}:")
        print(str(e))
        print(f"{'!' * 50}")
        raise
    finally:
        # Nettoyage
        if config['device_'] == 'cuda':
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
    # print_notification("fin de la CV")

    if ENV == 'pycharm':
        if keyboard.is_pressed('x'):  # Nécessite le package 'keyboard'
            study.stop()
    else:
        if os.path.exists('stop_optimization.txt'):
            study.stop()
    # Calculs finaux et métriques

    if config['device_'] == 'cuda':
        import cupy as cp

        winrates_by_fold_cpu = cp.asnumpy(cv_results['winrates_by_fold'])
        nb_trades_by_fold_cpu = cp.asnumpy(cv_results['nb_trades_by_fold'])
        scores_train_by_fold_cpu = cp.asnumpy(cv_results['scores_train_by_fold'])
        tp_train_by_fold_cpu = cp.asnumpy(cv_results['tp_train_by_fold'])
        fp_train_by_fold_cpu = cp.asnumpy(cv_results['fp_train_by_fold'])
        tp_val_by_fold_cpu = cp.asnumpy(cv_results['tp_val_by_fold'])
        fp_val_by_fold_cpu = cp.asnumpy(cv_results['fp_val_by_fold'])
        scores_val_by_fold_cpu = cp.asnumpy(cv_results['scores_val_by_fold'])

        # Nettoyage mémoire GPU
        cp.get_default_memory_pool().free_all_blocks()
    else:
        # Utilisation directe sur CPU avec NumPy ou Python natif
        winrates_by_fold_cpu = np.array(cv_results['winrates_by_fold']) if isinstance(cv_results['winrates_by_fold'],
                                                                                      list) else cv_results[
            'winrates_by_fold']
        nb_trades_by_fold_cpu = np.array(cv_results['nb_trades_by_fold']) if isinstance(cv_results['nb_trades_by_fold'],
                                                                                        list) else cv_results[
            'nb_trades_by_fold']
        scores_train_by_fold_cpu = np.array(cv_results['scores_train_by_fold']) if isinstance(
            cv_results['scores_train_by_fold'], list) else cv_results['scores_train_by_fold']
        tp_train_by_fold_cpu = np.array(cv_results['tp_train_by_fold']) if isinstance(cv_results['tp_train_by_fold'],
                                                                                      list) else cv_results[
            'tp_train_by_fold']
        fp_train_by_fold_cpu = np.array(cv_results['fp_train_by_fold']) if isinstance(cv_results['fp_train_by_fold'],
                                                                                      list) else cv_results[
            'fp_train_by_fold']
        tp_val_by_fold_cpu = np.array(cv_results['tp_val_by_fold']) if isinstance(cv_results['tp_val_by_fold'],
                                                                                  list) else cv_results[
            'tp_val_by_fold']
        fp_val_by_fold_cpu = np.array(cv_results['fp_val_by_fold']) if isinstance(cv_results['fp_val_by_fold'],
                                                                                  list) else cv_results[
            'fp_val_by_fold']
        scores_val_by_fold_cpu = np.array(cv_results['scores_val_by_fold']) if isinstance(
            cv_results['scores_val_by_fold'], list) else cv_results['scores_val_by_fold']

    if is_log_enabled:
        display_metrics(cv_results)

    # Conversion des fold_stats
    fold_stats = {}
    if config['device_'] == 'cuda':
        import cupy as cp
        for fold_num, stats in cv_results['fold_stats'].items():
            fold_stats[fold_num] = {
                'train_n_trades': float(cp.asnumpy(stats['train_n_trades'])) if isinstance(stats['train_n_trades'],
                                                                                           cp.ndarray) else stats[
                    'train_n_trades'],
                'train_n_class_1': float(cp.asnumpy(stats['train_n_class_1'])) if isinstance(stats['train_n_class_1'],
                                                                                             cp.ndarray) else stats[
                    'train_n_class_1'],
                'train_n_class_0': float(cp.asnumpy(stats['train_n_class_0'])) if isinstance(stats['train_n_class_0'],
                                                                                             cp.ndarray) else stats[
                    'train_n_class_0'],
                'train_class_ratio': float(cp.asnumpy(stats['train_class_ratio'])) if isinstance(
                    stats['train_class_ratio'],
                    cp.ndarray) else stats[
                    'train_class_ratio'],
                'train_success_rate': float(cp.asnumpy(stats['train_success_rate'])) if isinstance(
                    stats['train_success_rate'], cp.ndarray) else stats['train_success_rate'],
                'val_n_trades': float(cp.asnumpy(stats['val_n_trades'])) if isinstance(stats['val_n_trades'],
                                                                                       cp.ndarray) else stats[
                    'val_n_trades'],
                'val_n_class_1': float(cp.asnumpy(stats['val_n_class_1'])) if isinstance(stats['val_n_class_1'],
                                                                                         cp.ndarray) else stats[
                    'val_n_class_1'],
                'val_n_class_0': float(cp.asnumpy(stats['val_n_class_0'])) if isinstance(stats['val_n_class_0'],
                                                                                         cp.ndarray) else stats[
                    'val_n_class_0'],
                'val_class_ratio': float(cp.asnumpy(stats['val_class_ratio'])) if isinstance(stats['val_class_ratio'],
                                                                                             cp.ndarray) else stats[
                    'val_class_ratio'],
                'val_success_rate': float(cp.asnumpy(stats['val_success_rate'])) if isinstance(
                    stats['val_success_rate'],
                    cp.ndarray) else stats[
                    'val_success_rate'],
                'fold_num': stats['fold_num'],
                'train_size': stats['train_size'],
                'val_size': stats['val_size'],
                'best_iteration': stats['best_iteration'],
                'val_score': stats['val_score'],
                'train_score': stats['train_score']
            }

        # Nettoyage mémoire GPU
        cp.get_default_memory_pool().free_all_blocks()
    else:
        for fold_num, stats in cv_results['fold_stats'].items():
            fold_stats[fold_num] = {
                'train_n_trades': float(stats['train_n_trades']) if isinstance(stats['train_n_trades'], np.ndarray) else
                stats['train_n_trades'],
                'train_n_class_1': float(stats['train_n_class_1']) if isinstance(stats['train_n_class_1'],
                                                                                 np.ndarray) else stats[
                    'train_n_class_1'],
                'train_n_class_0': float(stats['train_n_class_0']) if isinstance(stats['train_n_class_0'],
                                                                                 np.ndarray) else stats[
                    'train_n_class_0'],
                'train_class_ratio': float(stats['train_class_ratio']) if isinstance(stats['train_class_ratio'],
                                                                                     np.ndarray) else stats[
                    'train_class_ratio'],
                'train_success_rate': float(stats['train_success_rate']) if isinstance(stats['train_success_rate'],
                                                                                       np.ndarray) else stats[
                    'train_success_rate'],
                'val_n_trades': float(stats['val_n_trades']) if isinstance(stats['val_n_trades'], np.ndarray) else
                stats['val_n_trades'],
                'val_n_class_1': float(stats['val_n_class_1']) if isinstance(stats['val_n_class_1'], np.ndarray) else
                stats['val_n_class_1'],
                'val_n_class_0': float(stats['val_n_class_0']) if isinstance(stats['val_n_class_0'], np.ndarray) else
                stats['val_n_class_0'],
                'val_class_ratio': float(stats['val_class_ratio']) if isinstance(stats['val_class_ratio'],
                                                                                 np.ndarray) else stats[
                    'val_class_ratio'],
                'val_success_rate': float(stats['val_success_rate']) if isinstance(stats['val_success_rate'],
                                                                                   np.ndarray) else stats[
                    'val_success_rate'],
                'fold_num': stats['fold_num'],
                'train_size': stats['train_size'],
                'val_size': stats['val_size'],
                'best_iteration': stats['best_iteration'],
                'val_score': stats['val_score'],
                'train_score': stats['train_score']
            }

    # Conversion des totaux validation
    total_tp_val = float(cv_results['metrics']['total_tp_val'])  # Déjà en NumPy
    total_fp_val = float(cv_results['metrics']['total_fp_val'])
    total_tn_val = float(cv_results['metrics']['total_tn_val'])
    total_fn_val = float(cv_results['metrics']['total_fn_val'])

    # Conversion des totaux entraînement
    total_tp_train = float(cv_results['metrics']['total_tp_train'])
    total_fp_train = float(cv_results['metrics']['total_fp_train'])
    total_tn_train = float(cv_results['metrics']['total_tn_train'])
    total_fn_train = float(cv_results['metrics']['total_fn_train'])

    # Calculs des métriques finales
    total_samples_val = total_tp_val + total_fp_val + total_tn_val + total_fn_val
    total_samples_train = total_tp_train + total_fp_train + total_tn_train + total_fn_train
    total_trades_val = total_tp_val + total_fp_val
    total_trades_train = total_tp_train + total_fp_train

    total_pnl_val = sum(scores_val_by_fold_cpu)
    total_pnl_train = sum(scores_train_by_fold_cpu)

    val_pnl_perTrades = total_pnl_val / total_trades_val if total_trades_val > 0 else 0
    train_pnl_perTrades = total_pnl_train / total_trades_train if total_trades_train > 0 else 0

    pnl_perTrade_diff = abs(val_pnl_perTrades - train_pnl_perTrades)

    # Suggestions des paramètres finaux
    weight_split = trial.suggest_float('weight_split', weight_param['weight_split']['min'],
                                       weight_param['weight_split']['max'])
    nb_split_weight = trial.suggest_int('nb_split_weight', weight_param['nb_split_weight']['min'],
                                        weight_param['nb_split_weight']['max'])
    std_penalty_factor = trial.suggest_float('std_penalty_factor', weight_param['std_penalty_factor']['min'],
                                             weight_param['std_penalty_factor']['max'])

    # Calcul du score final ajusté
    score_adjustedStd_val, mean_cv_score, std_dev_score = calculate_weighted_adjusted_score_custom(
        scores_val_by_fold_cpu,
        weight_split=weight_split,
        nb_split_weight=nb_split_weight,
        std_penalty_factor=std_penalty_factor)

    # Calculs des métriques finales
    if total_samples_val > 0:
        tp_percentage = (total_tp_val / total_samples_val) * 100
    else:
        tp_percentage = 0

    win_rate = total_tp_val / total_trades_val * 100 if total_trades_val > 0 else 0
    tp_fp_diff_val = total_tp_val - total_fp_val
    cummulative_pnl_val = total_tp_val * weight_param['profit_per_tp']['min'] + total_fp_val * \
                          weight_param['loss_per_fp']['max']

    # print(f"cummulative_pnl_val: {cummulative_pnl_val}")

    # Mise à jour des attributs du trial
    trial.set_user_attr('params_optuna', params_optuna)
    trial.set_user_attr('model_weight_optuna', model_weight_optuna)

    trial.set_user_attr('total_tp_val', total_tp_val)
    trial.set_user_attr('total_fp_val', total_fp_val)
    trial.set_user_attr('total_tn_val', total_tn_val)
    trial.set_user_attr('total_fn_val', total_fn_val)
    trial.set_user_attr('weight_param', weight_param)
    trial.set_user_attr('scores_ens_val_list', scores_val_by_fold_cpu.tolist())
    trial.set_user_attr('nb_split_tscv', nb_split_tscv)
    trial.set_user_attr('mean_cv_score', mean_cv_score)
    trial.set_user_attr('std_dev_score', std_dev_score)
    trial.set_user_attr('std_penalty_factor', std_penalty_factor)
    trial.set_user_attr('score_adjustedStd_val', score_adjustedStd_val)
    trial.set_user_attr('train_pnl_perTrades', train_pnl_perTrades)
    trial.set_user_attr('val_pnl_perTrades', val_pnl_perTrades)
    trial.set_user_attr('pnl_perTrade_diff', pnl_perTrade_diff)
    trial.set_user_attr('total_samples_val', total_samples_val)
    trial.set_user_attr('n_trials_optuna', n_trials_optuna)
    trial.set_user_attr('tp_percentage', tp_percentage)
    trial.set_user_attr('win_rate', win_rate)
    trial.set_user_attr('tp_fp_diff_val', tp_fp_diff_val)
    trial.set_user_attr('cummulative_pnl_val', cummulative_pnl_val)

    trial.set_user_attr('winrates_by_fold', winrates_by_fold_cpu.tolist())
    trial.set_user_attr('nb_trades_by_fold', nb_trades_by_fold_cpu.tolist())

    trial.set_user_attr('weight_split', weight_split)
    trial.set_user_attr('nb_split_weight', nb_split_weight)
    # trial.set_user_attr('model', model)
    trial.set_user_attr('selected_feature_names', selected_feature_names)
    trial.set_user_attr('use_of_rfe_in_optuna', config.get('use_of_rfe_in_optuna', rfe_param.NO_RFE))
    trial.set_user_attr('optuna_objective_type',
                        config.get('optuna_objective_type', optuna_doubleMetrics.DISABLE))
    trial.set_user_attr('profit_per_tp', weight_param['profit_per_tp'])
    trial.set_user_attr('penalty_per_fn', weight_param['penalty_per_fn'])
    trial.set_user_attr('tp_val_list', tp_val_by_fold_cpu)
    trial.set_user_attr('fp_val_list', fp_val_by_fold_cpu)
    trial.set_user_attr('cv_method', cv_method)
    use_imbalance_penalty = config.get('use_imbalance_penalty', False)
    trial.set_user_attr('use_imbalance_penalty', use_imbalance_penalty)



    # 5. Retour du score ajusté (à minimiser) et de la différence de PnL par trade
    objectives = calculate_normalized_objectives(
        tp_train_list=tp_train_by_fold_cpu,
        fp_train_list=fp_train_by_fold_cpu,
        tp_val_list=tp_val_by_fold_cpu,
        fp_val_list=fp_val_by_fold_cpu,
        scores_train_list=scores_train_by_fold_cpu,
        scores_val_list=scores_val_by_fold_cpu,
        fold_stats=fold_stats,
        scale_objectives=False,
        use_imbalance_penalty=use_imbalance_penalty
    )

    # Extraction des métriques brutes pour analyse
    raw_metrics = objectives['raw_metrics']

    # Affichage des métriques détaillées
    print("\nMétriques brutes :")
    print(f"PnL moyen          : {raw_metrics['avg_pnl']:.4f}")
#    print(f"Winrate diff moyen : {raw_metrics['ecart_train_val']:.4%}")
    print(f"Pénalité imbalance : {raw_metrics['imbalance_penalty']:.4f}")

    print("\nObjectifs normalisés finaux :")
    print(f"PnL objectif  Normalisé        : {objectives['pnl_norm_objective']:.4f}")
    print(f"Winrate diff objectif Normalisé: {objectives['ecart_train_val']:.4f}")

    # Sauvegarde des métriques dans Optuna
    trial.set_user_attr('pnl_norm_objective', objectives['pnl_norm_objective'])
    trial.set_user_attr('ecart_train_val', objectives['ecart_train_val'])
    trial.set_user_attr('constraint_ecart_train_val', config.get('constraint_ecart_train_val',0))
    trial.set_user_attr('constraint_winrates_by_fold', config.get('constraint_winrates_by_fold', 0))
    trial.set_user_attr('constraint_min_trades_threshold_by_Fold',
                        config.get('constraint_min_trades_threshold_by_Fold', 0))
    trial.set_user_attr('raw_avg_pnl', raw_metrics['avg_pnl'])
    trial.set_user_attr('imbalance_penalty', raw_metrics['imbalance_penalty'])

    # Retourner les objectifs pour Optuna
    return [
        objectives['pnl_norm_objective'],
        objectives['ecart_train_val']
    ]
    # return score_adjustedStd_val, pnl_perTrade_diff  # Retour normal


########################################
#########   END FUNCTION DEF   #########
########################################


def train_and_evaluate_XGBOOST_model(
        df_init=None,
        config=None,  # Add config parameter here
        weight_param=None
):
    device = config.get('device_', 'cuda')
    n_trials_optimization = config.get('n_trials_optuna', 4)
    nb_split_tscv = config.get('nb_split_tscv_', 10)
    nanvalue_to_newval = config.get('nanvalue_to_newval_', np.nan)
    random_state_seed = config.get('random_state_seed', 30)
    #early_stopping_rounds = config.get('early_stopping_rounds', 70)
    cv_method = config.get('cv_method', cv_config.K_FOLD)
    #optuna_objective_type_value = config.get('optuna_objective_type ', optuna_doubleMetrics.USE_DIST_TO_IDEAL)
    is_log_enabled = config.get('is_log_enabled', False)
    selected_columns= config.get('selected_columns', None)
    chosen_scaler = config.get('scaler_choice', scalerChoice.SCALER_ROBUST)
    model = config.get('model_type', modelType.XGB)

    zeros = (df_init['class_binaire'] == 0).sum()
    ones = (df_init['class_binaire'] == 1).sum()
    total = zeros + ones
    print(f"Dimensions de df_init: {df_init.shape} (lignes, colonnes)")
    print("df_init:")
    print(df_init)

    (X_train_full, y_train_full_label, X_test_full, y_test_full_label,
     X_train, y_train_label, X_test, y_test_label,
     nb_SessionTrain, nb_SessionTest, nan_value) = (
        init_dataSet(df_init=df_init, nanvalue_to_newval=nanvalue_to_newval,
                     config=config, CUSTOM_SESSIONS_=CUSTOM_SESSIONS, results_directory=results_directory))

    print(
        f"\nValeurs NaN : X_train={X_train.isna().sum().sum()}, y_train_label={y_train_label.isna().sum()}, X_test={X_test.isna().sum().sum()}, y_test_label={y_test_label.isna().sum()}\n")

    print(f"Dimensions de X_train: {X_train.shape} (lignes, colonnes)")

    print(f"Nb de features après exlusion manuelle: {len(selected_columns)}\n")

    # Affichage des informations sur les NaN et zéros dans chaque colonne
    print(f"\nFeatures X_train_full après exclusion manuelle des features (short + 99)(a verivier AL)):")
    displaytNan_vifMiCorrFiltering(X=X_train_full, selected_columns=selected_columns,name="X_train_full",config=config)

    print(f"Features X_train après exclusion manuelle des features (sur trades short après exclusion de 99):")
    displaytNan_vifMiCorrFiltering(X=X_train, selected_columns=selected_columns, name="X_train",
                                   config=config)


    if chosen_scaler != scalerChoice.SCALER_DISABLE:
        # Sauvegarde des données originales pour réinsertion potentielle
        X_train_original = X_train
        X_test_original = X_test
        y_train_label_original = y_train_label
        y_test_label_original = y_test_label

        # Nettoyage des NaN et Inf
        X_train, y_train_label, mask_train = remove_nan_inf(X_train, y_train_label, "train")
        X_test, y_test_label, mask_test = remove_nan_inf(X_test, y_test_label, "test")

        save_sacler_dir = os.path.join(results_directory, 'optuna_results')

        is_coherence_ranges_problem = False
        # Vérification de la cohérence des distributions
        diff_features = check_distribution_coherence(X_train, X_test)
        if diff_features:
            print("Avertissement : certaines features ont des distributions très différentes entre X_train et X_test :")
            for f, stats in diff_features.items():
                print(f"Feature: {f}, KS-stat: {stats['statistic']:.3f}, p-value: {stats['p_value']:.3e}")

        # Vérification des bornes
        oob = check_value_ranges(X_train, X_test)
        if oob:
            print(
                "Avertissement : certaines features contiennent des valeurs en dehors des bornes observées dans X_train :")
            for f, vals in oob.items():
                print(f"\nFeature: {f}")
                print(f"Valeurs sous le min ({vals['train_min']}): {vals['below_min_count']}")
                if vals['below_min_count'] > 0:
                    print(f"Liste des valeurs sous le min: {vals['below_min_values']}")
                print(f"Valeurs au-dessus du max ({vals['train_max']}): {vals['above_max_count']}")
                if vals['above_max_count'] > 0:
                    print(f"Liste des valeurs au-dessus du max: {vals['above_max_values']}")

        if is_coherence_ranges_problem:
            raise ValueError("Un problème de valeurs hors bornes ou de distribution détecté")

        # Application du scaling
        X_train_scaled, X_test_scaled, scaler, scaler_params = apply_scaling(
            X_train,
            X_test,
            save_path=save_sacler_dir,
            chosen_scaler=chosen_scaler
        )
        print("\nScaler actif\n")

        # Réinsertion des valeurs NaN et Inf si demandé
        reinsert_nan_inf_afterScaling = config.get('reinsert_nan_inf_afterScaling', False)

        if reinsert_nan_inf_afterScaling:
            X_train = X_train_original.copy()
            X_test = X_test_original.copy()
            y_train_label = y_train_label_original
            y_test_label = y_test_label_original

            # Mise à jour uniquement des valeurs valides avec les données scalées
            X_train[mask_train] = X_train_scaled
            X_test[mask_test] = X_test_scaled

            print("\nRéinsertion des valeurs NaN et Inf effectuée")
            print(f"Train : {(~mask_train).sum()} lignes réinsérées")
            print(f"Test : {(~mask_test).sum()} lignes réinsérées")
        else:
            X_train = X_train_scaled
            X_test = X_test_scaled
    else:
        print("\nPas de scaler actif\n")

    if len(X_train) != len(y_train_label):
        raise ValueError(f"Mismatch des tailles (pas de scaler): "
                         f"X_train ({len(X_train)}) et y_train_label ({len(y_train_label)})")
    if len(X_test) != len(y_test_label):
        raise ValueError(f"Mismatch des tailles (pas de scaler): "
                         f"X_test ({len(X_test)}) et y_test_label ({len(y_test_label)})")
    print("X_train")
    print(X_train)
    print("X_test")
    print(X_test)


    enable_vif_corr_mi = config.get('enable_vif_corr_mi', None)
    if enable_vif_corr_mi:
        selected_columns_afterVifCorrMiFiltering = displaytNan_vifMiCorrFiltering(X=X_train,Y= y_train_label,selected_columns=selected_columns,name="X_train",config=config,enable_vif_corr_mi=enable_vif_corr_mi)
        print("\nRésumé:")
        print(f"Nombre total de features filtrées manuellement: {len(selected_columns)}")
        print(
            f"Nombre total de features après filtrage VIF, CORR et MI: {len(selected_columns_afterVifCorrMiFiltering)}")
    else :
        selected_columns_afterVifCorrMiFiltering=selected_columns
        print("\nRésumé:")
        print(f"Nombre total de features filtrées manuellement: {len(selected_columns)}")
        print(
            f"Nombre total de features (filtrage VIF, CORR et MI désactivé): {len(selected_columns_afterVifCorrMiFiltering)}")

    X_train=X_train[selected_columns_afterVifCorrMiFiltering]
    X_test=X_test[selected_columns_afterVifCorrMiFiltering]



    # Affichage de la distribution des classes
    print("Distribution des trades (excluant les 99):")
    trades_distribution = y_train_label.value_counts(normalize=True)
    trades_counts = y_train_label.value_counts()
    print(f"Trades échoués [0]: {trades_distribution.get(0, 0) * 100:.2f}% ({trades_counts.get(0, 0)} trades)")
    print(f"Trades réussis [1]: {trades_distribution.get(1, 0) * 100:.2f}% ({trades_counts.get(1, 0)} trades)")

    # Vérification de l'équilibre des classes
    total_trades_train = y_train_label.count()
    total_trades_test = y_test_label.count()

    print(f"Nombre total de trades pour l'ensemble d'entrainement (excluant les 99): {total_trades_train}")
    print(f"Nombre total de trades pour l'ensemble de test (excluant les 99): {total_trades_test}")

    thresholdClassImb = 0.06
    class_difference = abs(trades_distribution.get(0, 0) - trades_distribution.get(1, 0))
    if class_difference >= thresholdClassImb:
        print(f"Erreur : Les classes ne sont pas équilibrées. Différence : {class_difference:.2f}")
        # sys.exit(1)
    else:
        print(f"Les classes sont considérées comme équilibrées (différence : {class_difference:.2f})")

    # **Ajout de la réduction des features ici, avant l'optimisation**

    print_notification('###### FIN: CHARGER ET PRÉPARER LES DONNÉES  ##########', color="blue")

    # Début de l'optimisation
    print_notification('###### DÉBUT: OPTIMISATION BAYESIENNE ##########', color="blue")


    # Assurez-vous que X_test et y_test_label ont le même nombre de lignes
    assert X_test.shape[0] == y_test_label.shape[0], "X_test et y_test_label doivent avoir le même nombre de lignes"

    # Adjust the objective wrapper function
    def objective_wrapper(trial, study, model_weight_optuna):
        # Call your original objective function
        score_adjustedStd_val, pnl_perTrade_diff = objective_optuna(df_init=df_init,
                                                                    trial=trial, study=study, X_train=X_train,
                                                                    X_test=X_test,
                                                                    y_train_label=y_train_label,
                                                                    X_train_full=X_train_full,
                                                                    device=device,
                                                                    config=config, nb_split_tscv=nb_split_tscv,
                                                                    model_weight_optuna=model_weight_optuna,
                                                                    weight_param=weight_param,
                                                                    random_state_seed_=random_state_seed,
                                                                    is_log_enabled=is_log_enabled,
                                                                    cv_method=cv_method,
                                                                    selected_columns=selected_columns,
                                                                    model=model
                                                                    )

        if config.get('optuna_objective_type', optuna_doubleMetrics.DISABLE) == optuna_doubleMetrics.DISABLE:
            # Return only the first objective
            return score_adjustedStd_val
        else:
            # Return both objectives
            return score_adjustedStd_val, pnl_perTrade_diff

    model_weight_optuna = {}

    weightPareto_pnl_val = config.get('weightPareto_pnl_val', 0.6)
    weightPareto_pnl_diff = config.get('weightPareto_pnl_diff', 0.4)

    # Vérifier que la somme des poids est égale à 1
    if abs(weightPareto_pnl_val + weightPareto_pnl_diff - 1.0) > 1e-6:  # Tolérance d'erreur flottante
        raise ValueError("La somme des poids (weightPareto_pnl_val + weightPareto_pnl_diff) doit être égale à 1.0")

    # Créer l'étude
    # Assume 'optuna_objective_type' is the parameter that determines the optimization mode



    # Conditionally create the study
    if config.get('optuna_objective_type', optuna_doubleMetrics.DISABLE) == optuna_doubleMetrics.DISABLE:
        # Create a single-objective study
        # Définition de la fonction de contraintes

        def create_constraints_func():
            def constraints_func(trial):
                constraints=calculate_constraints_optuna(trial=trial,config=config)

                return constraints
            return constraints_func
        sampler = optuna.samplers.TPESampler(
            seed=42,
            constraints_func=create_constraints_func() if config.get('use_optuna_constraints_func', False) else None
        )
        study_optuna = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
    else:
        # Create a multi-objective study
        study_optuna = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )

    # Créer une fonction wrapper pour le callback qui inclut optuna

    def callback_wrapper(study, trial):
        return callback_optuna(study, trial, optuna, study_optuna, rfe_param, config,
                               results_directory)


    # Lancer l'optimisation avec le wrapper

    # Lancer l'optimisationxxxxxx
    study_optuna.optimize(
        lambda trial: objective_wrapper(trial, study_optuna, model_weight_optuna),
        n_trials=n_trials_optimization,
        callbacks=[callback_wrapper],
    )

    bestResult_dict = study_optuna.user_attrs['bestResult_dict']




    # Après l'optimisation
    #best_params = bestResult_dict["best_params"]
    params_optuna = bestResult_dict["params_optuna"]
    model_weight_optuna = bestResult_dict["model_weight_optuna"]

    optimal_threshold = model_weight_optuna['threshold']


    selected_feature_names = bestResult_dict["selected_feature_names"]
    rfe_param_value = bestResult_dict["use_of_rfe_in_optuna"]
    print("#################################")
    print("#################################")
    print(
        f"## Optimisation Optuna terminée Meilleur essai : {bestResult_dict['best_optunaTrial_number']}")
    print(f"## Meilleurs hyperparamètres trouvés pour params_optuna: ", params_optuna)
    print(f"## Meilleurs hyperparamètres trouvés pour model_weight_optuna: ", model_weight_optuna)


    print(f"## Seuil utilisé : {optimal_threshold:.4f}")
    print("## Meilleur score Objective 1 (pnl_norm_objective): ", bestResult_dict["pnl_norm_objective"])
    if config.get('optuna_objective_type', optuna_doubleMetrics.DISABLE) != optuna_doubleMetrics.DISABLE:
        print("## Meilleur score Objective 2 (ecart_train_val): ",
              bestResult_dict["ecart_train_val"])
    print("#################################")
    print("#################################\n")

    print_notification('###### FIN: OPTIMISATION BAYESIENNE ##########', color="blue")

    print_notification('###### DEBUT: ENTRAINEMENT MODELE FINAL ##########', color="blue")

    print(selected_feature_names)
    X_train = X_train[selected_feature_names]
    X_test = X_test[selected_feature_names]

    train_finalModel_analyse(xgb=xgb,
                             X_train=X_train, X_train_full=X_train_full, X_test=X_test, X_test_full=X_test_full,
                             y_train_label=y_train_label,y_test_label=y_test_label,
                             nb_SessionTest=nb_SessionTest, nan_value=nan_value, feature_names=selected_feature_names,
                             config=config,weight_param=weight_param,bestResult_dict=bestResult_dict)


############### main######################
if __name__ == "__main__":
    # Demander à l'utilisateur s'il souhaite afficher les graphiques
    check_gpu_availability()
    from parameters import get_path, get_weight_param,get_config
    DIRECTORY_PATH,FILE_PATH, base_results_path=get_path()
    weight_param=get_weight_param()
    config=get_config()
    directories = DIRECTORY_PATH.split(os.path.sep)
    print(directories)
    target_directory = directories[-2]
    results_directory=os.path.join(base_results_path, target_directory),

    # Obtenir l'heure et la date actuelles
    now = datetime.now()
    # Formater l'heure et la date au format souhaité
    time_suffix = now.strftime("_%H_%M_%d%m%y")
    # Ajouter le suffixe à target_directory
    target_directory += time_suffix

    config.update({
        'target_directory': target_directory})

    config.update({
        'results_directory': os.path.join(base_results_path, target_directory)})

    # Exemple d'utilisation
    print(f"Le répertoire cible est : {target_directory}")

    results_directory = config.get('results_directory', None)
    print(results_directory)
    # Créer le répertoire s'il n'existe pas
    os.makedirs(results_directory, exist_ok=True)

    # Définir les paramètres supplémentaires
    print_notification('###### DEBUT: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")

    df_init, CUSTOM_SESSIONS = load_features_and_sections(FILE_PATH)



    print("\nContenu de CUSTOM_SESSIONS (format tabulé) :")
    print(f"{'Section':<15} {'Start':>6} {'End':>6} {'Type':>6} {'Selected':>8} {'Description':<20}")
    print("-" * 70)
    for section, data in CUSTOM_SESSIONS.items():
        print(f"{section:<15} {data['start']:>6} {data['end']:>6} {data['session_type_index']:>6} "
              f"{str(data['selected']):>8} {data['description']:<20}")

    print_notification('###### FIN: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")
    # Utilisation
    # Définition des sections personnalisées
    print("df_init: ")
    zeros = (df_init['class_binaire'] == 0).sum()
    ones = (df_init['class_binaire'] == 1).sum()
    total = zeros + ones
    print(f"   - Dimensions de df_init: {df_init.shape} (lignes, colonnes)")
    print(
        f"   - Distribution des trades df_init - Échecs (0): {zeros} ({zeros / total * 100:.1f}%), Réussis (1): {ones} ({ones / total * 100:.1f}%), Total: {total}")

    print(f"   - Nb de features avant  selection manuelle: {len(df_init.columns)}\n")

    # Définition des colonnes de features et des colonnes exclues
    excluded_columns_principal = [
        'class_binaire', 'date', 'trade_category',
        'SessionStartEnd',

        'timeStampOpening',
        'deltaTimestampOpening',
        'candleDir',
        # 'deltaTimestampOpeningSession1min', bear_imbalance_high_3 4.49 NAN
        'deltaTimestampOpeningSession1index',
        'deltaTimestampOpeningSession5min',
        'deltaTimestampOpeningSession5index',
        'deltaTimestampOpeningSession15min',
        'deltaTimestampOpeningSession15index',
        'deltaTimestampOpeningSession30min',
        'deltaTimestampOpeningSession30index',
        'deltaCustomSessionMin',
        'deltaCustomSessionIndex',
        'meanVolx',
        'total_count_abv',
        'total_count_blw',
        'staked00_high',
        'staked00_low',
        'bear_imbalance_high_3', ## 4.8 % de NAN
        'bull_imbalance_high_0', #7.8%
        'bearish_absorption_ratio' #2.8nan,


    ]
    excluded_columns_tradeDirection= [
        'bullish_ask_bid_ratio',
        'bullish_ask_ratio',
        'bullish_bid_ratio',
        'bullish_ask_score',
        'bullish_bid_score',
        'bullish_imnbScore_score',
        'bullish_ask_abs_ratio_blw',
        'bullish_bid_abs_ratio_blw',
        'bullish_abs_diff_blw',
        'bullish_asc_dsc_ratio',
        'bullish_asc_dynamics',
        'bullish_dsc_dynamics',
        'bullish_asc_ask_bid_imbalance',
        'bullish_dsc_ask_bid_imbalance',
        'bullish_imbalance_evolution',
        'bullish_asc_ask_bid_delta_imbalance',
        'bullish_dsc_ask_bid_delta_imbalance',
        'bullish_absorption_ratio',
        'absorption_intensity_repeat_bullish_vol',
        'bullish_absorption_intensity_repeat_count',
        'bullish_repeatAskBid_ratio',
        'bullish_absorption_score',
        'bullish_market_context_score',
        'bullish_combined_pressure',
       'naked_poc_dist_above',
        'bull_imbalance_low_1',
        'bull_imbalance_low_2',
        'bull_imbalance_low_3',
        'bear_imbalance_low_0',
        'bear_imbalance_low_1',
        'bear_imbalance_low_2',
        ]

    excluded_columns_CorrCol = [
    ]

    # Liste des catégories à exclure
    excluded_categories = [
        '_special',
        '_6Tick',
        'BigHigh',
        'bigHigh',
        'big',
        'Big',
        'state',
        'State',
        'extrem',
        'Extrem',
        "bullish",
    ]

    # Créer la liste des colonnes à exclure
    excluded_columns_category = [
        col for col in df_init.columns
        if any(category in col for category in excluded_categories)
    ]

    print(excluded_columns_category)
    excluded_columns = excluded_columns_principal + excluded_columns_tradeDirection+excluded_columns_CorrCol+excluded_columns_category

    # ajoute les colonnes pour retraitement ultérieurs
    df_init = add_session_id(df_init, CUSTOM_SESSIONS)
    # Sélectionner les colonnes qui ne sont pas dans excluded_columns
    selected_columns = [col for col in df_init.columns if col not in excluded_columns]

    selected_columnsByFiltering = [

    ]

    if selected_columnsByFiltering != []:
        selected_columns = selected_columnsByFiltering  # Assigne les colonnes filtrées à selected_columns
        config.update({
            'enable_vif_corr_mi': False})


    config.update({
        'excluded_columns_principal':excluded_columns_principal ,
        'excluded_columns_tradeDirection':excluded_columns_tradeDirection ,
        'excluded_columns_CorrCol': excluded_columns_CorrCol,
        'excluded_columns_category': excluded_columns_category,
        'selected_columns':selected_columns
    })


    results = train_and_evaluate_XGBOOST_model(
        df_init=df_init,
        config=config,
        weight_param=weight_param
    )

    if results is not None:
        print("entrainement et analyse termisé")
    else:
        print("L'entraînement n'a pas produit de résultats.")



