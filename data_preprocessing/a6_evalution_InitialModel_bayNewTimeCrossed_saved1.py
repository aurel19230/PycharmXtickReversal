from numba import njit
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
from functools import partial
from typing import Dict, List, Tuple

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
import cupy as cp
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

class rfe_param(Enum):
    NO_RFE = 0
    RFE_WITH_OPTUNA = 1
    RFE_AUTO = 2


import os


def detect_environment():
    """
    Détecte l'environnement d'exécution
    Returns: 'colab', 'pycharm', ou 'other'
    """
    # Vérification pour Google Colab
    try:
        from google.colab import drive
        return 'colab'
    except ImportError:
        # Vérification pour PyCharm
        if 'PYCHARM_HOSTED' in os.environ or 'PYCHARM_MATPLOTLIB_PORT' in os.environ:
            return 'pycharm'
        return 'other'


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
import optuna
from colorama import Fore, Style, init

# Utilisation
ENV = detect_environment()

# Import des fonctions selon l'environnement
if ENV == 'colab':
    BASE_PATH = "/content/drive/MyDrive/..."
    print("le code s'éxécute sur collab")

    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is not None:
            print("le code s'éxécute sur collab1")
            ipython.run_line_magic('run', '/content/drive/MyDrive/Colab_Notebooks/xtickReversal/standardFunc.ipynb')
            from standardFunc_sauv import *  # Import après l'exécution du notebook
    except Exception as e:
        print(f"Erreur lors de l'exécution du notebook: {str(e)}")

elif ENV == 'pycharm':
    BASE_PATH = "C:/Users/aulac/OneDrive/..."
    from standardFunc import (load_data, split_sessions, print_notification,
                              plot_calibrationCurve_distrib, plot_fp_tp_rates, check_gpu_availability,
                              optuna_doubleMetrics,
                              timestamp_to_date_utc, calculate_and_display_sessions,
                              calculate_and_display_sessions, callback_optuna,
                              calculate_weighted_adjusted_score_custom, sigmoidCustom,
                              custom_metric_ProfitBased_gpu, create_weighted_logistic_obj_gpu,
                              xgb_metric, scalerChoice,ScalerMode,modeleType,
                              train_finalModel_analyse, init_dataSet, compute_confusion_matrix_cupy,
                              sessions_selection, calculate_normalized_objectives,
                              run_cross_validation, setup_metric_dict,
                              process_RFE_filteringg, calculate_fold_stats, add_session_id, update_fold_metrics,
                              initialize_metrics_dict, setup_model_params, cv_config,displaytNan_vifMiCorrFiltering,
                              load_features_and_sections,apply_scaling)
    import keyboard

    print("le code s'éxécute sur pycharm")

    # Exécution du notebook après l'installation des packages
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is not None:
            print("le code s'éxécute sur collab1")
            ipython.run_line_magic('run', '/content/drive/MyDrive/Colab_Notebooks/xtickReversal/standardFunc.ipynb')
    except Exception as e:
        print(f"Erreur lors de l'exécution du notebook: {str(e)}")

    print("le code s'éxécute sur pycharm")

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

"""
class NonAnchoredWalkForwardCV(BaseCrossValidator):


    def __init__(self, n_splits, r=1.0):
        self.n_splits = n_splits
        self.r = r

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        N = len(X)
        nb_split_tscv = self.n_splits
        r = self.r

        # Calcul du train_size de base
        # Equation : train_size * (1 + nb_split_tscv * r) <= N
        train_size = N // (1 + nb_split_tscv * r)
        val_size = int(train_size * r)

        required_size = train_size + nb_split_tscv * val_size
        leftover = N - required_size
        if leftover < 0:
            raise ValueError("Pas assez de données pour le nombre de splits et le ratio demandé.")

        train_size_first = train_size + leftover

        # Premier split
        current_index = 0
        train_indices = np.arange(current_index, current_index + train_size_first)
        current_index += train_size_first
        val_indices = np.arange(current_index, current_index + val_size)
        current_index += val_size

        yield train_indices, val_indices

        # Splits suivants
        for _ in range(1, nb_split_tscv):
            train_indices = val_indices
            val_indices = np.arange(current_index, current_index + val_size)
            current_index += val_size

            yield train_indices, val_indices
"""
class NonAnchoredWalkForwardCV(BaseCrossValidator):
    """
    Validation croisée Non-Anchored Walk-Forward avec ratio stable.

    - Le ratio r = val_size / train_size est maintenu constant à partir du second fold.
    - Le leftover est ajouté au premier train pour utiliser au mieux toutes les données.
    - Chaque split est déterminé à l'avance en fonction du ratio, puis on avance dans les données.
    """

    def __init__(self, n_splits, r=1.0):
        self.n_splits = n_splits
        self.r = r

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        N = len(X)
        nb_split_tscv = self.n_splits
        r = self.r

        # Calcul du train_size maximum
        # On veut (train_size + val_size)*nb_split_tscv <= N
        # val_size = r * train_size => (1 + r)*train_size * nb_split_tscv <= N
        # train_size <= N / (nb_split_tscv*(1+r))
        train_size = int(N // (nb_split_tscv * (1 + r)))
        val_size = int(train_size * r)

        required_size = nb_split_tscv * (train_size + val_size)
        leftover = N - required_size

        # Ajout du leftover au premier train
        train_size_first = train_size + leftover

        # Fold 1
        current_index = 0
        train_indices = np.arange(current_index, current_index + train_size_first)
        current_index += train_size_first
        val_indices = np.arange(current_index, current_index + val_size)
        current_index += val_size

        yield train_indices, val_indices

        # Folds suivants
        for i in range(1, nb_split_tscv):
            train_indices = np.arange(current_index, current_index + train_size)
            current_index += train_size
            val_indices = np.arange(current_index, current_index + val_size)
            current_index += val_size

            yield train_indices, val_indices
def convert_metrics_to_numpy(metrics_dict):
    """
    Convertit de manière sûre les métriques GPU en arrays NumPy

    Args:
        metrics_dict: Dictionnaire contenant les métriques CuPy
    Returns:
        Dict avec les mêmes métriques en NumPy
    """
    numpy_metrics = {}
    for key, value in metrics_dict.items():
        if isinstance(value, cp.ndarray):
            numpy_metrics[key] = cp.asnumpy(value)
        else:
            numpy_metrics[key] = value
    return numpy_metrics


def calculate_time_difference(start_date_str, end_date_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)
    diff = relativedelta(end_date, start_date)
    return diff


from sklearn.model_selection import BaseCrossValidator
import numpy as np


class CustomSessionTimeSeriesSplit_byID(BaseCrossValidator):
    def __init__(self, session_ids, n_splits=5):
        self.session_ids = np.array(session_ids)
        self.n_splits = n_splits
        # Trier les sessions uniques pour garantir l'ordre chronologique
        self.unique_sessions = np.sort(np.unique(self.session_ids))

    def split(self, X, y=None, groups=None):
        n_sessions = len(self.unique_sessions)
        print("____CustomSessionTimeSeriesSplit_byID____")
        if self.n_splits >= n_sessions:
            raise ValueError(
                f"Le nombre de splits ({self.n_splits}) doit être inférieur au nombre de sessions uniques ({n_sessions}).")

        # Calculer la taille minimale pour chaque fold
        min_sessions_per_fold = n_sessions // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # Calculer les indices pour ce fold
            train_end = min_sessions_per_fold * (fold + 1)
            val_start = train_end
            val_end = val_start + min_sessions_per_fold

            # Sélectionner les sessions
            train_sessions = self.unique_sessions[:train_end]
            val_sessions = self.unique_sessions[val_start:val_end]

            # Obtenir les indices correspondants
            train_mask = np.isin(self.session_ids, train_sessions)
            val_mask = np.isin(self.session_ids, val_sessions)

            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            """
            # Vérifications et logs
            print(f"\nFold {fold + 1}:")
            print(f"Sessions train: {len(train_sessions)} ({train_sessions[0]} à {train_sessions[-1]})")
            print(f"Sessions val: {len(val_sessions)} ({val_sessions[0]} à {val_sessions[-1]})")
            print(f"Indices train: {len(train_indices)}")
            print(f"Indices val: {len(val_indices)}")
            """
            yield train_indices, val_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def setup_cv_method(df_init=None,X_train=None,y_train_label=None,cv_method=None, nb_split_tscv=None, config=None):
    # D'abord, vérifier si la colonne existe dans df_init
    if 'timeStampOpening' not in df_init.columns:
        raise ValueError("La colonne 'timeStampOpening' est absente de df_init.")

    """Configure la méthode de validation croisée"""
    if cv_method == cv_config.TIME_SERIE_SPLIT:
        return TimeSeriesSplit(n_splits=nb_split_tscv)
    elif cv_method == cv_config.TIME_SERIE_SPLIT_NON_ANCHORED:
        r = config.get('non_acnhored_val_ratio', 1)

        cv=NonAnchoredWalkForwardCV(n_splits=nb_split_tscv, r=r)
        return cv

    elif cv_method == cv_config.TIMESERIES_SPLIT_BY_ID:
        # Ensuite, appliquer la logique en fonction de cv_method
        # On garde une trace des colonnes originales
        X_train_ = X_train.copy()

        original_columns = set(X_train_.columns)

        # Ajout des colonnes nécessaires si elles n'existent pas déjà
        if 'timeStampOpening' not in X_train_.columns:
            X_train_['timeStampOpening'] = df_init.loc[X_train_.index, 'timeStampOpening']

        if 'session_type_index' not in X_train_.columns:
            X_train_['session_type_index'] = df_init.loc[X_train_.index, 'session_type_index']

        # Vérification des valeurs
        columns_to_check = ['timeStampOpening']
        comparison_results = X_train_[columns_to_check].eq(df_init.loc[X_train_.index, columns_to_check])
        discrepancies = comparison_results[~comparison_results.all(axis=1)]

        if not discrepancies.empty:
            print("\nDétails des divergences :")
            print(discrepancies)
            raise ValueError("divergence df_init X_train_")


        # Création d'une colonne temporaire session_type_index_utc
        X_train_['session_type_index_utc'] = timestamp_to_date_utc(X_train_['timeStampOpening']).str[:10] + '_' + X_train_[
            'session_type_index'].astype(str)

        session_ids = X_train_['session_type_index_utc'].values

        # Suppression de la colonne temporaire

        cv = CustomSessionTimeSeriesSplit_byID(session_ids=session_ids, n_splits=nb_split_tscv)


        return cv
    elif cv_method == cv_config.K_FOLD:
        return KFold(n_splits=nb_split_tscv, shuffle=False)
    elif cv_method == cv_config.K_FOLD_SHUFFLE:
        return KFold(n_splits=nb_split_tscv, shuffle=True, random_state=42)
    else:
        raise ValueError(f"Unknown cv_method: {cv_method}")

def select_features_ifRfe(X_train, y_train_label, trial, config, params, weight_param, metric_dict,    use_of_rfe_in_optuna=None):
    """Sélection des features avec RFE"""
    if use_of_rfe_in_optuna == rfe_param.RFE_WITH_OPTUNA:
        n_features_to_select = trial.suggest_int("n_features_to_select", 1, X_train.shape[1])
    elif use_of_rfe_in_optuna == rfe_param.RFE_AUTO:
        n_features_to_select = config.get('min_features_if_RFE_AUTO', 5)

    selected_feature_names = process_RFE_filteringg(params, trial, weight_param, metric_dict,
                                                    selected_columns, n_features_to_select,
                                                    X_train, y_train_label)
    X_train_selected = X_train[selected_feature_names]

    return X_train_selected, selected_feature_names


def objective_optuna(df_init=None, trial=None, study=None, X_train=None, X_test=None, y_train_label=None,
                     X_train_full=None,
                     device=None, modele_param_optuna_range=None, config=None, nb_split_tscv=None,
                     optima_score=None, metric_dict=None, weight_param=None,
                     random_state_seed_=None, is_log_enabled=None, cv_method=cv_config.K_FOLD, selected_columns=None):
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

        params, num_boost_round = setup_model_params(trial, modele_param_optuna_range, random_state_seed_, device)

        metric_dict = setup_metric_dict(trial, weight_param, optima_score,metric_dict)

        cv = setup_cv_method(df_init=df_init,X_train=X_train, y_train_label=y_train_label,cv_method=cv_method,
                             nb_split_tscv=nb_split_tscv,config=config)

        use_of_rfe_in_optuna = config.get('use_of_rfe_in_optuna', rfe_param.NO_RFE)

        if use_of_rfe_in_optuna!=rfe_param.NO_RFE:
            print(f"\n------- RFE activé activé:")
            X_train, selected_feature_names = select_features_ifRfe(
                X_train, y_train_label, trial, config, params, weight_param, metric_dict,use_of_rfe_in_optuna=use_of_rfe_in_optuna
            )
            print(f"              - Features sélectionnées avec rfe: {len(selected_feature_names)}")
        else :
            selected_feature_names=X_train.columns.tolist()

        cv_results = run_cross_validation(
            X_train=X_train, X_train_full=X_train_full,
            y_train_label=y_train_label,
            trial=trial,
            params=params,
            num_boost_round=num_boost_round,
            metric_dict=metric_dict,
            cv=cv,
            nb_split_tscv=nb_split_tscv,
            weight_param=weight_param,
            optima_score=optima_score,
            xgb_metric=xgb_metric,
            is_log_enabled=is_log_enabled,
            framework='xgboost'
        )

        # 8. Sauvegarde de l'état
        state = {
            'iteration': objective_optuna.iteration_counter,
            'trial_number': trial.number,
            'params': params,
            'threshold': metric_dict['threshold'],
            'mean_score': cv_results['mean_val_score'],
            'std_score': cv_results['std_val_score'],
            'metrics': {k: float(v) for k, v in cv_results['metrics'].items()},
            'memory_usage': cp.get_default_memory_pool().used_bytes() / 1024 ** 2
        }
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
        cp.get_default_memory_pool().free_all_blocks()
    # print_notification("fin de la CV")

    if ENV == 'pycharm':
        if keyboard.is_pressed('q'):  # Nécessite le package 'keyboard'
            study.stop()
    else:
        if os.path.exists('stop_optimization.txt'):
            study.stop()
    # Calculs finaux et métriques

    # Conversion des métriques par fold
    winrates_by_fold_cpu = cp.asnumpy(cv_results['winrates_by_fold'])
    nb_trades_by_fold_cpu = cp.asnumpy(cv_results['nb_trades_by_fold'])
    scores_train_by_fold_cpu = cp.asnumpy(cv_results['scores_train_by_fold'])
    tp_train_by_fold_cpu = cp.asnumpy(cv_results['tp_train_by_fold'])
    fp_train_by_fold_cpu = cp.asnumpy(cv_results['fp_train_by_fold'])
    tp_val_by_fold_cpu = cp.asnumpy(cv_results['tp_val_by_fold'])
    fp_val_by_fold_cpu = cp.asnumpy(cv_results['fp_val_by_fold'])
    scores_val_by_fold_cpu = cp.asnumpy(cv_results['scores_val_by_fold'])

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

    def display_metrics(cv_results):
        """
        Affiche toutes les métriques de manière organisée
        """
        # Métriques par fold
        # print("\n=== Métriques par Fold ===")
        winrates_by_fold_cpu = cp.asnumpy(cv_results['winrates_by_fold'])
        nb_trades_by_fold_cpu = cp.asnumpy(cv_results['nb_trades_by_fold'])
        scores_train_by_fold_cpu = cp.asnumpy(cv_results['scores_train_by_fold'])
        tp_train_by_fold_cpu = cp.asnumpy(cv_results['tp_train_by_fold'])
        fp_train_by_fold_cpu = cp.asnumpy(cv_results['fp_train_by_fold'])
        tp_val_by_fold_cpu = cp.asnumpy(cv_results['tp_val_by_fold'])
        fp_val_by_fold_cpu = cp.asnumpy(cv_results['fp_val_by_fold'])
        scores_val_by_fold_cpu = cp.asnumpy(cv_results['scores_val_by_fold'])

        # print("\nWinrates par fold      :", winrates_by_fold_cpu)
        # print("Nombre trades par fold :", nb_trades_by_fold_cpu)
        # print("\nScores Train par fold  :", scores_train_by_fold_cpu)
        # print("TP Train par fold      :", tp_train_by_fold_cpu)
        # print("FP Train par fold      :", fp_train_by_fold_cpu)
        # print("\nTP Val par fold        :", tp_val_by_fold_cpu)
        # print("FP Val par fold        :", fp_val_by_fold_cpu)
        # print("Scores Val par fold    :", scores_val_by_fold_cpu)

        # Totaux validation
        # print("\n=== Totaux Validation ===")
        total_tp_val = float(cv_results['metrics']['total_tp_val'])
        total_fp_val = float(cv_results['metrics']['total_fp_val'])
        total_tn_val = float(cv_results['metrics']['total_tn_val'])
        total_fn_val = float(cv_results['metrics']['total_fn_val'])

        # print(f"Total TP : {total_tp_val}")
        # print(f"Total FP : {total_fp_val}")
        # print(f"Total TN : {total_tn_val}")
        # print(f"Total FN : {total_fn_val}")

        # Totaux entraînement
        # print("\n=== Totaux Entraînement ===")
        total_tp_train = float(cv_results['metrics']['total_tp_train'])
        total_fp_train = float(cv_results['metrics']['total_fp_train'])
        total_tn_train = float(cv_results['metrics']['total_tn_train'])
        total_fn_train = float(cv_results['metrics']['total_fn_train'])

        # print(f"Total TP : {total_tp_train}")
        # print(f"Total FP : {total_fp_train}")
        # print(f"Total TN : {total_tn_train}")
        # print(f"Total FN : {total_fn_train}")

        # Statistiques supplémentaires
        # print("\n=== Statistiques Globales ===")
        # print(f"Taux de succès validation : {(total_tp_val / (total_tp_val + total_fp_val)) * 100:.2f}%" if (
        #                                                                                                               total_tp_val + total_fp_val) > 0 else "N/A")
        # print(f"Taux de succès entraînement : {(total_tp_train / (total_tp_train + total_fp_train)) * 100:.2f}%" if (
        #                                                                                                                       total_tp_train + total_fp_train) > 0 else "N/A")

    # Utilisation :
    # display_metrics(cv_results)
    # Conversion des fold_stats
    fold_stats = {}
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
            'train_class_ratio': float(cp.asnumpy(stats['train_class_ratio'])) if isinstance(stats['train_class_ratio'],
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
            'val_success_rate': float(cp.asnumpy(stats['val_success_rate'])) if isinstance(stats['val_success_rate'],
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
    # print(f"PnL par trade - Train: {train_pnl_perTrades:.2f}")
    # print(f"PnL par trade - Validation: {val_pnl_perTrades:.2f}")
    # print(f"Différence absolue: {abs(val_pnl_perTrades - train_pnl_perTrades):.2f}")

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
        modele_param_optuna_range=None,
        user_input=None,
        weight_param=None,
        CUSTOM_SECTIONS=None
):
    xgb_metric_method = config.get('xgb_metric_method', xgb_metric.XGB_METRIC_CUSTOM_METRIC_PROFITBASED)
    device = config.get('device_', 'cuda')
    n_trials_optimization = config.get('n_trials_optuna', 4)
    nb_split_tscv = config.get('nb_split_tscv_', 10)
    nanvalue_to_newval = config.get('nanvalue_to_newval_', np.nan)
    random_state_seed = config.get('random_state_seed', 30)
    early_stopping_rounds = config.get('early_stopping_rounds', 70)
    cv_method = config.get('cv_method', cv_config.K_FOLD)
    optuna_objective_type_value = config.get('optuna_objective_type ', optuna_doubleMetrics.USE_DIST_TO_IDEAL)
    is_log_enabled = config.get('is_log_enabled', False)
    selected_columns= config.get('selected_columns', None)

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

    print(X_train)
    print(X_train_full)
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

    chosen_scaler = config.get('scaler_choice', scalerChoice.SCALER_ROBUST)
    if (chosen_scaler!=scalerChoice.SCALER_DISABLE):
        # Calculer le nombre de lignes initiales
        initial_count = len(X_train)

        # Garder les indices des lignes à conserver (celles sans inf ou nan)
        mask = ~X_train.replace([np.inf, -np.inf], np.nan).isna().any(axis=1)

        # Appliquer le même masque à X_train et y_train_label pour maintenir l'homogénéité
        X_train = X_train[mask]
        y_train_label = y_train_label[mask]

        # Calculer le nombre de lignes après nettoyage
        final_count = len(X_train)

        # Calculer le nombre de lignes supprimées
        lines_removed = initial_count - final_count

        # Calculer le pourcentage de lignes supprimées
        percentage_removed = (lines_removed / initial_count) * 100

        # Afficher les résultats
        print(f"Nombre initial de trades : {initial_count}")
        print(f"Nombre de trades supprimés après nettoyage des inf et nan : {lines_removed}")
        print(f"Pourcentage de trades supprimés : {percentage_removed:.2f}%")

        save_sacler_dir = os.path.join(results_directory, 'optuna_results')
        # Pour les données d'entraînement
        X_train, X_test, scaler, scaler_params = apply_scaling(
            X_train,
            X_test,
            save_path=save_sacler_dir,chosen_scaler=chosen_scaler
        )
        print("\nSacler actif\n")
    else :
        print("\nPas de sacler actif\n")

    print("X_train:")
    print(X_train)
    print("X_test:")
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
    def objective_wrapper(trial, study, metric_dict):
        # Call your original objective function
        score_adjustedStd_val, pnl_perTrade_diff = objective_optuna(df_init=df_init,
                                                                    trial=trial, study=study, X_train=X_train,
                                                                    X_test=X_test,
                                                                    y_train_label=y_train_label,
                                                                    X_train_full=X_train_full,
                                                                    device=device,
                                                                    modele_param_optuna_range=modele_param_optuna_range,
                                                                    config=config, nb_split_tscv=nb_split_tscv,
                                                                    optima_score=xgb_metric_method,
                                                                    metric_dict=metric_dict, weight_param=weight_param,
                                                                    random_state_seed_=random_state_seed,
                                                                    is_log_enabled=is_log_enabled,
                                                                    cv_method=cv_method,
                                                                    selected_columns=selected_columns
                                                                    )

        if config.get('optuna_objective_type', optuna_doubleMetrics.DISABLE) == optuna_doubleMetrics.DISABLE:
            # Return only the first objective
            return score_adjustedStd_val
        else:
            # Return both objectives
            return score_adjustedStd_val, pnl_perTrade_diff

    metric_dict = {}

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
                """
                Contraintes:
                1. écart train-val <= seuil
                2. nombre minimal de trades par fold >= 20
                """
                ecart_train_val = trial.user_attrs.get('ecart_train_val', float('inf'))
                nb_trades_by_fold_list = trial.user_attrs.get('nb_trades_by_fold', [float('inf')])
                constraint_min_trades_threshold_by_Fold = trial.user_attrs.get('constraint_min_trades_threshold_by_Fold', [float('inf')])  # Seuil minimal de trades par fold
                constraint_ecart_train_val = trial.user_attrs.get('constraint_ecart_train_val', float('inf'))
                winrates_by_fold = trial.user_attrs.get('winrates_by_fold', None)
                constraint_winrates_by_fold = trial.user_attrs.get('constraint_winrates_by_fold', None)

                # Contrainte sur l'écart train-val

                constraint_ecart = max(0, ecart_train_val - constraint_ecart_train_val)

                # Contrainte sur le nombre minimal de trades par fold
                min_trades = min(nb_trades_by_fold_list) if isinstance(nb_trades_by_fold_list, list) else float('inf')
                min_winrate = min(winrates_by_fold) if isinstance(winrates_by_fold, list) else float('inf')

                constraint_min_trades = max(0, constraint_min_trades_threshold_by_Fold - min_trades)
                constraint_winrates_by_fold = max(0, constraint_winrates_by_fold - min_winrate)


                constraints = [
                    constraint_ecart,  # Contrainte 1: écart train-val
                    constraint_min_trades,  # Contrainte 2: nombre minimal de trades
                    constraint_winrates_by_fold,  # Contrainte 3: min min_winrate

                ]

                return constraints

            return constraints_func
        sampler = optuna.samplers.TPESampler(
            seed=42,
            constraints_func=create_constraints_func() if config.get('use_optuna_constraints_func', False) else None
        )
        study_xgb = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
    else:
        # Create a multi-objective study
        study_xgb = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )

    # Créer une fonction wrapper pour le callback qui inclut optuna
    def callback_wrapper(study, trial):
        return callback_optuna(study, trial, optuna, study_xgb, rfe_param, config, modele_param_optuna_range,
                               results_directory)


    # Lancer l'optimisation avec le wrapper

    # Lancer l'optimisation
    study_xgb.optimize(
        lambda trial: objective_wrapper(trial, study_xgb, metric_dict),
        n_trials=n_trials_optimization,
        callbacks=[callback_wrapper],
    )

    bestResult_dict = study_xgb.user_attrs['bestResult_dict']

    # Après l'optimisation
    best_params = bestResult_dict["best_params"]
    selected_feature_names = bestResult_dict["selected_feature_names"]
    rfe_param_value = bestResult_dict["use_of_rfe_in_optuna"]
    print("#################################")
    print("#################################")
    print(
        f"## Optimisation Optuna terminée avec distance euclidienne. Meilleur essai : {bestResult_dict['best_optunaTrial_number']}")
    print(f"## Meilleurs hyperparamètres trouvés: ", best_params)
    #if (rfe_param_value != rfe_param.NO_RFE):
       # feature_names = selected_feature_names
    #     print(
    #       f"## Nb des features lectionnées par RFECE({len(selected_feature_names)}) : {list(selected_feature_names)}")
   # print(f"##       - Rappel avant RFECE nombre de feature: {len(X_train.columns)}")

    optimal_threshold = best_params['threshold']
    print(f"## Seuil utilisé : {optimal_threshold:.4f}")
    print("## Meilleur score Objective 1 (pnl_norm_objective): ", bestResult_dict["pnl_norm_objective"])
    if config.get('optuna_objective_type', optuna_doubleMetrics.DISABLE) != optuna_doubleMetrics.DISABLE:
        print("## Meilleur score Objective 2 (ecart_train_val): ",
              bestResult_dict["ecart_train_val"])
    print("#################################")
    print("#################################\n")

    print_notification('###### FIN: OPTIMISATION BAYESIENNE ##########', color="blue")

    print_notification('###### DEBUT: ENTRAINEMENT MODELE FINAL ##########', color="blue")

    """
    if cv_method == cv_config.TIMESERIES_SPLIT_BY_ID:
        if 'session_type_index' not in selected_columns:
            feature_names.remove('session_type_index')
            # Vérification pour session_type_index
            if 'session_type_index' in X_train.columns and 'session_type_index' in X_test.columns:
                X_train.drop('session_type_index', axis=1, inplace=True)
                X_test.drop('session_type_index', axis=1, inplace=True)

    # Vérifie si les colonnes sont identiques
    if not set(X_train.columns) == set(selected_columns) or not set(X_test.columns) == set(selected_columns):
        raise ValueError(
            f"Les colonnes ne correspondent pas:\nColonnes attendues: {sorted(selected_columns)}\nColonnes X_train: {sorted(X_train.columns)}\nColonnes X_test: {sorted(X_test.columns)}")
    """

    # Réduire X_train à seulement les colonnes sélectionnées
    use_of_rfe_in_optuna = config.get('use_of_rfe_in_optuna', rfe_param.NO_RFE)

    #if (use_of_rfe_in_optuna!=rfe_param.NO_RFE): # retrive best parameter form optuma ussing RFE
    print(selected_feature_names)
    X_train = X_train[selected_feature_names]
    X_test = X_test[selected_feature_names]

    # Créer les DMatrix pour l'entraînement
    sample_weights_train = compute_sample_weight('balanced', y=y_train_label)
    print(X_train)
    dtrain = xgb.DMatrix(X_train, label=y_train_label, weight=sample_weights_train)

    # Créer les DMatrix pour le test
    dtest = xgb.DMatrix(X_test, label=y_test_label)

    train_finalModel_analyse(xgb=xgb,
                             X_train=X_train, X_train_full=X_train_full, X_test=X_test, X_test_full=X_test_full,
                             y_train_label=y_train_label,
                             y_test_label=y_test_label,
                             dtrain=dtrain, dtest=dtest,
                             nb_SessionTest=nb_SessionTest, nan_value=nan_value, feature_names=selected_feature_names,
                             best_params=best_params, config=config, weight_param=weight_param,
                             user_input=user_input)


############### main######################
if __name__ == "__main__":
    # Demander à l'utilisateur s'il souhaite afficher les graphiques
    check_gpu_availability()

    if ENV == 'pycharm':
        FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorized_oldclean.csv"
        FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
        # FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorizedScaledWithNanVal.csv"
        #FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
        FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
        FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
        FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnly220LastFullSession_OnlyShort_feat_winsorized.csv"
        FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
        FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnly900LastFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"

        DIRECTORY_PATH_ = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_5TP_1SL_newBB\merge"

    FILE_PATH_ = os.path.join(DIRECTORY_PATH_, FILE_NAME_)
    directories = DIRECTORY_PATH_.split(os.path.sep)
    target_directory = directories[-2]
    from datetime import datetime

    # Obtenir l'heure et la date actuelles
    now = datetime.now()

    # Formater l'heure et la date au format souhaité
    time_suffix = now.strftime("_%H_%M_%d%m%y")

    # Ajouter le suffixe à target_directory
    target_directory += time_suffix

    # Exemple d'utilisation
    print(f"Le répertoire cible est : {target_directory}")
    # Extraire le répertoire contenant la chaîne "4_0_4TP_1SL"

    # Création du dictionnaire de config
    # Définir le chemin de base selon l'environnement
    if ENV == 'colab':
        base_results_path = r"/content/drive/MyDrive/Colab_Notebooks/xtickReversal/results_optim/"
    else:  # pycharm ou autre
        base_results_path = r"C:/Users/aulac/OneDrive/Documents/Trading/PyCharmProject/MLStrategy/data_preprocessing/results_optim/"

    # Configuration
    config = {
        'target_directory': target_directory,
        'xgb_metric_method': xgb_metric.XGB_METRIC_CUSTOM_METRIC_PROFITBASED,
        'device_': 'cuda',
        'n_trials_optuna': 3000,
        'nb_split_tscv_': 6,
        'test_size_ratio':0.15,
        'nanvalue_to_newval_': np.nan,
        'random_state_seed': 35,
        'early_stopping_rounds': 60,
        #'use_shapeImportance_file': r"C:\Users\aulac\OneDrive\Documents\Trading\PyCharmProject\MLStrategy\data_preprocessing\shap_dependencies_results\shap_values_Training_Set.csv",
        'results_directory': os.path.join(base_results_path, target_directory),
        'cv_method': cv_config.TIME_SERIE_SPLIT,  # cv_config.K_FOLD, #,  TIME_SERIE_SPLIT TIMESERIES_SPLIT_BY_ID TIME_SERIE_SPLIT_NON_ANCHORED
        'non_acnhored_val_ratio':0.5,
        'weightPareto_pnl_val': 0.4,
        'weightPareto_pnl_diff': 0.6,
        'use_of_rfe_in_optuna': rfe_param.NO_RFE,
        'min_features_if_RFE_AUTO': 3,
        'optuna_objective_type': optuna_doubleMetrics.DISABLE, #USE_DIST_TO_IDEAL,
        'use_optuna_constraints_func':True,
        'constraint_min_trades_threshold_by_Fold': 25,
        'constraint_ecart_train_val': 0.25,
        'constraint_winrates_by_fold':0.53,
        'use_imbalance_penalty': False,
        'is_log_enabled': False,
        'enable_vif_corr_mi':False,
        'vif_threshold' : 15,
        'corr_threshold' : 1,
        'mi_threshold': 0.001,
        'scaler_choice': scalerChoice.SCALER_DISABLE,  # ou  ou SCALER_DISABLE SCALER_ROBUST SCALER_STANDARD
        'modele_type': modeleType.XGB
    }

    if (config['use_optuna_constraints_func'] == True and
            config['optuna_objective_type'] != optuna_doubleMetrics.DISABLE):
        raise ValueError(
            "Configuration invalide : Impossible d'utiliser à la fois les contraintes Optuna "
            "et l'optimisation multi-objectif. "
            "Si use_optuna_constraints_func=True, alors optuna_objective_type "
            "doit être optuna_doubleMetrics.DISABLE"
        )

    results_directory = config.get('results_directory', None)
    user_input = ''
    """

    user_input = input(
        f"Pour afficher les graphiques, appuyez sur 'd',\n "
        f"Repertoire d'enregistrrepentt des resultat par défaut:\n {results_directory}\n pour le modifier taper 'r'\n"
        "Sinon, appuyez sur 'Entrée' pour les enregistrer sans les afficher: ")

    print(f"Les résultats seront saugardés dans : {results_directory}")

    if user_input.lower() == 'r':
        new_output_dir = input("Entrez le nouveau répertoire de sortie des résultats : ")
        results_directory = new_output_dir

    if results_directory == None:
        exit(35)
    """

    # Créer le répertoire s'il n'existe pas
    os.makedirs(results_directory, exist_ok=True)

    # Vérifier si le répertoire existe déjà
    """
    if os.path.exists(results_directory):
        overwrite = input(
            f"Le répertoire '{results_directory}'  \n existe déjà. Voulez-vous le supprimer et continuer ? (Appuyez sur Entrée pour continuer, ou tapez une autre touche pour arrêter le programme) ")
        if overwrite == "":
            shutil.rmtree(results_directory)
        else:
            print("Le programme a été arrêté.")
            exit()
    """

    # Définir les paramètres supplémentaires

    weight_param = {
        'threshold': {'min': 0.50, 'max': 0.65},  # total_trades_val = tp + fp
        'w_p': {'min': 0.8, 'max': 2},  # poid pour la class 1 dans objective
        'w_n': {'min': 0.7, 'max': 1.5},  # poid pour la class 0 dans objective
        'profit_per_tp': {'min': 1.25, 'max': 1.25},  # fixe, dépend des profits par trade
        'loss_per_fp': {'min': -1.25, 'max': -1.25},  # fixe, dépend des pertes par trade
        'penalty_per_fn': {'min': 0, 'max': 0},
        'weight_split': {'min': 0.65, 'max': 0.65},
        'nb_split_weight': {'min': 0, 'max': 0},  # si 0, pas d'utilisation de weight_split
        'std_penalty_factor': {'min': 0, 'max': 0}

    }

    # 'profit_ratio_weight': {'min': 0.4, 'max': 0.4},  # profit_ratio = (tp - fp) / total_trades_val
    # 'win_rate_weight': {'min': 0.45, 'max': 0.45},  # win_rate = tp / total_trades_val if total_trades_val
    # 'selectivity_weight': {'min': 0.075, 'max': 0.075},  # selectivity = total_trades_val / total_samples

    modele_param_optuna_range = {
        'num_boost_round': {'min': 600, 'max': 1000},  # Étendre vers le haut

        'max_depth': {'min': 7, 'max': 9},  # Étendre vers le haut

        'learning_rate': {'min': 0.001, 'max': 0.009,  # Resserrer autour de 0.025
                          'log': True},

        'min_child_weight': {'min': 1, 'max': 4},  # Resserrer vers le bas

        'subsample': {'min': 0.45, 'max': 0.75},  # Resserrer autour de 0.57

        'colsample_bytree': {'min': 0.6, 'max': 0.80},  # Resserrer vers le haut

        'colsample_bylevel': {'min': 0.4, 'max': 0.6},  # Étendre vers le bas

        'colsample_bynode': {'min': 0.65, 'max': 0.95},  # Resserrer vers le haut

        'gamma': {'min': 5, 'max': 13},  # Resserrer autour de 4.18

        'reg_alpha': {'min': 1, 'max': 2,  # Resserrer vers le bas
                      'log': True},

        'reg_lambda': {'min': 0.1, 'max': 0.9,  # Resserrer vers le bas
                       'log': True}
    }

    print_notification('###### DEBUT: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")
    file_path = FILE_PATH_
    if ENV == 'pycharm':
        #df_init = load_data(file_path)
        df_init, CUSTOM_SESSIONS = load_features_and_sections(file_path)

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
       # 'bear_imbalance_high_3', ## 4.8 % de NAN
        #'bull_imbalance_high_0', #7.8%
        #'bearish_absorption_ratio' #2.8nan,


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
      # 'naked_poc_dist_above',
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
        'Extrem'
        "bullish"
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
        modele_param_optuna_range=modele_param_optuna_range,
        user_input=user_input,
        weight_param=weight_param,
        CUSTOM_SECTIONS=CUSTOM_SESSIONS
    )

    if results is not None:
        print("entrainement et analyse termisé")
    else:
        print("L'entraînement n'a pas produit de résultats.")



