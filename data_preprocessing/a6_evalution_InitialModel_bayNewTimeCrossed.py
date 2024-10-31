import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from standardFunc import (load_data, split_sessions, print_notification,
                          plot_calibrationCurve_distrib, plot_fp_tp_rates, check_gpu_availability,
                          timestamp_to_date_utc, calculate_and_display_sessions,
                          timestamp_to_date_utc, calculate_and_display_sessions,
                          calculate_weighted_adjusted_score_custom, sigmoidCustom,
                          custom_metric_ProfitBased,create_weighted_logistic_obj,
                          optuna_options,train_finalModel_analyse, init_dataSet)
import optuna
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
import shutil
from sklearn.model_selection import KFold, TimeSeriesSplit
import sys
import json
import tempfile
import shutil
import shap
from colorama import Fore, Style, init
import seaborn as sns

# Define the custom_metric class using Enum
from enum import Enum


class cv_config(Enum):
    TIME_SERIE_SPLIT = 0
    TIME_SERIE_SPLIT_NON_ANCHORED = 1
    K_FOLD = 2
    K_FOLD_SHUFFLE = 3


class optuna_doubleMetrics(Enum):
    USE_DIST_TO_IDEAL = 0
    USE_WEIGHTED_AVG = 1


# Variable globale pour suivre si la fonction a déjà été appelée
_first_call_save_r_trialesults = True


########################################
#########    FUNCTION DEF      #########
########################################


def verify_session_integrity(df, context=""):
    starts = df[df['SessionStartEnd'] == 10].index
    ends = df[df['SessionStartEnd'] == 20].index

    print(f"Vérification pour {context}")
    print(f"Nombre total de débuts de session : {len(starts)}")
    print(f"Nombre total de fins de session : {len(ends)}")

    if len(starts) != len(ends):
        print(f"Erreur : Le nombre de débuts et de fins de session ne correspond pas pour {context}.")
        return False

    for i, (start, end) in enumerate(zip(starts, ends), 1):
        if start >= end:
            print(f"Erreur : Session {i} invalide détectée. Début : {start}, Fin : {end}")
            return False
        elif df.loc[start, 'SessionStartEnd'] != 10 or df.loc[end, 'SessionStartEnd'] != 20:
            print(
                f"Erreur : Session {i} mal formée. Début : {df.loc[start, 'SessionStartEnd']}, Fin : {df.loc[end, 'SessionStartEnd']}")
            return False

    print(f"OK : Toutes les {len(starts)} sessions sont correctement formées pour {context}.")
    return True


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


"""

class CustomCallback(TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        train_score = evals_log['train']['auc'][-1]
        valid_score = evals_log['eval']['auc'][-1]
        if epoch % 10 == 0 and train_score - valid_score > 1:  # on le met à 1 pour annuler ce test. On se base sur l'early stopping désormais
            print(f"Arrêt de l'entraînement à l'itération {epoch}. Écart trop important.")
            return True
        return False
"""


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


"""
# Fonction pour calculer les scores d'entraînement et de validation
def calculate_scores_for_cv_split_learning_curve(
        params, num_boost_round, X_train, y_train_label, X_val,
        y_val, weight_dict, combined_metric, metric_dict, custom_metric):
    sample_weights = np.array([weight_dict[label] for label in y_train_label])
    dtrain = xgb.DMatrix(X_train, label=y_train_label, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)

booster = xgb.train(params, dtrain, num_boost_round=num_boost_round, maximize=True, custom_metric=custom_metric)

    train_pred = booster.predict(dtrain)
    val_pred = booster.predict(dval)

    train_score = combined_metric(y_train_label, train_pred, metric_dict=metric_dict)
    val_score_best = combined_metric(y_val, val_pred, metric_dict=metric_dict)

    return {
        'train_sizes': [len(X_train)],
        'train_scores_mean': [train_score],
        'val_scores_mean': [val_score_best]
    }
"""

from sklearn.model_selection import train_test_split

"""
def print_callback(study, trial, X_train, y_train_label, config):
    trial_values = trial.values  # [score_adjustedStd_val, pnl_perTrade_diff]

    learning_curve_data = trial.user_attrs.get('learning_curve_data')
    best_val_score= trial_values[0]  # Premier objectif (maximize)
    pnl_diff = trial_values[1]   # Deuxième objectif (minimize)
    std_dev_score = trial.user_attrs['std_dev_score']
    total_train_size = len(X_train)

    n_trials_optuna = config.get('n_trials_optuna', 4)

    print(f"\nSur les differents ensembles de validation :")
    print(f"   -Essai terminé Optuna: {trial.number+1}/{n_trials_optuna}")
    print(f"   -Score de validation moyen (score_adjustedStd_val) : {best_val_score:.2f}")
    print(f"   -Écart-type des scores : {std_dev_score:.2f}")
    print(
        f"   -Intervalle de confiance (±1 écart-type) : [{best_val_score - std_dev_score:.2f}, {best_val_score + std_dev_score:.2f}]")
    print(f"   -Différence PnL train-val par sample (pnl_diff): {pnl_diff:.2f}")


    # Récupérer les valeurs de TP et FP
    total_tp_val = trial.user_attrs.get('total_tp_val', 0)
    total_fp_val = trial.user_attrs.get('total_fp_val', 0)
    total_tn_train = trial.user_attrs.get('total_tn_train', 0)
    total_fn_train = trial.user_attrs.get('total_fn_train', 0)
    tp_fp_diff = trial.user_attrs.get('tp_fp_diff', 0)
    cummulative_pnl = trial.user_attrs.get('cummulative_pnl', 0)

    weight_param_FromAttr = trial.user_attrs['weight_param']

    cummulative_pnl_FromAttr = total_tp_val * weight_param_FromAttr['profit_per_tp']['min'] + total_fp_val * weight_param_FromAttr['loss_per_fp'][
        'max']

    print(f"{cummulative_pnl} {cummulative_pnl_FromAttr}")
    exit(23)
    tp_percentage = trial.user_attrs.get('tp_percentage', 0)
    total_trades_val = total_tp_val + total_fp_val
    win_rate = total_tp_val / total_trades_val * 100 if total_trades_val > 0 else 0
    print(f"\nEnsemble de validation (somme de l'ensemble des splits) :")
    print(f"   -Nombre de: TP (True Positives) : {total_tp_val}, FP (False Positives) : {total_fp_val}, "
          f"TN (True Negative) : {total_tn_train}, FN (False Negative) : {total_fn_train},")
    print(f"   -Pourcentage Winrate           : {win_rate:.2f}%")
    print(f"   -Pourcentage de TP             : {tp_percentage:.2f}%")
    print(f"   -Différence (TP - FP)          : {tp_fp_diff}")
    print(f"   -PNL                           : {cummulative_pnl}")
    print(f"   -Nombre de trades              : {total_tp_val+total_fp_val+total_tn_train+total_fn_train}")

    if learning_curve_data:
        train_scores = learning_curve_data['train_scores_mean']
        val_scores = learning_curve_data['val_scores_mean']
        train_sizes = learning_curve_data['train_sizes']

        print("\nCourbe d'apprentissage :")
        for size, train_score, val_score_best in zip(train_sizes, train_scores, val_scores):
            print(f"Taille d'entraînement: {size} ({size / total_train_size * 100:.2f}%)")
            print(f"  Score d'entraînement : {train_score:.4f}")
            print(f"  Score de validation  : {val_score_best:.4f}")
            if val_score_best != 0:
                diff_percentage = ((train_score - val_score_best) / val_score_best) * 100
                print(f"  Différence en % : {diff_percentage:.2f}%")
            print()

        max_train_size_index = np.argmax(train_sizes)
        best_train_size = train_sizes[max_train_size_index]
        best_train_score = train_scores[max_train_size_index]
        corresponding_val_score = val_scores[max_train_size_index]

        print(f"Meilleur score d'entraînement : {best_train_score:.4f}")
        print(f"Score de validation correspondant : {corresponding_val_score:.4f}")
        if corresponding_val_score != 0:
            diff_percentage = ((best_train_score - corresponding_val_score) / corresponding_val_score) * 100
            print(f"Différence en % entre entraînement et validation : {diff_percentage:.2f}%")
        print(
            f"Nombre d'échantillons d'entraînement utilisés : {int(best_train_size)} ({best_train_size / total_train_size * 100:.2f}% du total)")
    #else:
     #   print("Option Courbe d'Apprentissage non activé")

    # Afficher les trials sur le front de Pareto
    print("\nTrials sur le front de Pareto :") #ensemble d'essais dits optimaux qui constituent le front de Pareto
    #Un essai est considéré sur ce front s'il n'est pas "dominé" par un autre essai sur tous les objectifs.

    for trial in study.best_trials:
        pnl_val = trial.values[0]
        pnl_perTrade_diff = trial.values[1]
        trial_number = trial.number + 1
        print(f"Trial numéro {trial_number}:")
        print(f"  pnl sur validation : {pnl_val:.4f}")
        print(f"  Différence pnl per trade : {pnl_perTrade_diff:.4f}\n")

    # Trouver et afficher le trial avec le meilleur pnl sur validation
    best_trial_pnl = max(study.trials, key=lambda t: t.values[0])
    print(
        f"\nMeilleure valeur de pnl sur validation jusqu'à présent : {best_trial_pnl.values[0]:.4f} "
        f"(obtenue lors de l'essai numéro : {best_trial_pnl.number + 1})"
    )

    # Trouver et afficher le trial avec la plus petite différence de pnl per trade
    best_trial_pnl_diff = min(study.trials, key=lambda t: t.values[1])
    print(
        f"Meilleur score de différence de pnl per trade : {best_trial_pnl_diff.values[1]:.4f} "
        f"(obtenu lors de l'essai numéro : {best_trial_pnl_diff.number + 1})"
    )

    print("------")
"""


# Fonctions supplémentaires pour l'analyse des erreurs et SHAP
""""
def calculate_precision_recall_tp_ratio_gpu(y_true, y_pred_threshold, metric_dict):
    y_true_gpu = cp.array(y_true)
    y_pred_gpu = cp.array(y_pred_threshold)

    tp = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 1))
    fp = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 1))
    total_samples = len(y_true_gpu)
    total_trades_val = tp + fp

    # Calcul des métriques
    profit_ratio = (tp - fp) / total_trades_val if total_trades_val > 0 else 0
    win_rate = tp / total_trades_val if total_trades_val > 0 else 0
    selectivity = total_trades_val / total_samples

    # Récupérer les poids et la préférence de sélectivité
    profit_ratio_weight = metric_dict.get('profit_ratio_weight', 0.5)
    win_rate_weight = metric_dict.get('win_rate_weight', 0.3)
    selectivity_weight = metric_dict.get('selectivity_weight', 0.2)

    # Calculer le score combiné
   # if (profit_ratio <= 0 or win_rate < 0.5):
    #    return 0.0  # Retourner directement 0.0 si ces conditions sont remplies

    combined_score = (profit_ratio * profit_ratio_weight +
                      win_rate * win_rate_weight +
                      selectivity * selectivity_weight)

    # Normaliser le score
    sum_of_weights = profit_ratio_weight + win_rate_weight + selectivity_weight
    normalized_score = combined_score / sum_of_weights if sum_of_weights > 0 else 0

    return float(normalized_score)  # Retourner un float pour XGBoost
"""


# Ajoutez cette variable globale au début de votre script
global lastBest_score
lastBest_score = float('-inf')


def select_features_shap(model, X, threshold):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Pour les modèles de classification binaire, shap_values peut être une liste
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Sélectionner les valeurs SHAP pour la classe positive

    shap_importance = np.abs(shap_values).mean(axis=0)

    # Calcul de l'importance cumulative
    sorted_indices = np.argsort(shap_importance)[::-1]
    cumulative_importance = np.cumsum(shap_importance[sorted_indices])

    # Déterminer le nombre de features nécessaires pour atteindre le seuil
    total_importance = shap_importance.sum()
    n_features = np.searchsorted(cumulative_importance, threshold * total_importance) + 1

    # Retourner les indices des features les plus importantes
    return sorted_indices[:n_features]


def timestamp_to_date_utc_(timestamp):
    date_format = "%Y-%m-%d %H:%M:%S"
    if isinstance(timestamp, pd.Series):
        return timestamp.apply(lambda x: time.strftime(date_format, time.gmtime(x)))
    else:
        return time.strftime(date_format, time.gmtime(timestamp))


def get_val_cv_time_range(X_train_full, X_train, X_val_cv):
    # Trouver les index de X_val_cv dans X_train
    val_cv_indices_in_train = X_train.index.get_indexer(X_val_cv.index)

    # Trouver les index correspondants dans X_train_full
    original_indices = X_train.index[val_cv_indices_in_train]

    # Obtenir les temps de début et de fin de X_val_cv dans X_train_full
    start_time = X_train_full.loc[original_indices[0], 'timeStampOpening']
    end_time = X_train_full.loc[original_indices[-1], 'timeStampOpening']

    # Supposons que original_indices contient les indices que nous voulons utiliser
    start_index = original_indices[0]
    end_index = original_indices[-1]

    # Extrayons les données de XTrain
    df_extracted = X_train_full.loc[start_index:end_index]

    num_sessions_XTest, _, _ = calculate_and_display_sessions(df_extracted)

    return start_time, end_time, num_sessions_XTest


from dateutil.relativedelta import relativedelta

from sklearn.model_selection import BaseCrossValidator


class CustomTimeSeriesSplitter(BaseCrossValidator):
    def __init__(self, n_splits, train_window, val_window):
        self.n_splits = n_splits
        self.train_window = train_window
        self.val_window = val_window

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        max_start = n_samples - self.n_splits * self.val_window - self.train_window
        for i in range(self.n_splits):
            train_start = i * self.val_window
            train_end = train_start + self.train_window
            val_start = train_end
            val_end = val_start + self.val_window
            yield (range(train_start, train_end), range(val_start, val_end))

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def calculate_time_difference(start_date_str, end_date_str):
    date_format = "%Y-%m-%d %H:%M:%S"
    start_date = datetime.strptime(start_date_str, date_format)
    end_date = datetime.strptime(end_date_str, date_format)
    diff = relativedelta(end_date, start_date)
    return diff


def report_trial_optuna(trial, best_trial_with_2_obj):
    # Récupération des valeurs depuis trial.user_attrs
    total_tp_val = trial.user_attrs['total_tp_val']
    total_fp_val = trial.user_attrs['total_fp_val']
    total_tn_val = trial.user_attrs['total_tn_val']
    total_fn_val = trial.user_attrs['total_fn_val']
    weight_param = trial.user_attrs['weight_param']
    nb_split_tscv = trial.user_attrs['nb_split_tscv']
    mean_cv_score = trial.user_attrs['mean_cv_score']
    std_dev_score = trial.user_attrs['std_dev_score']
    std_penalty_factor = trial.user_attrs['std_penalty_factor']
    score_adjustedStd_val = trial.user_attrs['score_adjustedStd_val']
    train_pnl_perTrades = trial.user_attrs['train_pnl_perTrades']
    val_pnl_perTrades = trial.user_attrs['val_pnl_perTrades']
    pnl_perTrade_diff = trial.user_attrs['pnl_perTrade_diff']
    total_samples_val = trial.user_attrs['total_samples_val']
    n_trials_optuna = trial.user_attrs['n_trials_optuna']
    total_samples_val = trial.user_attrs['total_samples_val']
    cummulative_pnl_val = trial.user_attrs['cummulative_pnl_val']
    scores_ens_val_list = trial.user_attrs['scores_ens_val_list']
    tp_fp_diff_val = trial.user_attrs['tp_fp_diff_val']
    tp_percentage = trial.user_attrs['tp_percentage']
    win_rate = trial.user_attrs['win_rate']

    trial.set_user_attr('win_rate', win_rate)
    weight_split = trial.user_attrs['weight_split']
    nb_split_weight = trial.user_attrs['nb_split_weight']

    print(f"   ##Essai actuel: ")
    print(
        f"    =>Objective 1, cummulative_pnl sur ens de Val : {cummulative_pnl_val} avec weight_split {weight_split} nb_split_weight {nb_split_weight}")

    print(
        f"     -Moyenne des pnl des {nb_split_tscv} iterations : {mean_cv_score:.2f}, std_dev_score : {std_dev_score}, std_penalty_factor={std_penalty_factor} => score_adjustedStd_val(objective 1) : {score_adjustedStd_val:.2f}")
    print(f"    =>Objective 2, pnl per trade: train {train_pnl_perTrades} // Val {val_pnl_perTrades} "
          f"donc diff val-train PNL per trade {pnl_perTrade_diff}")

    print(f"    =>Principal métrique pour l'essai en cours :")
    print(f"     -Nombre de: TP (True Positives) : {total_tp_val}, FP (False Positives) : {total_fp_val}, "
          f"TN (True Negative) : {total_tn_val}, FN (False Negative) : {total_fn_val},")
    print(f"     -Pourcentage Winrate           : {win_rate:.2f}%")
    print(f"     -Pourcentage de TP             : {tp_percentage:.2f}%")
    print(f"     -Différence (TP - FP)          : {tp_fp_diff_val}")
    print(
        f"     -PNL                           : {cummulative_pnl_val}, orginal: (scores_ens_val_list: {scores_ens_val_list})")
    print(f"     -Nombre de trades              : {total_tp_val + total_fp_val + total_tn_val + total_fn_val}")

    if total_samples_val > 0:
        tp_percentage = (total_tp_val / total_samples_val) * 100
    else:
        tp_percentage = 0
    total_trades_val = total_tp_val + total_fp_val
    win_rate = total_tp_val / total_trades_val * 100 if total_trades_val > 0 else 0
    result_dict_trialOptuna = {
        "cummulative_pnl": cummulative_pnl_val,
        "win_rate_percentage": round(win_rate, 2),
        "scores_ens_val_list": scores_ens_val_list,
        "score_adjustedStd_val": score_adjustedStd_val,
        "std_dev_score": std_dev_score,
        "tp_fp_diff_val": tp_fp_diff_val,
        "total_trades_val": total_trades_val,
        "tp_percentage": round(tp_percentage, 3),
        "total_tp_val": total_tp_val,
        "total_fp_val": total_fp_val,
        "total_tn_val": total_tn_val,
        "total_fn_val": total_fn_val,
        "current_trial_number": trial.number + 1,
        "best_trial_with_2_Obj": best_trial_with_2_obj
    }

    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def convert_to_serializable_config(obj):
        """Convert non-serializable objects to a format suitable for JSON."""
        if isinstance(obj, optuna_options):
            return str(obj)  # or obj.name or obj.value depending on the enum or custom class
        try:
            json.dumps(obj)  # Try to serialize it
            return obj  # If no error, return the object itself
        except (TypeError, ValueError):
            return str(obj)  # If it's not serializable, convert it to string

    def save_trial_results(trial, result_dict_trialOptuna, config=None,
                           xgb_param_optuna_range=None, weight_param=None, selected_columns=None,
                           save_dir="optuna_results",
                           result_file="optuna_results.json"):
        global _first_call_save_r_trialesults
        params = trial.params
        model = trial.user_attrs['model']
        trial_number = trial.number
        # Suppression du contenu du répertoire seulement au premier appel
        if _first_call_save_r_trialesults:
            if os.path.exists(save_dir) and os.listdir(save_dir):
                for filename in os.listdir(save_dir):
                    file_path = os.path.join(save_dir, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')
            _first_call_save_r_trialesults = False  # Marquer que le premier appel est terminé

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        result_file_path = os.path.join(save_dir, result_file)

        # Load existing results or start with an empty dict
        results_data = {}
        if os.path.exists(result_file_path) and os.path.getsize(result_file_path) > 0:
            try:
                with open(result_file_path, 'r') as f:
                    results_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error reading existing results: {e}")
                # Create a backup of the corrupted file
                backup_file = f"{result_file_path}.bak"
                shutil.copy2(result_file_path, backup_file)
                print(f"Backup of corrupted file created: {backup_file}")

        if 'selected_columns' not in results_data:
            results_data['selected_columns'] = selected_columns

        if 'config' not in results_data:
            results_data['config'] = {k: convert_to_serializable_config(v) for k, v in config.items()}

        # Add xgb_param_optuna_range to the results data only if it doesn't exist
        if 'weight_param' not in results_data:
            results_data['weight_param'] = weight_param

        # Add xgb_param_optuna_range to the results data only if it doesn't exist
        if 'xgb_param_optuna_range' not in results_data:
            results_data['xgb_param_optuna_range'] = xgb_param_optuna_range

        # Add new trial results
        results_data[f"trial_{trial_number + 1}"] = {
            "best_result": {k: convert_to_serializable(v) for k, v in result_dict_trialOptuna.items()},
            "params": {k: convert_to_serializable(v) for k, v in params.items()}
        }

        # Write results atomically using a temporary file
        with tempfile.NamedTemporaryFile('w', dir=save_dir, delete=False) as tf:
            json.dump(results_data, tf, indent=4)
            temp_filename = tf.name

        # Rename the temporary file to the actual result file
        os.replace(temp_filename, result_file_path)


    # print(f"   Config: {config}")

    # Appel de la fonction save_trial_results
    save_trial_results(
        trial,
        result_dict_trialOptuna,
        config=config,
        xgb_param_optuna_range=xgb_param_optuna_range, selected_columns=selected_columns, weight_param=weight_param,
        save_dir=os.path.join(results_directory, 'optuna_results'),  # 'optuna_results' should be a string
        result_file="optuna_results.json"
    )
    print(f"   {trial.number + 1}/{n_trials_optuna} Optuna results and model saved successfully.")

    """"
    if learning_curve_enabled and learning_curve_data_list:
        avg_learning_curve_data = average_learning_curves(learning_curve_data_list)
        if avg_learning_curve_data is not None:
            trial.set_user_attr('learning_curve_data', avg_learning_curve_data)
            if trial.number == 0 or score_adjustedStd_val > trial.study.best_value:
                plot_learning_curve(
                    avg_learning_curve_data,
                    title=f"Courbe d'apprentissage moyenne (Meilleur essai {trial.number})",
                    filename=f'learning_curve_best_trial_{trial.number}.png'
                )
    """


def objective_optuna(trial, study, X_train, y_train_label, X_train_full,
                     device,
                     xgb_param_optuna_range, config=None, nb_split_tscv=None,
                     learning_curve_enabled=None,
                     optima_score=None, metric_dict=None, weight_param=None, random_state_seed_=None,
                     early_stopping_rounds=None,
                     cv_method=cv_config.TIME_SERIE_SPLIT):
    np.random.seed(random_state_seed_)

    global lastBest_score
    params = {
        'max_depth': trial.suggest_int('max_depth', xgb_param_optuna_range['max_depth']['min'],
                                       xgb_param_optuna_range['max_depth']['max']),
        'learning_rate': trial.suggest_float('learning_rate', xgb_param_optuna_range['learning_rate']['min'],
                                             xgb_param_optuna_range['learning_rate']['max'],
                                             log=xgb_param_optuna_range['learning_rate'].get('log', False)),
        'min_child_weight': trial.suggest_int('min_child_weight', xgb_param_optuna_range['min_child_weight']['min'],
                                              xgb_param_optuna_range['min_child_weight']['max']),
        'subsample': trial.suggest_float('subsample', xgb_param_optuna_range['subsample']['min'],
                                         xgb_param_optuna_range['subsample']['max']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', xgb_param_optuna_range['colsample_bytree']['min'],
                                                xgb_param_optuna_range['colsample_bytree']['max']),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel',
                                                 xgb_param_optuna_range['colsample_bylevel']['min'],
                                                 xgb_param_optuna_range['colsample_bylevel']['max']),
        'colsample_bynode': trial.suggest_float('colsample_bynode', xgb_param_optuna_range['colsample_bynode']['min'],
                                                xgb_param_optuna_range['colsample_bynode']['max']),
        'gamma': trial.suggest_float('gamma', xgb_param_optuna_range['gamma']['min'],
                                     xgb_param_optuna_range['gamma']['max']),
        'reg_alpha': trial.suggest_float('reg_alpha', xgb_param_optuna_range['reg_alpha']['min'],
                                         xgb_param_optuna_range['reg_alpha']['max'],
                                         log=xgb_param_optuna_range['reg_alpha'].get('log', False)),
        'reg_lambda': trial.suggest_float('reg_lambda', xgb_param_optuna_range['reg_lambda']['min'],
                                          xgb_param_optuna_range['reg_lambda']['max'],
                                          log=xgb_param_optuna_range['reg_lambda'].get('log', False)),
        'random_state': random_state_seed_,
        'tree_method': 'hist',
        'device': device,
    }
    n_trials_optuna = config.get('n_trials_optuna', 4)

    print(f"\n## Optuna {trial.number + 1}/{n_trials_optuna} ##")

    # Initialiser les compteurs
    total_tp_val = total_fp_val = total_tn_val = total_fn_val = total_samples_val = 0
    total_tp_train = total_fp_train = total_tn_train = total_fn_train = total_samples_train = 0

    # Fonction englobante qui intègre metric_dict

    threshold_value = trial.suggest_float('threshold', weight_param['threshold']['min'],
                                          weight_param['threshold']['max'])

    # Sélection de la fonction de métrique appropriée
    if optima_score == optuna_options.USE_OPTIMA_CUSTOM_METRIC_TP_FP:

        # Suggérer les poids pour la métrique combinée
        profit_ratio_weight = trial.suggest_float('profit_ratio_weight', weight_param['profit_ratio_weight']['min'],
                                                  weight_param['profit_ratio_weight']['max'])

        win_rate_weight = trial.suggest_float('win_rate_weight', weight_param['win_rate_weight']['min'],
                                              weight_param['win_rate_weight']['max'])

        selectivity_weight = trial.suggest_float('selectivity_weight', weight_param['selectivity_weight']['min'],
                                                 weight_param['selectivity_weight']['max'])

        # Normaliser les poids pour qu'ils somment à 1
        total_weight = profit_ratio_weight + win_rate_weight + selectivity_weight
        profit_ratio_weight /= total_weight
        win_rate_weight /= total_weight
        selectivity_weight /= total_weight

        # Définir la préférence de sélectivité

        metric_dict = {
            'profit_ratio_weight': profit_ratio_weight,
            'win_rate_weight': win_rate_weight,
            'selectivity_weight': selectivity_weight
        }

    elif optima_score == optuna_options.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED:

        # Définir les paramètres spécifiques pour la métrique basée sur le profit
        metric_dict = {
            'profit_per_tp': trial.suggest_float('profit_per_tp', weight_param['profit_per_tp']['min'],
                                                 weight_param['profit_per_tp']['max']),
            'loss_per_fp': trial.suggest_float('loss_per_fp', weight_param['loss_per_fp']['min'],
                                               weight_param['loss_per_fp']['max']),
            'penalty_per_fn': trial.suggest_float('penalty_per_fn', weight_param['penalty_per_fn']['min'],
                                                  weight_param['penalty_per_fn']['max'])
        }
    metric_dict['threshold'] = threshold_value

    num_boost_round = trial.suggest_int('num_boost_round', xgb_param_optuna_range['num_boost_round']['min'],
                                        xgb_param_optuna_range['num_boost_round']['max'])

    scores_ens_val_list = []
    scores_ens_train_list = []

    last_score = None
    learning_curve_data_list = []

    if nb_split_tscv < 2:
        print("nb_split_tscv < 2")
        exit(1)
    else:
        # Choisir la méthode de validation croisée en fonction du paramètre cv_method
        if cv_method == cv_config.TIME_SERIE_SPLIT:
            cv = TimeSeriesSplit(n_splits=nb_split_tscv)
            typeCV = 'timeSerie'
        elif cv_method == cv_config.TIME_SERIE_SPLIT_NON_ANCHORED:

            x_percent = 50  # La fenêtre d'entraînement est 50 % plus grande que celle de validation
            n_samples = len(X_train)
            n_splits = nb_split_tscv  # Nombre de splits (nb_split_tscv)
            size_per_split = n_samples / (n_splits + 1)  # = 1000 / (5 + 1) = 166.67 (rounded down to 166)

            validation_size = size_per_split / (1 + x_percent / 100)  # = 166 / 1.5 = 110
            train_window = int((1 + x_percent / 100) * validation_size)  # = int(1.5 * 110) = 165

            cv = CustomTimeSeriesSplitter(n_splits=n_splits, train_window=train_window, val_window=validation_size)

        elif cv_method == cv_config.K_FOLD:
            cv = KFold(n_splits=nb_split_tscv, shuffle=False)
            typeCV = 'kfold'
        elif cv_method == cv_config.K_FOLD_SHUFFLE:
            cv = KFold(n_splits=nb_split_tscv, shuffle=True, random_state=42)
            typeCV = 'kfold_shuffle'
        else:
            raise ValueError(f"Unknown cv_method: {cv_method}")

        # Initialisation du timing
        for_loop_start_time = time.time()
        xgboost_train_time_cum = 0
        i = 0

        for train_index, val_index in cv.split(X_train):
            # 1. Optimisation de la préparation des données
            X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_cv, y_val_cv = y_train_label.iloc[train_index], y_train_label.iloc[val_index]
            i += 1

            # 2. Traitement des informations temporelles
            start_time, end_time, val_sessions = get_val_cv_time_range(X_train_full, X_train, X_val_cv)
            start_time_str = timestamp_to_date_utc_(start_time)
            end_time_str = timestamp_to_date_utc_(end_time)
            time_diff = calculate_time_difference(start_time_str, end_time_str)
            print(
                f" ->Split ({typeCV} {i}/{nb_split_tscv}) X_val_cv: de {start_time_str} à {end_time_str}. "
                f"Temps écoulé: {time_diff.months} mois, {time_diff.days} jours, {time_diff.minutes} minutes sur {val_sessions} sessions")
            print(f'   X_train_cv:{len(X_train_cv)} // X_val_cv:{len(X_val_cv)}')

            if len(X_train_cv) == 0 or len(y_train_cv) == 0:
                print("Warning: Empty training set after filtering")
                continue

            # 3. Optimisation de la création des données d'entraînement
            sample_weights = compute_sample_weight('balanced', y=y_train_cv)
            dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv, weight=sample_weights, nthread=-1)
            dval = xgb.DMatrix(X_val_cv, label=y_val_cv, nthread=-1)

            # 4. Configuration des paramètres
            w_p = trial.suggest_float('w_p', weight_param['w_p']['min'], weight_param['w_p']['max'])
            w_n = trial.suggest_float('w_n', weight_param['w_n']['min'], weight_param['w_n']['max'])

            if optima_score == optuna_options.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED:
                custom_metric = lambda predtTrain, dtrain: custom_metric_ProfitBased(predtTrain, dtrain, metric_dict)
                obj_function = create_weighted_logistic_obj(w_p, w_n)
                params['disable_default_eval_metric'] = 1
                params['nthread'] = -1
            else:
                params.update({
                    'objective': 'binary:logistic',
                    'eval_metric': ['aucpr', 'logloss'],
                    'disable_default_eval_metric': 0,
                    'nthread': -1
                })
                obj_function = None
                custom_metric = None

            try:
                # 5. Entraînement avec mesure du temps
                evals_result = {}
                xgboost_start_time = time.time()
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtrain, 'train'), (dval, 'eval')],
                    obj=obj_function,
                    custom_metric=custom_metric,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False,
                    evals_result=evals_result,
                    maximize=True
                )
                xgboost_end_time = time.time()
                xgboost_train_time = xgboost_end_time - xgboost_start_time
                xgboost_train_time_cum += xgboost_train_time

                # 6. Optimisation des calculs de scores
                eval_scores = evals_result['eval']['custom_metric_ProfitBased']
                val_score_best = max(eval_scores)
                val_score_bestIdx = eval_scores.index(val_score_best)
                best_iteration = val_score_bestIdx + 1

                # 7. Optimisation des prédictions
                # Validation
                y_val_pred_proba = model.predict(dval, iteration_range=(0, best_iteration))
                y_val_pred_proba = sigmoidCustom(cp.asarray(y_val_pred_proba))
                y_val_pred_proba = cp.clip(y_val_pred_proba, 0.0, 1.0)
                y_val_pred = (y_val_pred_proba.get() > metric_dict['threshold']).astype(int)

                tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val_cv, y_val_pred).ravel()
                print(f"   *Val, ensemble des scores val_scores: {eval_scores}")
                print(
                    f"      -Meilleur résultat {val_score_best} à l'iteration num_boost {best_iteration} / {num_boost_round}")
                print(f"      -TP (validation): {tp_val}, FP (validation): {fp_val}")

                # Entrainement
                y_train_pred_proba = model.predict(dtrain, iteration_range=(0, best_iteration))
                y_train_pred_proba = sigmoidCustom(cp.asarray(y_train_pred_proba))
                y_train_pred_proba = cp.clip(y_train_pred_proba, 0.0, 1.0)
                y_train_pred = (y_train_pred_proba.get() > metric_dict['threshold']).astype(int)

                tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train_cv, y_train_pred).ravel()
                train_scores = evals_result['train']['custom_metric_ProfitBased']
                train_score_at_val_best = train_scores[val_score_bestIdx]

                print(f"   *Train, ensemble des scores train_scores: {train_scores}")
                print(
                    f"      -Meilleur résultat {train_score_at_val_best} équivalent de l'itération de Val {best_iteration} / {num_boost_round}")
                print(f"      -TP (entraînement): {tp_train}, FP (entraînement): {fp_train}")

                # 8. Mise à jour des scores
                scores_ens_val_list.append(val_score_best)
                scores_ens_train_list.append(train_score_at_val_best)

                # 9. Mise à jour des totaux de manière optimisée
                total_samples_val += len(y_val_cv)
                total_tp_val += tp_val
                total_fp_val += fp_val
                total_tn_val += tn_val
                total_fn_val += fn_val
                total_tp_train += tp_train
                total_fp_train += fp_train
                total_tn_train += tn_train
                total_fn_train += fn_train

            except Exception as e:
                print(f"Error during training or evaluation: {e}")
                exit(4)

        # Calculs finaux et métriques
        for_loop_end_time = time.time()
        total_for_loop_time = for_loop_end_time - for_loop_start_time

        print(
            f"\n <Total time spent in for loop: {total_for_loop_time:.2f} sec // Cummulative time spent in xgb.train {xgboost_train_time_cum:.2f} sec>")

        # Calculs des métriques finales
        total_samples_val = total_tp_val + total_fp_val + total_tn_val + total_fn_val
        total_samples_train = total_tp_train + total_fp_train + total_tn_train + total_fn_train
        total_trades_val = total_tp_val + total_fp_val
        total_trades_train = total_tp_train + total_fp_train

        total_pnl_val = sum(scores_ens_val_list)
        total_pnl_train = sum(scores_ens_train_list)

        val_pnl_perTrades = total_pnl_val / total_trades_val if total_trades_val > 0 else 0
        train_pnl_perTrades = total_pnl_train / total_trades_train if total_trades_val > 0 else 0

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
            scores_ens_val_list,
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

        # Mise à jour des attributs du trial
        trial.set_user_attr('total_tp_val', total_tp_val)
        trial.set_user_attr('total_fp_val', total_fp_val)
        trial.set_user_attr('total_tn_val', total_tn_val)
        trial.set_user_attr('total_fn_val', total_fn_val)
        trial.set_user_attr('weight_param', weight_param)
        trial.set_user_attr('scores_ens_val_list', scores_ens_val_list)
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
        trial.set_user_attr('scores_ens_val_list', scores_ens_val_list)
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
        trial.set_user_attr('scores_ens_val_list', scores_ens_val_list)
        trial.set_user_attr('weight_split', weight_split)
        trial.set_user_attr('nb_split_weight', nb_split_weight)
        trial.set_user_attr('model', model)

        # 5. Retour du score ajusté (à minimiser) et de la différence de PnL par trade
        return score_adjustedStd_val, pnl_perTrade_diff



########################################
#########   END FUNCTION DEF   #########
########################################




def train_and_evaluate_XGBOOST_model(
        df=None,

        config=None,  # Add config parameter here
        xgb_param_optuna_range=None,
        selected_columns=None,
        use_shapeImportance_file=None,
        user_input=None,
        weight_param=None,
):
    optuna_options_method = config.get('optuna_options_method', optuna_options.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED)
    device = config.get('device_', 'cuda')
    n_trials_optimization = config.get('n_trials_optuna', 4)
    nb_split_tscv = config.get('nb_split_tscv_', 10)
    nanvalue_to_newval = config.get('nanvalue_to_newval_', np.nan)
    learning_curve_enabled = config.get('learning_curve_enabled', False)
    random_state_seed = config.get('random_state_seed', 30)
    early_stopping_rounds = config.get('early_stopping_rounds', 60)
    preShapImportance = config.get('preShapImportance', 1)
    use_shapeImportance_file = config.get('use_shapeImportance_file', r'C:\Users\aulac\Downloads')
    cv_method = config.get('cv_method', cv_config.K_FOLD_SHUFFLE)

    X_train_full,X_train,y_train_label,X_test,y_test_label,nb_SessionTrain,nb_SessionTest,nan_value=(
        init_dataSet(df,nanvalue_to_newval,selected_columns))
    print(
        f"\nValeurs NaN : X_train={X_train.isna().sum().sum()}, y_train_label={y_train_label.isna().sum()}, X_test={X_test.isna().sum().sum()}, y_test_label={y_test_label.isna().sum()}\n")


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
        sys.exit(1)
    else:
        print(f"Les classes sont considérées comme équilibrées (différence : {class_difference:.2f})")

    # **Ajout de la réduction des features ici, avant l'optimisation**

    print_notification('###### FIN: CHARGER ET PRÉPARER LES DONNÉES  ##########', color="blue")

    # Début de l'optimisation
    print_notification('###### DÉBUT: OPTIMISATION BAYESIENNE ##########', color="blue")
    start_time = time.time()

    maskShap = np.ones(X_train.shape[1], dtype=bool)
    selected_features = X_train.columns
    print(f"Utilisation de toutes les features ({len(selected_features)}) : {list(selected_features)}")

    X_train = X_train.loc[:, maskShap]
    X_test = X_test.loc[:, maskShap]
    feature_names = X_train.columns.tolist()

    # Assurez-vous que X_test et y_test_label ont le même nombre de lignes
    assert X_test.shape[0] == y_test_label.shape[0], "X_test et y_test_label doivent avoir le même nombre de lignes"

    # Créer les DMatrix pour l'entraînement
    sample_weights_train = compute_sample_weight('balanced', y=y_train_label)
    dtrain = xgb.DMatrix(X_train, label=y_train_label, weight=sample_weights_train)

    # Créer les DMatrix pour le test
    sample_weights_test = compute_sample_weight('balanced', y=y_test_label)
    dtest = xgb.DMatrix(X_test, label=y_test_label, weight=sample_weights_test)

    def objective_wrapper(trial, study, metric_dict):
        score_adjustedStd_val, pnl_perTrade_diff = objective_optuna(
            trial=trial, study=study, X_train=X_train, y_train_label=y_train_label, X_train_full=X_train_full,
            device=device,
            xgb_param_optuna_range=xgb_param_optuna_range, config=config, nb_split_tscv=nb_split_tscv,
            learning_curve_enabled=learning_curve_enabled,
            optima_score=optuna_options_method, metric_dict=metric_dict, weight_param=weight_param,
            random_state_seed_=random_state_seed,
            early_stopping_rounds=early_stopping_rounds, cv_method=cv_method
        )
        return score_adjustedStd_val, pnl_perTrade_diff

    metric_dict = {}

    weightPareto_pnl_val = config.get('weightPareto_pnl_val', 0.7)
    weightPareto_pnl_diff = config.get('weightPareto_pnl_diff', 0.3)

    # Vérifier que la somme des poids est égale à 1
    if abs(weightPareto_pnl_val + weightPareto_pnl_diff - 1.0) > 1e-6:  # Tolérance d'erreur flottante
        raise ValueError("La somme des poids (weightPareto_pnl_val + weightPareto_pnl_diff) doit être égale à 1.0")

    def callback_optuna(study, trial):
        """
        Callback fonction pour Optuna avec calculs optimisés.
        """
        current_trial = trial.number
        print(f"\n {Fore.CYAN}Optuna Callback essai {current_trial + 1}:{Style.RESET_ALL}")

        # Si l'essai n'est pas complet ou s'il n'y a pas d'essais complétés, retourner tôt
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            study_xgb.set_user_attr('bestResult_dict', {
                "best_optunaTrial_number": None,
                "best_pnl_val": None,
                "best_pnl_perTrade_diff": None,
                "best_params": None
            })
            print("Aucun essai complété n'a été trouvé.")
            return

        # Sélection du meilleur essai
        pareto_trials = study.best_trials
        weights = np.array([weightPareto_pnl_val, weightPareto_pnl_diff])

        if optuna_doubleMetrics.USE_DIST_TO_IDEAL:
            # Calcul optimisé des valeurs min/max
            values_0 = np.array([t.values[0] for t in pareto_trials])
            values_1 = np.array([t.values[1] for t in pareto_trials])

            min_pnl, max_pnl = values_0.min(), values_0.max()
            min_pnl_diff, max_pnl_diff = values_1.min(), values_1.max()

            pnl_range = max_pnl - min_pnl or 1
            pnl_diff_range = max_pnl_diff - min_pnl_diff or 1

            # Calcul vectorisé de la distance
            pnl_normalized = (max_pnl - values_0) / pnl_range
            pnl_diff_normalized = (values_1 - min_pnl_diff) / pnl_diff_range

            distances = np.sqrt(
                (weightPareto_pnl_val * pnl_normalized) ** 2 +
                (weightPareto_pnl_diff * pnl_diff_normalized) ** 2
            )
            best_trial = pareto_trials[np.argmin(distances)]
        else:
            # Calcul de la moyenne pondérée vectorisé
            df = pd.DataFrame({
                'PnL_val': [-t.values[0] for t in completed_trials],
                'pnl_perTrade_diff': [t.values[1] for t in completed_trials]
            })

            # Normalisation vectorisée
            for col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                range_val = max_val - min_val or 1
                df[f'{col}_normalized'] = (df[col] - min_val) / range_val

            weighted_avg = df[['PnL_val_normalized', 'pnl_perTrade_diff_normalized']].dot(weights)
            best_trial = completed_trials[weighted_avg.idxmin()]

        # Création du dictionnaire de résultats
        bestResult_dict = {
            "best_optunaTrial_number": best_trial.number + 1,
            "best_pnl_val": best_trial.values[0],
            "best_pnl_perTrade_diff": best_trial.values[1],
            "best_params": best_trial.params
        }

        # Récupération des métriques de l'essai
        metrics = {
            'total_tp_val': best_trial.user_attrs['total_tp_val'],
            'total_fp_val': best_trial.user_attrs['total_fp_val'],
            'total_tn_val': best_trial.user_attrs['total_tn_val'],
            'total_fn_val': best_trial.user_attrs['total_fn_val'],
            'cummulative_pnl_val': best_trial.user_attrs['cummulative_pnl_val'],
            'tp_fp_diff_val': best_trial.user_attrs['tp_fp_diff_val'],
            'tp_percentage': best_trial.user_attrs['tp_percentage'],
            'win_rate': best_trial.user_attrs['win_rate'],
            'scores_ens_val_list': best_trial.user_attrs['scores_ens_val_list'],
            'weight_split': best_trial.user_attrs['weight_split'],
            'nb_split_weight': best_trial.user_attrs['nb_split_weight']
        }

        # Rapport
        report_trial_optuna(trial, best_trial.number + 1)

        print(f"\n   {Fore.BLUE}##Meilleur essai jusqu'à present: {bestResult_dict['best_optunaTrial_number']}, "
              f"Méthode: '{optuna_doubleMetrics.USE_DIST_TO_IDEAL}##{Style.RESET_ALL}")
        print(f"    =>Objective 1: score_adjustedStd_val -> {bestResult_dict['best_pnl_val']:.4f} "
              f"avec weight_split: {metrics['weight_split']} nb_split_weight {metrics['nb_split_weight']}")
        print(
            f"    =>Objective 2: score différence par trade (train - val) -> {bestResult_dict['best_pnl_perTrade_diff']:.4f}")
        print(f"    Principal métrique pour le meilleur essai:")
        print(f"     -Nombre de: TP: {metrics['total_tp_val']}, FP: {metrics['total_fp_val']}, "
              f"TN: {metrics['total_tn_val']}, FN: {metrics['total_fn_val']}")
        print(f"     -Pourcentage Winrate           : {metrics['win_rate']:.2f}%")
        print(f"     -Pourcentage de TP             : {metrics['tp_percentage']:.2f}%")
        print(f"     -Différence (TP - FP)          : {metrics['tp_fp_diff_val']}")
        print(
            f"     -PNL                           : {metrics['cummulative_pnl_val']}, original: {metrics['scores_ens_val_list']}")
        print(
            f"     -Nombre de trades              : {sum([metrics['total_tp_val'], metrics['total_fp_val'], metrics['total_tn_val'], metrics['total_fn_val']])}")
        print(f"    =>Hyperparamètres du meilleur score trouvé à date: {bestResult_dict['best_params']}")

        study_xgb.set_user_attr('bestResult_dict', bestResult_dict)


    ##end callback_optuna

    # Créer l'étude
    study_xgb = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)  # Vous pouvez ajuster n_warmup_steps selon vos besoins

    )

    # Lancer l'optimisation
    study_xgb.optimize(
        lambda trial: objective_wrapper(trial, study_xgb, metric_dict),
        n_trials=n_trials_optimization,
        callbacks=[callback_optuna],
    )
    end_time = time.time()
    execution_time = end_time - start_time

    bestResult_dict = study_xgb.user_attrs['bestResult_dict']

    # Après l'optimisation
    best_params = bestResult_dict["best_params"]
    print(f"\nTemps d'exécution total : {execution_time:.2f} secondes")
    print("#################################")
    print("#################################")
    print(
        f"## Optimisation Optuna terminée avec distance euclidienne. Meilleur essai : {bestResult_dict['best_optunaTrial_number']}")
    print(f"## Meilleurs hyperparamètres trouvés: ", best_params)
    optimal_threshold = best_params['threshold']
    print(f"## Seuil utilisé : {optimal_threshold:.4f}")
    print("## Meilleur score Objective 1 (best_pnl_val): ", bestResult_dict["best_pnl_val"])
    print("## Meilleur score Objective 2 (best_pnl_perTrade_diff): ", bestResult_dict["best_pnl_perTrade_diff"])
    print("#################################")
    print("#################################\n")

    print_notification('###### FIN: OPTIMISATION BAYESIENNE ##########', color="blue")

    print_notification('###### DEBUT: ENTRAINEMENT MODELE FINAL ##########', color="blue")

    train_finalModel_analyse(xgb=xgb, X_train=X_train, X_test=X_test, y_train_label=y_train_label, y_test_label=y_test_label,
                                dtrain=dtrain, dtest=dtest,
                                nb_SessionTest=nb_SessionTest, nan_value=nan_value, feature_names=feature_names,
                                best_params=best_params, config=config,
                                user_input=user_input)

############### main######################
if __name__ == "__main__":
    # Demander à l'utilisateur s'il souhaite afficher les graphiques
    check_gpu_availability()

    FILE_NAME_ = "Step5_4_0_4TP_1SL_080919_091024_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
    # FILE_NAME_ = "Step5_4_0_4TP_1SL_080919_091024_extractOnly220LastFullSession_OnlyShort_feat_winsorized.csv"
    DIRECTORY_PATH_ = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL\merge"
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
    config = {
        'target_directory': target_directory,
        'optuna_options_method': optuna_options.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED,
        'device_': 'cuda',
        'n_trials_optuna': 200,
        'nb_split_tscv_': 8,
        'nanvalue_to_newval_': np.nan,
        'learning_curve_enabled': False,
        'random_state_seed': 30,
        'early_stopping_rounds': 60,
        'use_shapeImportance_file': r"C:\Users\aulac\OneDrive\Documents\Trading\PyCharmProject\MLStrategy\data_preprocessing\shap_dependencies_results\shap_values_Training_Set.csv",
        'results_directory' : \
        (
            r"C:\\Users\\aulac\OneDrive\\Documents\\Trading\\PyCharmProject\\MLStrategy\\data_preprocessing\\results_optim\\"
            f"{target_directory}{os.path.sep}"),
        'preShapImportance': 1,
        'cv_method': cv_config.K_FOLD_SHUFFLE,
        'weightPareto_pnl_val': 0.7,
        'weightPareto_pnl_diff': 0.3
    }
    results_directory = config.get('results_directory', None)

    user_input = input(
        f"Pour afficher les graphiques, appuyez sur 'd',\n "
        f"Repertoire d'enregistrrepentt des resultat par défaut:\n {results_directory}\n pour le modifier taper 'r'\n"
        "Sinon, appuyez sur 'Entrée' pour les enregistrer sans les afficher: ")


    print(f"Les résultats seront saugardés dans : {results_directory}")

    if user_input.lower() == 'r':
        new_output_dir = input("Entrez le nouveau répertoire de sortie des résultats : ")
        results_directory = new_output_dir

    if results_directory==None:
        exit(35)

        # Créer le répertoire s'il n'existe pas
    os.makedirs(results_directory, exist_ok=True)

    # Vérifier si le répertoire existe déjà
    if os.path.exists(results_directory):
        overwrite = input(
            f"Le répertoire '{results_directory}'  \n existe déjà. Voulez-vous le supprimer et continuer ? (Appuyez sur Entrée pour continuer, ou tapez une autre touche pour arrêter le programme) ")
        if overwrite == "":
            shutil.rmtree(results_directory)
        else:
            print("Le programme a été arrêté.")
            exit()

    # Définir les paramètres supplémentaires

    weight_param = {
        'threshold': {'min': 0.52, 'max': 0.72},  # total_trades_val = tp + fp
        'w_p': {'min': 1, 'max': 2.2},  # poid pour la class 1 dans objective
        'w_n': {'min': 1, 'max': 1},  # poid pour la class 0 dans objective
        'profit_per_tp': {'min': 1, 'max': 1},  # fixe, dépend des profits par trade
        'loss_per_fp': {'min': -1.1, 'max': -1.1},  # fixe, dépend des pertes par trade
        'penalty_per_fn': {'min': -0.000, 'max': -0.000},
        'weight_split': {'min': 0.65, 'max': 0.65},
        'nb_split_weight': {'min': 3, 'max': 3},  # si 0, pas d'utilisation de weight_split
        'std_penalty_factor': {'min': 1, 'max': 1.1}  # si 0, pas d'utilisation de weight_split

    }

    # 'profit_ratio_weight': {'min': 0.4, 'max': 0.4},  # profit_ratio = (tp - fp) / total_trades_val
    # 'win_rate_weight': {'min': 0.45, 'max': 0.45},  # win_rate = tp / total_trades_val if total_trades_val
    # 'selectivity_weight': {'min': 0.075, 'max': 0.075},  # selectivity = total_trades_val / total_samples

    xgb_param_optuna_range = {
        'num_boost_round': {'min': 200, 'max': 750},
        'max_depth': {'min': 4, 'max': 8},
        'learning_rate': {'min': 0.008, 'max': 0.2, 'log': True},
        'min_child_weight': {'min': 1, 'max': 5},
        'subsample': {'min': 0.7, 'max': 0.9},
        'colsample_bytree': {'min': 0.55, 'max': 0.85},
        'colsample_bylevel': {'min': 0.6, 'max': 0.85},
        'colsample_bynode': {'min': 0.5, 'max': 0.85},
        'gamma': {'min': 0, 'max': 10},
        'reg_alpha': {'min': 0.1, 'max': 9, 'log': True},
        'reg_lambda': {'min': 0.1, 'max': 9, 'log': True},
    }

    print_notification('###### DEBUT: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")
    file_path = FILE_PATH_
    initial_df = load_data(file_path)
    print_notification('###### FIN: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")

    # Chargement et préparation des données
    df = initial_df.copy()

    print(f"Nb de features avant  exlusion: {len(df.columns)}\n")

    # Définition des colonnes de features et des colonnes exclues
    excluded_columns = [
        'class_binaire', 'date', 'trade_category',
        'SessionStartEnd',
        'timeStampOpening',
        'deltaTimestampOpening',
        'candleDir',
        'deltaTimestampOpeningSection1min',
        'deltaTimestampOpeningSection1index',
        'deltaTimestampOpeningSection5min',
        # 'deltaTimestampOpeningSection5index',
        'deltaTimestampOpeningSection15min',
        'deltaTimestampOpeningSection15index',
        'deltaTimestampOpeningSection30min',
        'deltaTimestampOpeningSection30index',
        'deltaCustomSectionMin',
        'deltaCustomSectionIndex',
        'meanVolx',
        'total_count_abv',
        'total_count_blw',

        'bearish_big_trade_imbalance_extrem',  # nan 90.87
        'bearish_big_trade_ratio2_extrem',  # nan zero  90.88
        'bearish_big_trade_ratio_extrem',  # nan zero 90.87
        # 'extrem_ask_bid_imbalance_bearish_extrem',  # nan  51.64
        # 'extrem_asc_dsc_comparison_bearish_extrem',  # nan 54.89
        # 'bearish_extrem_abs_ratio_extrem',  # nan 54.27
        # 'extrem_asc_dsc_comparison_bearish_extrem',  # nan  54.89
        # 'bearish_extrem_pressure_ratio_extrem',  # nan 51.62
        # 'bearish_continuation_vs_reversal_extrem',  # nan 50.69
        'bearish_repeat_ticks_ratio_extrem',  # nan 50.69
        'bearish_bidBigStand_abs_ratio_abv',  # NaN+Zeros(%)  84.53  (bcp de 0)=> new
        'bearish_askBigStand_abs_ratio_abv',  # NaN+Zeros(%)  84.72  (bcp de 0)=> new
        'bullish_askBigStand_abs_ratio_blw',  # NaN+Zeros(%) 92.46 abev bcp de zerp
        'bullish_bidBigStand_abs_ratio_blw',
        'bearish_bigStand_abs_diff_abv',  # NaN+Zeros(%) 92.46 abev bcp de zerp
        'staked00_high',
        'staked00_low',
        """
            'perct_VA6P',
            'ratio_delta_vol_VA6P',
            'diffPriceClose_VA6PPoc',
            'diffPriceClose_VA6PvaH',
            'diffPriceClose_VA6PvaL',
            'perct_VA11P',
            'ratio_delta_vol_VA11P',
            'diffPriceClose_VA11PPoc',
            'diffPriceClose_VA11PvaH',
            'diffPriceClose_VA11PvaL',
            'perct_VA16P',
            'ratio_delta_vol_VA16P',
            'diffPriceClose_VA16PPoc',
            'diffPriceClose_VA16PvaH',
            'diffPriceClose_VA16PvaL',
            'overlap_ratio_VA_6P_11P',
            'overlap_ratio_VA_6P_16P',
            'overlap_ratio_VA_11P_16P',
            'poc_diff_6P_11P',
            'poc_diff_ratio_6P_11P',
            'poc_diff_6P_16P',
            'poc_diff_ratio_6P_16P',
            'poc_diff_11P_16P',
            'poc_diff_ratio_11P_16P'
           """
    ]
    selected_columns = [col for col in df.columns if col not in excluded_columns
                        and '_special' not in col
                        and '_6Tick' not in col
                        and 'BigHigh' not in col
                        and 'bigHigh' not in col
                        and 'big' not in col]
    print(f"Nb de features après exlusion: {len(selected_columns)}\n")

    # Affichage des informations sur les NaN et zéros dans chaque colonne
    print(f"\nAnalyses détaillée des features selectionnées:")
    print("=" * 100)
    print(f"{'Feature':<50} {'NaN Count':>10} {'NaN%':>8} {'Zeros%':>8} {'NaN+Zeros%':>12}")
    print("-" * 100)

    for column in selected_columns:
        nan_count = df[column].isna().sum()
        nan_percentage = (nan_count / len(df)) * 100

        try:
            zeros_count = (df[column] == 0).sum()
            zeros_percentage = (zeros_count / len(df)) * 100
        except:
            zeros_percentage = 0

        total_percentage = nan_percentage + zeros_percentage

        print(f"{column:<50} {nan_count:>10} {nan_percentage:>8.2f} {zeros_percentage:>8.2f} {total_percentage:>12.2f}")

    print("\nRésumé:")
    print(f"Nombre total de features sélectionnées: {len(selected_columns)}")
    print(f"Nombre total d'échantillons: {len(df)}")

    results = train_and_evaluate_XGBOOST_model(
        df=df,
        config=config,  # Pass the config here
        xgb_param_optuna_range=xgb_param_optuna_range,
        selected_columns=selected_columns,
        user_input=user_input,
        weight_param=weight_param,
    )

    if results is not None:
        print("entrainement et analyse termisé")
    else:
        print("L'entraînement n'a pas produit de résultats.")

"""
                if optima_score == optuna_options.USE_OPTIMA_ROCAUC:
                    val_score_best = roc_auc_score(y_val_cv, y_val_pred_proba_np)
                elif optima_score == optuna_options.USE_OPTIMA_AUCPR:
                    val_score_best = average_precision_score(y_val_cv, y_val_pred_proba_np)
                elif optima_score == optuna_options.USE_OPTIMA_F1:
                    val_score_best = f1_score(y_val_cv, y_val_pred)
                elif optima_score == optuna_options.USE_OPTIMA_PRECISION:
                    val_score_best = precision_score(y_val_cv, y_val_pred)
                elif optima_score == optuna_options.USE_OPTIMA_RECALL:
                    val_score_best = recall_score(y_val_cv, y_val_pred)
                elif optima_score == optuna_options.USE_OPTIMA_MCC:
                    val_score_best = matthews_corrcoef(y_val_cv, y_val_pred)
                elif optima_score == optuna_options.USE_OPTIMA_YOUDEN_J:
                    tn, fp, fn, tp = confusion_matrix(y_val_cv, y_val_pred).ravel()
                    sensitivity = tp / (tp + fn)
                    specificity = tn / (tn + fp)
                    val_score_best = sensitivity + specificity - 1
                elif optima_score == optuna_options.USE_OPTIMA_SHARPE_RATIO:
                    val_score_best = calculate_sharpe_ratio(y_val_cv, y_val_pred, price_changes_val)
                elif optima_score == optuna_options.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED:
                    #val_score_best = optuna_profitBased_score(y_val_cv, y_val_pred_proba_np, metric_dict=metric_dict)
                    val_score_best = max(evals_result['eval']['custom_metric_ProfitBased'])

                elif optima_score == optuna_options.USE_OPTIMA_CUSTOM_METRIC_TP_FP:
                    val_score_best = optuna_TP_FP_score(y_val_cv, y_val_pred_proba_np, metric_dict=metric_dict)
                else:
                    print("Invalid Optuna score")
                    exit(1)
"""