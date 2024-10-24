import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from standardFunc import load_data, split_sessions, print_notification, plot_calibrationCurve_distrib,plot_fp_tp_rates, check_gpu_availability, timestamp_to_date_utc,calculate_and_display_sessions
import torch
import optuna
import time
from sklearn.utils.class_weight import compute_sample_weight
import os
from numba import njit
from xgboost.callback import TrainingCallback
from sklearn.metrics import precision_recall_curve, log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import bisect
from sklearn.metrics import roc_auc_score ,precision_score, recall_score, f1_score, confusion_matrix, roc_curve,average_precision_score,matthews_corrcoef
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.ticker as ticker
from PIL import Image
from enum import Enum
from typing import Tuple
import cupy as cp
import shutil
from sklearn.model_selection import KFold, TimeSeriesSplit

# Define the custom_metric class using Enum
from enum import Enum

class cv_config(Enum):
    TIME_SERIE_SPLIT = 0
    TIME_SERIE_SPLIT_NON_ANCHORED=1
    K_FOLD = 2
    K_FOLD_SHUFFLE = 3

class optima_option(Enum):
    USE_OPTIMA_ROCAUC = 1
    USE_OPTIMA_AUCPR = 2
    USE_OPTIMA_F1 = 4
    USE_OPTIMA_PRECISION = 5
    USE_OPTIMA_RECALL = 6
    USE_OPTIMA_MCC = 7
    USE_OPTIMA_YOUDEN_J = 8
    USE_OPTIMA_SHARPE_RATIO = 9
    USE_OPTIMA_CUSTOM_METRIC_PROFITBASED = 10
    USE_OPTIMA_CUSTOM_METRIC_TP_FP = 11

global bestResult_dict
# Variable globale pour suivre si la fonction a déjà été appelée
_first_call_save_r_trialesults = True
########################################
#########    FUNCTION DEF      #########
########################################


def train_preliminary_model_with_tscv(X_train, y_train_label, preShapImportance,use_shapeImportance_file):
    params = {
        'max_depth': 10,
        'learning_rate': 0.005,
        'min_child_weight': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'tree_method': 'hist',


    }
    metric_dict_prelim = {
            'threshold': 0.5,
            'profit_per_tp': 1,
            'loss_per_fp': -1.1,
        }
    num_boost_round = 450  # Nombre de tours pour l'entraînement du modèle préliminaire

    nb_split_tscv = 4

    if preShapImportance == 1.0:
        # Utiliser toutes les features
        maskShap = np.ones(X_train.shape[1], dtype=bool)
        selected_features = X_train.columns
        print(f"Utilisation de toutes les features ({len(selected_features)}) : {list(selected_features)}")
        return maskShap
    elif use_shapeImportance_file:  # Ceci vérifie si use_shapeImportance_dir n'est pas vide
        def load_shap_df(dataset_name, cumulative_importance_threshold=0.8):
            """
            Charge le DataFrame SHAP à partir du fichier CSV enregistré et extrait les features importantes.

            :param dataset_name: Nom du fichier CSV contenant les données SHAP
            :param cumulative_importance_threshold: Seuil de pourcentage cumulatif pour l'extraction des features (par défaut 0.8 pour 80%)
            :return: Tuple (DataFrame pandas avec toutes les colonnes SHAP, Liste des features importantes)
            """

            # Charger le CSV en utilisant le point-virgule comme séparateur
            shap_df = pd.read_csv(dataset_name, sep=';')

            # S'assurer que le DataFrame est trié par importance décroissante
            shap_df = shap_df.sort_values('importance', ascending=False)

            # Vérifier que toutes les colonnes attendues sont présentes
            expected_columns = ['feature', 'importance', 'cumulative_importance',
                                'importance_percentage', 'cumulative_importance_percentage']
            for col in expected_columns:
                if col not in shap_df.columns:
                    print(f"Attention : la colonne '{col}' est manquante dans le fichier chargé.")

            # Extraire les features importantes basées sur le seuil de pourcentage cumulatif
            important_features = \
            shap_df[shap_df['cumulative_importance_percentage'] <= cumulative_importance_threshold * 100][
                'feature'].tolist()

            print(
                f"Nombre de features extraites pour {cumulative_importance_threshold * 100}% d'importance cumulée : {len(important_features)}")

            return shap_df, important_features

        print(f"Utilisation du fichier SHAP importance: {use_shapeImportance_file}")
        _, shap_important_features_col = load_shap_df(use_shapeImportance_file, preShapImportance)
        print(shap_important_features_col)
        return shap_important_features_col

    # Validation croisée temporelle
    tscv = TimeSeriesSplit(n_splits=nb_split_tscv)
    shap_values_list = []


    for train_index, val_index in tscv.split(X_train):

        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train_label.iloc[train_index], y_train_label.iloc[val_index]

        # Filtrer les valeurs 99 (valeurs "inutilisables")



        if len(X_train_cv) == 0 or len(y_train_cv) == 0:
            print("Warning: Empty training set after filtering")
            exit(1)

        # Recalculer les poids des échantillons pour l'ensemble d'entraînement du pli actuel
        sample_weights = compute_sample_weight('balanced', y=y_train_cv)

        # Créer les DMatrix pour XGBoost
        dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv, weight=sample_weights)
        dval = xgb.DMatrix(X_val_cv, label=y_val_cv)



        # Entraîner le modèle préliminaire
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=200,
            verbose_eval=False,
            maximize=True,
        )

        # Obtenir les valeurs SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_cv)
        shap_values_list.append(shap_values)



    # Calculer l'importance moyenne des features sur tous les splits
    shap_values_mean = np.mean([np.abs(shap_values).mean(axis=0) for shap_values in shap_values_list], axis=0)

    # Trier les features par importance et calculer l'importance cumulative
    sorted_indices = np.argsort(shap_values_mean)[::-1]
    shap_values_sorted = shap_values_mean[sorted_indices]
    cumulative_importance = np.cumsum(shap_values_sorted) / np.sum(shap_values_sorted)

    # Sélectionner les features qui représentent le pourcentage cumulé spécifié
    Nshap_dynamic = np.argmax(cumulative_importance >= preShapImportance) + 1

    # Créer un masque pour sélectionner uniquement ces features
    top_features_indices = sorted_indices[:Nshap_dynamic]
    maskShap = np.zeros(X_train.shape[1], dtype=bool)
    maskShap[top_features_indices] = True

    # Afficher les noms des features sélectionnées
    selected_features = X_train.columns[top_features_indices]
    print(
        f"Features sélectionnées (top {Nshap_dynamic} avec SHAP représentant {cumulative_importance[Nshap_dynamic - 1] * 100:.2f}% de l'importance cumulative) : {list(selected_features)}"
    )

    return maskShap


# You already have CuPy imported as cp, so we will use it.
def analyze_predictions_by_range(X_test, y_pred_proba, shap_values_all, prob_min=0.90, prob_max=1.00,
                                 top_n_features=None,
                                 output_dir=r'C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL_02102024\proba_predictions_analysis'):

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Helper function to convert CuPy arrays to NumPy
    def convert_to_numpy(arr):
        # Check if cp is available and arr is a cp array
        return arr.get() if isinstance(arr, cp.ndarray) else arr

    # Convert y_pred_proba and shap_values_all to NumPy if necessary
    y_pred_proba = convert_to_numpy(y_pred_proba)
    shap_values_all = convert_to_numpy(shap_values_all)

    # 1. Identifier les échantillons dans la plage de probabilités spécifiée
    prob_mask = (y_pred_proba >= prob_min) & (y_pred_proba <= prob_max)
    selected_samples = X_test[prob_mask]
    selected_proba = y_pred_proba[prob_mask]

    # Vérifier s'il y a des échantillons dans la plage spécifiée
    if len(selected_samples) == 0:
        print(f"Aucun échantillon trouvé dans la plage de probabilités {prob_min:.2f} - {prob_max:.2f}")
        return

    print(f"Nombre d'échantillons dans la plage {prob_min:.2f} - {prob_max:.2f}: {len(selected_samples)}")

    # Calculer l'importance des features basée sur les valeurs SHAP
    feature_importance = np.abs(shap_values_all).mean(0)

    # Sélectionner les top features si spécifié
    if top_n_features is not None and top_n_features < len(X_test.columns):
        top_features_indices = np.argsort(feature_importance)[-top_n_features:]
        top_features = X_test.columns[top_features_indices]
        selected_samples = selected_samples[top_features]
        X_test_top = X_test[top_features]
        shap_values_selected = shap_values_all[prob_mask][:, top_features_indices]
    else:
        top_features = X_test.columns
        X_test_top = X_test
        shap_values_selected = shap_values_all[prob_mask]

    # 2. Examiner ces échantillons
    with open(os.path.join(output_dir, 'selected_samples_details.txt'), 'w') as f:
        f.write(f"Échantillons avec probabilités entre {prob_min:.2f} et {prob_max:.2f}:\n")
        for i, (idx, row) in enumerate(selected_samples.iterrows()):
            f.write(f"\nÉchantillon {i + 1} (Probabilité: {selected_proba[i]:.4f}):\n")
            for feature, value in row.items():
                f.write(f"  {feature}: {value}\n")

    # 3. Analyser les statistiques de ces échantillons
    stats = selected_samples.describe()
    stats.to_csv(os.path.join(output_dir, 'selected_samples_statistics.csv'), index=False, sep=';')

    # 4. Comparer avec les statistiques globales
    comparison_list = []
    for feature in top_features:
        global_mean = X_test_top[feature].mean()
        selected_mean = selected_samples[feature].mean()
        diff = selected_mean - global_mean
        comparison_list.append({
            'Feature': feature,
            'Global_Mean': global_mean,
            'Selected_Mean': selected_mean,
            'Difference': diff
        })
    comparison = pd.DataFrame(comparison_list)
    comparison.to_csv(os.path.join(output_dir, 'global_vs_selected_comparison.csv'), index=False, sep=';')

    # 5. Visualiser les distributions
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(X_test_top[feature], kde=True, label='Global')
        sns.histplot(selected_samples[feature], kde=True, label=f'Selected ({prob_min:.2f} - {prob_max:.2f})')
        plt.title(f"Distribution de {feature}")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'distribution_{feature}.png'))
        plt.close()

    # 6. Analyse des corrélations entre les features pour ces échantillons
    correlation_matrix = selected_samples.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f"Matrice de corrélation pour les échantillons avec probabilités entre {prob_min:.2f} et {prob_max:.2f}")
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()

    # 7. Comparer les valeurs SHAP pour ces échantillons
    shap.summary_plot(shap_values_selected, selected_samples, plot_type="bar", show=False)
    plt.title(
        f"Importance des features SHAP pour les échantillons avec probabilités entre {prob_min:.2f} et {prob_max:.2f}")
    plt.savefig(os.path.join(output_dir, 'shap_importance.png'))
    plt.close()

    print(f"Analyse terminée. Les résultats ont été sauvegardés dans le dossier '{output_dir}'.")



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


def calculate_scores_for_cv_split_learning_curve(
        params, num_boost_round, X_train, y_train_label, X_val,
        y_val, weight_dict, combined_metric, metric_dict, custom_metric):
    """
    Calcule les scores d'entraînement et de validation pour un split de validation croisée.
    """
    # Créer des DMatrix pour l'entraînement et la validation
    sample_weights = np.array([weight_dict[label] for label in y_train_label])
    dtrain = xgb.DMatrix(X_train, label=y_train_label, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Entraîner le modèle
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round, maximize=True, custom_metric=custom_metric)

    # Prédire sur les ensembles d'entraînement et de validation
    train_pred = booster.predict(dtrain)
    val_pred = booster.predict(dval)

    # Calculer les scores
    train_score = combined_metric(y_train_label, train_pred, metric_dict=metric_dict)

    val_score_best = combined_metric(y_val, val_pred, metric_dict=metric_dict)

    return {
        'train_sizes': [len(X_train)],  # Ajout de cette ligne
        'train_scores_mean': [train_score],  # Modification ici
        'val_scores_mean': [val_score_best]  # Modification ici
    }


from sklearn.model_selection import train_test_split


def print_callback(study, trial, X_train, y_train_label, config):
    global bestResult_dict
    trial_values = trial.values  # [score_adjustedStd_val, pnl_perTrade_diff]

    learning_curve_data = trial.user_attrs.get('learning_curve_data')
    best_val_score= trial_values[0]  # Premier objectif (maximize)
    pnl_diff = trial_values[1]   # Deuxième objectif (minimize)
    std_dev_score = trial.user_attrs['std_dev_score']
    total_train_size = len(X_train)

    n_trials_optuna = config.get('n_trials_optuna', 4)

    print(f"\nSur les differents ensembles d'entrainement :")
    print(f"\nEssai terminé : {trial.number+1}/{n_trials_optuna}")
    print(f"Score de validation moyen (score_adjustedStd_val) : {best_val_score:.4f}")
    print(f"Écart-type des scores : {std_dev_score:.4f}")
    print(
        f"Intervalle de confiance (±1 écart-type) : [{best_val_score - std_dev_score:.4f}, {best_val_score + std_dev_score:.4f}]")
    print(f"Score du dernier pli : {trial.user_attrs['last_score']:.4f}")
    print(f"Différence PnL par trade: {pnl_diff:.4f}")


    # Récupérer les valeurs de TP et FP
    total_tp_val = trial.user_attrs.get('total_tp_val', 0)
    total_fp_val = trial.user_attrs.get('total_fp_val', 0)
    total_tn_train = trial.user_attrs.get('total_tn_train', 0)
    total_fn_train = trial.user_attrs.get('total_fn_train', 0)
    tp_fp_diff = trial.user_attrs.get('tp_fp_diff', 0)
    cummulative_pnl = trial.user_attrs.get('cummulative_pnl', 0)

    tp_percentage = trial.user_attrs.get('tp_percentage', 0)
    total_trades_val = total_tp_val + total_fp_val
    win_rate = total_tp_val / total_trades_val * 100 if total_trades_val > 0 else 0
    print(f"\nEnsemble de validation (somme de l'ensemble des splits :")
    print(f"Nombre de: TP (True Positives) : {total_tp_val}, FP (False Positives) : {total_fp_val}, "
          f"TN (True Negative) : {total_tn_train}, FN (False Negative) : {total_fn_train},")
    print(f"Pourcentage Winrate           : {win_rate:.2f}%")
    print(f"Pourcentage de TP             : {tp_percentage:.2f}%")
    print(f"Différence (TP - FP)          : {tp_fp_diff}")
    print(f"PNL                           : {cummulative_pnl}")
    print(f"Nombre de trades              : {total_tp_val+total_fp_val+total_tn_train+total_fn_train}")
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
    else:
        print("Option Courbe d'Apprentissage non activé")

    # Afficher les trials sur le front de Pareto
    print("\nTrials sur le front de Pareto :")
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


# Fonctions supplémentaires pour l'analyse des erreurs et SHAP

# 1. Fonction pour analyser les erreurs

def analyze_errors(X_test, y_test_label, y_pred_threshold, y_pred_proba, feature_names, save_dir=None, top_features=None):
    """
    Analyse les erreurs de prédiction du modèle et génère des visualisations.

    Parameters:
    -----------
    X_test : pd.DataFrame
        Les features de l'ensemble de test.
    y_test_label : array-like
        Les vraies étiquettes de l'ensemble de test.
    y_pred_threshold : array-like
        Les prédictions du modèle après application du seuil.
    y_pred_proba : array-like
        Les probabilités prédites par le modèle.
    feature_names : list
        Liste des noms des features.
    save_dir : str, optional
        Le répertoire où sauvegarder les résultats de l'analyse (par défaut './analyse_error/').
    top_features : list, optional
        Liste des 10 features les plus importantes (si disponible).

    Returns:
    --------
    tuple
        (results_df, error_df) : DataFrames contenant respectivement tous les résultats et les cas d'erreur.
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    # Vérifier et convertir les arrays CuPy en NumPy
    def convert_to_numpy(arr):
        return arr.get() if isinstance(arr, cp.ndarray) else arr

    # Convertir y_test_label, y_pred_threshold, y_pred_proba en NumPy si nécessaire
    y_test_label = convert_to_numpy(y_test_label)
    y_pred_threshold = convert_to_numpy(y_pred_threshold)
    y_pred_proba = convert_to_numpy(y_pred_proba)

    # Créer un dictionnaire pour stocker toutes les données
    data = {
        'true_label': y_test_label,
        'predicted_label': y_pred_threshold,
        'prediction_probability': y_pred_proba
    }

    # Ajouter les features au dictionnaire, en les convertissant en NumPy si nécessaire
    for feature in feature_names:
        data[feature] = convert_to_numpy(X_test[feature])

    # Créer le DataFrame en une seule fois
    results_df = pd.DataFrame(data)

    # Ajouter les colonnes d'erreur
    results_df['is_error'] = results_df['true_label'] != results_df['predicted_label']
    results_df['error_type'] = np.where(results_df['is_error'],
                                        np.where(results_df['true_label'] == 1, 'False Negative', 'False Positive'),
                                        'Correct')

    # Analyse des erreurs
    error_distribution = results_df['error_type'].value_counts(normalize=True)
    print("Distribution des erreurs:")
    print(error_distribution)

    # Sauvegarder la distribution des erreurs
    error_distribution.to_csv(os.path.join(save_dir, 'error_distribution.csv'), index=False, sep=';')

    # Analyser les features pour les cas d'erreur
    error_df = results_df[results_df['is_error']]

    print("\nMoyenne des features pour les erreurs vs. prédictions correctes:")
    feature_means = results_df.groupby('error_type')[feature_names].mean()
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(feature_means)

    # Sauvegarder les moyennes des features
    feature_means.to_csv(os.path.join(save_dir, 'feature_means_by_error_type.csv'), index=False, sep=';')

    # Visualiser la distribution des probabilités de prédiction pour les erreurs
    plt.figure(figsize=(10, 6))
    sns.histplot(data=error_df, x='prediction_probability', hue='true_label', bins=20)
    plt.title('Distribution des probabilités de prédiction pour les erreurs')
    plt.savefig(os.path.join(save_dir, 'error_probability_distribution.png'))
    plt.close()

    # Identifier les cas les plus confiants mais erronés
    most_confident_errors = error_df.sort_values('prediction_probability', ascending=False).head(5)
    print("\nLes 5 erreurs les plus confiantes:")
    print(most_confident_errors[['true_label', 'predicted_label', 'prediction_probability']])

    # Sauvegarder les erreurs les plus confiantes
    most_confident_errors.to_csv(os.path.join(save_dir, 'most_confident_errors.csv'), index=False, sep=';')

    # Visualisations supplémentaires
    plt.figure(figsize=(12, 10))
    sns.heatmap(error_df[feature_names].corr(), annot=False, cmap='coolwarm')
    plt.title('Corrélations des features pour les erreurs')
    plt.savefig(os.path.join(save_dir, 'error_features_correlation.png'))
    plt.close()

    # Identification des erreurs
    errors = X_test[y_test_label != y_pred_threshold]
    print("Nombre d'erreurs:", len(errors))

    # Créer un subplot pour chaque feature (si top_features est fourni)
    if top_features and len(top_features) >= 10:
        fig, axes = plt.subplots(5, 2, figsize=(20, 25))
        fig.suptitle('Distribution des 10 features les plus importantes par type d\'erreur', fontsize=16)

        for i, feature in enumerate(top_features[:10]):
            row = i // 2
            col = i % 2
            sns.boxplot(x='error_type', y=feature, data=results_df, ax=axes[row, col])
            axes[row, col].set_title(f'{i + 1}. {feature}')
            axes[row, col].set_xlabel('')
            if col == 0:
                axes[row, col].set_ylabel('Valeur')
            else:
                axes[row, col].set_ylabel('')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'top_10_features_distribution_by_error.png'), dpi=300, bbox_inches='tight')
        print("Graphique combiné sauvegardé sous 'top_10_features_distribution_by_error.png'")
        plt.close()

    # Sauvegarder les DataFrames spécifiques demandés
    results_df.to_csv(os.path.join(save_dir, 'model_results_analysis.csv'), index=False, sep=';')
    error_df.to_csv(os.path.join(save_dir, 'model_errors_analysis.csv'), index=False, sep=';')
    print(f"\nLes résultats de l'analyse ont été sauvegardés dans le répertoire : {save_dir}")

    return results_df, error_df


# 2. Fonction pour analyser les erreurs confiantes
def analyze_confident_errors(shap_values, confident_errors, X_test, feature_names, important_features, n=5):

    for idx in confident_errors.index[:n]:
        print(f"-----------------> Analyse de l'erreur à l'index {idx}:")
        print(f"Vrai label: {confident_errors.loc[idx, 'true_label']}")
        print(f"Label prédit: {confident_errors.loc[idx, 'predicted_label']}")
        print(f"Probabilité de prédiction: {confident_errors.loc[idx, 'prediction_probability']:.4f}")

        print("\nValeurs des features importantes:")
        for feature in important_features:
            value = X_test.loc[idx, feature]
            print(f"{feature}: {value:.4f}")

        print("\nTop 5 features influentes (SHAP) pour ce cas:")
        case_feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values[0])
        }).sort_values('importance', ascending=False)

        print(case_feature_importance.head())
        print(f"<----------------- Fin Analyse de l'erreur à l'index {idx}:")


# 3. Fonction pour visualiser les erreurs confiantes
def plot_confident_errors(shap_values, confident_errors, X_test, feature_names, n=5):

    # Vérifier le nombre d'erreurs confiantes disponibles
    num_errors = len(confident_errors)
    if num_errors == 0:
        print("Aucune erreur confiante trouvée.")
        return

    # Ajuster n si nécessaire
    n = min(n, num_errors)

    for i, idx in enumerate(confident_errors.index[:n]):
        plt.figure(figsize=(10, 6))

        # Vérifier si shap_values est une liste (pour les modèles à plusieurs classes)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Prendre les valeurs SHAP pour la classe positive

        shap.summary_plot(shap_values, X_test.loc[idx:idx], plot_type="bar", feature_names=feature_names, show=False)
        plt.title(f"Erreur {i + 1}: Vrai {confident_errors.loc[idx, 'true_label']}, "
                  f"Prédit {confident_errors.loc[idx, 'predicted_label']} "
                  f"(Prob: {confident_errors.loc[idx, 'prediction_probability']:.4f})")
        plt.tight_layout()
        plt.savefig(f'confident_error_shap_{i + 1}.png')

        plt.close()

    if n > 0:
        # Create a summary image combining all individual plots
        images = [Image.open(f'confident_error_shap_{i + 1}.png') for i in range(n)]
        widths, heights = zip(*(i.size for i in images))

        max_width = max(widths)
        total_height = sum(heights)

        new_im = Image.new('RGB', (max_width, total_height))

        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]

        new_im.save(os.path.join(results_directory, 'confident_errors_shap_combined.png'), dpi=(300, 300),
                    bbox_inches='tight')
        # Clean up individual images
        for i in range(n):
            os.remove(f'confident_error_shap_{i + 1}.png')

        print(f"Image combinée des {n} erreurs confiantes sauvegardée sous 'confident_errors_shap_combined.png'")
    else:
        print("Pas assez d'erreurs confiantes pour créer une image combinée.")


# 4. Fonction pour comparer les erreurs vs les prédictions correctes
def compare_errors_vs_correct(confident_errors, correct_predictions, X_test, important_features,results_directory):
    error_data = X_test.loc[confident_errors.index]
    correct_data = X_test.loc[correct_predictions.index]

    comparison_data = []
    for feature in important_features:
        error_mean = error_data[feature].mean()
        correct_mean = correct_data[feature].mean()
        difference = error_mean - correct_mean

        comparison_data.append({
            'Feature': feature,
            'Erreurs Confiantes (moyenne)': error_mean,
            'Prédictions Correctes (moyenne)': correct_mean,
            'Différence': difference
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\nComparaison des features importantes:")
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(comparison_df)

    # Visualisation
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(important_features))

    plt.bar(index, comparison_df['Erreurs Confiantes (moyenne)'], bar_width, label='Erreurs Confiantes')
    plt.bar(index + bar_width, comparison_df['Prédictions Correctes (moyenne)'], bar_width,
            label='Prédictions Correctes')

    plt.xlabel('Features')
    plt.ylabel('Valeur Moyenne')
    plt.title('Comparaison des features importantes: Erreurs vs Prédictions Correctes')
    plt.xticks(index + bar_width / 2, important_features, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, 'compare_errors_vs_correct.png'), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_xgboost_trees(model, feature_names, nan_value, max_trees=None):
    """
    Analyse les arbres XGBoost pour identifier tous les splits et ceux impliquant des NaN.

    :param model: Modèle XGBoost entraîné (Booster ou XGBClassifier/XGBRegressor)
    :param feature_names: Liste des noms des features
    :param nan_value: Valeur utilisée pour remplacer les NaN (peut être np.nan)
    :param max_trees: Nombre maximum d'arbres à analyser (None pour tous les arbres)
    :return: Tuple (DataFrame de tous les splits, DataFrame des splits NaN)
    """
    # Si le modèle est un Booster, l'utiliser directement ; sinon, obtenir le Booster depuis le modèle
    if isinstance(model, xgb.Booster):
        booster = model
    else:
        booster = model.get_booster()

    # Obtenir les arbres sous forme de DataFrame
    trees_df = booster.trees_to_dataframe()

    if max_trees is not None:
        trees_df = trees_df[trees_df['Tree'] < max_trees].copy()

    # Filtrer pour ne garder que les splits (nœuds non-feuilles)
    all_splits = trees_df[trees_df['Feature'] != 'Leaf'].copy()

    # Check if 'Depth' is already present; if not, calculate it
    if pd.isna(nan_value):
        # Lorsque nan_value est np.nan, identifier les splits impliquant des valeurs manquantes
        nan_splits = all_splits[all_splits['Missing'] != all_splits['Yes']].copy()
        print(f"Utilisation de la condition pour np.nan. Nombre de splits NaN trouvés : {len(nan_splits)}")
    else:
        # Lorsque nan_value est une valeur spécifique, identifier les splits sur cette valeur
        all_splits.loc[:, 'Split'] = all_splits['Split'].astype(float)
        nan_splits = all_splits[np.isclose(all_splits['Split'], nan_value, atol=1e-8)].copy()
        print(
            f"Utilisation de la condition pour nan_value={nan_value}. Nombre de splits NaN trouvés : {len(nan_splits)}")
        if len(nan_splits) > 0:
            print("Exemples de valeurs de split considérées comme NaN:")
            print(nan_splits['Split'].head())

    return all_splits, nan_splits


def extract_decision_rules(model, nan_value, importance_threshold=0.01):
    """
    Extrait les règles de décision importantes impliquant la valeur de remplacement des NaN ou les valeurs NaN.
    :param model: Modèle XGBoost entraîné (Booster ou XGBClassifier/XGBRegressor)
    :param nan_value: Valeur utilisée pour remplacer les NaN (peut être np.nan)
    :param importance_threshold: Seuil de gain pour inclure une règle
    :return: Liste des règles de décision importantes
    """
    if isinstance(model, xgb.Booster):
        booster = model
    else:
        booster = model.get_booster()

    trees_df = booster.trees_to_dataframe()

    # Ajouter la colonne Depth en comptant le nombre de tirets dans 'ID'
    if 'Depth' not in trees_df.columns:
        trees_df['Depth'] = trees_df['ID'].apply(lambda x: x.count('-'))

    # Vérifiez que les colonnes attendues sont présentes dans chaque ligne avant de traiter
    expected_columns = ['Tree', 'Depth', 'Feature', 'Gain', 'Split', 'Missing', 'Yes']

    # Filtrer les lignes avec des NaN si c'est nécessaire
    trees_df = trees_df.dropna(subset=expected_columns, how='any')

    if pd.isna(nan_value):
        important_rules = trees_df[
            (trees_df['Missing'] != trees_df['Yes']) & (trees_df['Gain'] > importance_threshold)
            ]
    else:
        important_rules = trees_df[
            (trees_df['Split'] == nan_value) & (trees_df['Gain'] > importance_threshold)
            ]

    rules = []
    for _, row in important_rules.iterrows():
        try:
            # Check if necessary columns exist in the row before constructing the rule
            if all(col in row for col in expected_columns):
                rule = f"Arbre {row['Tree']}, Profondeur {row['Depth']}"
                feature = row['Feature']
                gain = row['Gain']

                if pd.isna(nan_value):
                    missing_direction = row['Missing']
                    rule += f": SI {feature} < {row['Split']} OU {feature} est NaN (valeurs manquantes vont vers le nœud {missing_direction}) ALORS ... (gain: {gain:.4f})"
                else:
                    rule += f": SI {feature} == {nan_value} ALORS ... (gain: {gain:.4f})"

                rules.append(rule)
            else:
                print(f"Colonnes manquantes dans cette ligne : {row.to_dict()}")
                continue

        except IndexError as e:
            # Gérer l'erreur d'index hors limites
            # print(f"Erreur lors de l'extraction des informations de la règle: {e}")
            continue
        except Exception as e:
            # Gérer toute autre erreur inattendue
            print(f"Erreur inattendue: {e}")
            continue

    return rules


def analyze_nan_impact(model, X_train, feature_names, nan_value, shap_values=None,
                       features_per_plot=35, verbose_nan_rule=False,
                       save_dir='./nan_analysis_results/'):

    """
    Analyse l'impact des valeurs NaN ou des valeurs de remplacement des NaN sur le modèle XGBoost.

    Parameters:
    -----------
    model : xgboost.Booster ou xgboost.XGBClassifier/XGBRegressor
        Modèle XGBoost entraîné
    X_train : pandas.DataFrame
        Données d'entrée
    feature_names : list
        Liste des noms des features
    nan_value : float ou np.nan
        Valeur utilisée pour remplacer les NaN
    shap_values : numpy.array, optional
        Valeurs SHAP pré-calculées
    features_per_plot : int, default 35
        Nombre de features à afficher par graphique
    verbose_nan_rule : bool, default False
        Si True, affiche les règles de décision détaillées
    save_dir : str, default './nan_analysis_results/'
        Répertoire où sauvegarder les résultats

    Returns:
    --------
    dict
        Dictionnaire contenant les résultats de l'analyse
    """
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    # 1. Analyser les splits impliquant les valeurs NaN
    all_splits, nan_splits = analyze_xgboost_trees(model, feature_names, nan_value)
    results['total_splits'] = len(all_splits)
    results['nan_splits'] = len(nan_splits)
    results['nan_splits_percentage'] = (len(nan_splits) / len(all_splits)) * 100

    # Stocker les distributions pour une utilisation ultérieure
    all_splits_dist = all_splits['Feature'].value_counts()
    nan_splits_dist = nan_splits['Feature'].value_counts()

    # 2. Visualiser la distribution des splits NaN
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Feature', data=nan_splits, order=nan_splits_dist.index)
    plt.title("Distribution des splits impliquant des valeurs NaN par feature")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'nan_splits_distribution.png'))
    plt.close()

    # 3. Analyser la profondeur des splits NaN
    if 'Depth' not in nan_splits.columns and 'ID' in nan_splits.columns:
        nan_splits['Depth'] = nan_splits['ID'].apply(lambda x: x.count('-'))

    if 'Depth' in nan_splits.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Feature', y='Depth', data=nan_splits)
        plt.title("Profondeur des splits impliquant des valeurs NaN par feature")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'nan_splits_depth.png'))
        plt.close()

    # 4. Extraire les règles de décision importantes
    important_rules = extract_decision_rules(model, nan_value)
    results['important_rules'] = important_rules

    if verbose_nan_rule:
        print("\nRègles de décision importantes impliquant des valeurs NaN :")
        for rule in important_rules:
            print(rule)

    # 5. Analyser l'importance des features avec des valeurs NaN

    # Calculs pour l'analyse SHAP et NaN
    shap_mean = np.abs(shap_values).mean(axis=0)
    nan_counts = X_train.isna().sum() if pd.isna(nan_value) else (X_train == nan_value).sum()
    nan_percentages = (nan_counts / len(X_train)) * 100

    nan_fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_mean,
        'Total_NaN': nan_counts,
        'Percentage_NaN': nan_percentages
    }).sort_values('Importance', ascending=False)

    # Générer les graphiques par lots
    num_features = len(nan_fi_df)
    for i in range((num_features + features_per_plot - 1) // features_per_plot):
        subset_df = nan_fi_df.iloc[i * features_per_plot: (i + 1) * features_per_plot]

        fig, ax1 = plt.subplots(figsize=(14, 8))
        sns.barplot(x='Feature', y='Importance', data=subset_df, ax=ax1, color='skyblue')
        ax1.set_xlabel('Feature', fontsize=12)
        ax1.set_ylabel('Importance (SHAP)', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        sns.lineplot(x='Feature', y='Percentage_NaN', data=subset_df, ax=ax2, color='red', marker='o')
        ax2.set_ylabel('Pourcentage de NaN (%)', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter())

        plt.title(f"Importance des features (SHAP) et pourcentage de NaN (X_Train)\n"
                  f"(Features {i * features_per_plot + 1} à {min((i + 1) * features_per_plot, num_features)})",
                  fontsize=14)

        # Incliner les étiquettes de l'axe x à 45 degrés
        ax1.set_xticks(range(len(subset_df)))
        ax1.set_xticklabels(subset_df['Feature'], rotation=45, ha='right', va='top')

        # Ajuster l'espace pour améliorer la visibilité
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.2)  # Ajustez cette valeur si nécessaire

        plt.savefig(os.path.join(save_dir, f'nan_features_shap_importance_percentage_{i + 1}.png'))
        plt.close()

    # Calcul et visualisation de la corrélation
    correlation = nan_fi_df['Importance'].corr(nan_fi_df['Percentage_NaN'])
    results['shap_nan_correlation'] = correlation

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Percentage_NaN', y='Importance', data=nan_fi_df)
    plt.title('Relation entre le pourcentage de NaN et l\'importance des features (SHAP)')
    plt.xlabel('Pourcentage de NaN (%)')
    plt.ylabel('Importance (SHAP)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'shap_importance_vs_percentage_nan.png'))
    plt.close()

    return results


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

# Fonctions CPU (inchangées)
def weighted_logistic_gradient_cpu(predt: np.ndarray, dtrain: xgb.DMatrix, w_p: float, w_n: float) -> np.ndarray:
    """Calcule le gradient pour la perte logistique pondérée (CPU)."""
    y = dtrain.get_label()
    predt = 1.0 / (1.0 + np.exp(-predt))  # Fonction sigmoïde
    weights = np.where(y == 1, w_p, w_n)
    grad = weights * (predt - y)
    return grad


def weighted_logistic_hessian_cpu(predt: np.ndarray, dtrain: xgb.DMatrix, w_p: float, w_n: float) -> np.ndarray:
    """Calcule le hessien pour la perte logistique pondérée (CPU)."""
    y = dtrain.get_label()
    predt = 1.0 / (1.0 + np.exp(-predt))  # Fonction sigmoïde
    weights = np.where(y == 1, w_p, w_n)
    hess = weights * predt * (1.0 - predt)
    return hess


# Fonctions GPU mises à jour

def sigmoidCustom_simple(x):
    """Custom sigmoid function."""
    return 1 / (1 + cp.exp(-x))

def sigmoidCustom(x):
    """Numerically stable sigmoid function."""
    x = cp.asarray(x)
    return cp.where(
        x >= 0,
        1 / (1 + cp.exp(-x)),
        cp.exp(x) / (1 + cp.exp(x))
    )


def weighted_logistic_gradient_Cupygpu(predt, dtrain, w_p, w_n):
    predt_gpu = cp.asarray(predt)
    y_gpu = cp.asarray(dtrain.get_label())

    predt_sigmoid = sigmoidCustom(predt_gpu)
    grad = predt_sigmoid - y_gpu
    # Appliquer les poids après le calcul initial du gradient
    weights = cp.where(y_gpu == 1, w_p, w_n)
    grad *= weights

    return cp.asnumpy(grad)


def weighted_logistic_hessian_Cupygpu(predt, dtrain, w_p, w_n):
    predt_gpu = cp.asarray(predt)
    y_gpu = cp.asarray(dtrain.get_label())

    predt_sigmoid = sigmoidCustom(predt_gpu)
    hess = predt_sigmoid * (1 - predt_sigmoid)
    # Appliquer les poids après le calcul initial de la hessienne
    weights = cp.where(y_gpu == 1, w_p, w_n)
    hess *= weights

    return cp.asnumpy(hess)
    return cp.asnumpy(hess)

def create_weighted_logistic_obj(w_p: float, w_n: float):
    def weighted_logistic_obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        grad = weighted_logistic_gradient_Cupygpu(predt, dtrain, w_p, w_n)
        hess = weighted_logistic_hessian_Cupygpu(predt, dtrain, w_p, w_n)
        return grad, hess
    return weighted_logistic_obj

# Fonction pour vérifier la disponibilité du GPU




def calculate_profitBased_gpu(y_true, y_pred_threshold, metric_dict):
    y_true_gpu = cp.array(y_true)
    y_pred_gpu = cp.array(y_pred_threshold)
    tp = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 1))
    fp = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 1))
    fn = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 0))
    profit_per_tp = metric_dict.get('profit_per_tp', 1.0)
    loss_per_fp = metric_dict.get('loss_per_fp', -1.1)
    penalty_per_fn = metric_dict.get('penalty_per_fn', -0.1)  # Include FN penalty
    total_profit = (tp * profit_per_tp) + (fp * loss_per_fp) + (fn * penalty_per_fn)
    total_trades_val = tp + fp  # Typically, total executed trades

    # Utiliser une condition pour éviter la division par zéro
    """"
    if total_trades_val > 0:
        normalized_profit = total_profit / total_trades_val
    else:
        normalized_profit = total_profit  # Reflect penalties from FNs when no trades are made



    return float(normalized_profit)  # Assurez-vous que c'est un float Python
    """
    return float(total_profit),tp,fp
"""
def optuna_profitBased_score(y_true, y_pred_proba, metric_dict):

    threshold = metric_dict.get('threshold', 0.7)
    print(metric_dict)
    print(f"--- optuna_profitBased_score avec seuil {threshold} ---")
    y_pred_threshold = (y_pred_proba > threshold).astype(int)
    return calculate_profitBased_gpu(y_true, y_pred_threshold, metric_dict)
"""

import logging
global_predt = None  # Variable globale pour stocker predt


def custom_metric_Profit(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict, normalize: bool = False) -> Tuple[
    str, float]:
    """
    Fonction commune pour calculer les métriques de profit (normalisée ou non)

    Args:
        predt: prédictions brutes
        dtrain: données d'entraînement
        metric_dict: dictionnaire des paramètres de métrique
        normalize: si True, normalise le profit par le nombre de trades
    """
    global global_predt
    global_predt = predt

    y_true = dtrain.get_label()
    CHECK_THRESHOLD = 0.55555555

    threshold = metric_dict.get('threshold', CHECK_THRESHOLD)

    if 'threshold' not in metric_dict:
        logging.warning("Aucun seuil personnalisé n'a été défini. Utilisation du seuil par défaut de 0.55555555.")

    predt = cp.asarray(predt)
    predt = sigmoidCustom(predt)
    predt = cp.clip(predt, 0.0, 1.0)

    mean_pred = cp.mean(predt).item()
    std_pred = cp.std(predt).item()
    min_val = cp.min(predt).item()
    max_val = cp.max(predt).item()

    if min_val < 0 or max_val > 1:
        logging.warning(f"Les prédictions sont hors de l'intervalle [0, 1]: [{min_val:.4f}, {max_val:.4f}]")
        exit(12)

    y_pred_threshold = (predt > threshold).astype(int)

    # Calcul du profit et des TP/FP
    total_profit, tp, fp = calculate_profitBased_gpu(y_true, y_pred_threshold, metric_dict)

    if normalize:
        # Version normalisée
        total_trades_val = tp + fp
        if total_trades_val > 0:
            final_profit = total_profit / total_trades_val
        else:
            final_profit = 0.0
        metric_name = 'custom_metric_ProfitBased_norm'
    else:
        # Version non normalisée
        final_profit = total_profit
        metric_name = 'custom_metric_ProfitBased'

    return metric_name, final_profit


# Création des deux fonctions spécifiques à partir de la fonction commune
def custom_metric_ProfitBased(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict) -> Tuple[str, float]:
    return custom_metric_Profit(predt, dtrain, metric_dict, normalize=False)


def custom_metric_ProfitBased_norm(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict) -> Tuple[str, float]:
    return custom_metric_Profit(predt, dtrain, metric_dict, normalize=True)


def create_custom_metric_wrapper(metric_dict):
    def custom_metric_wrapper(predt, dtrain):
        return custom_metric_ProfitBased(predt, dtrain, metric_dict)
    return custom_metric_wrapper


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
def objective_optuna(trial, study, X_train, y_train_label, X_train_full,
                     device,
                     xgb_param_optuna_range,config=None,nb_split_tscv= None,
                     learning_curve_enabled=None,
                     optima_score=None, metric_dict=None, bestResult_dict=None, weight_param=None, random_state_seed_=None,
                     early_stopping_rounds=None, std_penalty_factor_=None,
                     cv_method=cv_config.TIME_SERIE_SPLIT ):  # Ajouter le paramètre cv_method
    np.random.seed(random_state_seed_)
    global global_predt

    global lastBest_score
    params = {
        'max_depth': trial.suggest_int('max_depth', xgb_param_optuna_range['max_depth']['min'], xgb_param_optuna_range['max_depth']['max']),
        'learning_rate': trial.suggest_float('learning_rate', xgb_param_optuna_range['learning_rate']['min'],
                                             xgb_param_optuna_range['learning_rate']['max'],
                                             log=xgb_param_optuna_range['learning_rate'].get('log', False)),
        'min_child_weight': trial.suggest_int('min_child_weight', xgb_param_optuna_range['min_child_weight']['min'],
                                              xgb_param_optuna_range['min_child_weight']['max']),
        'subsample': trial.suggest_float('subsample', xgb_param_optuna_range['subsample']['min'], xgb_param_optuna_range['subsample']['max']),
        'colsample_bytree': trial.suggest_float('colsample_bytree', xgb_param_optuna_range['colsample_bytree']['min'],
                                                xgb_param_optuna_range['colsample_bytree']['max']),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', xgb_param_optuna_range['colsample_bylevel']['min'],
                                                 xgb_param_optuna_range['colsample_bylevel']['max']),
        'colsample_bynode': trial.suggest_float('colsample_bynode', xgb_param_optuna_range['colsample_bynode']['min'],
                                                xgb_param_optuna_range['colsample_bynode']['max']),
        'gamma': trial.suggest_float('gamma', xgb_param_optuna_range['gamma']['min'], xgb_param_optuna_range['gamma']['max']),
        'reg_alpha': trial.suggest_float('reg_alpha', xgb_param_optuna_range['reg_alpha']['min'], xgb_param_optuna_range['reg_alpha']['max'],
                                         log=xgb_param_optuna_range['reg_alpha'].get('log', False)),
        'reg_lambda': trial.suggest_float('reg_lambda', xgb_param_optuna_range['reg_lambda']['min'], xgb_param_optuna_range['reg_lambda']['max'],
                                          log=xgb_param_optuna_range['reg_lambda'].get('log', False)),
        'random_state': random_state_seed_,
        'tree_method': 'hist',
        'device': device,
    }

    # Initialiser les compteurs
    total_tp_val = total_fp_val =total_tn_val = total_fn_val= total_samples_val = 0
    total_tp_train = total_fp_train = total_tn_train=total_fn_train=total_samples_train = 0


    # Fonction englobante qui intègre metric_dict

    threshold_value = trial.suggest_float('threshold', weight_param['threshold']['min'],
                                          weight_param['threshold']['max'])

    # Sélection de la fonction de métrique appropriée
    if optima_score == optima_option.USE_OPTIMA_CUSTOM_METRIC_TP_FP:

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

    elif optima_score == optima_option.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED:


        # Définir les paramètres spécifiques pour la métrique basée sur le profit
        metric_dict = {
            'profit_per_tp': trial.suggest_float('profit_per_tp', weight_param['profit_per_tp']['min'],
                                                 weight_param['profit_per_tp']['max']),
            'loss_per_fp': trial.suggest_float('loss_per_fp', weight_param['loss_per_fp']['min'],
                                               weight_param['loss_per_fp']['max']),
            'penalty_per_fn': trial.suggest_float('penalty_per_fn', weight_param['penalty_per_fn']['min'],
                                               weight_param['penalty_per_fn']['max'])
        }
    metric_dict['threshold']=threshold_value

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
            size_per_split = n_samples / (n_splits   + 1)  # = 1000 / (5 + 1) = 166.67 (rounded down to 166)

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

        i = 0
        xgboost_train_time_cum = 0
        for_loop_start_time = time.time()

        for train_index, val_index in cv.split(X_train):
            X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_cv, y_val_cv = y_train_label.iloc[train_index], y_train_label.iloc[val_index]
            # Votre code d'entraînement et de validation ici

            #X_train_cv_full, X_val_cv_full = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
            i += 1

            #if cv_method ==cv_config.TIME_SERIE_SPLIT :
            start_time, end_time, val_sessions = get_val_cv_time_range(X_train_full, X_train, X_val_cv)
            start_time_str = timestamp_to_date_utc_(start_time)
            end_time_str = timestamp_to_date_utc_(end_time)
            time_diff = calculate_time_difference(start_time_str, end_time_str)
            n_trials_optuna = config.get('n_trials_optuna', 4)


            print(
                    f"--->Essai {trial.number + 1}/{n_trials_optuna} , split {typeCV} {i}/{nb_split_tscv} X_val_cv: de {start_time_str} à {end_time_str}. "
                    f"Temps écoulé: {time_diff.months} mois, {time_diff.days} jours, {time_diff.minutes} minutes sur {val_sessions} sessions")
            print(f'X_train_cv:{len(X_train_cv)} // X_val_cv:{len(X_val_cv)}')

            #else:
             #   print(f"---> split {i}/{nb_split_tscv}")

            if len(X_train_cv) == 0 or len(y_train_cv) == 0:
                print("Warning: Empty training set after filtering")
                continue

            # Recalculer les poids des échantillons
            sample_weights = compute_sample_weight('balanced', y=y_train_cv)

            # Créer les DMatrix pour XGBoost
            dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv, weight=sample_weights)
            dval = xgb.DMatrix(X_val_cv, label=y_val_cv)

            # Optimiser les poids de l'objectif
            w_p = trial.suggest_float('w_p', weight_param['w_p']['min'],
                                      weight_param['w_p']['max'])
            w_n = 1  # Vous pouvez également l'optimiser si nécessaire

            # Mettre à jour la fonction objective avec les poids optimisés
            if optima_score == optima_option.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED:
                custom_metric = lambda predtTrain, dtrain: custom_metric_ProfitBased(predtTrain, dtrain, metric_dict)


                obj_function = create_weighted_logistic_obj(w_p, w_n)
                params['disable_default_eval_metric'] = 1
            else:
                params['objective'] = 'binary:logistic'
                params['eval_metric'] = ['aucpr', 'logloss']
                obj_function = None
                custom_metrics = None
                params['disable_default_eval_metric'] = 0

            try:
                # Entraîner le modèle
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
                eval_scores = evals_result['eval']['custom_metric_ProfitBased']

                # Trouver le meilleur score de validation et son indice
                val_score_best = max(eval_scores) #equivanlent de model.best_score
                val_score_bestIdx = eval_scores.index(val_score_best)
                best_iteration = val_score_bestIdx + 1 #equivant de model.best_iteration+1

                print(f"XGBoost training time for this fold: {xgboost_train_time:.2f} seconds")

                print("Evaluation Results:", evals_result)
                print(
                    f"Best Results: {val_score_best} à l'iteration num_boost : {best_iteration} / {num_boost_round}")



                # Faire des prédictions sur l'ensemble de validation à l'itération du meilleur score
                y_val_pred_proba = model.predict(dval, iteration_range=(0, best_iteration))
                y_val_pred_proba = cp.asarray(y_val_pred_proba)
                y_val_pred_proba = sigmoidCustom(y_val_pred_proba)
                y_val_pred_proba = cp.clip(y_val_pred_proba, 0.0, 1.0)
                y_val_pred_proba_np = y_val_pred_proba.get()
                y_val_pred = (y_val_pred_proba_np > metric_dict['threshold']).astype(int)

                # Calculer TP et FP pour l'ensemble de validation
                tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val_cv, y_val_pred).ravel()

                print(f"Pour le meilleur score de validation à l'itération {best_iteration}:")
                print(f"TP (validation): {tp_val}, FP (validation): {fp_val}")

                # Faire des prédictions sur l'ensemble d'entraînement à la même itération
                y_train_pred_proba = model.predict(dtrain, iteration_range=(0, best_iteration))
                y_train_pred_proba = cp.asarray(y_train_pred_proba)
                y_train_pred_proba = sigmoidCustom(y_train_pred_proba)
                y_train_pred_proba = cp.clip(y_train_pred_proba, 0.0, 1.0)
                y_train_pred_proba_np = y_train_pred_proba.get()
                y_train_pred = (y_train_pred_proba_np > metric_dict['threshold']).astype(int)

                # Calculer TP et FP pour l'ensemble d'entraînement
                tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train_cv, y_train_pred).ravel()

                print(f"Pour le score d'entraînement correspondant à l'itération {best_iteration}:")
                print(f"TP (entraînement): {tp_train}, FP (entraînement): {fp_train}")

                # Accéder aux scores d'entraînement pour la métrique personnalisée
                train_scores = evals_result['train']['custom_metric_ProfitBased']
                # Obtenir le score d'entraînement à l'indice où le meilleur score de validation a été atteint
                train_score_at_val_best = train_scores[val_score_bestIdx]
                print(f"Score d'entraînement correspondant au meilleur score de validation : {train_score_at_val_best}")

                scores_ens_val_list.append(val_score_best)
                scores_ens_train_list.append(train_score_at_val_best)

                last_score = val_score_best

                if learning_curve_enabled:
                    exit(4)
                    """
                    # Calculer les scores pour ce split CV
                    split_scores = calculate_scores_for_cv_split_learning_curve(
                        params,
                        num_boost_round,
                        X_train_cv, y_train_cv,
                        X_val_cv, y_val_cv,
                        weight_dict, optuna_score, metric_dict,custom_metric
                    )

                    # Ajouter les données pour ce split
                    learning_curve_data_list.append(split_scores)
                    """
                total_samples_val += len(y_val_cv)

                pnl_val = tp_val * weight_param['profit_per_tp']['min'] + fp_val * \
                       weight_param['loss_per_fp']['min']
                pnl_train = tp_train * weight_param['profit_per_tp']['min'] + fp_train * \
                          weight_param['loss_per_fp']['min']

                print(f"----Split croisé {i}/{nb_split_tscv}//  Val: {len(y_val_pred)} trades, PNL : {pnl_val} // Train: {len(y_train_pred)} trades, PNL : {pnl_train}")

                total_tp_val += tp_val
                total_fp_val += fp_val
                total_tn_val += tn_val
                total_fn_val += fn_val
                total_tp_train += tp_train
                total_fp_train += fp_train
            except Exception as e:
                print(f"Error during training or evaluation: {e}")
                exit(4)

    total_samples_val = total_tp_val + total_fp_val
    total_samples_train = total_tp_train + total_fp_train

    total_pnl_val = sum(scores_ens_val_list)
    total_pnl_train = sum(scores_ens_train_list)

    val_pnl_perSample = total_pnl_val / total_samples_val if total_samples_val > 0 else 0
    train_pnl_perSample = total_pnl_train / total_samples_train if total_samples_train > 0 else 0

    # Calculer la différence absolue du PnL par trade
    pnl_perSample_diff = abs(val_pnl_perSample - train_pnl_perSample)
    print(f"---Val, pnl per sample: {val_pnl_perSample} // Train, pnl per sample: {train_pnl_perSample} => diff val-train PNL per sample={pnl_perSample_diff}")

    # Ajustement du score avec l'écart-type
    std_penalty_factor = std_penalty_factor_  # À ajuster selon vos besoins
    for_loop_end_time = time.time()
    total_for_loop_time = for_loop_end_time - for_loop_start_time
    print(f"\nTotal time spent in for loop: {total_for_loop_time:.2f} sec // Cullative time spent in xgb.train {xgboost_train_time_cum:.2f} sec")

    if not scores_ens_val_list:
        return float('-inf'), metric_dict, bestResult_dict  # Retourne les trois valeurs même en cas d'erreur

    mean_cv_score = np.mean(scores_ens_val_list)
    std_dev_score = np.std(scores_ens_val_list, ddof=1)  # ddof=1 pour l'estimation non biaisée

    score_adjustedStd_val = mean_cv_score - std_penalty_factor * std_dev_score
    score_variance = np.var(scores_ens_val_list)
    tp_fp_diff = total_tp_val-total_fp_val
    cummulative_pnl=total_tp_val*weight_param['profit_per_tp']['min'] +total_fp_val*weight_param['loss_per_fp']['max']

    print("cummulative_pnl : ", cummulative_pnl)


    print("scores_ens_val_list:", scores_ens_val_list)
    print(f"Score mean sur les {nb_split_tscv} iterations : {mean_cv_score:.3f} et std_dev_score : {std_dev_score}")
    print(f"-> std_penalty_factor de {std_penalty_factor} donc score_adjustedStd_val : {score_adjustedStd_val:.3f}")

    print(f"std_dev_score: {std_dev_score:.6f}")

    if total_samples_val > 0:
        tp_percentage = (total_tp_val / total_samples_val) * 100
    else:
        tp_percentage = 0
    total_trades_val = total_tp_val + total_fp_val
    win_rate = total_tp_val / total_trades_val * 100 if total_trades_val > 0 else 0
    result_dict = {
        "cummulative_pnl": cummulative_pnl,
        "win_rate_percentage": round(win_rate, 2),
        "scores_ens_val_list": scores_ens_val_list,
        "score_adjustedStd_val": score_adjustedStd_val,
        "std_dev_score":std_dev_score,
        "tp_fp_diff": tp_fp_diff,
        "total_trades_val": total_trades_val,
        "tp_percentage": round(tp_percentage, 3),
        "total_tp_val": total_tp_val,
        "total_fp_val": total_fp_val,
        "total_tn_train": total_tn_train,
        "total_fn_train": total_fn_train,
        "current_trial_number": trial.number+1
    }
    # Ajoutez le meilleur numéro d'essai seulement s'il y a des essais complétés
    if study.trials:
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            result_dict["best_trial_number"] = study.best_trial.number+1
            result_dict["best_score"] = study.best_value
        else:
            result_dict["best_trial_number"] = None
            result_dict["best_score"] = None
    else:
        result_dict["best_trial_number"] = None
        result_dict["best_score"] = None

    if score_adjustedStd_val > lastBest_score:
        lastBest_score = score_adjustedStd_val
        bestResult_dict.update(result_dict)  # Correction de la faute de frappe
        print(f"Nouveau meilleur score trouvé : {score_adjustedStd_val:.6f}")
        print(f"Updated bestResult_dict: {bestResult_dict}")



    trial.set_user_attr('last_score', last_score)
    trial.set_user_attr('score_variance', score_variance)
    trial.set_user_attr('std_dev_score', std_dev_score)

    # Après la boucle de validation croisée
    if total_samples_val > 0:
        tp_percentage = (total_tp_val / total_samples_val) * 100
    else:
        tp_percentage = 0

    # Stocker les valeurs dans trial.user_attrs
    trial.set_user_attr('total_tp_val', total_tp_val)
    trial.set_user_attr('total_fp_val', total_fp_val)
    trial.set_user_attr('total_tn_train', total_tn_train)
    trial.set_user_attr('total_fn_train', total_fn_train)
    trial.set_user_attr('tp_fp_diff', tp_fp_diff)
    trial.set_user_attr('cummulative_pnl', cummulative_pnl)

    trial.set_user_attr('tp_percentage', tp_percentage)

    import json
    import tempfile
    import shutil

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
        if isinstance(obj, optima_option):
            return str(obj)  # or obj.name or obj.value depending on the enum or custom class
        try:
            json.dumps(obj)  # Try to serialize it
            return obj  # If no error, return the object itself
        except (TypeError, ValueError):
            return str(obj)  # If it's not serializable, convert it to string

    def save_trial_results(trial_number, result_dict, params, model, config=None,xgb_param_optuna_range=None,weight_param=None,selected_columns=None, save_dir="optuna_results",
                           result_file="optuna_results.json"):
        global _first_call_save_r_trialesults

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
        results_data[f"trial_{trial_number+1}"] = {
            "best_result": {k: convert_to_serializable(v) for k, v in result_dict.items()},
            "params": {k: convert_to_serializable(v) for k, v in params.items()}
        }

        # Write results atomically using a temporary file
        with tempfile.NamedTemporaryFile('w', dir=save_dir, delete=False) as tf:
            json.dump(results_data, tf, indent=4)
            temp_filename = tf.name

        # Rename the temporary file to the actual result file
        os.replace(temp_filename, result_file_path)

        # Save the XGBoost model
        model_file = os.path.join(save_dir, f"model_trial_{trial_number+1}.json")
        model.save_model(model_file)

        print(f"Trial {trial_number+1} results and model saved successfully.")

    print(config)

    # Appel de la fonction save_trial_results
    save_trial_results(
        trial.number,
        result_dict,
        trial.params,
        model,config=config,
        xgb_param_optuna_range=xgb_param_optuna_range,selected_columns=selected_columns,weight_param=weight_param,
        save_dir=os.path.join(results_directory, 'optuna_results'),  # 'optuna_results' should be a string
        result_file="optuna_results.json"
    )
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

    return score_adjustedStd_val, pnl_perTrade_diff,metric_dict,bestResult_dict


########################################
#########   END FUNCTION DEF   #########
########################################
def prepare_xgboost_params(study, device):
    """Prépare les paramètres XGBoost à partir des résultats de l'étude Optuna."""
    best_params = study.best_params.copy()
    num_boost_round = best_params.pop('num_boost_round', None)
    if num_boost_round is None:
        raise ValueError("num_boost_round n'est pas présent dans best_params")

    xgb_valid_params = [
        'max_depth', 'learning_rate', 'min_child_weight', 'subsample',
        'colsample_bytree', 'colsample_bylevel', 'objective', 'eval_metric',
        'random_state', 'tree_method', 'device'
    ]
    best_params = {key: value for key, value in best_params.items() if key in xgb_valid_params}
    best_params['objective'] = 'binary:logistic'
    best_params['tree_method'] = 'hist'
    best_params['device'] = device

    return best_params, num_boost_round


import shap


def analyze_shap_values(model, X, y, dataset_name, create_dependence_plots=False, max_dependence_plots=3,
                        save_dir='./shap_dependencies_results/'):
    """
    Analyse les valeurs SHAP pour un ensemble de données et génère des visualisations.

    Parameters:
    -----------
    model : object
        Le modèle entraîné pour lequel calculer les valeurs SHAP.
    X : pandas.DataFrame
        Les features de l'ensemble de données.
    y : pandas.Series
        Les labels de l'ensemble de données (non utilisés directement dans la fonction,
        mais inclus pour une cohérence future).
    dataset_name : str
        Le nom de l'ensemble de données, utilisé pour nommer les fichiers de sortie.
    create_dependence_plots : bool, optional (default=False)
        Si True, crée des graphiques de dépendance pour les features les plus importantes.
    max_dependence_plots : int, optional (default=3)
        Le nombre maximum de graphiques de dépendance à créer si create_dependence_plots est True.
    save_dir : str, optional (default='./shap_dependencies_results/')
        Le répertoire où sauvegarder les graphiques générés et le fichier CSV.

    Returns:
    --------
    numpy.ndarray
        Les valeurs SHAP calculées pour l'ensemble de données.

    Side Effects:
    -------------
    - Sauvegarde un graphique résumé de l'importance des features SHAP.
    - Sauvegarde un fichier CSV contenant les valeurs SHAP et les importances cumulées.
    - Si create_dependence_plots est True, sauvegarde des graphiques de dépendance
      pour les features les plus importantes.
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    # Créer un explainer SHAP
    explainer = shap.TreeExplainer(model)

    # Calculer les valeurs SHAP
    shap_values = explainer.shap_values(X)

    # Pour les problèmes de classification binaire, prendre le deuxième élément si nécessaire
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    # Créer le résumé des valeurs SHAP
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {dataset_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'shap_importance_{dataset_name}.png'))
    plt.close()

    # Créer un DataFrame avec les valeurs SHAP
    feature_importance = np.abs(shap_values).mean(0)
    shap_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    })

    # Trier le DataFrame par importance décroissante
    shap_df = shap_df.sort_values('importance', ascending=False)

    # Calculer la somme cumulée de l'importance
    total_importance = shap_df['importance'].sum()
    shap_df['cumulative_importance'] = shap_df['importance'].cumsum() / total_importance

    # S'assurer que la dernière valeur est exactement 1.0
    shap_df.loc[shap_df.index[-1], 'cumulative_importance'] = 1.0

    # Ajouter une colonne pour le pourcentage
    shap_df['importance_percentage'] = shap_df['importance'] / total_importance * 100
    shap_df['cumulative_importance_percentage'] = shap_df['cumulative_importance'] * 100

    # Sauvegarder le DataFrame dans un fichier CSV
    csv_path = os.path.join(save_dir, f'shap_values_{dataset_name}.csv')
    shap_df.to_csv(csv_path, index=False, sep=';')
    print(f"Les valeurs SHAP ont été sauvegardées dans : {csv_path}")

    if create_dependence_plots:
        most_important_features = shap_df['feature'].head(max_dependence_plots)

        for feature in most_important_features:
            shap.dependence_plot(feature, shap_values, X, show=False)
            plt.title(f"SHAP Dependence Plot - {feature} - {dataset_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'shap_dependence_{feature}_{dataset_name}.png'))
            plt.close()

    print(f"Les graphiques SHAP ont été sauvegardés dans le répertoire : {save_dir}")

    return shap_values

def compare_feature_importance(shap_values_train, shap_values_test, X_train, X_test, save_dir='./shap_dependencies_results/', top_n=20):
    """
    Compare l'importance des features entre l'ensemble d'entraînement et de test.

    Parameters:
    -----------
    shap_values_train : numpy.ndarray
        Valeurs SHAP pour l'ensemble d'entraînement
    shap_values_test : numpy.ndarray
        Valeurs SHAP pour l'ensemble de test
    X_train : pandas.DataFrame
        DataFrame contenant les features d'entraînement
    X_test : pandas.DataFrame
        DataFrame contenant les features de test
    save_dir : str, optional
        Répertoire où sauvegarder le graphique (par défaut './shap_dependencies_results/')
    top_n : int, optional
        Nombre de top features à afficher dans le graphique (par défaut 20)

    Returns:
    --------
    pandas.DataFrame
        DataFrame contenant les importances des features et leurs différences
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    # Vérifier que les colonnes sont identiques dans X_train et X_test
    if not all(X_train.columns == X_test.columns):
        raise ValueError("Les colonnes de X_train et X_test doivent être identiques.")

    importance_train = np.abs(shap_values_train).mean(0)
    importance_test = np.abs(shap_values_test).mean(0)

    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance_Train': importance_train,
        'Importance_Test': importance_test
    })

    importance_df['Difference'] = importance_df['Importance_Train'] - importance_df['Importance_Test']
    importance_df = importance_df.sort_values('Difference', key=abs, ascending=False)

    # Sélectionner les top_n features pour la visualisation
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Difference', y='Feature', data=top_features, palette='coolwarm')
    plt.title(f"Top {top_n} Differences in Feature Importance (Train - Test)")
    plt.xlabel("Difference in Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance_difference.png'))
    plt.close()

    print(f"Graphique de différence d'importance des features sauvegardé dans {save_dir}")
    print("\nTop differences in feature importance:")
    print(importance_df.head(10))

    return importance_df


import seaborn as sns


def compare_shap_distributions(shap_values_train, shap_values_test, X_train, X_test, top_n=10,
                               save_dir='./shap_dependencies_results/'):
    """
    Compare les distributions des valeurs SHAP entre l'ensemble d'entraînement et de test.

    Parameters:
    -----------
    shap_values_train : numpy.ndarray
        Valeurs SHAP pour l'ensemble d'entraînement
    shap_values_test : numpy.ndarray
        Valeurs SHAP pour l'ensemble de test
    X_train : pandas.DataFrame
        DataFrame contenant les features d'entraînement
    X_test : pandas.DataFrame
        DataFrame contenant les features de test
    top_n : int, optional
        Nombre de top features à comparer (par défaut 10)
    save_dir : str, optional
        Répertoire où sauvegarder les graphiques (par défaut './shap_dependencies_results/')

    Returns:
    --------
    None
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    # Vérifier que les colonnes sont identiques dans X_train et X_test
    if not all(X_train.columns == X_test.columns):
        raise ValueError("Les colonnes de X_train et X_test doivent être identiques.")

    feature_importance = np.abs(shap_values_train).mean(0)
    top_features = X_train.columns[np.argsort(feature_importance)[-top_n:]]

    for feature in top_features:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(shap_values_train[:, X_train.columns.get_loc(feature)], label='Train', fill=True)
        sns.kdeplot(shap_values_test[:, X_test.columns.get_loc(feature)], label='Test', fill=True)
        plt.title(f"SHAP Value Distribution - {feature}")
        plt.xlabel("SHAP Value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()

        # Sauvegarder le graphique dans le répertoire spécifié
        plt.savefig(os.path.join(save_dir, f'shap_distribution_{feature}.png'))
        plt.close()

    print(f"Les graphiques de distribution SHAP ont été sauvegardés dans {save_dir}")


def compare_mean_shap_values(shap_values_train, shap_values_test, X_train, save_dir='./shap_dependencies_results/'):
    """
    Compare les valeurs SHAP moyennes entre l'ensemble d'entraînement et de test.

    Parameters:
    -----------
    shap_values_train : numpy.ndarray
        Valeurs SHAP pour l'ensemble d'entraînement
    shap_values_test : numpy.ndarray
        Valeurs SHAP pour l'ensemble de test
    X_train : pandas.DataFrame
        DataFrame contenant les features d'entraînement
    save_dir : str, optional
        Répertoire où sauvegarder le graphique (par défaut './shap_dependencies_results/')

    Returns:
    --------
    pandas.DataFrame
        DataFrame contenant la comparaison des valeurs SHAP moyennes
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    os.makedirs(save_dir, exist_ok=True)

    mean_shap_train = shap_values_train.mean(axis=0)
    mean_shap_test = shap_values_test.mean(axis=0)

    shap_comparison = pd.DataFrame({
        'Feature': X_train.columns,
        'Mean SHAP Train': mean_shap_train,
        'Mean SHAP Test': mean_shap_test,
        'Difference': mean_shap_train - mean_shap_test
    })

    shap_comparison = shap_comparison.sort_values('Difference', key=abs, ascending=False)

    plt.figure(figsize=(12, 8))
    plt.scatter(shap_comparison['Mean SHAP Train'], shap_comparison['Mean SHAP Test'], alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'r--')  # ligne de référence y=x
    for i, txt in enumerate(shap_comparison['Feature']):
        plt.annotate(txt, (shap_comparison['Mean SHAP Train'].iloc[i], shap_comparison['Mean SHAP Test'].iloc[i]))
    plt.xlabel('Mean SHAP Value (Train)')
    plt.ylabel('Mean SHAP Value (Test)')
    plt.title('Comparison of Mean SHAP Values: Train vs Test')
    plt.tight_layout()

    # Sauvegarder le graphique dans le répertoire spécifié
    plt.savefig(os.path.join(save_dir, 'mean_shap_comparison.png'))
    plt.close()

    print("Top differences in mean SHAP values:")
    print(shap_comparison.head(10))

    return shap_comparison


def main_shap_analysis(final_model, X_train, y_train_label, X_test, y_test_label,  save_dir='./shap_dependencies_results/'):
    """Fonction principale pour l'analyse SHAP."""

    # Analyse SHAP sur l'ensemble d'entraînement et de test
    shap_values_train = analyze_shap_values(final_model, X_train, y_train_label, "Training_Set",
                                            create_dependence_plots=True,
                                            max_dependence_plots=3,save_dir=save_dir)
    shap_values_test = analyze_shap_values(final_model, X_test, y_test_label, "Test_Set", create_dependence_plots=True,
                                           max_dependence_plots=3,save_dir=save_dir)

    # Comparaison des importances de features et des distributions SHAP
    importance_df = compare_feature_importance(shap_values_train, shap_values_test, X_train, X_test,save_dir=save_dir)
    compare_shap_distributions(shap_values_train, shap_values_test, X_train, X_test, top_n=10,save_dir=save_dir)

    # Comparaison des valeurs SHAP moyennes
    shap_comparison = compare_mean_shap_values(shap_values_train, shap_values_test, X_train,save_dir=save_dir)

    return importance_df, shap_comparison,shap_values_train,shap_values_test


def train_and_evaluate_XGBOOST_model(
        df=None,
        config=None,  # Add config parameter here
        xgb_param_optuna_range=None,
        selected_columns=None,
        use_shapeImportance_file=None,
        results_directory=None,
        user_input=None,
        weight_param=None,
):

    optima_option_method = config.get('optima_option_method', optima_option.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED)
    device = config.get('device_', 'cuda')
    n_trials_optimization = config.get('n_trials_optuna', 4)
    nb_split_tscv = config.get('nb_split_tscv_', 10)
    nanvalue_to_newval = config.get('nanvalue_to_newval_', np.nan)
    learning_curve_enabled = config.get('learning_curve_enabled', False)
    random_state_seed = config.get('random_state_seed', 30)
    early_stopping_rounds = config.get('early_stopping_rounds', 60)
    std_penalty_factor = config.get('std_penalty_factor_', 0.5)
    preShapImportance = config.get('preShapImportance', 1)
    use_shapeImportance_file = config.get('use_shapeImportance_file', r'C:\Users\aulac\Downloads')
    cv_method = config.get('cv_method', cv_config.K_FOLD_SHUFFLE)


    # Gestion des valeurs NaN
    if nanvalue_to_newval is not None:
        # Remplacer les NaN par la valeur spécifiée
        df = df.fillna(nanvalue_to_newval)
        nan_value = nanvalue_to_newval
    else:
        # Garder les NaN tels quels
        nan_value = np.nan


    # Affichage des informations sur les NaN dans chaque colonne
    print(f"Analyses des Nan:)")
    for column in df.columns:
        nan_count = df[column].isna().sum()
        print(f"Colonne: {column}, Nombre de NaN: {nan_count}")

    # Division en ensembles d'entraînement et de test
    print("Division des données en ensembles d'entraînement et de test...")
    try:
        train_df, nb_SessionTrain,test_df,nb_SessionTest = split_sessions(df, test_size=0.2, min_train_sessions=2, min_test_sessions=2)

    except ValueError as e:
        print(f"Erreur lors de la division des sessions : {e}")
        sys.exit(1)

    #num_sessions_XTest = calculate_and_display_sessions(test_df)

    print(f"Nombre de session dans x_train  {nb_SessionTrain}  ")
    print(f"Nombre de session dans x_test  {nb_SessionTest}  ")

    X_train_full=train_df.copy()


    # Préparation des features et de la cible
    X_train = train_df[selected_columns]
    y_train_label = train_df['class_binaire']
    X_test = test_df[selected_columns]
    y_test_label = test_df['class_binaire']



    print(
        f"\nValeurs NaN : X_train={X_train.isna().sum().sum()}, y_train_label={y_train_label.isna().sum()}, X_test={X_test.isna().sum().sum()}, y_test_label={y_test_label.isna().sum()}\n")

    # Suppression des échantillons avec la classe 99
    mask_train = y_train_label != 99
    X_train, y_train_label = X_train[mask_train], y_train_label[mask_train]
    mask_test = y_test_label != 99
    X_test, y_test_label = X_test[mask_test], y_test_label[mask_test]


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


    maskShap = train_preliminary_model_with_tscv(X_train=X_train, y_train_label=y_train_label, preShapImportance=preShapImportance,use_shapeImportance_file=use_shapeImportance_file)

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

    # Au début de votre script principal, ajoutez ceci :
    global metric_dict
    metric_dict = {}

    global bestResult_dict
    bestResult_dict = {}

    # Ensuite, modifiez votre code comme suit :

    def objective_wrapper(trial,study):
        global bestResult_dict
        global metric_dict
        score_adjustedStd_val,pnl_perTrade_diff, updated_metric_dict, updated_bestResult_dict = objective_optuna(
            trial=trial, study=study,X_train=X_train, y_train_label=y_train_label,X_train_full=X_train_full,
            device=device,
            xgb_param_optuna_range=xgb_param_optuna_range,config=config, nb_split_tscv=nb_split_tscv,
            learning_curve_enabled=learning_curve_enabled,
            optima_score=optima_option_method, metric_dict=metric_dict,bestResult_dict=bestResult_dict, weight_param=weight_param, random_state_seed_=random_state_seed,
            early_stopping_rounds=early_stopping_rounds,std_penalty_factor_=std_penalty_factor,cv_method=cv_method
        )
        if score_adjustedStd_val != float('-inf'):
            metric_dict.update(updated_metric_dict)
            bestResult_dict.update(updated_bestResult_dict)
        return score_adjustedStd_val,pnl_perTrade_diff

    study_xgb  = optuna.create_study(
        directions=["maximize", "minimize"],  # Comme vous retournez le PnL négatif pour le maximiser
        sampler=optuna.samplers.NSGAIISampler(seed=42)
    )

    study_xgb .optimize(
        lambda trial: objective_wrapper(trial, study_xgb ),
        n_trials=n_trials_optimization,
        callbacks=[lambda study, trial: print_callback(study, trial, X_train, y_train_label,config=config)],
        gc_after_trial=True

    )
    end_time = time.time()
    execution_time = end_time - start_time

    # Après l'optimisation


    print("Optimisation terminée.")
    print("Meilleurs hyperparamètres trouvés: ", study_xgb.best_params)
    print("Meilleur score: ", study_xgb.best_value)
    optimal_threshold = study_xgb.best_params['threshold']
    print(f"Seuil utilisé : {optimal_threshold:.4f}")
    print(f"Temps d'exécution total : {execution_time:.2f} secondes")
    print_notification('###### FIN: OPTIMISATION BAYESIENNE ##########', color="blue")

    print_notification('###### DEBUT: ENTRAINEMENT MODELE FINAL ##########', color="blue")

    best_params = study_xgb.best_params.copy()
    num_boost_round = best_params.pop('num_boost_round', None)

    best_params['tree_method'] = 'hist'
    best_params['device'] = device


    # Configurer custom_metric et obj_function si nécessaire
    if optima_option_method == optima_option.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED:
        custom_metric = lambda preds, dtrain: custom_metric_ProfitBased(preds, dtrain, metric_dict)
        obj_function = create_weighted_logistic_obj(best_params['w_p'], 1)
        best_params['disable_default_eval_metric'] = 1
    else:
        custom_metric = None
        obj_function = None
        best_params['objective'] = 'binary:logistic'
        best_params['eval_metric'] = ['aucpr', 'logloss']
    print(f"Seuil optimal: {best_params['threshold']}")
    # Supprimer les paramètres non utilisés par XGBoost mais uniquement dans l'optimisation
    parameters_to_removetoAvoidXgboostError = ['loss_per_fp', 'penalty_per_fn', 'profit_per_tp', 'threshold', 'w_p']
    for param in parameters_to_removetoAvoidXgboostError:
        best_params.pop(param, None)  # None est la valeur par défaut si la clé n'existe pas

    print(f"best_params dans les parametres d'optimisations non xgboost: \n{best_params}")


    print(
        f"Num Boost : {num_boost_round}")

    # Entraîner le modèle final
    evals_result = {}  # Créez un dictionnaire vide pour stocker les résultats
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        obj=obj_function,
        custom_metric=custom_metric,
        early_stopping_rounds=early_stopping_rounds,
        maximize=True,
        verbose_eval=20,
        evals_result=evals_result  # Assurez-vous d'inclure ceci
    )

    # Affichage des résultats clés de l'entraînement du modèle final

    print(
        f"Meilleur nombre d'itérations : {final_model.best_iteration}")  # Indique le point optimal avant surapprentissage
    print(f"Meilleur score : {final_model.best_score}")  # Score optimal atteint sur l'ensemble d'évaluation


    def add_early_stopping_zone(ax, best_iteration, color='orange', alpha=0.2):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.axvspan(best_iteration, xmax, facecolor=color, alpha=alpha)
        ax.text(best_iteration + (xmax - best_iteration) / 2, ymax, 'Zone post early stopping',
                horizontalalignment='center', verticalalignment='top', fontsize=12, color='orange')

    def plot_custom_metric_evolution_with_trade_info(model, evals_result, metric_name='custom_metric_ProfitBased',
                                                     n_train_trades=None, n_test_trades=None,results_directory=None):
        if not evals_result or 'train' not in evals_result or 'test' not in evals_result:
            print("Résultats d'évaluation incomplets ou non disponibles.")
            return

        train_metric = evals_result['train'][metric_name]
        test_metric = evals_result['test'][metric_name]

        iterations = list(range(1, len(train_metric) + 1))
        best_test_iteration = np.argmax(test_metric)

        fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(24, 14))
        fig.suptitle(f'Entraînement du modèle final avec les paramètres optimaux (Optuna) :\n'
                    f'Évaluation du score {metric_name} sur l\'ensemble d\'entraînement (X_train) '
                     f'et un nouvel ensemble de test indépendant (X_test)', fontsize=12)

        def add_vertical_line_and_annotations(ax, is_train, is_normalized=False):
            ax.axvline(x=best_test_iteration, color='green', linestyle='--')
            y_pos = ax.get_ylim()[1] if is_train else ax.get_ylim()[0]
            score = train_metric[best_test_iteration] if is_train else test_metric[best_test_iteration]
            if is_normalized:
                score = (score - min(train_metric if is_train else test_metric)) / (
                        max(train_metric if is_train else test_metric) - min(
                    train_metric if is_train else test_metric))
            ax.annotate(f'{"Train" if is_train else "Test"} Score: {score:.2f}',
                        (best_test_iteration, y_pos), xytext=(5, 5 if is_train else -5),
                        textcoords='offset points', ha='left', va='bottom' if is_train else 'top',
                        fontsize=8, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        # X_train (Non Normalisé)
        ax1.plot(iterations, train_metric, label='Train', color='blue')
        ax1.set_title(f'X_train (Non Normalized)', fontsize=14)
        ax1.set_xlabel('Number of Iterations', fontsize=12)
        ax1.set_ylabel(f'{metric_name} Score', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.text(0.5, 0.1, f'Profit cumulés réalisé sur {n_train_trades} trades',
                 horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=12,
                 color='blue')
        add_vertical_line_and_annotations(ax1, is_train=True)
        add_early_stopping_zone(ax1, best_test_iteration)

        # X_test (Non Normalisé)
        ax2.plot(iterations, test_metric, label='Test', color='red')
        ax2.set_title(f'X_test (Non Normalized)', fontsize=14)
        ax2.set_xlabel('Number of Iterations', fontsize=12)
        ax2.set_ylabel(f'{metric_name} Score', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.text(0.5, 0.1, f'Profit cumulés réalisé sur {n_test_trades} trades',
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=12,
                 color='red')
        add_vertical_line_and_annotations(ax2, is_train=False)
        add_early_stopping_zone(ax2, best_test_iteration)

        # Normalisation
        train_min, train_max = min(train_metric), max(train_metric)
        test_min, test_max = min(test_metric), max(test_metric)
        train_normalized = [(val - train_min) / (train_max - train_min) for val in train_metric]
        test_normalized = [(val - test_min) / (test_max - test_min) for val in test_metric]

        # X_train (Normalisé)
        ax3.plot(iterations, train_normalized, label='Train (Normalized)', color='blue')
        ax3.set_title(f'X_train (Normalized)', fontsize=14)
        ax3.set_xlabel('Number of Iterations', fontsize=12)
        ax3.set_ylabel(f'Normalized {metric_name} Score', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.text(0.5, 0.1,
                 f'Profit cumulés réalisé sur {n_train_trades} trades\nNorm Ratio: [{train_min:.4f}, {train_max:.4f}]',
                 horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize=12,
                 color='blue')
        add_vertical_line_and_annotations(ax3, is_train=True, is_normalized=True)
        add_early_stopping_zone(ax3, best_test_iteration)

        # X_test (Normalisé)
        ax4.plot(iterations, test_normalized, label='Test (Normalized)', color='red')
        ax4.set_title(f'X_test (Normalized)', fontsize=14)
        ax4.set_xlabel('Number of Iterations', fontsize=12)
        ax4.set_ylabel(f'Normalized {metric_name} Score', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.text(0.5, 0.1,
                 f'Profit cumulés réalisé sur {n_test_trades} trades\nNorm Ratio: [{test_min:.4f}, {test_max:.4f}]',
                 horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes, fontsize=12,
                 color='red')
        add_vertical_line_and_annotations(ax4, is_train=False, is_normalized=True)
        add_early_stopping_zone(ax4, best_test_iteration)

        # X_test (Non Normalisé) jusqu'à best_test_iteration
        ax5.plot(iterations[:best_test_iteration + 1], test_metric[:best_test_iteration + 1], label='Test', color='red')
        ax5.set_title(f'X_test (Non Normalized) until best Test Score', fontsize=14)
        ax5.set_xlabel('Number of Iterations', fontsize=12)
        ax5.set_ylabel(f'{metric_name} Score', fontsize=12)
        ax5.legend(fontsize=10)
        ax5.grid(True, linestyle='--', alpha=0.7)
        add_vertical_line_and_annotations(ax5, is_train=False)

        # X_test (Normalisé) jusqu'à best_test_iteration
        ax6.plot(iterations[:best_test_iteration + 1], test_normalized[:best_test_iteration + 1],
                 label='Test (Normalized)', color='red')
        ax6.set_title(f'X_test (Normalized) until best Test Score', fontsize=14)
        ax6.set_xlabel('Number of Iterations', fontsize=12)
        ax6.set_ylabel(f'Normalized {metric_name} Score', fontsize=12)
        ax6.legend(fontsize=10)
        ax6.grid(True, linestyle='--', alpha=0.7)
        add_vertical_line_and_annotations(ax6, is_train=False, is_normalized=True)
        plt.savefig(os.path.join(results_directory, f'Evolution of {metric_name} Score with Trade Information'),
                    dpi=300,
                    bbox_inches='tight')
        plt.tight_layout()
        plt.show()
    # Utilisation de la fonction
    plot_custom_metric_evolution_with_trade_info(final_model, evals_result, n_train_trades=len(X_train), n_test_trades=len(X_test),results_directory=results_directory)

    """
    ANALYSE DES RÉSULTATS :

    1. Meilleur nombre d'itérations vs Nombre total d'itérations :
       - Si proches : Le modèle pourrait bénéficier de plus d'itérations.
       - Si très différents : L'early stopping a bien fonctionné pour prévenir le surapprentissage.

    2. Meilleur score :
       - Comparer avec les scores de la validation croisée pour vérifier la cohérence.
       - Un écart important peut indiquer des problèmes de généralisation.

    3. Métrique utilisée :
       - Confirmer que c'est la métrique attendue (ex: custom_metric_ProfitBased).
       - Assure la cohérence entre l'optimisation et l'évaluation finale.

    4. Nombre total d'itérations :
       - Si proche de num_boost_round : Considérer augmenter num_boost_round.
       - Si beaucoup plus petit : L'early stopping est efficace, les paramètres semblent bien réglés.

    ACTIONS POTENTIELLES :
    - Si le meilleur nombre d'itérations est proche du total : Augmenter num_boost_round.
    - Si le meilleur score diffère significativement des résultats de validation croisée : 
      Revoir la stratégie de split des données ou les hyperparamètres.
    - Si la métrique utilisée n'est pas celle attendue : Vérifier la configuration du custom_metric.
    - Utiliser ces informations pour affiner les paramètres dans de futurs entraînements.

    Ces résultats aident à comprendre le comportement du modèle, sa convergence, 
    et fournissent des pistes pour l'optimisation future du processus d'entraînement.
    """

    print_notification('###### FIN: ENTRAINEMENT MODELE FINAL ##########', color="blue")

    print_notification('###### DEBUT: ANALYSE DES DEPENDENCES SHAP DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")
    importance_df, shap_comparison,shap_values_train,shap_values_test = main_shap_analysis(
        final_model, X_train, y_train_label, X_test, y_test_label, save_dir=os.path.join(results_directory, 'shap_dependencies_results'))
    print_notification('###### FIN: ANALYSE DES DEPENDENCES SHAP DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")


    print_notification('###### DEBUT: ANALYSE DE L\'IMPACT DES VALEURS NaN DU MOBEL FINAL (ENTRAINEMENT) ##########', color="blue")

    # Appeler la fonction d'analyse
    analyze_nan_impact(model=final_model,X_train= X_train, feature_names=feature_names,
                       shap_values=shap_values_train, nan_value=nan_value, save_dir = os.path.join(results_directory, 'nan_analysis_results'))

    print_notification('###### FIN: ANALYSE DE L\'IMPACT DES VALEURS NaN DU MOBEL FINAL (ENTRAINEMENT) ##########', color="blue")

    # Prédiction et évaluation
    print_notification('###### DEBUT: GENERATION PREDICTION AVEC MOBEL FINAL (TEST) ##########', color="blue")


    # Obtenir les probabilités prédites pour la classe positive
    y_test_predProba = final_model.predict(dtest)

    # Convertir les prédictions en CuPy avant transformation
    y_test_predProba = cp.asarray(y_test_predProba)
    y_test_predProba = sigmoidCustom(y_test_predProba)  # Appliquer la transformation sigmoïde sur CuPy

    # Vérification des prédictions après transformation
    min_val = cp.min(y_test_predProba).item()
    max_val = cp.max(y_test_predProba).item()
    #print(f"Plage de valeurs après transformation sigmoïde : [{min_val:.4f}, {max_val:.4f}]")

    print(f"Plage de valeurs : [{min_val:.4f}, {max_val:.4f}]")

    # Vérifier si les valeurs sont dans l'intervalle [0, 1]
    if min_val < 0 or max_val > 1:
        print("ERREUR : Les prédictions ne sont pas dans l'intervalle [0, 1] attendu pour une classification binaire.")
        print("Vous devez appliquer une transformation (comme sigmoid) aux prédictions.")
        print("Exemple : y_test_predProba = sigmoidCustom(final_model.predict(dtest))")
        exit(11)
    else:
        print("Les prédictions sont dans l'intervalle [0, 1] attendu pour une classification binaire.")

    # Appliquer un seuil optimal pour convertir les probabilités en classes
    y_test_pred_threshold = (y_test_predProba > optimal_threshold).astype(int)

    print_notification('###### FIN: GENERATION PREDICTION AVEC MOBEL FINAL (TEST) ##########', color="blue")




    print_notification('###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur (XTEST) ##########', color="blue")

    ###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur XTEST ##########

    # Pour la courbe de calibration et l'histogramme
    plot_calibrationCurve_distrib(y_test_label, y_test_predProba, optimal_threshold=optimal_threshold, user_input=user_input,
                                  num_sessions=nb_SessionTest,results_directory=results_directory)

    # Pour le graphique des taux FP/TP par feature
    plot_fp_tp_rates(X_test, y_test_label, y_test_predProba, 'deltaTimestampOpeningSection5index',
                     optimal_threshold,user_input=user_input,index_size=5,results_directory=results_directory)

    print("\nDistribution des probabilités prédites sur XTest:")
    print(f"seuil: {optimal_threshold}")
    print(f"Min : {y_test_predProba.min():.4f}")
    print(f"Max : {y_test_predProba.max():.4f}")
    print(f"Moyenne : {y_test_predProba.mean():.4f}")
    print(f"Médiane : {np.median(y_test_predProba):.4f}")

    # Compter le nombre de prédictions dans différentes plages de probabilité
    # Définir le pas pour les intervalles en dessous de optimal_threshold
    step_below = 0.1  # Vous pouvez ajuster ce pas selon vos besoins

    # Créer les intervalles en dessous de optimal_threshold
    ranges_below = np.arange(0, optimal_threshold, step_below)
    ranges_below = np.append(ranges_below, optimal_threshold)

    # Définir le pas pour les intervalles au-dessus de optimal_threshold
    step_above = 0.02  # Taille des intervalles souhaitée au-dessus du seuil

    # Calculer le prochain multiple de step_above au-dessus de optimal_threshold
    next_multiple = np.ceil(optimal_threshold / step_above) * step_above

    # Créer les intervalles au-dessus de optimal_threshold
    ranges_above = np.arange(next_multiple, 1.0001, step_above)

    # Combiner les intervalles
    ranges = np.concatenate((ranges_below, ranges_above))
    ranges = np.unique(ranges)  # Supprimer les doublons et trier

    # Maintenant, vous pouvez utiliser ces ranges pour votre histogramme
    hist, _ = np.histogram(y_test_predProba, bins=ranges)

    # Convertir les tableaux CuPy en NumPy si nécessaire
    y_test_predProba_np = cp.asnumpy(y_test_predProba) if isinstance(y_test_predProba, cp.ndarray) else y_test_predProba
    y_test_label_np = cp.asnumpy(y_test_label) if isinstance(y_test_label, cp.ndarray) else y_test_label

    print("\nDistribution des probabilités prédites avec TP et FP sur XTest:")
    cum_tp=0
    cum_fp = 0
    for i in range(len(ranges) - 1):
        mask = (y_test_predProba_np >= ranges[i]) & (y_test_predProba_np < ranges[i + 1])
        predictions_in_range = y_test_predProba_np[mask]
        true_values_in_range = y_test_label_np[mask]
        tp = np.sum((predictions_in_range >= optimal_threshold) & (true_values_in_range == 1))
        fp = np.sum((predictions_in_range >= optimal_threshold) & (true_values_in_range == 0))
        total_trades_val = tp + fp
        win_rate = tp / total_trades_val * 100 if total_trades_val > 0 else 0
        cum_tp = cum_tp+tp
        cum_fp = cum_tp+fp
        print(
            f"Probabilité {ranges[i]:.2f} - {ranges[i + 1]:.2f} : {hist[i]} prédictions, TP: {tp}, FP: {fp}, Winrate: {win_rate:.2f}%")

    total_trades_cum=cum_tp+cum_fp
    Winrate=cum_tp/total_trades_cum
    print(f"Test final: X_test avec model final optimisé : TP: {cum_tp}, FP: {cum_fp}, Winrate: {Winrate:.2f}%")

    print("Statistiques de y_pred_proba:")
    print(f"Nombre d'éléments: {len(y_test_predProba)}")
    print(f"Min: {np.min(y_test_predProba)}")
    print(f"Max: {np.max(y_test_predProba)}")
    print(f"Valeurs uniques: {np.unique(y_test_predProba)}")
    print(f"Y a-t-il des NaN?: {np.isnan(y_test_predProba).any()}")

    # Définissez min_precision si vous voulez l'utiliser, sinon laissez-le à None
    min_precision = None  # ou une valeur comme 0.7 si vous voulez l'utiliser

    # Création de la figure avec trois sous-graphiques côte à côte
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

    # Sous-graphique 1 : Courbe ROC

    # Convertir les tableaux CuPy en NumPy
    y_test_label_np = cp.asnumpy(y_test_label) if isinstance(y_test_label, cp.ndarray) else y_test_label
    y_test_predProba_np = cp.asnumpy(y_test_predProba) if isinstance(y_test_predProba, cp.ndarray) else y_test_predProba

    # Calculer la courbe ROC et le score AUC
    fpr, tpr, _ = roc_curve(y_test_label_np, y_test_predProba_np)
    auc_score = roc_auc_score(y_test_label_np, y_test_predProba_np)

    ax1.plot(fpr, tpr, color='blue', linestyle='-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.grid(True)
    ax1.legend(loc='lower right', fontsize=10)

    # Sous-graphique 2 : Distribution des probabilités prédites
    bins = np.linspace(y_test_predProba.min(), y_test_predProba.max(), 100)

    # Assurez-vous que y_test_predProba est en NumPy

    # Conversion de y_test_predProba en NumPy
    y_test_predProba_np = cp.asnumpy(y_test_predProba) if isinstance(y_test_predProba, cp.ndarray) else y_test_predProba

    # Assurez-vous que optimal_threshold est un scalaire Python
    optimal_threshold = float(optimal_threshold)

    # Créez les masques pour les valeurs au-dessus et en dessous du seuil
    mask_below = y_test_predProba_np <= optimal_threshold
    mask_above = y_test_predProba_np > optimal_threshold

    # Créez bins comme un tableau NumPy
    bins = np.linspace(np.min(y_test_predProba_np), np.max(y_test_predProba_np), 100)

    # Utilisez ces masques avec y_test_predProba_np pour l'histogramme
    ax2.hist(y_test_predProba_np[mask_below], bins=bins, color='orange',
             label=f'Prédictions ≤ {optimal_threshold:.4f}', alpha=0.7)
    ax2.hist(y_test_predProba_np[mask_above], bins=bins, color='blue',
             label=f'Prédictions > {optimal_threshold:.4f}', alpha=0.7)

    ax2.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Seuil de décision ({optimal_threshold:.4f})')
    ax2.set_title('Proportion de prédictions négatives (fonction du choix du seuil) sur XTest', fontsize=14,
                  fontweight='bold')
    ax2.set_xlabel('Proportion de prédictions négatives (fonction du choix du seuil)', fontsize=12)
    ax2.set_ylabel('Nombre de prédictions', fontsize=12)

    # Ajout des annotations pour les comptes

    # Convertir y_test_predProba en NumPy si c'est un tableau CuPy
    y_test_predProba_np = cp.asnumpy(y_test_predProba) if isinstance(y_test_predProba, cp.ndarray) else y_test_predProba

    # Utiliser la version NumPy pour les calculs
    num_below = np.sum(y_test_predProba_np <= optimal_threshold)
    num_above = np.sum(y_test_predProba_np > optimal_threshold)

    ax2.text(0.05, 0.95, f'Count ≤ {optimal_threshold:.4f}: {num_below}', color='orange', transform=ax2.transAxes,
             va='top')
    ax2.text(0.05, 0.90, f'Count > {optimal_threshold:.4f}: {num_above}', color='blue', transform=ax2.transAxes,
             va='top')

    ax2.legend(fontsize=10)

    def to_numpy(arr):
        if isinstance(arr, cp.ndarray):
            return arr.get()
        elif isinstance(arr, np.ndarray):
            return arr
        else:
            return np.array(arr)

    # Convertir y_test_label et y_test_predProba en tableaux NumPy
    y_test_label_np = to_numpy(y_test_label)
    y_test_predProba_np = to_numpy(y_test_predProba)

    # Sous-graphique 3 : Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test_label_np, y_test_predProba_np)
    ax3.plot(recall, precision, color='green', marker='.', linestyle='-', linewidth=2)
    ax3.set_title('Courbe Precision-Recall', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Recall (Taux de TP)', fontsize=12)
    ax3.set_ylabel('Precision (1 - Taux de FP)', fontsize=12)
    ax3.grid(True)

    # Ajout de la ligne de précision minimale si définie
    if min_precision is not None:
        ax3.axhline(y=min_precision, color='r', linestyle='--', label=f'Précision minimale ({min_precision:.2f})')
        ax3.legend(fontsize=10)

    # Ajustement de la mise en page
    plt.tight_layout()

    # Sauvegarde et affichage du graphique
    plt.savefig(os.path.join(results_directory, 'roc_distribution_precision_recall_combined.png'), dpi=300,
                bbox_inches='tight')

    # Afficher ou fermer la figure selon l'entrée de l'utilisateur
    if user_input.lower() == 'd':
        plt.show()  # Afficher les graphiques
    plt.close()  # Fermer après l'affichage ou sans affichage

    print_notification('###### FIN: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur (XTEST) ##########',
                       color="blue")


    ###### DEBUT: ANALYSE SHAP ##########
    print_notification('###### DEBUT: ANALYSE SHAP ##########', color="blue")
    def analyze_shap_feature_importance(shap_values, X, ensembleType='train', save_dir='./shap_feature_importance/'):
        """
        Analyse l'importance des features basée sur les valeurs SHAP et génère des visualisations.

        Parameters:
        -----------
        shap_values : np.array
            Les valeurs SHAP calculées pour l'ensemble de données.
        X : pd.DataFrame
            Les features de l'ensemble de données.
        ensembleType : str, optional
            Type d'ensemble de données ('train' ou 'test'). Détermine le suffixe des fichiers sauvegardés.
        save_dir : str, optional
            Le répertoire où sauvegarder les graphiques générés (par défaut './shap_feature_importance/').

        Returns:
        --------
        dict
            Un dictionnaire contenant les résultats clés de l'analyse.
        """
        if ensembleType not in ['train', 'test']:
            raise ValueError("ensembleType doit être 'train' ou 'test'")

        os.makedirs(save_dir, exist_ok=True)
        suffix = f"_{ensembleType}"

        # Calcul des valeurs SHAP moyennes pour chaque feature
        shap_mean = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': shap_mean,
            'effect': np.mean(shap_values, axis=0)  # Effet moyen (positif ou négatif)
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        # Graphique des 20 features les plus importantes
        top_20_features = feature_importance.head(20)
        plt.figure(figsize=(12, 10))
        colors = ['#FF9999', '#66B2FF']  # Rouge clair pour négatif, bleu clair pour positif
        bars = plt.barh(top_20_features['feature'], top_20_features['importance'],
                        color=[colors[1] if x > 0 else colors[0] for x in top_20_features['effect']])

        plt.title(f"Feature Importance Determined By SHAP Values ({ensembleType.capitalize()} Set)", fontsize=16)
        plt.xlabel('Mean |SHAP Value| (Average Impact on Model Output Magnitude)', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.legend([plt.Rectangle((0, 0), 1, 1, fc=colors[0]), plt.Rectangle((0, 0), 1, 1, fc=colors[1])],
                   ['Diminue la probabilité de succès', 'Augmente la probabilité de succès'],
                   loc='lower right', fontsize=10)
        plt.text(0.5, 1.05, "La longueur de la barre indique l'importance globale de la feature.\n"
                            "La couleur indique si la feature tend à augmenter (bleu) ou diminuer (rouge) la probabilité de succès du trade.",
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_importance_binary_trade{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Visualisation des 30 valeurs SHAP moyennes absolues les plus importantes
        plt.figure(figsize=(12, 6))
        plt.bar(feature_importance['feature'][:30], feature_importance['importance'][:30])
        plt.xticks(rotation=45, ha='right')
        plt.title(f"Top 30 Features par Importance SHAP (valeurs moyennes absolues) - {ensembleType.capitalize()} Set")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'shap_importance_mean_abs{suffix}.png'))
        plt.close()

        # Analyse supplémentaire : pourcentage cumulatif de l'importance
        feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum() / feature_importance[
            'importance'].sum()

        # Top 10 features
        top_10_features = feature_importance['feature'].head(10).tolist()

        # Nombre de features nécessaires pour expliquer 80% de l'importance
        features_for_80_percent = feature_importance[feature_importance['cumulative_importance'] <= 0.8].shape[0]

        results = {
            'feature_importance': feature_importance,
            'top_10_features': top_10_features,
            'features_for_80_percent': features_for_80_percent
        }

        print(f"Graphiques SHAP pour l'ensemble {ensembleType} sauvegardés sous:")
        print(f"- {os.path.join(save_dir, f'shap_importance_binary_trade{suffix}.png')}")
        print(f"- {os.path.join(save_dir, f'shap_importance_mean_abs{suffix}.png')}")
        print(f"\nTop 10 features basées sur l'analyse SHAP ({ensembleType}):")
        print(top_10_features)
        print(
            f"\nNombre de features nécessaires pour expliquer 80% de l'importance ({ensembleType}) : {features_for_80_percent}")

        return results

    resulat_train_shap_feature_importance =analyze_shap_feature_importance(shap_values_train, X_train, ensembleType='train', save_dir=os.path.join(results_directory, 'shap_feature_importance'))
    resulat_test_shap_feature_importance=analyze_shap_feature_importance(shap_values_test, X_test, ensembleType='test', save_dir=os.path.join(results_directory, 'shap_feature_importance'))
    ###### FIN: ANALYSE SHAP ##########

     ###### DEBUT: ANALYSE DES ERREURS ##########
    print_notification('###### DEBUT: ANALYSE DES ERREURS ##########', color="blue")
    # Analyse des erreurs


    results_df, error_df = analyze_errors(X_test, y_test_label, y_test_pred_threshold, y_test_predProba, feature_names,
                                          save_dir=os.path.join(results_directory, 'analyse_error'),
                                          top_features=resulat_test_shap_feature_importance['top_10_features'])


    print_notification('###### FIN: ANALYSE DES ERREURS ##########', color="blue")
    ###### FIN: ANALYSE DES ERREURS ##########

    ###### DEBUT: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########
    print_notification('###### DEBUT: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########', color="blue")

    # Exemple d'utilisation :
    analyze_predictions_by_range(X_test, y_test_predProba, shap_values_test, prob_min=0.5, prob_max=1.00, top_n_features=20)


    feature_importance = np.abs(shap_values_test).mean(axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': feature_importance
    })

    # 1. Identifier les erreurs les plus confiantes
    errors = results_df[results_df['true_label'] != results_df['predicted_label']]
    confident_errors = errors.sort_values('prediction_probability', ascending=False)

    # 2. Récupérer les features importantes à partir de l'analyse SHAP
    important_features = feature_importance_df['feature'].head(10).tolist()

    print("Visualisation des erreurs confiantes:")
    plot_confident_errors(
       shap_values_test,
        confident_errors=confident_errors,
        X_test=X_test,
        feature_names=feature_names,
        n=5)
    #plot_confident_errors(xgb_classifier, confident_errors, X_test, X_test.columns,explainer_Test)


    # Exécution des analyses
    print("\nAnalyse des erreurs confiantes:")

    analyze_confident_errors(shap_values_test,confident_errors=confident_errors,X_test=X_test,feature_names=feature_names,important_features=important_features,n=5)
    correct_predictions = results_df[results_df['true_label'] == results_df['predicted_label']]
    print("\nComparaison des erreurs vs prédictions correctes:")
    compare_errors_vs_correct(confident_errors.head(30), correct_predictions, X_test, important_features,results_directory)
    print("\nAnalyse SHAP terminée. Les visualisations ont été sauvegardées.")
    print("\nAnalyse terminée. Les visualisations ont été sauvegardées.")
    print_notification('###### FIN: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########', color="blue")
    ###### FIN: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########

    ###### DEBUT: CALCUL DES VALEURS D'INTERACTION SHAP ##########
    print_notification("###### DEBUT: CALCUL DES VALEURS D'INTERACTION SHAP ##########", color="blue")
    # Calcul des valeurs d'interaction SHAP
    shap_interaction_values = final_model.predict(xgb.DMatrix(X_test), pred_interactions=True)
    # Exclure le biais en supprimant la dernière ligne et la dernière colonne
    shap_interaction_values = shap_interaction_values[:, :-1, :-1]

    # Vérification de la compatibilité des dimensions
    print("Shape of shap_interaction_values:", shap_interaction_values.shape)
    print("Number of features in X_test:", len(X_test.columns))

    if shap_interaction_values.shape[1:] != (len(X_test.columns), len(X_test.columns)):
        print("Erreur : Incompatibilité entre les dimensions des valeurs d'interaction SHAP et le nombre de features.")
        print(f"Dimensions des valeurs d'interaction SHAP : {shap_interaction_values.shape}")
        print(f"Nombre de features dans X_test : {len(X_test.columns)}")

        # Afficher les features de X_test
        print("Features de X_test:")
        print(list(X_test.columns))

        # Tenter d'accéder aux features du modèle
        try:
            model_features = final_model.feature_names
            print("Features du modèle:")
            print(model_features)

            # Comparer les features
            x_test_features = set(X_test.columns)
            model_features_set = set(model_features)

            missing_features = x_test_features - model_features_set
            extra_features = model_features_set - x_test_features

            if missing_features:
                print("Features manquantes dans le modèle:", missing_features)
            if extra_features:
                print("Features supplémentaires dans le modèle:", extra_features)
        except AttributeError:
            print("Impossible d'accéder aux noms des features du modèle.")
            print("Type du modèle:", type(final_model))
            print("Attributs disponibles:", dir(final_model))

        print("Le calcul des interactions SHAP est abandonné.")
        return  # ou sys.exit(1) si vous voulez quitter le programme entièrement

    # Si les dimensions sont compatibles, continuez avec le reste du code
    interaction_matrix = np.abs(shap_interaction_values).sum(axis=0)
    feature_names = X_test.columns
    interaction_df = pd.DataFrame(interaction_matrix, index=feature_names, columns=feature_names)

    # Masquer la diagonale (interactions d'une feature avec elle-même)
    np.fill_diagonal(interaction_df.values, 0)

    # Sélection des top N interactions (par exemple, top 10)
    N = 60
    top_interactions = interaction_df.unstack().sort_values(ascending=False).head(N)

    # Visualisation des top interactions
    plt.figure(figsize=(24, 16))
    top_interactions.plot(kind='bar')
    plt.title(f"Top {N} Feature Interactions (SHAP Interaction Values)", fontsize=16)
    plt.xlabel("Feature Pairs", fontsize=12)
    plt.ylabel("Total Interaction Strength", fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_directory, 'top_feature_interactions.png'), dpi=300, bbox_inches='tight')

    plt.close()
    seen_pairs = set()
    print(f"Top {N // 2} Feature Interactions:")  # On divise N par 2 pour l'affichage

    for (f1, f2), value in top_interactions.items():
        # Créer une paire triée pour garantir que (A,B) et (B,A) sont considérées comme identiques
        pair = tuple(sorted([f1, f2]))

        # Si la paire n'a pas encore été vue et ce n'est pas une interaction avec soi-même
        if pair not in seen_pairs and f1 != f2:
            print(f"{f1} <-> {f2}: {value:.4f}")
            seen_pairs.add(pair)

    # Heatmap des interactions pour les top features
    top_features = interaction_df.sum().sort_values(ascending=False).head(N).index
    plt.figure(figsize=(26, 20))  # Augmenter la taille de la figure

    # Créer la heatmap avec des paramètres ajustés
    sns.heatmap(interaction_df.loc[top_features, top_features].round(0).astype(int),
                annot=True,
                cmap='coolwarm',
                fmt='d',  # Afficher les valeurs comme des entiers
                annot_kws={'size': 7},  # Réduire la taille de la police des annotations
                square=True,  # Assurer que les cellules sont carrées
                linewidths=0.5,  # Ajouter des lignes entre les cellules
                cbar_kws={'shrink': .8})  # Ajuster la taille de la barre de couleur

    plt.title(f"SHAP Interaction Values for Top {N} Features", fontsize=16)
    plt.tight_layout()
    plt.xticks(rotation=90, ha='center')  # Rotation verticale des labels de l'axe x
    plt.yticks(rotation=0)  # S'assurer que les labels de l'axe y sont horizontaux
    plt.savefig(
        os.path.join(results_directory, 'feature_interaction_heatmap.png'), dpi=300,
                bbox_inches='tight')
    plt.close()

    print("Graphique d'interaction sauvegardé sous 'feature_interaction_heatmap.png'")
    print_notification("###### FIN: CALCUL DES VALEURS D'INTERACTION SHAP ##########", color="blue")
    ###### FIN: CALCUL DES VALEURS D'INTERACTION SHAP ##########

    # Retourner un dictionnaire avec les résultats
    return {
        'study': study_xgb,
        'optimal_threshold': optimal_threshold,
        'final_model': final_model,
        'feature_importance_df': feature_importance_df,
        'X_test': X_test,
        'y_test_label': y_test_label,
        'y_test_pred_threshold': y_test_pred_threshold,
        'y_test_predProba': y_test_predProba
    }



############### main######################
if __name__ == "__main__":
 # Demander à l'utilisateur s'il souhaite afficher les graphiques
 check_gpu_availability()




 FILE_NAME_ = "Step5_4_0_4TP_1SL_080919_091024_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
 #FILE_NAME_ = "Step5_Step4_Step3_Step2_MergedAllFile_Step1_4_merged_extractOnly220LastFullSession_OnlyShort_feat_winsorized.csv"
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

 results_directory = \
     ("C:\\Users\\aulac\OneDrive\\Documents\\Trading\\PyCharmProject\\MLStrategy\\data_preprocessing\\results_optim\\"
      f"{target_directory}{os.path.sep}")

 # Extraire le répertoire contenant la chaîne "4_0_4TP_1SL"

 user_input = input(
     f"Pour afficher les graphiques, appuyez sur 'd',\n "
     f"Repertoire d'enregistrrepentt des resultat par défaut:\n {results_directory}\n pour le modifier taper 'r'\n"
     "Sinon, appuyez sur 'Entrée' pour les enregistrer sans les afficher: ")

 if user_input.lower() == 'r':
    new_output_dir = input("Entrez le nouveau répertoire de sortie des résultats : ")
    results_directory =new_output_dir
 else :
    results_directory=results_directory

 # Vérifier si le répertoire existe déjà
 if os.path.exists(results_directory):
     overwrite = input(
         f"Le répertoire '{results_directory}' existe déjà. Voulez-vous le supprimer et continuer ? (Appuyez sur Entrée pour continuer, ou tapez une autre touche pour arrêter le programme) ")
     if overwrite == "":
         shutil.rmtree(results_directory)
     else:
         print("Le programme a été arrêté.")
         exit()



 # Créer le répertoire s'il n'existe pas
 os.makedirs(results_directory, exist_ok=True)

 print(f"Les résultats seront saugardés dans : {results_directory}")

 # Création du dictionnaire de config
 config = {
     'target_directory':target_directory,
     'optima_option_method': optima_option.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED,
     'device_': 'cuda',
     'n_trials_optuna': 300,
     'nb_split_tscv_': 6,
     'nanvalue_to_newval_': np.nan,
     'learning_curve_enabled': False,
     'random_state_seed': 30,
     'early_stopping_rounds': 80,
     'std_penalty_factor_': 1,
     'use_shapeImportance_file': r"C:\Users\aulac\OneDrive\Documents\Trading\PyCharmProject\MLStrategy\data_preprocessing\shap_dependencies_results\shap_values_Training_Set.csv",
     'preShapImportance': 1,
     'cv_method': cv_config.K_FOLD
 }

 # Définir les paramètres supplémentaires

 weight_param = {
     'threshold': {'min': 0.55, 'max': 0.72},  # total_trades_val = tp + fp
     'w_p': {'min': 1, 'max': 2.2},  # poid pour la class 1 dans objective
     'profit_per_tp': {'min': 1, 'max': 1}, #fixe, dépend des profits par trade
     'loss_per_fp': {'min': -1.1, 'max': -1.1}, #fixe, dépend des pertes par trade
    'penalty_per_fn': {'min': -0.000, 'max': -0.000}
 }

 # 'profit_ratio_weight': {'min': 0.4, 'max': 0.4},  # profit_ratio = (tp - fp) / total_trades_val
 # 'win_rate_weight': {'min': 0.45, 'max': 0.45},  # win_rate = tp / total_trades_val if total_trades_val
 # 'selectivity_weight': {'min': 0.075, 'max': 0.075},  # selectivity = total_trades_val / total_samples

 xgb_param_optuna_range = {
     'num_boost_round': {'min': 200, 'max': 700},
     'max_depth': {'min': 6, 'max': 11},
     'learning_rate': {'min': 0.01, 'max': 0.2, 'log': True},
     'min_child_weight': {'min': 3, 'max': 10},
     'subsample': {'min': 0.7, 'max': 0.9},
     'colsample_bytree': {'min': 0.55, 'max': 0.8},
     'colsample_bylevel': {'min': 0.6, 'max': 0.85},
     'colsample_bynode': {'min': 0.5, 'max': 1.0},
     'gamma': {'min': 0, 'max': 5},
     'reg_alpha': {'min': 1, 'max': 15.0, 'log': True},
     'reg_lambda': {'min': 2, 'max': 20.0, 'log': True},
 }

 print_notification('###### DEBUT: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")
 file_path = FILE_PATH_
 initial_df = load_data(file_path)
 print_notification('###### FIN: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")

 # Chargement et préparation des données
 df = initial_df.copy()


 print(f"Nb de features avant  exlusion: {len(df.columns)}\n")

 # Définition des colonnes de features et des colonnes exclues
 excluded_columns  = [
         'class_binaire', 'date', 'trade_category',
         'SessionStartEnd',
         'timeStampOpening',
         'deltaTimestampOpening',
         'candleDir',
         'deltaTimestampOpeningSection1min',
         'deltaTimestampOpeningSection1index',
         'deltaTimestampOpeningSection5min',
         #'deltaTimestampOpeningSection5index',
         'deltaTimestampOpeningSection15min',
         'deltaTimestampOpeningSection15index',
         'deltaTimestampOpeningSection30min',
         'deltaTimestampOpeningSection30index',
         'deltaCustomSectionMin',
         'deltaCustomSectionIndex',
         'meanVolx',
         'total_count_abv',
         'total_count_blw',
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
 selected_columns = [col for col in df.columns if col not in excluded_columns and '_special' not in col]

 print(f"Nb de features après exlusion: {len(selected_columns)}\n")

 results = train_and_evaluate_XGBOOST_model(
     df=df,
     config=config,  # Pass the config here
     xgb_param_optuna_range=xgb_param_optuna_range,
     selected_columns=selected_columns,
     results_directory=results_directory,
     user_input=user_input,
     weight_param=weight_param,
 )

 if results is not None:
     print("Meilleurs hyperparamètres trouvés:", results['study'].best_params)
     print("Meilleur score:", results['study'].best_value)
     print("Seuil optimal:", results['optimal_threshold'])
 else:
     print("L'entraînement n'a pas produit de résultats.")


"""
                if optima_score == optima_option.USE_OPTIMA_ROCAUC:
                    val_score_best = roc_auc_score(y_val_cv, y_val_pred_proba_np)
                elif optima_score == optima_option.USE_OPTIMA_AUCPR:
                    val_score_best = average_precision_score(y_val_cv, y_val_pred_proba_np)
                elif optima_score == optima_option.USE_OPTIMA_F1:
                    val_score_best = f1_score(y_val_cv, y_val_pred)
                elif optima_score == optima_option.USE_OPTIMA_PRECISION:
                    val_score_best = precision_score(y_val_cv, y_val_pred)
                elif optima_score == optima_option.USE_OPTIMA_RECALL:
                    val_score_best = recall_score(y_val_cv, y_val_pred)
                elif optima_score == optima_option.USE_OPTIMA_MCC:
                    val_score_best = matthews_corrcoef(y_val_cv, y_val_pred)
                elif optima_score == optima_option.USE_OPTIMA_YOUDEN_J:
                    tn, fp, fn, tp = confusion_matrix(y_val_cv, y_val_pred).ravel()
                    sensitivity = tp / (tp + fn)
                    specificity = tn / (tn + fp)
                    val_score_best = sensitivity + specificity - 1
                elif optima_score == optima_option.USE_OPTIMA_SHARPE_RATIO:
                    val_score_best = calculate_sharpe_ratio(y_val_cv, y_val_pred, price_changes_val)
                elif optima_score == optima_option.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED:
                    #val_score_best = optuna_profitBased_score(y_val_cv, y_val_pred_proba_np, metric_dict=metric_dict)
                    val_score_best = max(evals_result['eval']['custom_metric_ProfitBased'])

                elif optima_score == optima_option.USE_OPTIMA_CUSTOM_METRIC_TP_FP:
                    val_score_best = optuna_TP_FP_score(y_val_cv, y_val_pred_proba_np, metric_dict=metric_dict)
                else:
                    print("Invalid Optuna score")
                    exit(1)
"""