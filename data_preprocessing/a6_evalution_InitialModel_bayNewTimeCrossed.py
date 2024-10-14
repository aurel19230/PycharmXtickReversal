import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from standardFunc import load_data, split_sessions, print_notification, plot_calibrationCurve_distrib,plot_fp_tp_rates, check_gpu_availability, calculate_and_display_sessions
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

# Define the custom_metric class using Enum
class xgb_custom_metric(Enum):
    # Constantes numériques
    PR_RECALL_TP = 1
    PROFIT_BASED = 2

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

global bestRessult_dict
########################################
#########    FUNCTION DEF      #########
########################################
def analyze_predictions_by_range(X_test, y_pred_proba, shap_values_all, prob_min=0.90, prob_max=1.00,
                                 top_n_features=None,
                                 output_dir=r'C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL_02102024\proba_predictions_analysis'):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shap
    import os

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # 1. Identifier les échantillons dans la plage de probabilités spécifiée
    prob_mask = (y_pred_proba >= prob_min) & (y_pred_proba <= prob_max)
    selected_samples = X_test[prob_mask]
    selected_proba = y_pred_proba[prob_mask]

    # Vérifier s'il y a des échantillons dans la plage spécifiée
    if len(selected_samples) == 0:
        print(f"Aucun échantillon trouvé dans la plage de probabilités {prob_min:.2f} - {prob_max:.2f}")
        return

    print(f"Nombre d'échantillons dans la plage {prob_min:.2f} - {prob_max:.2f}: {len(selected_samples)}")

    # Calculer les valeurs SHAP une seule fois pour tout X_test


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
    stats.to_csv(os.path.join(output_dir, 'selected_samples_statistics.csv'))

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
    comparison.to_csv(os.path.join(output_dir, 'global_vs_selected_comparison.csv'), index=False)

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


def optimize_threshold(y_true, y_pred_proba):
    # Calcul des courbes de précision et de rappel en fonction du seuil
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Calcul du F1-score pour chaque seuil
    f1_scores = 2 * (precisions * recalls) / (
            precisions + recalls + 1e-8)  # Ajout d'une petite valeur pour éviter la division par zéro

    # Trouver l'index du seuil optimal qui maximise le F1-score
    optimal_threshold_index = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_threshold_index]

    print(f"Seuil optimal trouvé: {optimal_threshold:.4f}")

    return optimal_threshold


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

    val_score = combined_metric(y_val, val_pred, metric_dict=metric_dict)

    return {
        'train_sizes': [len(X_train)],  # Ajout de cette ligne
        'train_scores_mean': [train_score],  # Modification ici
        'val_scores_mean': [val_score]  # Modification ici
    }


from sklearn.model_selection import train_test_split


def print_callback(study, trial, X_train, y_train_label):
    global bestRessult_dict
    learning_curve_data = trial.user_attrs.get('learning_curve_data')
    best_val_score = trial.value
    score_std = trial.user_attrs['score_std']
    total_train_size = len(X_train)

    print(f"\nSur les differents ensembles d'entrainement :")
    print(f"\nEssai terminé : {trial.number}")
    print(f"Score de validation moyen : {best_val_score:.4f}")
    print(f"Écart-type des scores : {score_std:.4f}")
    print(
        f"Intervalle de confiance (±1 écart-type) : [{best_val_score - score_std:.4f}, {best_val_score + score_std:.4f}]")
    print(f"Score du dernier pli : {trial.user_attrs['last_score']:.4f}")
    print(f"Variance des scores : {trial.user_attrs['score_variance']:.4f}")

    # Récupérer les valeurs de TP et FP
    total_tp = trial.user_attrs.get('total_tp', 0)
    total_fp = trial.user_attrs.get('total_fp', 0)
    total_tn = trial.user_attrs.get('total_tn', 0)
    total_fn = trial.user_attrs.get('total_fn', 0)
    tp_difference_raw = trial.user_attrs.get('tp_difference_raw', 0)
    tp_difference_pnl = trial.user_attrs.get('tp_difference_pnl', 0)

    tp_percentage = trial.user_attrs.get('tp_percentage', 0)
    total_trades = total_tp + total_fp
    win_rate = total_tp / total_trades * 100 if total_trades > 0 else 0
    print(f"\nEnsemble de validation (somme de l'ensemble des splits :")
    print(f"Nombre de: TP (True Positives) : {total_tp}, FP (False Positives) : {total_fp}, "
          f"TN (True Negative) : {total_tn}, FN (False Negative) : {total_fn},")
    print(f"Pourcentage Winrate           : {win_rate:.2f}%")
    print(f"Pourcentage de TP             : {tp_percentage:.2f}%")
    print(f"Différence (TP - FP)          : {tp_difference_raw}")
    print(f"PNL                           : {tp_difference_pnl}")
    print(f"Nombre de trades              : {total_tp+total_fp+total_tn+total_fn}")
    if learning_curve_data:
        train_scores = learning_curve_data['train_scores_mean']
        val_scores = learning_curve_data['val_scores_mean']
        train_sizes = learning_curve_data['train_sizes']

        print("\nCourbe d'apprentissage :")
        for size, train_score, val_score in zip(train_sizes, train_scores, val_scores):
            print(f"Taille d'entraînement: {size} ({size / total_train_size * 100:.2f}%)")
            print(f"  Score d'entraînement : {train_score:.4f}")
            print(f"  Score de validation  : {val_score:.4f}")
            if val_score != 0:
                diff_percentage = ((train_score - val_score) / val_score) * 100
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

    print(
        f"\nMeilleure valeur jusqu'à présent : {study.best_value:.4f} (obtenue lors de l'essai numéro : {study.best_trial.number})")
    print(f"Meilleurs paramètres jusqu'à présent : {study.best_params}")
    print(f"Score bestRessult_dict  corresoond au Best Score jusqu'à présent : {bestRessult_dict}")
    print("------")


# Fonctions supplémentaires pour l'analyse des erreurs et SHAP

# 1. Fonction pour analyser les erreurs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_errors(X_test, y_test_label, y_pred_threshold, y_pred_proba, feature_names, save_dir='./analyse_error/', top_features=None):
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

    # Créer un dictionnaire pour stocker toutes les données
    data = {
        'true_label': y_test_label,
        'predicted_label': y_pred_threshold,
        'prediction_probability': y_pred_proba
    }

    # Ajouter les features au dictionnaire
    for feature in feature_names:
        data[feature] = X_test[feature]

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
    error_distribution.to_csv(os.path.join(save_dir, 'error_distribution.csv'))

    # Analyser les features pour les cas d'erreur
    error_df = results_df[results_df['is_error']]

    print("\nMoyenne des features pour les erreurs vs. prédictions correctes:")
    feature_means = results_df.groupby('error_type')[feature_names].mean()
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(feature_means)

    # Sauvegarder les moyennes des features
    feature_means.to_csv(os.path.join(save_dir, 'feature_means_by_error_type.csv'))

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
    most_confident_errors.to_csv(os.path.join(save_dir, 'most_confident_errors.csv'), index=False)

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
    results_df.to_csv(os.path.join(save_dir, 'model_results_analysis.csv'), index=False)
    error_df.to_csv(os.path.join(save_dir, 'model_errors_analysis.csv'), index=False)

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

        new_im.save('confident_errors_shap_combined.png')

        # Clean up individual images
        for i in range(n):
            os.remove(f'confident_error_shap_{i + 1}.png')

        print(f"Image combinée des {n} erreurs confiantes sauvegardée sous 'confident_errors_shap_combined.png'")
    else:
        print("Pas assez d'erreurs confiantes pour créer une image combinée.")


# 4. Fonction pour comparer les erreurs vs les prédictions correctes
def compare_errors_vs_correct(confident_errors, correct_predictions, X_test, important_features):
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
    plt.savefig('compare_errors_vs_correct.png')
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
    import os
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


def train_preliminary_model_with_tscv(X_train, y_train_label, preShapImportance):
    params = {
        'max_depth': 10,
        'learning_rate': 0.005,
        'min_child_weight': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'tree_method': 'hist'
    }
    num_boost_round = 450  # Nombre de tours pour l'entraînement du modèle préliminaire

    nb_split_tscv = 4

    if preShapImportance == 1.0:
        # Utiliser toutes les features
        maskShap = np.ones(X_train.shape[1], dtype=bool)
        selected_features = X_train.columns
        print(f"Utilisation de toutes les features ({len(selected_features)}) : {list(selected_features)}")
        return maskShap

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
            early_stopping_rounds=150,
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


def calculate_precision_recall_tp_ratio_gpu(y_true, y_pred_threshold, metric_dict):
    """
    Calculate TP and FP on GPU and compute precision, recall, and combined score.
    """
    y_true_gpu = cp.array(y_true)
    y_pred_gpu = cp.array(y_pred_threshold)

    tp = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 1))
    fp = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 1))
    total_samples = len(y_true_gpu)
    total_trades = tp + fp

    # Calcul des métriques
    profit_ratio = (tp - fp) / total_trades if total_trades > 0 else 0
    win_rate = tp / total_trades if total_trades > 0 else 0
    selectivity = total_trades / total_samples

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
def weighted_logistic_gradient_Torchgpu(predt: np.ndarray, dtrain: xgb.DMatrix, w_p: float, w_n: float) -> np.ndarray:
    """Calcule le gradient pour la perte logistique pondérée (GPU)."""
    device = torch.device("cuda")
    predt_torch = torch.tensor(predt, dtype=torch.float32, device=device)
    y_torch = torch.tensor(dtrain.get_label(), dtype=torch.float32, device=device)

    predt_sigmoid = torch.sigmoid(predt_torch)
    weights = torch.where(y_torch == 1, w_p, w_n)
    grad = weights * (predt_sigmoid - y_torch)

    torch.cuda.synchronize()
    return grad.cpu().numpy()


def weighted_logistic_hessian_Torchgpu(predt: np.ndarray, dtrain: xgb.DMatrix, w_p: float, w_n: float) -> np.ndarray:
    """Calcule le hessien pour la perte logistique pondérée (GPU)."""
    device = torch.device("cuda")
    predt_torch = torch.tensor(predt, dtype=torch.float32, device=device)
    y_torch = torch.tensor(dtrain.get_label(), dtype=torch.float32, device=device)

    predt_sigmoid = torch.sigmoid(predt_torch)
    weights = torch.where(y_torch == 1, w_p, w_n)
    hess = weights * predt_sigmoid * (1 - predt_sigmoid)

    torch.cuda.synchronize()
    return hess.cpu().numpy()


def sigmoidCustom(x):
    """Custom sigmoid function."""
    return 1 / (1 + cp.exp(-x))


def weighted_logistic_gradient_Cupygpu(predt: np.ndarray, dtrain: xgb.DMatrix, w_p: float, w_n: float) -> np.ndarray:
    """Compute the gradient for the weighted logistic loss (GPU) using CuPy."""
    predt_gpu = cp.asarray(predt, dtype=cp.float32)
    y_gpu = cp.asarray(dtrain.get_label(), dtype=cp.float32)

    # Compute sigmoid using the custom function
    predt_sigmoid = sigmoidCustom(predt_gpu)
    weights = cp.where(y_gpu == 1, w_p, w_n)
    grad = weights * (predt_sigmoid - y_gpu)

    # Synchronize (optional, if necessary)
    cp.cuda.Stream.null.synchronize()

    return cp.asnumpy(grad)


def weighted_logistic_hessian_Cupygpu(predt: np.ndarray, dtrain: xgb.DMatrix, w_p: float, w_n: float) -> np.ndarray:
    """Compute the Hessian for the weighted logistic loss (GPU) using CuPy."""
    predt_gpu = cp.asarray(predt, dtype=cp.float32)
    y_gpu = cp.asarray(dtrain.get_label(), dtype=cp.float32)

    # Compute sigmoid using the custom function
    predt_sigmoid = sigmoidCustom(predt_gpu)
    weights = cp.where(y_gpu == 1, w_p, w_n)
    hess = weights * predt_sigmoid * (1 - predt_sigmoid)

    # Synchronize (optional, if necessary)
    cp.cuda.Stream.null.synchronize()

    return cp.asnumpy(hess)

def create_weighted_logistic_obj(w_p: float, w_n: float):
    def weighted_logistic_obj(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        grad = weighted_logistic_gradient_Cupygpu(predt, dtrain, w_p, w_n)
        hess = weighted_logistic_hessian_Cupygpu(predt, dtrain, w_p, w_n)
        return grad, hess
    return weighted_logistic_obj

# Fonction pour vérifier la disponibilité du GPU
def check_gpu_availability():
    torch_available = torch.cuda.is_available()
    cupy_available = cp.cuda.is_available()

    if not (torch_available and cupy_available):
        print("Erreur : GPU n'est pas disponible pour PyTorch et/ou CuPy. Le programme va s'arrêter.")
        if not torch_available:
            print("PyTorch ne détecte pas de GPU.")
        if not cupy_available:
            print("CuPy ne détecte pas de GPU.")
        exit(1)

    print("GPU est disponible. Utilisation de CUDA pour les calculs.")
    print(f"GPU détecté par PyTorch : {torch.cuda.get_device_name(0)}")
    print(f"GPU détecté par CuPy : {cp.cuda.runtime.getDeviceProperties(cp.cuda.Device())['name'].decode()}")

    # Vérification de la version CUDA
    torch_cuda_version = torch.version.cuda
    cupy_cuda_version = cp.cuda.runtime.runtimeGetVersion()

    print(f"Version CUDA pour PyTorch : {torch_cuda_version}")
    print(f"Version CUDA pour CuPy : {cupy_cuda_version}")

    if torch_cuda_version != cupy_cuda_version:
        print("Attention : Les versions CUDA pour PyTorch et CuPy sont différentes.")
        print("Cela pourrait causer des problèmes de compatibilité.")

    # Affichage de la mémoire GPU disponible
    torch_memory = torch.cuda.get_device_properties(0).total_memory
    cupy_memory = cp.cuda.runtime.memGetInfo()[1]

    print(f"Mémoire GPU totale (PyTorch) : {torch_memory / 1e9:.2f} GB")
    print(f"Mémoire GPU totale (CuPy) : {cupy_memory / 1e9:.2f} GB")
def optuna_TP_FP_score(y_true, y_pred_proba, metric_dict):
    """
    Custom Optuna score based on TP/FP ratio, win_rate, and selectivity.
    """
    threshold = metric_dict.get('threshold', 0.7)  # Default threshold for converting probabilities to binary
    y_pred_threshold = (y_pred_proba > threshold).astype(int)

    return calculate_precision_recall_tp_ratio_gpu(y_true, y_pred_threshold, metric_dict)


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
    total_trades = tp + fp  # Typically, total executed trades

    # Utiliser une condition pour éviter la division par zéro
    """"
    if total_trades > 0:
        normalized_profit = total_profit / total_trades
    else:
        normalized_profit = total_profit  # Reflect penalties from FNs when no trades are made



    return float(normalized_profit)  # Assurez-vous que c'est un float Python
    """
    return float(total_profit)


def custom_metric_ProfitBased(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y_true = dtrain.get_label()
    threshold = 0.7  # Vous pouvez ajuster ce seuil si nécessaire

    # Convertir les prédictions en décisions binaires
    y_pred_threshold = (predt > threshold).astype(int)

    # Définir les paramètres de profit/perte
    metric_dict = {
        'profit_per_tp': 1.0,
        'loss_per_fp': -1.1,
        'penalty_per_fn': -0.1
    }

    # Utiliser calculate_profitBased_gpu pour calculer le profit total
    total_profit = calculate_profitBased_gpu(y_true, y_pred_threshold, metric_dict)

    # Retourner directement le profit total sans modification
    return 'custom_metric_ProfitBased', total_profit,True

def optuna_profitBased_score(y_true, y_pred_proba, metric_dict):
    """
    Custom scoring function for Optuna, calculating profit based on TP, FP, and FN.
    Args:
        y_true: Actual labels.
        y_pred_proba: Predicted probabilities from the model.
        metric_dict: Dictionary containing profit and loss parameters.
    Returns:
        Profit score calculated based on TP, FP, and FN.
    """
    threshold = metric_dict.get('threshold', 0.7)
    print(metric_dict)
    print(f"--- optuna_profitBased_score avec seuil {threshold} ---")
    y_pred_threshold = (y_pred_proba > threshold).astype(int)
    return calculate_profitBased_gpu(y_true, y_pred_threshold, metric_dict)


def custom_metric_ProfitBased(predt: np.ndarray, dtrain: xgb.DMatrix, metric_dict) -> Tuple[str, float]:
    y_true = dtrain.get_label()
    threshold = metric_dict.get('threshold', 0.7)

    # Convertir les prédictions en décisions binaires
    y_pred_threshold = (predt > threshold).astype(int)

    # Utiliser calculate_profitBased_gpu pour calculer le profit total
    total_profit = calculate_profitBased_gpu(y_true, y_pred_threshold, metric_dict)

    # Retourner directement le profit total sans modification
    return 'custom_metric_ProfitBased', total_profit

def create_custom_metric_wrapper(metric_dict):
    def custom_metric_wrapper(predt, dtrain):
        return custom_metric_ProfitBased(predt, dtrain, metric_dict)
    return custom_metric_wrapper


# Ajoutez cette variable globale au début de votre script
global lastBest_score
lastBest_score = float('-inf')

def objective_optuna(trial, X_train_full, y_train_label_full, device,
                     num_boost_min, num_boost_max, nb_split_tscv,
                     learning_curve_enabled,
                     optima_score, metric_dict,bestRessult_dict, weight_param):
    global lastBest_score
    params = {
        'max_depth': trial.suggest_int('max_depth', 8, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.05, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 0.9),
        'random_state': random_state_seed,

        'tree_method': 'hist',
        'device': device,
        'base_score': 0.5  # ou une autre valeur appropriée
    }



    # Initialiser les compteurs de TP et FP
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    total_samples = 0
    total_nb_session_val=0

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
            'threshold': threshold_value,
            'profit_ratio_weight': profit_ratio_weight,
            'win_rate_weight': win_rate_weight,
            'selectivity_weight': selectivity_weight
        }

    elif optima_score == optima_option.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED:


        # Définir les paramètres spécifiques pour la métrique basée sur le profit
        metric_dict = {
            'threshold': threshold_value,
            'profit_per_tp': trial.suggest_float('profit_per_tp', weight_param['profit_per_tp']['min'],
                                                 weight_param['profit_per_tp']['max']),
            'loss_per_fp': trial.suggest_float('loss_per_fp', weight_param['loss_per_fp']['min'],
                                               weight_param['loss_per_fp']['max']),
            'penalty_per_fn': trial.suggest_float('penalty_per_fn', weight_param['penalty_per_fn']['min'],
                                               weight_param['penalty_per_fn']['max'])
        }

    else:
        raise ValueError(f"Méthode de métrique combinée non reconnue: {xgb_custom_metric}")


    num_boost_round = trial.suggest_int('num_boost_round', num_boost_min, num_boost_max)

    scores = []
    last_score = None
    optimal_thresholds = []
    learning_curve_data_list = []

    if nb_split_tscv < 2:
        exit(1)
    else:
        # Validation croisée avec TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=nb_split_tscv)
        i = 0

        for train_index, val_index in tscv.split(X_train_full):
            X_train_cv, X_val_cv = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
            y_train_cv, y_val_cv = y_train_label_full.iloc[train_index], y_train_label_full.iloc[val_index]

            """
            count_10 = (X_val_cv_full['SessionStartEnd'] == 10.0).sum()
            count_20 = (X_val_cv_full['SessionStartEnd'] == 20.0).sum()
            print(f"Nombre de 10.0 dans SessionStartEnd : {count_10}")
            print(f"Nombre de 20.0 dans SessionStartEnd : {count_20}")

            
            # Suppression des échantillons avec la classe 99
            mask_train = y_train_cv_full != 99
            X_train_cv, y_train_cv = X_train_cv_full[mask_train], y_train_cv_full[mask_train]
            mask_val = y_val_cv_full != 99
            X_val_cv, y_val_cv = X_val_cv_full[mask_val], y_val_cv_full[mask_val]

            nb_session_val=calculate_and_display_sessions(X_val_cv_full)
            total_nb_session_val += nb_session_val

            i = i + 1
            y_val_cv_size = len(y_val_cv_full)
            y_train_cv_size = len(y_train_cv_full)

            print(
                f"Taille pour le split {i} ->ensemble d'entrainement : {y_train_cv_size} / ensemble de validation : {y_val_cv_size}")
            print(
                f"->nombre de session pour le split : {nb_session_val} / nombre de session cumulés : {total_nb_session_val}")
           

            # Liste des colonnes à exclure
            columns_to_exclude = ['SessionStartEnd', 'deltaTimestampOpening']

            # Exclure les colonnes de X_train
            X_val_cv = X_val_cv.drop(columns=columns_to_exclude)


            # Filtrer les valeurs 99
            mask_train = y_train_cv != 99
            X_train_cv, y_train_cv = X_train_cv[mask_train], y_train_cv[mask_train]
            mask_val = y_val_cv != 99
            X_val_cv, y_val_cv = X_val_cv[mask_val], y_val_cv[mask_val]
             """


            if len(X_train_cv) == 0 or len(y_train_cv) == 0:
                print("Warning: Empty training set after filtering")
                continue

            # Recalculer les poids des échantillons pour l'ensemble d'entraînement du pli actuel
            sample_weights = compute_sample_weight('balanced', y=y_train_cv)

            # Créer les DMatrix pour XGBoost
            dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv, weight=sample_weights)
            dval = xgb.DMatrix(X_val_cv, label=y_val_cv)

            # Optimiser les poids


            w_p = trial.suggest_float('w_p', weight_param['w_p']['min'],
                                weight_param['w_p']['max'])
            w_n = 1  # Vous pouvez également l'optimiser si nécessaire

            # Mettre à jour la fonction objective avec les poids optimisés
            obj_function = create_weighted_logistic_obj(w_p, w_n)
            if(optima_score == optima_option.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED):
                custom_metric = lambda predtTrain, dtrain: custom_metric_ProfitBased(predtTrain, dtrain,
                                                                                 metric_dict)
                params['disable_default_eval_metric'] = 1
            else:
                params['objective']= 'binary:logistic', #la supprimimer si obj est défini dans xjboost
                params['eval_metric']='aucpr', #aucpr' logloss# peuvent etre conservé en plus de custum metric. sont stockées dans evals_result

            try:
                # Entraîner le modèle préliminaire
                # Use evals_result to track metrics during training
                evals_result = {}
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtrain, 'train'), (dval, 'eval')],
                    obj=obj_function,
                    custom_metric=custom_metric,
                    early_stopping_rounds=20,
                    verbose_eval=False,
                    evals_result=evals_result,
                    maximize=True
                )
                # Access and print the custom metric results
                print("Evaluation Results:", evals_result)
                y_val_pred_proba = model.predict(dval)
                y_val_pred = (y_val_pred_proba > metric_dict['threshold']).astype(int)
                # Calculer la matrice de confusion
                tn, fp, fn, tp = confusion_matrix(y_val_cv, y_val_pred).ravel()
                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn
                total_samples += len(y_val_cv)

                if optima_score == optima_option.USE_OPTIMA_ROCAUC:
                    val_score = roc_auc_score(y_val_cv, y_val_pred_proba)
                elif optima_score == optima_option.USE_OPTIMA_AUCPR:
                    val_score = average_precision_score(y_val_cv, y_val_pred_proba)
                elif optima_score == optima_option.USE_OPTIMA_F1:
                    val_score = f1_score(y_val_cv, y_val_pred)
                elif optima_score == optima_option.USE_OPTIMA_PRECISION:
                    val_score = precision_score(y_val_cv, y_val_pred)
                elif optima_score == optima_option.USE_OPTIMA_RECALL:
                    val_score = recall_score(y_val_cv, y_val_pred)
                elif optima_score == optima_option.USE_OPTIMA_MCC:
                    val_score = matthews_corrcoef(y_val_cv, y_val_pred)
                elif optima_score == optima_option.USE_OPTIMA_YOUDEN_J:
                    tn, fp, fn, tp = confusion_matrix(y_val_cv, y_val_pred).ravel()
                    sensitivity = tp / (tp + fn)
                    specificity = tn / (tn + fp)
                    val_score = sensitivity + specificity - 1
                elif optima_score == optima_option.USE_OPTIMA_SHARPE_RATIO:
                    val_score = calculate_sharpe_ratio(y_val_cv, y_val_pred, price_changes_val)
                elif optima_score == optima_option.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED:
                    val_score = optuna_profitBased_score(y_val_cv, y_val_pred_proba, metric_dict=metric_dict)
                elif optima_score == optima_option.USE_OPTIMA_CUSTOM_METRIC_TP_FP:
                    val_score = optuna_TP_FP_score(y_val_cv, y_val_pred_proba, metric_dict=metric_dict)
                else:
                    print("Invalid Optuna score")
                    exit(1)
                scores.append(val_score)
                last_score = val_score

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

            except Exception as e:
                print(f"Error during training or evaluation: {e}")
                continue

    if not scores:
        return float('-inf'), metric_dict, bestRessult_dict  # Retourne les trois valeurs même en cas d'erreur

    mean_cv_score = np.mean(scores)
    score_variance = np.var(scores)
    score_std = np.std(scores)
    tp_difference_raw = total_tp-total_fp
    tp_difference_pnl=total_tp*weight_param['profit_per_tp']['min'] +total_fp*weight_param['loss_per_fp']['max']

    if total_samples > 0:
        tp_percentage = (total_tp / total_samples) * 100
    else:
        tp_percentage = 0
    total_trades = total_tp + total_fp
    win_rate = total_tp / total_trades * 100 if total_trades > 0 else 0
    if mean_cv_score>lastBest_score:
        lastBest_score = mean_cv_score
        bestRessult_dict.update({
            "win_rate_percentage": round(win_rate, 2),
            "tp_difference_raw": tp_difference_raw,
            "tp_difference_pnl": tp_difference_pnl,
            "tp_percentage": round(tp_percentage, 3),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_tn": total_tn,
            "total_fn": total_fn,
            "total_trades": total_trades,
            "lastBest_score": mean_cv_score
        })
        print(f"Nouveau meilleur score trouvé : {mean_cv_score:.6f}")
        print(f"Updated bestRessult_dict: {bestRessult_dict}")

    print("Scores:", scores)
    print(f"Score mean sur les {nb_split_tscv} iterations : {mean_cv_score:.6f}")
    print(f"Score variance: {score_variance:.6f}")
    print(f"Score standard deviation: {score_std:.6f}")

    trial.set_user_attr('last_score', last_score)
    trial.set_user_attr('score_variance', score_variance)
    trial.set_user_attr('score_std', score_std)

    # Après la boucle de validation croisée
    if total_samples > 0:
        tp_percentage = (total_tp / total_samples) * 100
    else:
        tp_percentage = 0

    # Stocker les valeurs dans trial.user_attrs
    trial.set_user_attr('total_tp', total_tp)
    trial.set_user_attr('total_fp', total_fp)
    trial.set_user_attr('total_tn', total_tn)
    trial.set_user_attr('total_fn', total_fn)
    trial.set_user_attr('tp_difference_raw', tp_difference_raw)
    trial.set_user_attr('tp_difference_pnl', tp_difference_pnl)

    trial.set_user_attr('tp_percentage', tp_percentage)


    if learning_curve_enabled and learning_curve_data_list:
        avg_learning_curve_data = average_learning_curves(learning_curve_data_list)
        if avg_learning_curve_data is not None:
            trial.set_user_attr('learning_curve_data', avg_learning_curve_data)
            if trial.number == 0 or mean_cv_score > trial.study.best_value:
                plot_learning_curve(
                    avg_learning_curve_data,
                    title=f"Courbe d'apprentissage moyenne (Meilleur essai {trial.number})",
                    filename=f'learning_curve_best_trial_{trial.number}.png'
                )

    return mean_cv_score, metric_dict,bestRessult_dict


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
import numpy as np
import matplotlib.pyplot as plt

import shap
import numpy as np
import matplotlib.pyplot as plt
import os

import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def analyze_shap_values(model, X, y, dataset_name, create_dependence_plots=False, max_dependence_plots=3, save_dir='./shap_dependencies_results/'):
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


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


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


import pandas as pd
import matplotlib.pyplot as plt
import os


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
    shap_values_train = analyze_shap_values(final_model, X_train, y_train_label, "Training Set",
                                            create_dependence_plots=True,
                                            max_dependence_plots=3,save_dir=save_dir)
    shap_values_test = analyze_shap_values(final_model, X_test, y_test_label, "Test Set", create_dependence_plots=True,
                                           max_dependence_plots=3,save_dir=save_dir)

    # Comparaison des importances de features et des distributions SHAP
    importance_df = compare_feature_importance(shap_values_train, shap_values_test, X_train, X_test,save_dir=save_dir)
    compare_shap_distributions(shap_values_train, shap_values_test, X_train, X_test, top_n=10,save_dir=save_dir)

    # Comparaison des valeurs SHAP moyennes
    shap_comparison = compare_mean_shap_values(shap_values_train, shap_values_test, X_train,save_dir=save_dir)

    return importance_df, shap_comparison,shap_values_train,shap_values_test


def train_and_evaluate_XGBOOST_model(
        df=None,
        n_trials_optimization=4,
        device='cuda',
        num_boost_min=400,
        num_boost_max=1000,
        nb_split_tscv=12,
        nanvalue_to_newval=None,
        learning_curve_enabled=False,
        optima_option_method = optima_option.USE_OPTIMA_F1,
        feature_columns=None,
        xgb_custom_metric=xgb_custom_metric.PR_RECALL_TP,
        preShapImportance=1,
        user_input=None,
        weight_param=None

):
    # Gestion des valeurs NaN
    if nanvalue_to_newval is not None:
        # Remplacer les NaN par la valeur spécifiée
        df = df.fillna(nanvalue_to_newval)
        nan_value = nanvalue_to_newval
    else:
        # Garder les NaN tels quels
        nan_value = np.nan

    print(f"Nb de features: {len(df.columns)}\n")

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

    num_sessions_XTest = calculate_and_display_sessions(test_df)

    print(f"{num_sessions_XTest}  {nb_SessionTest}  ")

    # Préparation des features et de la cible
    X_train = train_df[feature_columns]
    y_train_label = train_df['class_binaire']
    X_test = test_df[feature_columns]
    y_test_label = test_df['class_binaire']

    X_train_full=X_train.copy()
    y_train_label_full=y_train_label.copy()
    X_test_full=X_test.copy()
    y_test_label_full=y_test_label.copy()

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

    if xgb_custom_metric == xgb_custom_metric.PR_RECALL_TP:
        metric_dict_prelim = {
            'threshold': 0.5,
            'profit_ratio_weight': 0.3,
            'win_rate_weight': 0.5,
            'selectivity_weight': 0.2
        }

    elif xgb_custom_metric == xgb_custom_metric.PROFIT_BASED:
        # Définir les paramètres spécifiques pour la métrique basée sur le profit
        metric_dict_prelim = {
            'threshold': 0.5,
            'profit_per_tp': 1,
            'loss_per_fp': -1.1,
        }

    maskShap = train_preliminary_model_with_tscv(X_train, y_train_label, preShapImportance)

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

    global bestRessult_dict
    bestRessult_dict = {}

    # Ensuite, modifiez votre code comme suit :

    def objective_wrapper(trial):
        global bestRessult_dict
        global metric_dict
        score, updated_metric_dict, updated_bestRessult_dict = objective_optuna(
            trial, X_train, y_train_label, device,
            num_boost_min, num_boost_max, nb_split_tscv,
            learning_curve_enabled,
            optima_option_method, metric_dict, bestRessult_dict, weight_param
        )
        if score != float('-inf'):
            metric_dict.update(updated_metric_dict)
            bestRessult_dict.update(updated_bestRessult_dict)
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(
        objective_wrapper,
        n_trials=n_trials_optimization,
        callbacks=[lambda study, trial: print_callback(study, trial, X_train, y_train_label)]
    )

    # Après l'optimisation
    optimal_threshold = metric_dict.get('threshold', 0.5)
    print(f"Seuil utilisé : {optimal_threshold:.4f}")

    end_time = time.time()
    execution_time = end_time - start_time

    print("Optimisation terminée.")
    print("Meilleurs hyperparamètres trouvés: ", study.best_params)
    print("Meilleur score: ", study.best_value)
    print(f"Temps d'exécution total : {execution_time:.2f} secondes")
    print_notification('###### FIN: OPTIMISATION BAYESIENNE ##########', color="blue")

    print_notification('###### DEBUT: ENTRAINEMENT MODELE FINAL ##########', color="blue")
    best_params, num_boost_round = prepare_xgboost_params(study, device)

    # Entraîner le modèle avec xgb.train
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train')],
        early_stopping_rounds=500,
        maximize=True,  # pour maximiser feval_func
        #custom_metric=custom_metric,
        verbose_eval=False
    )
    print_notification('###### FIN: ENTRAINEMENT MODELE FINAL ##########', color="blue")

    print_notification('###### DEBUT: ANALYSE DES DEPENDENCES SHAP DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")
    importance_df, shap_comparison,shap_values_train,shap_values_test = main_shap_analysis(
        final_model, X_train, y_train_label, X_test, y_test_label, save_dir='./shap_dependencies_results/')
    print_notification('###### FIN: ANALYSE DES DEPENDENCES SHAP DU MOBEL FINAL (ENTRAINEMENT) ##########',
                       color="blue")


    print_notification('###### DEBUT: ANALYSE DE L\'IMPACT DES VALEURS NaN DU MOBEL FINAL (ENTRAINEMENT) ##########', color="blue")

    # Appeler la fonction d'analyse
    analyze_nan_impact(model=final_model,X_train= X_train, feature_names=feature_names,shap_values=shap_values_train, nan_value=nan_value)

    print_notification('###### FIN: ANALYSE DE L\'IMPACT DES VALEURS NaN DU MOBEL FINAL (ENTRAINEMENT) ##########', color="blue")

    # Prédiction et évaluation
    print_notification('###### DEBUT: GENERATION PREDICTION AVEC MOBEL FINAL (TEST) ##########', color="blue")


    # Obtenir les probabilités prédites pour la classe positive
    y_test_predProba = final_model.predict(dtest)

    # Appliquer un seuil optimal pour convertir les probabilités en classes
    y_test_pred_threshold = (y_test_predProba > optimal_threshold).astype(int)

    print_notification('###### FIN: GENERATION PREDICTION AVEC MOBEL FINAL (TEST) ##########', color="blue")




    print_notification('###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur (XTEST) ##########', color="blue")

    ###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur XTEST ##########

    # Pour la courbe de calibration et l'histogramme
    plot_calibrationCurve_distrib(y_test_label, y_test_predProba, optimal_threshold=optimal_threshold, user_input=user_input,
                                  num_sessions=nb_SessionTest)

    # Pour le graphique des taux FP/TP par feature
    plot_fp_tp_rates(X_test, y_test_label, y_test_predProba, 'deltaTimestampOpeningSection5index', optimal_threshold,user_input=user_input,index_size=5)

    print("\nDistribution des probabilités prédites sur XTest:")
    print(f"seuil: {optimal_threshold}")
    print(f"Min : {y_test_predProba.min():.4f}")
    print(f"Max : {y_test_predProba.max():.4f}")
    print(f"Moyenne : {y_test_predProba.mean():.4f}")
    print(f"Médiane : {np.median(y_test_predProba):.4f}")

    # Compter le nombre de prédictions dans différentes plages de probabilité
    ranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bisect.insort(ranges, optimal_threshold)
    ranges = sorted(set(ranges))
    hist, _ = np.histogram(y_test_predProba, bins=ranges)

    print("\nDistribution des probabilités prédites avec TP et FP sur XTest:")
    for i in range(len(ranges) - 1):
        mask = (y_test_predProba >= ranges[i]) & (y_test_predProba < ranges[i + 1])
        predictions_in_range = y_test_predProba[mask]
        true_values_in_range = y_test_label[mask]

        tp = np.sum((predictions_in_range >= optimal_threshold) & (true_values_in_range == 1))
        fp = np.sum((predictions_in_range >= optimal_threshold) & (true_values_in_range == 0))
        total_trades=tp+fp
        win_rate = tp / total_trades * 100 if total_trades > 0 else 0
        print(f"Probabilité {ranges[i]:.4f} - {ranges[i + 1]:.4f} : {hist[i]} prédictions, TP: {tp}, FP: {fp}, Winrate: {win_rate:.2f}%")

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
    fpr, tpr, _ = roc_curve(y_test_label, y_test_predProba)
    auc_score = roc_auc_score(y_test_label, y_test_predProba)

    ax1.plot(fpr, tpr, color='blue', linestyle='-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.grid(True)
    ax1.legend(loc='lower right', fontsize=10)

    # Sous-graphique 2 : Distribution des probabilités prédites
    bins = np.linspace(y_test_predProba.min(), y_test_predProba.max(), 100)

    ax2.hist(y_test_predProba[y_test_predProba <= optimal_threshold], bins=bins, color='orange',
             label=f'Prédictions ≤ {optimal_threshold:.4f}', alpha=0.7)
    ax2.hist(y_test_predProba[y_test_predProba > optimal_threshold], bins=bins, color='blue',
             label=f'Prédictions > {optimal_threshold:.4f}', alpha=0.7)
    ax2.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Seuil de décision ({optimal_threshold:.4f})')
    ax2.set_title('Proportion de prédictions négatives (fonction du choix du seuil) sur XTest', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Proportion de prédictions négatives (fonction du choix du seuil)', fontsize=12)
    ax2.set_ylabel('Nombre de prédictions', fontsize=12)

    # Ajout des annotations pour les comptes
    num_below = np.sum(y_test_predProba <= optimal_threshold)
    num_above = np.sum(y_test_predProba > optimal_threshold)
    ax2.text(0.05, 0.95, f'Count ≤ {optimal_threshold:.4f}: {num_below}', color='orange', transform=ax2.transAxes,
             va='top')
    ax2.text(0.05, 0.90, f'Count > {optimal_threshold:.4f}: {num_above}', color='blue', transform=ax2.transAxes,
             va='top')

    ax2.legend(fontsize=10)

    # Sous-graphique 3 : Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test_label, y_test_predProba)
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
    plt.savefig('roc_distribution_precision_recall_combined.png', dpi=300, bbox_inches='tight')
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

    resulat_train_shap_feature_importance =analyze_shap_feature_importance(shap_values_train, X_train, ensembleType='train', save_dir='./shap_feature_importance/')
    resulat_test_shap_feature_importance=analyze_shap_feature_importance(shap_values_test, X_test, ensembleType='test', save_dir='./shap_feature_importance/')
    ###### FIN: ANALYSE SHAP ##########

     ###### DEBUT: ANALYSE DES ERREURS ##########
    print_notification('###### DEBUT: ANALYSE DES ERREURS ##########', color="blue")
    # Analyse des erreurs


    results_df, error_df = analyze_errors(X_test, y_test_label, y_test_pred_threshold, y_test_predProba, feature_names,
                                          save_dir='./analyse_error/',
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
    compare_errors_vs_correct(confident_errors.head(30), correct_predictions, X_test, important_features)
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
    N = 30
    top_interactions = interaction_df.unstack().sort_values(ascending=False).head(N)

    # Visualisation des top interactions
    plt.figure(figsize=(12, 8))
    top_interactions.plot(kind='bar')
    plt.title(f"Top {N} Feature Interactions (SHAP Interaction Values)", fontsize=16)
    plt.xlabel("Feature Pairs", fontsize=12)
    plt.ylabel("Total Interaction Strength", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top_feature_interactions.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Top {N} Feature Interactions:")
    for (f1, f2), value in top_interactions.items():
        if f1 != f2:  # Vérification supplémentaire pour exclure les interactions avec soi-même
            print(f"{f1} <-> {f2}: {value:.4f}")

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
    plt.savefig('feature_interaction_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Graphique d'interaction sauvegardé sous 'feature_interaction_heatmap.png'")
    print_notification("###### FIN: CALCUL DES VALEURS D'INTERACTION SHAP ##########", color="blue")
    ###### FIN: CALCUL DES VALEURS D'INTERACTION SHAP ##########

    # Retourner un dictionnaire avec les résultats
    return {
        'study': study,
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
 user_input = input(
     "Pour afficher les graphiques, appuyez sur 'd'. Sinon, appuyez sur 'Entrée' pour les enregistrer sans les afficher: ")

 FILE_NAME_ = "Step5_Step4_Step3_Step2_MergedAllFile_Step1_4_merged_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
 #FILE_NAME_ = "Step5_Step4_Step3_Step2_MergedAllFile_Step1_4_merged_extractOnly220LastFullSession_OnlyShort_feat_winsorized.csv"
 DIRECTORY_PATH_ = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL_04102024\merge"
 FILE_PATH_ = os.path.join(DIRECTORY_PATH_, FILE_NAME_)

 DEVICE_ = 'cuda'

 xgb_custom_metric = xgb_custom_metric.PR_RECALL_TP #oas finalisé a date. implémente  objectuf et eval_metric sont ceux pas defaut dans params
 optima_option_method=optima_option.USE_OPTIMA_CUSTOM_METRIC_PROFITBASED

 NUM_BOOST_MIN_ = 400
 NUM_BOOST_MAX_ = 850
 N_TRIALS_OPTIMIZATION_ = 500
 NB_SPLIT_TSCV_ = 4
 NANVALUE_TO_NEWVAL_ = np.nan  # 900000.123456789
 LEARNING_CURVE_ENABLED = False
 random_state_seed = 30
 # Définir les paramètres supplémentaires

 weight_param = {
     'threshold': {'min': 0.50, 'max': 0.53},  # total_trades = tp + fp
     'profit_ratio_weight': {'min': 0.4, 'max': 0.4},  # profit_ratio = (tp - fp) / total_trades
     'win_rate_weight': {'min': 0.45, 'max': 0.45},  # win_rate = tp / total_trades if total_trades
     'selectivity_weight': {'min': 0.075, 'max': 0.075},  # selectivity = total_trades / total_samples
     'profit_per_tp': {'min': 1, 'max': 1}, #fixe, dépend des profits par trade
     'loss_per_fp': {'min': -1.1, 'max': -1.1}, #fixe, dépend des pertes par trade
    'penalty_per_fn': {'min': -0.000, 'max': -0.000},

     'w_p': {'min': 1, 'max': 3} #poid pour la class 1 dans objective
 }

 print_notification('###### DEBUT: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")
 file_path = FILE_PATH_
 initial_df = load_data(file_path)
 print_notification('###### FIN: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")

 # Chargement et préparation des données
 df = initial_df.copy()

 # Définition des colonnes de features et des colonnes exclues
 feature_columns = [
     col for col in df.columns if col not in [
         'class_binaire', 'date', 'trade_category',
         'SessionStartEnd', #exclu plus tard
         'deltaTimestampOpening', #exclu plus tard
         'candleDir',
         'deltaTimestampOpeningSection5min',
         # 'deltaTimestampOpeningSection5index',
         'deltaTimestampOpeningSection30min',
         'deltaTimestampOpeningSection30index',
         'deltaCustomSectionMin',
         'deltaCustomSectionIndex',
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
 ]

 results = train_and_evaluate_XGBOOST_model(
     df=df,
     n_trials_optimization=N_TRIALS_OPTIMIZATION_,
     device=DEVICE_,
     num_boost_min=NUM_BOOST_MIN_,
     num_boost_max=NUM_BOOST_MAX_,
     nb_split_tscv=NB_SPLIT_TSCV_,
     nanvalue_to_newval=NANVALUE_TO_NEWVAL_,
     learning_curve_enabled=LEARNING_CURVE_ENABLED,
     optima_option_method=optima_option_method,
     feature_columns=feature_columns,
     xgb_custom_metric=xgb_custom_metric,
     preShapImportance=1,
     user_input=user_input,
     weight_param=weight_param
 )

 if results is not None:
     print("Meilleurs hyperparamètres trouvés:", results['study'].best_params)
     print("Meilleur score:", results['study'].best_value)
     print("Seuil optimal:", results['optimal_threshold'])
 else:
     print("L'entraînement n'a pas produit de résultats.")

"""
"# Un seul split : 80% pour l'entraînement, 20% pour la validation
X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(X_train, y_train_label, test_size=0.2, shuffle=False)

# Filtrer les valeurs 99
mask_train = y_train_cv != 99
X_train_cv, y_train_cv = X_train_cv[mask_train], y_train_cv[mask_train]
mask_val = y_val_cv != 99
X_val_cv, y_val_cv = X_val_cv[mask_val], y_val_cv[mask_val]

if len(X_train_cv) == 0 or len(y_train_cv) == 0:
    print("Warning: Empty training set after filtering")
    return float('-inf')  # Retourne un score très faible en cas d'erreur

# Calculer les poids des échantillons pour l'ensemble d'entraînement du pli actuel
sample_weights = compute_sample_weight('balanced', y=y_train_cv)

# Créer les DMatrix pour XGBoost
dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv, weight=sample_weights)
dval = xgb.DMatrix(X_val_cv, label=y_val_cv)

try:
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=100,
        verbose_eval=False,
        maximize=True,  # Changé à True car nous voulons maximiser le profit
        #obj=obj,
        #custom_metric=custom_metric
    )

    y_val_pred_proba = model.predict(dval)
    y_val_pred = (y_val_pred_proba > metric_dict['threshold']).astype(int)

    # Calculer la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_val_cv, y_val_pred).ravel()
    total_tp += tp
    total_fp += fp
    total_tn += tn
    total_fn += fn
    total_samples += len(y_val_cv)

except Exception as e:
    print(f"Erreur lors de l'entraînement : {e}")
    # Gérer l'erreur selon vos besoins

    if use_optimized_threshold:
        optimal_threshold = optimize_threshold(y_val_cv, y_val_pred_proba)
        optimal_thresholds.append(optimal_threshold)

    if use_auc_roc:
        val_score = roc_auc_score(y_val_cv, y_val_pred_proba)
    else:
        val_score = optuna_score(
            y_val_cv,
            y_val_pred_proba,
            metric_dict=metric_dict
        )
    scores.append(val_score)
    last_score = val_score

except Exception as e:
    print(f"Error during training or evaluation: {e}")
    return float('-inf')  # Retourne un score très faible en cas d'erreur
"""
"""
# Fonction englobante qui intègre metric_dict
if xgb_custom_metric == xgb_custom_metric.PR_RECALL_TP:
    custom_metric = lambda predtTrain, dtrain: xgb_custom_metric_TP_FP(predtTrain, dtrain,
                                                                          metric_dict)
    obj = lambda preds, dtrain: get_xgb_objective_TP_FP(metric_dict)
elif xgb_custom_metric == xgb_custom_metric.PROFIT_BASED:
    custom_metric = lambda predtTrain, dtrain: xgb_custom_metric_profitBased(predtTrain, dtrain, metric_dict)
    obj=lambda preds, dtrain: xgb_objective_profitBased(preds, dtrain, metric_dict)
else:
    raise ValueError(f"Méthode de métrique combinée non reconnue: {xgb_custom_metric}")
"""