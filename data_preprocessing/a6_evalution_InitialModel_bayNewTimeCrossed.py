import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from standardFunc import load_data, split_sessions, print_notification,plot_calibration_curve

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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.ticker as ticker
from enum import Enum, auto
from PIL import Image

class CombinedMetric(Enum):
    # Constantes numériques
    PR_RECALL_TP = 1
    PROFIT_BASED =2
########################################
#########    FUNCTION DEF      #########
########################################
def analyze_predictions_by_range(X_test, y_pred_proba, xgb_classifier, prob_min=0.90, prob_max=1.00,
                                 top_n_features=None, output_dir=r'C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL_02102024\proba_predictions_analysis'):
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
    explainer = shap.TreeExplainer(xgb_classifier)
    shap_values_all = explainer.shap_values(X_test)

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
            print(f"Erreur : Session {i} mal formée. Début : {df.loc[start, 'SessionStartEnd']}, Fin : {df.loc[end, 'SessionStartEnd']}")
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
        params, num_boost_round, X_train, y_train, X_val,
        y_val, weight_dict, combined_metric,metric_dict,custom_metric):
    """
    Calcule les scores d'entraînement et de validation pour un split de validation croisée.
    """
    # Créer des DMatrix pour l'entraînement et la validation
    sample_weights = np.array([weight_dict[label] for label in y_train])
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Entraîner le modèle
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round, maximize=True, custom_metric=custom_metric)

    # Prédire sur les ensembles d'entraînement et de validation
    train_pred = booster.predict(dtrain)
    val_pred = booster.predict(dval)

    # Calculer les scores
    train_score = combined_metric(y_train, train_pred, metric_dict=metric_dict)

    val_score = combined_metric(y_val, val_pred, metric_dict=metric_dict)

    return {
        'train_sizes': [len(X_train)],  # Ajout de cette ligne
        'train_scores_mean': [train_score],  # Modification ici
        'val_scores_mean': [val_score]  # Modification ici
    }

from sklearn.model_selection import train_test_split



def print_callback(study, trial, X_train, y_train):
    total_train_size = len(X_train)
    learning_curve_data = trial.user_attrs.get('learning_curve_data')
    best_val_score = trial.value
    score_std = trial.user_attrs['score_std']
    total_train_size = len(X_train)

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
    tp_difference = trial.user_attrs.get('tp_difference', 0)
    tp_percentage = trial.user_attrs.get('tp_percentage', 0)
    total_trades = total_tp + total_fp
    win_rate = total_tp / total_trades*100 if total_trades > 0 else 0
    print(f"\nNombre (ens de validation) de: TP (True Positives) : {total_tp}, FP (False Positives) : {total_fp}, "
          f"TN (True Negative) : {total_tn}, FN (False Negative) : {total_fn},")
    print(f"Pourcentage Winrate           : {win_rate:.2f}%")
    print(f"Pourcentage de TP             : {tp_percentage:.2f}%")
    print(f"Différence (TP - FP)          : {tp_difference}")
    print(f"Nombre de trades             : {len(y_train)}")
    if learning_curve_data:
        train_scores = learning_curve_data['train_scores_mean']
        val_scores = learning_curve_data['val_scores_mean']
        train_sizes = learning_curve_data['train_sizes']

        print("\nCourbe d'apprentissage :")
        for size, train_score, val_score in zip(train_sizes, train_scores, val_scores):
            print(f"Taille d'entraînement: {size} ({size/total_train_size*100:.2f}%)")
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
        print(f"Nombre d'échantillons d'entraînement utilisés : {int(best_train_size)} ({best_train_size/total_train_size*100:.2f}% du total)")
    else:
        print("Option Courbe d'Apprentissage non activé")

    print(
        f"\nMeilleure valeur jusqu'à présent : {study.best_value:.4f} (obtenue lors de l'essai numéro : {study.best_trial.number})")
    print(f"Meilleurs paramètres jusqu'à présent : {study.best_params}")
    print("------")

# Fonctions supplémentaires pour l'analyse des erreurs et SHAP

# 1. Fonction pour analyser les erreurs
def analyze_errors(X_test, y_test, y_pred, y_pred_proba, feature_names):
    # Créer un dictionnaire pour stocker toutes les données
    data = {
        'true_label': y_test,
        'predicted_label': y_pred,
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
    print("Distribution des erreurs:")
    print(results_df['error_type'].value_counts(normalize=True))

    # Analyser les features pour les cas d'erreur
    error_df = results_df[results_df['is_error']]

    print("\nMoyenne des features pour les erreurs vs. prédictions correctes:")
    feature_means = results_df.groupby('error_type')[feature_names].mean()
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(feature_means)

    # Visualiser la distribution des probabilités de prédiction pour les erreurs
    plt.figure(figsize=(10, 6))
    sns.histplot(data=error_df, x='prediction_probability', hue='true_label', bins=20)
    plt.title('Distribution des probabilités de prédiction pour les erreurs')
    plt.savefig('error_probability_distribution.png')
    plt.close()

    # Identifier les cas les plus confiants mais erronés
    most_confident_errors = error_df.sort_values('prediction_probability', ascending=False).head(5)
    print("\nLes 5 erreurs les plus confiantes:")
    print(most_confident_errors[['true_label', 'predicted_label', 'prediction_probability']])

    return results_df, error_df

# 2. Fonction pour analyser les erreurs confiantes
def analyze_confident_errors(xgb_classifier, confident_errors, X_test, feature_names, important_features, n=5,explainer=None):
    if explainer is None:
        explainer = shap.TreeExplainer(xgb_classifier)
    for idx in confident_errors.index[:n]:
        print(f"-----------------> Analyse de l'erreur à l'index {idx}:")
        print(f"Vrai label: {confident_errors.loc[idx, 'true_label']}")
        print(f"Label prédit: {confident_errors.loc[idx, 'predicted_label']}")
        print(f"Probabilité de prédiction: {confident_errors.loc[idx, 'prediction_probability']:.4f}")

        print("\nValeurs des features importantes:")
        for feature in important_features:
            value = X_test.loc[idx, feature]
            print(f"{feature}: {value:.4f}")

        shap_values = explainer.shap_values(X_test.loc[idx:idx])

        print("\nTop 5 features influentes (SHAP) pour ce cas:")
        case_feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values[0])
        }).sort_values('importance', ascending=False)

        print(case_feature_importance.head())
        print(f"<----------------- Fin Analyse de l'erreur à l'index {idx}:")

# 3. Fonction pour visualiser les erreurs confiantes
def plot_confident_errors(xgb_classifier, confident_errors, X_test, feature_names, n=5, explainer=None):
    if explainer is None:
        explainer = shap.TreeExplainer(xgb_classifier)

    # Vérifier le nombre d'erreurs confiantes disponibles
    num_errors = len(confident_errors)
    if num_errors == 0:
        print("Aucune erreur confiante trouvée.")
        return

    # Ajuster n si nécessaire
    n = min(n, num_errors)

    for i, idx in enumerate(confident_errors.index[:n]):
        plt.figure(figsize=(10, 6))
        shap_values = explainer.shap_values(X_test.loc[idx:idx])

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
            #print(f"Erreur lors de l'extraction des informations de la règle: {e}")
            continue
        except Exception as e:
            # Gérer toute autre erreur inattendue
            print(f"Erreur inattendue: {e}")
            continue

    return rules




def analyze_nan_impact(model, X, feature_names, nan_value, shap_values=None):
    """
    Analyse l'impact des valeurs NaN ou des valeurs de remplacement des NaN sur le modèle XGBoost en utilisant les valeurs SHAP.

    :param model: Modèle XGBoost entraîné (Booster ou XGBClassifier/XGBRegressor)
    :param X: Données d'entrée (DataFrame)
    :param feature_names: Liste des noms des features
    :param nan_value: Valeur utilisée pour remplacer les NaN (peut être np.nan)
    :param shap_values: Valeurs SHAP pré-calculées (optionnel)
    """
    # 1. Analyser les splits impliquant les valeurs NaN ou la valeur de remplacement
    all_splits,nan_splits = analyze_xgboost_trees(model, feature_names, nan_value)
    print(f"Nombre total de splits : {len(all_splits)}")
    print(f"Nombre de splits impliquant des NaN : {len(nan_splits)}")
    print(f"Pourcentage de splits impliquant des NaN : {(len(nan_splits) / len(all_splits)) * 100:.2f}%")

    print("\nAperçu de tous les splits :")
    print(all_splits.head())

    print("\nAperçu des splits impliquant des NaN :")
    print(nan_splits.head())

    # Analyses supplémentaires possibles
    print("\nDistribution des features pour tous les splits :")
    print(all_splits['Feature'].value_counts())

    print("\nDistribution des features pour les splits NaN :")
    print(nan_splits['Feature'].value_counts())

    # 2. Visualiser la distribution des splits NaN
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Feature', data=nan_splits, order=nan_splits['Feature'].value_counts().index)
    plt.title("Distribution des splits impliquant des valeurs NaN par feature")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('nan_splits_distribution.png')
    plt.close()

    # 3. Analyze the depth of NaN splits
    if 'Depth' not in nan_splits.columns:
        if 'ID' in nan_splits.columns:
            # Calculate depth from ID column
            nan_splits['Depth'] = nan_splits['ID'].apply(lambda x: x.count('-'))
            print("'Depth' column added based on 'ID' column.")
        else:
            print("Warning: 'Depth' column is missing and cannot be calculated. Skipping depth analysis.")

    if 'Depth' in nan_splits.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Feature', y='Depth', data=nan_splits)
        plt.title("Depth of splits involving NaN values by feature")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('nan_splits_depth.png')
        plt.close()
    else:
        print("Skipping creation of 'nan_splits_depth.png' due to missing 'Depth' column.")

    # 4. Extraire les règles de décision importantes
    all_splits, nan_splits = analyze_xgboost_trees(model, feature_names, nan_value)
    total_trees = len(set(all_splits['Tree']))
    trees_with_nan = len(set(nan_splits['Tree']))

    important_rules = extract_decision_rules(model, nan_value)
    print("\nRègles de décision importantes impliquant des valeurs NaN :")
    verbose_nan_rule = True
    if verbose_nan_rule:
        print("\nRègles de décision importantes impliquant des valeurs NaN :")
        for rule in important_rules:
            # Safely split and parse depth
            parts = rule.split(', ')

            try:
                # Check if parts[1] has a valid depth value (should be an integer)
                depth_str = parts[1].split()[1].replace(':', '')  # Remove any colon or special characters
                if depth_str.isdigit():
                    depth = int(depth_str)
                   #XXX print(f"Parsed depth: {depth}")
                else:
                    # If the depth contains invalid characters, raise an error
                    raise ValueError(f"Invalid depth value encountered: {parts[1]}")
            except (IndexError, ValueError) as e:
                print(f"Erreur lors du parsing de la règle: {e}")
                continue

        print("\nAnalyse des règles de décision importantes:")
        feature_counts = {}
        depth_counts = {}
        gain_sum = 0
        print('ERROR SUR LES RULES 0 CORRGIGER XXX')
        for rule in important_rules:
            # Extraction des informations de la règle
            parts = rule.split(', ')

            try:
                tree = int(parts[0].split()[1])

                # Same parsing for depth with error handling
                depth_str = parts[1].split()[1].replace(':', '')
                if depth_str.isdigit():
                    depth = int(depth_str)
                else:
                    raise ValueError(f"Invalid depth value encountered in rule: {rule}")

                feature = parts[2].split()[1]
                gain = float(parts[-1].split()[-1].strip(')'))

                # Comptage des features
                feature_counts[feature] = feature_counts.get(feature, 0) + 1

                # Comptage des profondeurs
                depth_counts[depth] = depth_counts.get(depth, 0) + 1

                # Somme des gains
                gain_sum += gain

            except (IndexError, ValueError) as e:
                #print(f"Erreur lors de l'extraction des informations de la règle: {e}")
                continue
    print(f"\nStatistiques sur les règles de décision impliquant des NaN (sur xtrain final):")
    print(f"Nombre total d'arbres : {total_trees}")
    print(f"Nombre d'arbres impliquant des NaN : {trees_with_nan}")
    print(f"Pourcentage d'arbres impliquant des NaN : {(trees_with_nan / total_trees) * 100:.2f}%")
    print(f"Nombre de règles de décision importantes impliquant des NaN : {len(important_rules)}")
    print(f"Nombre moyen de règles importantes par arbre impliqué : {len(important_rules) / trees_with_nan:.2f}")

    # 5. Analyser l'importance des features avec des valeurs NaN en utilisant les valeurs SHAP
    # Calcul des valeurs SHAP si elles ne sont pas fournies
    if shap_values is None:
        print("Calcul des valeurs SHAP (sur xtrain final)...")
        # Si le modèle est un Booster, convertir en XGBClassifier pour utiliser SHAP
        if isinstance(model, xgb.Booster):
            # Créer un XGBClassifier et charger le Booster
            xgb_classifier = xgb.XGBClassifier()
            xgb_classifier._Booster = model
        else:
            xgb_classifier = model
        explainer = shap.TreeExplainer(xgb_classifier)
        shap_values = explainer.shap_values(X)
    else:
        print("Utilisation des valeurs SHAP fournies.")

    # Calcul de l'importance moyenne absolue des valeurs SHAP pour chaque feature
    shap_mean = np.abs(shap_values).mean(axis=0)
    shap_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_mean
    })

    # Identifier les features avec des valeurs NaN ou la valeur de remplacement
    if pd.isna(nan_value):
        nan_features = X.columns[X.isna().any()].tolist()
    else:
        nan_features = X.columns[(X == nan_value).any()].tolist()

    # Calculer le nombre total de valeurs par feature
    total_counts = X.shape[0]

    # Calculer le nombre de NaN par feature
    if pd.isna(nan_value):
        nan_counts = X.isna().sum()
    else:
        nan_counts = (X == nan_value).sum()

    # Calculer le pourcentage de NaN par feature
    nan_percentages = (nan_counts / total_counts) * 100

    # Créer un DataFrame avec ces informations
    nan_info_df = pd.DataFrame({
        'Feature': X.columns,
        'Total_NaN': nan_counts,
        'Percentage_NaN': nan_percentages
    })

    # Filtrer les features avec au moins une valeur NaN
    nan_info_df = nan_info_df[nan_info_df['Total_NaN'] > 0]

    # Joindre les importances SHAP avec les informations sur les NaN
    nan_fi_df = pd.merge(nan_info_df, shap_importance_df, on='Feature', how='left')
    nan_fi_df['Importance'] = nan_fi_df['Importance'].fillna(0)

    # Trier par ordre décroissant d'importance
    nan_fi_df = nan_fi_df.sort_values('Importance', ascending=False)

    # Paramètres pour la division des features
    num_features = len(nan_fi_df)
    features_per_plot = 35
    num_plots = (num_features + features_per_plot - 1) // features_per_plot  # Calcul du nombre de graphiques

    # Générer les graphiques par lots
    for i in range(num_plots):
        start_idx = i * features_per_plot
        end_idx = start_idx + features_per_plot
        subset_df = nan_fi_df.iloc[start_idx:end_idx]

        fig, ax1 = plt.subplots(figsize=(14, 8))
        sns.barplot(x='Feature', y='Importance', data=subset_df, ax=ax1, color='skyblue')
        ax1.set_xlabel('Feature', fontsize=12)
        ax1.set_ylabel('Importance (SHAP)', color='blue', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='blue')

        ax1.set_xticks(range(len(subset_df)))
        ax1.set_xticklabels(subset_df['Feature'], rotation=45, ha='right')

        ax2 = ax1.twinx()
        sns.lineplot(x='Feature', y='Percentage_NaN', data=subset_df, ax=ax2, color='red', marker='o')
        ax2.set_ylabel('Pourcentage de NaN (%)', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.yaxis.set_major_formatter(ticker.PercentFormatter())

        plt.title(
            f"Importance des features (SHAP) et pourcentage de NaN (Features {start_idx + 1} à {min(end_idx, num_features)} (sur xtrain final)\n (ensemble d'entrainement avec model final))",
            fontsize=14)
        plt.tight_layout()
        plt.savefig(f'nan_features_shap_importance_percentage_{i + 1}.png')
        plt.close()

    # Calcul de la corrélation entre l'importance (SHAP) et le pourcentage de NaN
    correlation = nan_fi_df['Importance'].corr(nan_fi_df['Percentage_NaN'])
    print(f"\nCorrélation entre l'importance des features (SHAP) et le pourcentage de NaN (sur xtrain final) : {correlation:.4f}")

    # Visualisation de la relation entre l'importance (SHAP) et le pourcentage de NaN
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Percentage_NaN', y='Importance', data=nan_fi_df)
    plt.title('Relation entre le pourcentage de NaN et l\'importance des features (SHAP)')
    plt.xlabel('Pourcentage de NaN (%)')
    plt.ylabel('Importance (SHAP)')
    plt.tight_layout()
    plt.savefig('shap_importance_vs_percentage_nan.png')
    plt.close()


def train_preliminary_model_with_tscv(X_train, y_train, preShapImportance,custom_metric):
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
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

        # Filtrer les valeurs 99 (valeurs "inutilisables")
        mask_train = y_train_cv != 99
        X_train_cv, y_train_cv = X_train_cv[mask_train], y_train_cv[mask_train]
        mask_val = y_val_cv != 99
        X_val_cv, y_val_cv = X_val_cv[mask_val], y_val_cv[mask_val]

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
            early_stopping_rounds=50,
            verbose_eval=False,
            maximize=True,
            custom_metric=custom_metric
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



"""
def calculate_precision_recall_tp_ratio(y_true, y_pred, metric_dict):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    tp_ratio = np.sum((y_true == 1) & (y_pred == 1)) / len(y_true)

    precision_weight = metric_dict.get('precision_weight', 0.4)
    recall_weight = metric_dict.get('recall_weight', 0.3)
    tp_ratio_weight = metric_dict.get('tp_ratio_weight', 0.3)
   # print(f"--- calculate_precision_recall_tp_ratio: precision:{precision} / recall:{recall} / "
    #      f"precision:{tp_ratio} ---")

    combined_score = (
        (precision * precision_weight) +
        (recall * recall_weight) +
        (tp_ratio * tp_ratio_weight)
    )

    sum_of_weights = precision_weight + recall_weight + tp_ratio_weight
    return combined_score / sum_of_weights

def calculate_profit_based(y_true, y_pred, metric_dict):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    profit_per_tp = metric_dict.get('profit_per_tp', 1.0)
    loss_per_fp = metric_dict.get('loss_per_fp', -1.0)
   # print(f"--- calculate_profit_based: profit_per_tp:{profit_per_tp} / loss_per_fp:{loss_per_fp} ---")
    total_profit = (tp * profit_per_tp) + (fp * loss_per_fp)
    return total_profit / len(y_true)
"""

# Définissez d'abord un callback personnalisé
def custom_callback_early_stopping(env):
    if env.iteration == env.best_iteration:
        print(f"Best iteration so far: {env.best_iteration} with score: {env.best_score}")
    if env.iteration == env.end_iteration - 1:
        print(f"Early stopping at iteration {env.iteration}")
    return False

import cupy as cp

def optunaScore_precision_recall_tp_ratio(y_true, y_pred_proba, metric_dict):
    threshold = metric_dict.get('threshold', 0.5)
    print(f"--- optunaScore_precision_recall_tp_ratio avec {threshold}---")
    y_pred = (y_pred_proba > threshold).astype(int)
    return calculate_precision_recall_tp_ratio_gpu(y_true, y_pred, metric_dict)

def calculate_precision_recall_tp_ratio_gpu(y_true, y_pred, metric_dict):
    y_true_gpu = cp.array(y_true)
    y_pred_gpu = cp.array(y_pred)

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
    if (profit_ratio <= 0
            or win_rate < 0.5
    )\
            :  # Ajout de la condition win_rate < 0.5
        return 0.0  # Retourner directement 0.0 si ces conditions sont remplies

    combined_score = (profit_ratio * profit_ratio_weight +
                      win_rate * win_rate_weight +
                      selectivity * selectivity_weight)

    # Normaliser le score
    sum_of_weights = profit_ratio_weight + win_rate_weight + selectivity_weight
    normalized_score = combined_score / sum_of_weights if sum_of_weights > 0 else 0

    return float(normalized_score)  # Assurer que nous retournons un float Python

def xgb_custom_metric_precision_recall_tp_ratio(preds, dtrain, metric_dict):
    y_true = dtrain.get_label()
    threshold = metric_dict.get('threshold', 0.5)
    y_pred = (preds > threshold).astype(int)
    score = calculate_precision_recall_tp_ratio_gpu(y_true, y_pred, metric_dict)
    return 'custom_metric', score
    #return 'combined_metric', score
def calculate_profit_based_gpu(y_true, y_pred, metric_dict):
    # Convertir les données d'entrée en matrices GPU
    y_true_gpu = cp.array(y_true)
    y_pred_gpu = cp.array(y_pred)

    # Calculer TN, FP, FN, TP sur le GPU
    tp = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 1))
    fp = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 1))
    tn = cp.sum((y_true_gpu == 0) & (y_pred_gpu == 0))
    fn = cp.sum((y_true_gpu == 1) & (y_pred_gpu == 0))

    # Récupérer les valeurs monétaires depuis metric_dict
    profit_per_tp = metric_dict.get('profit_per_tp', 1.0)
    loss_per_fp = metric_dict.get('loss_per_fp', -1.0)


    total_samples = len(y_true_gpu)
    #total_trades = tp + fp
    if total_samples == 0:
        normalized_profit = 0.0
    else:
        # Calculer le profit total
        total_profit = (tp * profit_per_tp) + (fp * loss_per_fp)

        # Normaliser le profit par le nombre total de prédictions
        #normalized_profit = total_profit / total_samples
        normalized_profit = total_profit / total_samples
    # Retourner le résultat en tant que valeur CPU
    return normalized_profit.get()
def optunaScore_profit_based(y_true, y_pred_proba, metric_dict):
    threshold = metric_dict.get('threshold', 0.7)
    print(f"--- optunaScore_profit_based avec {threshold}---")
    y_pred = (y_pred_proba > threshold).astype(int)
    return calculate_profit_based_gpu(y_true, y_pred, metric_dict)

def xgb_custom_metric_profit_based(preds, dtrain, metric_dict):
    y_true = dtrain.get_label()
    threshold = metric_dict.get('threshold', 0.7)
    y_pred = (1.0 / (1.0 + np.exp(-preds)) > threshold).astype(int)
    score = calculate_profit_based_gpu(y_true, y_pred, metric_dict)
    return 'profit_metric', score


def objective(trial, X_train, y_train, device,
              num_boost_min, num_boost_max, nb_split_tscv,
              use_optimized_threshold, learning_curve_enabled,
              use_auc_roc, score_methodAL, maskShap, metric_dict,weight_param):
    params = {
        'max_depth': trial.suggest_int('max_depth', 8, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.05, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.9),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 0.9),
        'objective': 'binary:logistic',
        'random_state': random_state_seed,
        'tree_method': 'hist',
        'device': device,
    }

    X_train = X_train.loc[:, maskShap]

    # Initialiser les compteurs de TP et FP
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    total_samples = 0

    # Fonction englobante qui intègre metric_dict




    threshold_value = trial.suggest_float('threshold', weight_param['threshold']['min'],
                                              weight_param['threshold']['max'])

    # Sélection de la fonction de métrique appropriée
    if score_methodAL == CombinedMetric.PR_RECALL_TP:
        selected_metric = optunaScore_precision_recall_tp_ratio

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

    elif score_methodAL == CombinedMetric.PROFIT_BASED:
        selected_metric = optunaScore_profit_based
        # Définir les paramètres spécifiques pour la métrique basée sur le profit
        metric_dict = {
            'threshold': threshold_value,
            'profit_per_tp': trial.suggest_float('profit_per_tp', weight_param['profit_per_tp']['min'], weight_param['profit_per_tp']['max']),
            'loss_per_fp': trial.suggest_float('loss_per_fp', weight_param['loss_per_fp']['min'], weight_param['loss_per_fp']['max']),
        }

    else:
        raise ValueError(f"Méthode de métrique combinée non reconnue: {score_methodAL}")

    # Enrichir le metric_dict avec les poids


    num_boost_round = trial.suggest_int('num_boost_round', num_boost_min, num_boost_max)

    scores = []
    last_score = None
    optimal_thresholds = []
    learning_curve_data_list = []

    # Fonction englobante qui intègre metric_dict
    if score_methodAL == CombinedMetric.PR_RECALL_TP:
        custom_metric = lambda predtTrain, dtrain: xgb_custom_metric_precision_recall_tp_ratio(predtTrain, dtrain, metric_dict)
    elif score_methodAL == CombinedMetric.PROFIT_BASED:
        custom_metric = lambda predtTrain, dtrain: xgb_custom_metric_profit_based(predtTrain, dtrain, metric_dict)
    else:
        raise ValueError(f"Méthode de métrique combinée non reconnue: {score_methodAL}")


    if nb_split_tscv < 2:
        # Un seul split : 80% pour l'entraînement, 20% pour la validation
        X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

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
                maximize=True,  # pour maximiser feval_func
                early_stopping_rounds=100,
                custom_metric=custom_metric,
                verbose_eval=False,

                # callbacks=[custom_callback_early_stopping]
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
                val_score = selected_metric(
                    y_val_cv,
                    y_val_pred_proba,
                    metric_dict=metric_dict
                )
            scores.append(val_score)
            last_score = val_score

        except Exception as e:
            print(f"Error during training or evaluation: {e}")
            return float('-inf')  # Retourne un score très faible en cas d'erreur

    else:
        # Validation croisée avec TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=nb_split_tscv)

        for train_index, val_index in tscv.split(X_train):
            X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

            # Filtrer les valeurs 99
            mask_train = y_train_cv != 99
            X_train_cv, y_train_cv = X_train_cv[mask_train], y_train_cv[mask_train]
            mask_val = y_val_cv != 99
            X_val_cv, y_val_cv = X_val_cv[mask_val], y_val_cv[mask_val]

            if len(X_train_cv) == 0 or len(y_train_cv) == 0:
                print("Warning: Empty training set after filtering")
                continue

            # Recalculer les poids des échantillons pour l'ensemble d'entraînement du pli actuel
            sample_weights = compute_sample_weight('balanced', y=y_train_cv)

            # Créer les DMatrix pour XGBoost
            dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv, weight=sample_weights)
            dval = xgb.DMatrix(X_val_cv, label=y_val_cv)
            try:
                # Entraîner le modèle préliminaire
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dtrain, 'train'), (dval, 'eval')],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                    maximize=True,
                    custom_metric=custom_metric
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
                if use_optimized_threshold:
                    optimal_threshold = optimize_threshold(y_val_cv, y_val_pred_proba)
                    optimal_thresholds.append(optimal_threshold)

                if use_auc_roc:
                    val_score = roc_auc_score(y_val_cv, y_val_pred_proba)
                else:
                    val_score = selected_metric(
                        y_val_cv,
                        y_val_pred_proba,
                        metric_dict=metric_dict
                    )
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
                        weight_dict, selected_metric, metric_dict,custom_metric
                    )
                
                    # Ajouter les données pour ce split
                    learning_curve_data_list.append(split_scores)
                    """

            except Exception as e:
                print(f"Error during training or evaluation: {e}")
                continue

    if not scores:
        return float('-inf')  # Retourne un score très faible si tous les plis ont échoué

    mean_cv_score = np.mean(scores)
    score_variance = np.var(scores)
    score_std = np.std(scores)

    print("Scores:", scores)
    print(f"Score mean: {mean_cv_score:.6f}")
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
    trial.set_user_attr('tp_difference', total_tp - total_fp)
    trial.set_user_attr('tp_percentage', tp_percentage)

    if use_optimized_threshold and optimal_thresholds:
        average_optimal_threshold = np.mean(optimal_thresholds)
        trial.set_user_attr('average_optimal_threshold', average_optimal_threshold)
    else:
        trial.set_user_attr('average_optimal_threshold', metric_dict['threshold'])

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

    return mean_cv_score, metric_dict

########################################
#########   END FUNCTION DEF   #########
########################################

def train_and_evaluate_XGBOOST_model(
        df=None,
        n_trials_optimization=4,
        device='cuda',
        use_optimized_threshold=False,
        num_boost_min=400,
        num_boost_max=1000,
        nb_split_tscv=12,
        nanvalue_to_newval=None,
        learning_curve_enabled=False,
        use_auc_roc=False,
        feature_columns=None,
        score_methodAL=CombinedMetric.PR_RECALL_TP,
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
        train_df, test_df = split_sessions(df, test_size=0.2, min_train_sessions=2, min_test_sessions=2)
    except ValueError as e:
        print(f"Erreur lors de la division des sessions : {e}")
        sys.exit(1)

    # Préparation des features et de la cible
    X_train = train_df[feature_columns]
    y_train = train_df['class_binaire']
    X_test = test_df[feature_columns]
    y_test = test_df['class_binaire']


    print(
        f"\nValeurs NaN : X_train={X_train.isna().sum().sum()}, y_train={y_train.isna().sum()}, X_test={X_test.isna().sum().sum()}, y_test={y_test.isna().sum()}\n")



    # Suppression des échantillons avec la classe 99
    mask_train = y_train != 99
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    mask_test = y_test != 99
    X_test, y_test = X_test[mask_test], y_test[mask_test]

    # Affichage de la distribution des classes
    print("Distribution des trades (excluant les 99):")
    trades_distribution = y_train.value_counts(normalize=True)
    trades_counts = y_train.value_counts()
    print(f"Trades échoués [0]: {trades_distribution.get(0, 0) * 100:.2f}% ({trades_counts.get(0, 0)} trades)")
    print(f"Trades réussis [1]: {trades_distribution.get(1, 0) * 100:.2f}% ({trades_counts.get(1, 0)} trades)")

    # Vérification de l'équilibre des classes
    total_trades_train = y_train.count()
    total_trades_test = y_test.count()

    print(f"Nombre total de trades pour l'ensemble d'entrainement (excluant les 99): {total_trades_train}")
    print(f"Nombre total de trades pour l'ensemble de test (excluant les 99): {total_trades_test}")

    thresholdClassImb = 0.06
    class_difference = abs(trades_distribution.get(0, 0) - trades_distribution.get(1, 0))
    if class_difference >= thresholdClassImb:
        print(f"Erreur : Les classes ne sont pas équilibrées. Différence : {class_difference:.2f}")
        sys.exit(1)
    else:
        print(f"Les classes sont considérées comme équilibrées (différence : {class_difference:.2f})")
    feature_names = X_train.columns.tolist()

    # **Ajout de la réduction des features ici, avant l'optimisation**

    feature_names = X_train.columns.tolist()

    print_notification('###### FIN: CHARGER ET PRÉPARER LES DONNÉES  ##########', color="blue")

    # Début de l'optimisation
    print_notification('###### DÉBUT: OPTIMISATION BAYESIENNE ##########', color="blue")
    start_time = time.time()

    if score_methodAL == CombinedMetric.PR_RECALL_TP:
        metric_dict_prelim = {
            'threshold': 0.5,
            'profit_ratio_weight': 0.3,
            'win_rate_weight': 0.5,
            'selectivity_weight': 0.2
        }

    elif score_methodAL == CombinedMetric.PROFIT_BASED:
        selected_metric = optunaScore_profit_based
        # Définir les paramètres spécifiques pour la métrique basée sur le profit
        metric_dict_prelim = {
            'threshold': 0.5,
            'profit_per_tp': 1,
            'loss_per_fp': -1.1,
        }

    if score_methodAL == CombinedMetric.PR_RECALL_TP:
        custom_metric = lambda predtTest, dtest: xgb_custom_metric_precision_recall_tp_ratio(predtTest, dtest, metric_dict_prelim)
    elif score_methodAL == CombinedMetric.PROFIT_BASED:
        custom_metric = lambda predtTest, dtest: xgb_custom_metric_profit_based(predtTest, dtest, metric_dict)
    else:
        raise ValueError(f"Méthode de métrique combinée non reconnue: {score_methodAL}")

    maskShap=train_preliminary_model_with_tscv(X_train, y_train, preShapImportance,custom_metric)

    # Au début de votre script principal, ajoutez ceci :
    global metric_dict
    metric_dict = {}

    # Ensuite, modifiez votre code comme suit :


    def objective_wrapper(trial):
        global metric_dict
        score, updated_metric_dict = objective(
            trial,
            X_train,
            y_train,
            device,
            num_boost_min,
            num_boost_max,
            nb_split_tscv,
            use_optimized_threshold,
            learning_curve_enabled,
            use_auc_roc,
            score_methodAL,
            maskShap,
            metric_dict.copy(),
            weight_param
        )
        metric_dict.update(updated_metric_dict)
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(
        objective_wrapper,
        n_trials=n_trials_optimization,
        callbacks=[lambda study, trial: print_callback(study, trial, X_train, y_train)]
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

    # Entraînement du modèle final avec les features réduites
    best_trial = study.best_trial

    # Créer une copie de best_params
    best_params = study.best_params.copy()

    # Extraire num_boost_round de best_params
    num_boost_round = best_params.pop('num_boost_round', None)

    if num_boost_round is None:
        raise ValueError("num_boost_round n'est pas présent dans best_params")

    # Filtrer best_params pour ne conserver que les paramètres reconnus par XGBoost
    xgb_valid_params = [
        'max_depth', 'learning_rate', 'min_child_weight', 'subsample',
        'colsample_bytree', 'colsample_bylevel', 'objective', 'eval_metric',
        'random_state',  'tree_method', 'device'
    ]

    best_params = {key: value for key, value in best_params.items() if key in xgb_valid_params}

    # Ajouter les autres paramètres nécessaires si besoin
    best_params['objective'] = 'binary:logistic'
    #best_params['eval_metric'] = 'auc'
    # Fonction englobante qui intègre metric_dict
    if score_methodAL == CombinedMetric.PR_RECALL_TP:
        custom_metric = lambda predtTest, dtest: xgb_custom_metric_precision_recall_tp_ratio(predtTest, dtest, metric_dict)
    elif score_methodAL == CombinedMetric.PROFIT_BASED:
        custom_metric = lambda predtTest, dtest: xgb_custom_metric_profit_based(predtTest, dtest, metric_dict)
    else:
        raise ValueError(f"Méthode de métrique combinée non reconnue: {score_methodAL}")
    best_params['tree_method'] = 'hist'
    best_params['device'] = device

    # Afficher les paramètres finaux pour vérification
    print("Paramètres utilisés pour l'entraînement du modèle final:")
    print(best_params)

    # Calcul des poids des classes
    # Recalculer les poids des échantillons pour l'ensemble d'entraînement
    sample_weights = compute_sample_weight('balanced', y=y_train)

    # Créer les DMatrix pour XGBoost en utilisant sample_weights
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)

    # Entraîner le modèle avec les paramètres optimaux
    final_model = xgb.train(
        best_params,
        dtrain,  # 'dtrain' contient déjà les poids des échantillons
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train')],
        early_stopping_rounds=100,
        maximize=True,  # pour maximiser feval_func
        custom_metric=custom_metric,
        verbose_eval=False
        # callbacks=[custom_callback_early_stopping]
    )

    print_notification('###### FIN: ENTRAINEMENT DU MODÈLE FINAL ##########', color="blue")
    # Analyser l'impact des valeurs NaN
    print_notification('###### DEBUT: ANALYSE DE L\'IMPACT DES VALEURS NaN DU MOBEL FINAL (ENTRAINEMENT) ##########', color="blue")


    # Appeler la fonction d'analyse
    analyze_nan_impact(final_model, X_train, feature_names, nan_value)

    print_notification('###### FIN: ANALYSE DE L\'IMPACT DES VALEURS NaN DU MOBEL FINAL (ENTRAINEMENT) ##########', color="blue")
    
    # Prédiction et évaluation
    print_notification('###### DEBUT: PREDICTION ET EVALUATION DU MOBEL FINAL (TEST) ##########', color="blue")
    X_test = X_test.loc[:, maskShap]

    # Créer le DMatrix pour les données de test
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Obtenir les probabilités prédites pour la classe positive
    y_pred_proba = final_model.predict(dtest)

    # Appliquer un seuil optimal pour convertir les probabilités en classes
    y_pred = (y_pred_proba > optimal_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("Accuracy sur les données de test:", accuracy)
    print("AUC-ROC sur les données de test:", auc)
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred))
    print_notification('###### FIN: PREDICTION ET EVALUATION DU MODELE FINAL (TEST) ##########', color="blue")


    
    ###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES sur XTEST ##########

    plot_calibration_curve(y_test, y_pred_proba, X_test=X_test,
                           feature_name='deltaTimestampOpeningSection5index',
                           n_bins=200, strategy='uniform',
                           optimal_threshold=optimal_threshold, show_histogram=True,user_input=user_input)
    print_notification('###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES ET COURBE ROC ##########',
                       color="blue")



    print("\nDistribution des probabilités prédites sur XTest:")
    print(f"seuil: {optimal_threshold}")
    print(f"Min : {y_pred_proba.min():.4f}")
    print(f"Max : {y_pred_proba.max():.4f}")
    print(f"Moyenne : {y_pred_proba.mean():.4f}")
    print(f"Médiane : {np.median(y_pred_proba):.4f}")

    # Compter le nombre de prédictions dans différentes plages de probabilité
    ranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bisect.insort(ranges, optimal_threshold)
    ranges = sorted(set(ranges))
    hist, _ = np.histogram(y_pred_proba, bins=ranges)

    print("\nDistribution des probabilités prédites avec TP et FP:")
    for i in range(len(ranges) - 1):
        mask = (y_pred_proba >= ranges[i]) & (y_pred_proba < ranges[i + 1])
        predictions_in_range = y_pred_proba[mask]
        true_values_in_range = y_test[mask]

        tp = np.sum((predictions_in_range >= optimal_threshold) & (true_values_in_range == 1))
        fp = np.sum((predictions_in_range >= optimal_threshold) & (true_values_in_range == 0))
        total_trades=tp+fp
        win_rate = tp / total_trades * 100 if total_trades > 0 else 0
        print(f"Probabilité {ranges[i]:.4f} - {ranges[i + 1]:.4f} : {hist[i]} prédictions, TP: {tp}, FP: {fp}, Winrate: {win_rate:.2f}%")

    print("Statistiques de y_pred_proba:")
    print(f"Nombre d'éléments: {len(y_pred_proba)}")
    print(f"Min: {np.min(y_pred_proba)}")
    print(f"Max: {np.max(y_pred_proba)}")
    print(f"Valeurs uniques: {np.unique(y_pred_proba)}")
    print(f"Y a-t-il des NaN?: {np.isnan(y_pred_proba).any()}")

    # Définissez min_precision si vous voulez l'utiliser, sinon laissez-le à None
    min_precision = None  # ou une valeur comme 0.7 si vous voulez l'utiliser

    # Création de la figure avec trois sous-graphiques côte à côte
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

    # Sous-graphique 1 : Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    ax1.plot(fpr, tpr, color='blue', linestyle='-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.grid(True)
    ax1.legend(loc='lower right', fontsize=10)

    # Sous-graphique 2 : Distribution des probabilités prédites
    bins = np.linspace(y_pred_proba.min(), y_pred_proba.max(), 100)

    ax2.hist(y_pred_proba[y_pred_proba <= optimal_threshold], bins=bins, color='orange',
             label=f'Prédictions ≤ {optimal_threshold:.4f}', alpha=0.7)
    ax2.hist(y_pred_proba[y_pred_proba > optimal_threshold], bins=bins, color='blue',
             label=f'Prédictions > {optimal_threshold:.4f}', alpha=0.7)
    ax2.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Seuil de décision ({optimal_threshold:.4f})')
    ax2.set_title('Proportion de prédictions négatives (fonction du choix du seuil) sur XTest', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Proportion de prédictions négatives (fonction du choix du seuil)', fontsize=12)
    ax2.set_ylabel('Nombre de prédictions', fontsize=12)

    # Ajout des annotations pour les comptes
    num_below = np.sum(y_pred_proba <= optimal_threshold)
    num_above = np.sum(y_pred_proba > optimal_threshold)
    ax2.text(0.05, 0.95, f'Count ≤ {optimal_threshold:.4f}: {num_below}', color='orange', transform=ax2.transAxes,
             va='top')
    ax2.text(0.05, 0.90, f'Count > {optimal_threshold:.4f}: {num_above}', color='blue', transform=ax2.transAxes,
             va='top')

    ax2.legend(fontsize=10)

    # Sous-graphique 3 : Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
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

    print_notification('###### FIN: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES ET COURBE ROC ##########',
                       color="blue")

    ###### DEBUT: COMPARAISON DES DISTRIBUTIONS RÉELLES ET PRÉDITES ##########
    print_notification('###### DEBUT: COMPARAISON DES DISTRIBUTIONS RÉELLES ET PRÉDITES ##########', color="blue")
    # Distribution réelle des classes dans l'ensemble de test
    real_distribution = y_test.value_counts(normalize=True)
    print("Distribution réelle des classes dans l'ensemble de test:")
    print(real_distribution)
    print(f"Nombre total d'échantillons dans l'ensemble de test: {len(y_test)}")

    # Distribution des classes prédites
    predicted_distribution = pd.Series(y_pred).value_counts(normalize=True)
    print("\nDistribution des classes prédites:")
    print(predicted_distribution)
    print(f"Nombre total de prédictions: {len(y_pred)}")

    # Comparaison côte à côte
    comparison = pd.DataFrame({
        'Réelle (%)': real_distribution * 100,
        'Prédite (%)': predicted_distribution * 100
    })
    print("\nComparaison côte à côte (en pourcentage):")
    print(comparison)

    # Visualisation de la comparaison
    plt.figure(figsize=(10, 6))
    comparison.plot(kind='bar')
    plt.title('Comparaison des distributions de classes réelles et prédites')
    plt.xlabel('Classe')
    plt.ylabel('Pourcentage')
    plt.legend(['Réelle', 'Prédite'])
    plt.tight_layout()
    plt.savefig('class_distribution_comparison.png')
    plt.close()

    # Calcul de la différence absolue entre les distributions
    diff = abs(real_distribution - predicted_distribution)
    print("\nDifférence absolue entre les distributions:")
    print(diff)
    print(f"Somme des différences absolues: {diff.sum():.4f}")
    print_notification('###### FIN: COMPARAISON DES DISTRIBUTIONS RÉELLES ET PRÉDITES ##########', color="blue")
    ###### FIN: COMPARAISON DES DISTRIBUTIONS RÉELLES ET PRÉDITES ##########

    ###### DEBUT: ANALYSE DES ERREURS ##########
    print_notification('###### DEBUT: ANALYSE DES ERREURS ##########', color="blue")
    # Analyse des erreurs
    feature_names = X_test.columns.tolist()

    results_df, error_df = analyze_errors(X_test, y_test, y_pred, y_pred_proba, feature_names)

    # Sauvegarde des résultats pour une analyse ultérieure si nécessaire
    results_df.to_csv('model_results_analysis.csv', index=False)
    error_df.to_csv('model_errors_analysis.csv', index=False)

    # Visualisations supplémentaires
    plt.figure(figsize=(12, 10))
    sns.heatmap(error_df[feature_names].corr(), annot=False, cmap='coolwarm')
    plt.title('Corrélations des features pour les erreurs')
    plt.savefig('error_features_correlation.png')
    plt.close()

    # Identification des erreurs
    errors = X_test[y_test != y_pred]
    print("Nombre d'erreurs:", len(errors))

    print_notification('###### FIN: ANALYSE DES ERREURS ##########', color="blue")
    ###### FIN: ANALYSE DES ERREURS ##########

    ###### DEBUT: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########
    print_notification('###### DEBUT: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########', color="blue")
    # Conversion du modèle en XGBClassifier:
    print("Conversion du modèle en XGBClassifier:")
    # Copier les meilleurs hyperparamètres optimisés pour les utiliser avec XGBClassifier.
    best_params_sklearn = best_params.copy()
    # Ajouter le nombre d'arbres (n_estimators) déterminé par num_boost_round.
    best_params_sklearn['n_estimators'] = num_boost_round
    # Ajuster le ratio de poids des classes pour traiter les déséquilibres (classe 1 par rapport à classe 0).

    # Calculer les poids des échantillons pour l'ensemble d'entraînement du pli actuel
    sample_weights = compute_sample_weight('balanced', y=y_train)

    # Instancier un classificateur XGBoost avec les hyperparamètres optimisés.
    xgb_classifier = xgb.XGBClassifier(**best_params_sklearn)
    # Entraîner le modèle sur les données d'entraînement avec des poids pour chaque échantillon.
    xgb_classifier.fit(X_train, y_train, sample_weight=sample_weights)

    # Exemple d'utilisation :
    analyze_predictions_by_range(X_test, y_pred_proba, xgb_classifier, prob_min=0.7, prob_max=1.00, top_n_features=20)

    # Utilisation de SHAP pour évaluer l'importance des features
    explainer_Test = shap.TreeExplainer(xgb_classifier)
    shap_values = explainer_Test.shap_values(X_test)

    feature_importance = np.abs(shap_values).mean(axis=0)
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
        xgb_classifier=xgb_classifier,
        confident_errors=confident_errors,
        X_test=X_test,
        feature_names=feature_names,
        n=5,
        explainer=explainer_Test)
    #plot_confident_errors(xgb_classifier, confident_errors, X_test, X_test.columns,explainer_Test)


    # Exécution des analyses
    print("\nAnalyse des erreurs confiantes:")

    analyze_confident_errors(xgb_classifier=xgb_classifier,confident_errors=confident_errors,X_test=X_test,feature_names=feature_names,important_features=important_features,n=5,explainer=explainer_Test)
    correct_predictions = results_df[results_df['true_label'] == results_df['predicted_label']]
    print("\nComparaison des erreurs vs prédictions correctes:")
    compare_errors_vs_correct(confident_errors.head(30), correct_predictions, X_test, important_features)
    print("\nAnalyse SHAP terminée. Les visualisations ont été sauvegardées.")
    print("\nAnalyse terminée. Les visualisations ont été sauvegardées.")
    print_notification('###### FIN: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########', color="blue")
    ###### FIN: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########

    ###### DEBUT: ANALYSE SHAP ##########
    print_notification('###### DEBUT: ANALYSE SHAP ##########', color="blue")

    # Calcul des valeurs SHAP moyennes pour chaque feature
    shap_mean = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': shap_mean,
        'effect': np.mean(shap_values, axis=0)  # Effet moyen (positif ou négatif)
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(12, 10))

    # Utilisation de deux couleurs pour montrer l'effet positif/négatif
    colors = ['#FF9999', '#66B2FF']  # Rouge clair pour négatif, bleu clair pour positif
    bars = plt.barh(feature_importance['feature'], feature_importance['importance'],
                    color=[colors[1] if x > 0 else colors[0] for x in feature_importance['effect']])

    plt.title("Feature Importance Determined By SHAP Values", fontsize=16)
    plt.xlabel('Mean |SHAP Value| (Average Impact on Model Output Magnitude)', fontsize=12)
    plt.ylabel('Features', fontsize=12)

    # Ajout de la légende
    plt.legend([plt.Rectangle((0, 0), 1, 1, fc=colors[0]), plt.Rectangle((0, 0), 1, 1, fc=colors[1])],
               ['Diminue la probabilité de succès', 'Augmente la probabilité de succès'],
               loc='lower right', fontsize=10)

    # Annotations pour expliquer l'interprétation
    plt.text(0.5, 1.05, "La longueur de la barre indique l'importance globale de la feature.\n"
                        "La couleur indique si la feature tend à augmenter (bleu) ou diminuer (rouge) la probabilité de succès du trade.",
             horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig('shap_importance_binary_trade.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Graphique SHAP sauvegardé sous 'shap_importance_binary_trade.png'")

    # 2. Valeurs SHAP moyennes absolues
    print("\nAnalyse des valeurs SHAP moyennes absolues:")

    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    # Visualisation des valeurs SHAP moyennes absolues
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance_df['feature'][:30], feature_importance_df['importance'][:30])
    plt.xticks(rotation=45, ha='right')
    plt.title("Top 30 Features par Importance SHAP (valeurs moyennes absolues)")
    plt.tight_layout()
    plt.savefig('shap_importance_mean_abs.png')
    plt.close()

    # Analyse supplémentaire : pourcentage cumulatif de l'importance
    feature_importance_df['cumulative_importance'] = feature_importance_df['importance'].cumsum() / feature_importance_df['importance'].sum()

    # Utiliser les résultats de l'analyse SHAP pour identifier les top features
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    top_10_features = feature_importance_df['feature'].head(10).tolist()

    print("Top 10 features basées sur l'analyse SHAP:")
    print(top_10_features)

    # Créer un subplot pour chaque feature
    fig, axes = plt.subplots(5, 2, figsize=(20, 25))
    fig.suptitle('Distribution des 10 features les plus importantes par type d\'erreur', fontsize=16)

    for i, feature in enumerate(top_10_features):
        row = i // 2
        col = i % 2
        sns.boxplot(x='error_type', y=feature, data=results_df, ax=axes[row, col])
        axes[row, col].set_title(f'{i+1}. {feature}')
        axes[row, col].set_xlabel('')
        if col == 0:
            axes[row, col].set_ylabel('Valeur')
        else:
            axes[row, col].set_ylabel('')

    plt.tight_layout()

    # Sauvegarde du graphique combiné des 10 features
    plt.savefig('top_10_features_distribution_by_error.png', dpi=300, bbox_inches='tight')
    print("Graphique combiné sauvegardé sous 'top_10_features_distribution_by_error.png'")
    plt.close()

    # Trouver combien de features sont nécessaires pour expliquer 80% de l'importance
    features_for_80_percent = feature_importance_df[feature_importance_df['cumulative_importance'] <= 0.8].shape[0]
    print(f"\nNombre de features nécessaires pour expliquer 80% de l'importance : {features_for_80_percent}")
    print_notification('###### FIN: ANALYSE SHAP ##########', color="blue")
    ###### FIN: ANALYSE SHAP ##########

    ###### DEBUT: COMPARAISON AVEC FEATURE_IMPORTANCES_ DE XGBOOST ##########
    print_notification('###### DEBUT: COMPARAISON AVEC FEATURE_IMPORTANCES_ DE XGBOOST ##########', color="blue")
    # 3. Comparaison avec feature_importances_ de XGBoost
    xgb_feature_importance = xgb_classifier.feature_importances_
    xgb_importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'xgb_importance': xgb_feature_importance
    })
    xgb_importance_df = xgb_importance_df.sort_values('xgb_importance', ascending=False)

    print("\nComparaison des top 30 features : SHAP vs XGBoost feature_importances_")
    comparison_df = pd.merge(feature_importance_df, xgb_importance_df, on='feature')
    comparison_df = comparison_df.sort_values('importance', ascending=False).head(30)
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(comparison_df)

    # Visualisation de la comparaison
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(comparison_df))
    plt.bar(x - width/2, comparison_df['importance'], width, label='SHAP')
    plt.bar(x + width/2, comparison_df['xgb_importance'], width, label='XGBoost')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Comparaison de l\'importance des features : SHAP vs XGBoost')
    plt.xticks(x, comparison_df['feature'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png')
    plt.close()
    print_notification('###### FIN: COMPARAISON AVEC FEATURE_IMPORTANCES_ DE XGBOOST ##########', color="blue")
    ###### FIN: COMPARAISON AVEC FEATURE_IMPORTANCES_ DE XGBOOST ##########
    #exit(1)
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
    sns.heatmap(interaction_df.loc[top_features, top_features],
                annot=True,
                cmap='coolwarm',
                fmt='d',#'.2f',
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
        'xgb_classifier': xgb_classifier,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

if __name__ == "__main__":
    # Demander à l'utilisateur s'il souhaite afficher les graphiques
    user_input = input(
        "Pour afficher les graphiques, appuyez sur 'd'. Sinon, appuyez sur 'Entrée' pour les enregistrer sans les afficher: ")

    FILE_NAME_ = "Step5_Step4_Step3_Step2_MergedAllFile_Step0_4_merged_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
    #FILE_NAME_ = "Step5_Step4_Step3_Step2_MergedAllFile_Step1_2_merged_extractOnlyFullSession_OnlyShort_feat.csv"
    DIRECTORY_PATH_ = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL_02102024\merge"
    FILE_PATH_ = os.path.join(DIRECTORY_PATH_, FILE_NAME_)

    DEVICE_ = 'cuda'
    USE_OPTIMIZED_THRESHOLD_ = False
    score_methodAL=CombinedMetric.PROFIT_BASED
    NUM_BOOST_MIN_ = 400
    NUM_BOOST_MAX_ = 850
    N_TRIALS_OPTIMIZATION_ =10
    NB_SPLIT_TSCV_ = 4
    NANVALUE_TO_NEWVAL_ = np.nan #900000.123456789
    LEARNING_CURVE_ENABLED = False
    random_state_seed = 30
    USE_AUC_ROC = False  # False combined metric
    # Définir les paramètres supplémentaires


    weight_param = {
        'threshold': {'min': 0.505, 'max': 0.505}, # total_trades = tp + fp
        'profit_ratio_weight': {'min': 0.4, 'max': 0.4}, # profit_ratio = (tp - fp) / total_trades
        'win_rate_weight': {'min': 0.45, 'max': 0.45}, #win_rate = tp / total_trades if total_trades
        'selectivity_weight': {'min': 0.075, 'max': 0.075}, #selectivity = total_trades / total_samples
        'profit_per_tp': {'min': 1, 'max': 1},
        'loss_per_fp': {'min': -1.1, 'max': -1.1}
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
            'class_binaire', 'date', 'trade_category', 'SessionStartEnd','candleDir',
             'deltaTimestampOpening',
             'deltaTimestampOpeningSection5min',
            #'deltaTimestampOpeningSection5index',
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

            ]]


    for top_N in [200]:
        top_N=None
        print(f"\n\n### Entraînement avec les top {top_N} features ###\n")
        results = train_and_evaluate_XGBOOST_model(
            df=df,
            n_trials_optimization=N_TRIALS_OPTIMIZATION_,
            device=DEVICE_,
            use_optimized_threshold=USE_OPTIMIZED_THRESHOLD_,
            num_boost_min=NUM_BOOST_MIN_,
            num_boost_max=NUM_BOOST_MAX_,
            nb_split_tscv=NB_SPLIT_TSCV_,
            nanvalue_to_newval=NANVALUE_TO_NEWVAL_,
            learning_curve_enabled=LEARNING_CURVE_ENABLED,
            use_auc_roc=USE_AUC_ROC,
            feature_columns=feature_columns,
            score_methodAL=score_methodAL,preShapImportance=1,
            user_input=user_input,
            weight_param=weight_param
        )

    if results is not None:
        print("Meilleurs hyperparamètres trouvés:", results['study'].best_params)
        print("Meilleur score:", results['study'].best_value)
        print("Seuil optimal:", results['optimal_threshold'])
    else:
        print("L'entraînement n'a pas produit de résultats.")

