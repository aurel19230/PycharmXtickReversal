import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from standardFunc import load_data, split_sessions, print_notification

import optuna
import time
from sklearn.utils.class_weight import compute_class_weight
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


def combined_metric(y_true, y_pred_proba, metric_dict):
    # Extraire les paramètres depuis le dictionnaire
    threshold = metric_dict.get('threshold', 0.5)
    profit_ratio = metric_dict.get('profit_ratio', 1.1)
    tp_weight = metric_dict.get('tp_weight', 0.4)
    fp_penalty = metric_dict.get('fp_penalty', 0.2)

    # Imprimer les valeurs
    #print(f"Threshold: {threshold}")
    #print(f"Profit Ratio: {profit_ratio}")
    #print(f"TP Weight: {tp_weight}")
    #print(f"FP Penalty: {fp_penalty}")

    # Convertir les probabilités en prédictions binaires avec le seuil donné
    y_pred = (y_pred_proba > threshold).astype(int)

    # Calculer l'AUC directement avec les probabilités
    auc = roc_auc_score(y_true, y_pred_proba)

    # Calculer les autres métriques avec les prédictions binaires
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculer la matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculer le taux de faux positifs et le taux de vrais positifs
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculer le profit potentiel (hypothétique)
    potential_profit = (tp * profit_ratio - fp) / (tp + fp) if (tp + fp) > 0 else 0

    # Calculer un score pour les vrais positifs et faux positifs
    tp_score = tpr * tp_weight
    fp_score = (1 - fpr) * fp_penalty

    # Définir les poids utilisés
    auc_weight = 0.05  # Réduit pour laisser plus de place aux métriques financières
    precision_weight = 0.15
    recall_weight = 0.25 #accorde une importance significative à la capacité du modèle à capturer des opportunités de trading réussies.
    f1_weight = 0.05
    potential_profit_weight = 0.3
    tp_score_weight = tp_weight
    fp_score_weight = fp_penalty
    # Calculer la somme de tous les poids
    sum_of_weights = (auc_weight + precision_weight + recall_weight + f1_weight +
                      potential_profit_weight + tp_score_weight + fp_score_weight)

    # Calculer le score combiné
    combined_score = (
        (auc * auc_weight) +
        (precision * precision_weight) +
        (recall * recall_weight) +
        (f1 * f1_weight) +
        (potential_profit * potential_profit_weight) +
        tp_score +
        fp_score
    )

    # Normaliser le score
    normalized_score = combined_score / sum_of_weights

    return normalized_score


class CustomCallback(TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        train_score = evals_log['train']['aucpr'][-1]
        valid_score = evals_log['eval']['aucpr'][-1]
        if epoch % 10 == 0 and train_score - valid_score > 1:  # on le met à 1 pour annuler ce test. On se base sur l'early stopping désormais
            print(f"Arrêt de l'entraînement à l'itération {epoch}. Écart trop important.")
            return True
        return False


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

def calculate_scores_for_cv_split(params, num_boost_round, X_train, y_train, X_val, y_val, weight_dict, combined_metric,metric_dict):
    """
    Calcule les scores d'entraînement et de validation pour un split de validation croisée.
    """
    # Créer des DMatrix pour l'entraînement et la validation
    sample_weights = np.array([weight_dict[label] for label in y_train])
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Entraîner le modèle
    booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    # Prédire sur les ensembles d'entraînement et de validation
    train_pred = booster.predict(dtrain)
    val_pred = booster.predict(dval)

    # Calculer les scores
    #train_score = combined_metric(y_train, train_pred, threshold=threshold)
    #val_score = combined_metric(y_val, val_pred, threshold=threshold)


    train_score = combined_metric(y_train, train_pred, metric_dict=metric_dict)

    val_score = combined_metric(y_val, val_pred, metric_dict=metric_dict)

    return {
        'train_sizes': [len(X_train)],  # Ajout de cette ligne
        'train_scores_mean': [train_score],  # Modification ici
        'val_scores_mean': [val_score]  # Modification ici
    }

def objective(trial, class_weights, weight_dict, X_train, y_train, device, num_boost_min, num_boost_max, nb_split_tscv, use_optimized_threshold, learning_curve_enabled=False, metric_dict=None):
    params = {
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.001, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 50, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.85),
        'gamma': trial.suggest_float('gamma', 1, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1, 20, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 30.0, log=True),
        'max_delta_step': trial.suggest_float('max_delta_step', 2, 20),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 0.9),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 0.9),
        'max_leaves': trial.suggest_int('max_leaves', 10, 100),
        'max_bin': trial.suggest_int('max_bin', 128, 512),
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'random_state': 42,
        'scale_pos_weight': class_weights[1] / class_weights[0],
        'tree_method': 'hist',
        'device': device,
    }


    fixed_threshold = metric_dict.get('threshold', 0.5)
    # Paramètres supplémentaires optimisables



    # Vous pouvez choisir de laisser profit_ratio, tp_weight, et fp_penalty fixes ou les inclure dans l'optimisation
    # Ici, nous les laissons fixes via metric_dict


    num_boost_round = trial.suggest_int('num_boost_round', num_boost_min, num_boost_max)

    tscv = TimeSeriesSplit(n_splits=nb_split_tscv)
    scores = []
    last_score = None
    optimal_thresholds = []
    learning_curve_data_list = []

    for train_index, val_index in tscv.split(X_train):
        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

        # Filtrage des valeurs 99
        mask_train = y_train_cv != 99
        X_train_cv, y_train_cv = X_train_cv[mask_train], y_train_cv[mask_train]
        mask_val = y_val_cv != 99
        X_val_cv, y_val_cv = X_val_cv[mask_val], y_val_cv[mask_val]

        if len(X_train_cv) == 0 or len(y_train_cv) == 0:
            print("Warning: Empty training set after filtering")
            continue

        sample_weights = np.array([weight_dict[label] for label in y_train_cv])
        dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv, weight=sample_weights)
        dval = xgb.DMatrix(X_val_cv, label=y_val_cv)

        custom_callback = CustomCallback()

        try:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, 'train'), (dval, 'eval')],
                early_stopping_rounds=100,
                callbacks=[custom_callback],
                verbose_eval=False,
            )

            y_val_pred_proba = model.predict(dval)

            if use_optimized_threshold:
                optimal_threshold = optimize_threshold(y_val_cv, y_val_pred_proba)
                optimal_thresholds.append(optimal_threshold)

            # Calcul du score combiné avec les paramètres optimisés
            val_score = combined_metric(
                y_val_cv,
                y_val_pred_proba,
                metric_dict=metric_dict
            )
            scores.append(val_score)
            last_score = val_score

            if learning_curve_enabled:
                # Calculer les scores pour ce split CV
                split_scores = calculate_scores_for_cv_split(
                    params,
                    num_boost_round,
                    X_train_cv, y_train_cv,
                    X_val_cv, y_val_cv,
                    weight_dict, combined_metric,metric_dict
                )

                # Ajouter les données pour ce split
                learning_curve_data_list.append(split_scores)

        except Exception as e:
            print(f"Error during training or evaluation: {e}")
            continue

    if not scores:
        return float('-inf')  # Return worst possible score if all folds failed

    mean_cv_score = np.mean(scores)
    score_variance = np.var(scores)
    score_std = np.std(scores)  # Calcul de l'écart-type

    print("Scores:", scores)
    print(f"Score mean: {mean_cv_score:.6f}")
    print(f"Score variance: {score_variance:.6f}")
    print(f"Score standard deviation: {score_std:.6f}")

    trial.set_user_attr('last_score', scores[-1])
    trial.set_user_attr('score_variance', score_variance)
    trial.set_user_attr('score_std', score_std)  # Enregistrement de l'écart-type")


    trial.set_user_attr('last_score', last_score)
    mean_cv_score = np.mean(scores)


    if use_optimized_threshold:
        average_optimal_threshold = np.mean(optimal_thresholds)
        trial.set_user_attr('average_optimal_threshold', average_optimal_threshold)
    else:
        trial.set_user_attr('average_optimal_threshold', fixed_threshold)

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

    return mean_cv_score

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

# Définir un callback pour afficher les progrès



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
import matplotlib.pyplot as plt
import shap
from PIL import Image
import os
import numpy as np


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
    print(f"\nStatistiques sur les règles de décision impliquant des NaN :")
    print(f"Nombre total d'arbres : {total_trees}")
    print(f"Nombre d'arbres impliquant des NaN : {trees_with_nan}")
    print(f"Pourcentage d'arbres impliquant des NaN : {(trees_with_nan / total_trees) * 100:.2f}%")
    print(f"Nombre de règles de décision importantes impliquant des NaN : {len(important_rules)}")
    print(f"Nombre moyen de règles importantes par arbre impliqué : {len(important_rules) / trees_with_nan:.2f}")

    # 5. Analyser l'importance des features avec des valeurs NaN en utilisant les valeurs SHAP
    # Calcul des valeurs SHAP si elles ne sont pas fournies
    if shap_values is None:
        print("Calcul des valeurs SHAP...")
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
            f"Importance des features (SHAP) et pourcentage de NaN (Features {start_idx + 1} à {min(end_idx, num_features)} \n (ensemble d'entrainement avec model final))",
            fontsize=14)
        plt.tight_layout()
        plt.savefig(f'nan_features_shap_importance_percentage_{i + 1}.png')
        plt.close()

    # Calcul de la corrélation entre l'importance (SHAP) et le pourcentage de NaN
    correlation = nan_fi_df['Importance'].corr(nan_fi_df['Percentage_NaN'])
    print(f"\nCorrélation entre l'importance des features (SHAP) et le pourcentage de NaN : {correlation:.4f}")

    # Visualisation de la relation entre l'importance (SHAP) et le pourcentage de NaN
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Percentage_NaN', y='Importance', data=nan_fi_df)
    plt.title('Relation entre le pourcentage de NaN et l\'importance des features (SHAP)')
    plt.xlabel('Pourcentage de NaN (%)')
    plt.ylabel('Importance (SHAP)')
    plt.tight_layout()
    plt.savefig('shap_importance_vs_percentage_nan.png')
    plt.close()



########################################
#########   END FUNCTION DEF   #########
########################################

def train_and_evaluate_XGBOOST_model(
        initial_df,
        n_trials_optimization=4,
        device='cuda',
        use_optimized_threshold=False,
        num_boost_min=400,
        num_boost_max=1000,
        nb_split_tscv=12,
        nanvalue_to_newval=None,
        learning_curve_enabled=False,
        metric_dict=None  # Nouvel argument pour les paramètres supplémentaires

):
    # Chargement et préparation des données
    df = initial_df
    # Gestion des valeurs NaN
    if nanvalue_to_newval is not None:
        # Remplacer les NaN par la valeur spécifiée
        df = df.fillna(nanvalue_to_newval)
        nan_value = nanvalue_to_newval
    else:
        # Garder les NaN tels quels
        nan_value = np.nan

    # Affichage des informations sur les NaN dans chaque colonne
    for column in df.columns:
        nan_count = df[column].isna().sum()
        print(f"Colonne: {column}, Nombre de NaN: {nan_count}")

    # Définition des colonnes de features et des colonnes exclues
    feature_columns = [col for col in df.columns if col not in [
        'class_binaire', 'date', 'trade_category', 'SessionStartEnd',
        'deltaTimestampOpening', 'deltaTimestampOpeningSection5min',
        'deltaTimestampOpeningSection5index', 'deltaTimestampOpeningSection30min',
        'range_strength', 'market_regimeADX'
    ]]

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

    # Calcul des poids des classes
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(zip(np.unique(y_train), class_weights))

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
    total_trades = y_train.count()
    print(f"Nombre total de trades (excluant les 99): {total_trades}")
    threshold = 0.06
    class_difference = abs(trades_distribution.get(0, 0) - trades_distribution.get(1, 0))
    if class_difference >= threshold:
        print(f"Erreur : Les classes ne sont pas équilibrées. Différence : {class_difference:.2f}")
        sys.exit(1)
    else:
        print(f"Les classes sont considérées comme équilibrées (différence : {class_difference:.2f})")

    print_notification('###### FIN: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")

    # Début de l'optimisation
    print_notification('###### DEBUT: OPTIMISATION BAYESIENNE ##########', color="blue")
    start_time = time.time()

    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(
            trial,
            class_weights,
            weight_dict,
            X_train,
            y_train,
            device,
            num_boost_min,
            num_boost_max,
            nb_split_tscv,
            use_optimized_threshold,
            learning_curve_enabled=learning_curve_enabled,
            metric_dict=metric_dict  # Passer les paramètres supplémentaires

        ),
        n_trials=n_trials_optimization,
        callbacks=[lambda study, trial: print_callback(study, trial, X_train, y_train)]
    )

    end_time = time.time()
    execution_time = end_time - start_time

    print("Optimisation terminée.")
    print("Meilleurs hyperparamètres trouvés: ", study.best_params)
    print("Meilleur score: ", study.best_value)
    print(f"Temps d'exécution total : {execution_time:.2f} secondes")
    print_notification('###### FIN: OPTIMISATION BAYESIENNE ##########', color="blue")

    # Entraînement du modèle final
    best_trial = study.best_trial
    optimal_threshold = best_trial.user_attrs['average_optimal_threshold']
    print(f"Seuil utilisé : {optimal_threshold:.4f}")

    best_params = study.best_params.copy()
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'auc'
    best_params['tree_method'] = 'hist'
    best_params['device'] = device

    num_boost_round = best_params.pop('num_boost_round')

    sample_weights = np.array([weight_dict[label] for label in y_train])
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dtest = xgb.DMatrix(X_test, label=y_test)

    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train')],
        verbose_eval=False
    )
    print_notification('###### FIN: ENTRAINEMENT DU MODELE FINAL ##########', color="blue")
    # Analyser l'impact des valeurs NaN
    print_notification('###### DEBUT: ANALYSE DE L\'IMPACT DES VALEURS NaN DU MOBEL FINAL (ENTRAINEMENT) ##########', color="blue")

    feature_names = X_train.columns.tolist()

    # Appeler la fonction d'analyse
    analyze_nan_impact(final_model, X_train, feature_names, nan_value)

    print_notification('###### FIN: ANALYSE DE L\'IMPACT DES VALEURS NaN DU MOBEL FINAL (ENTRAINEMENT) ##########', color="blue")

    # Prédiction et évaluation
    print_notification('###### DEBUT: PREDICTION ET EVALUATION DU MOBEL FINAL (TEST) ##########', color="blue")
    y_pred_proba = final_model.predict(dtest)
    y_pred = (y_pred_proba > optimal_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("Accuracy sur les données de test:", accuracy)
    print("AUC-ROC sur les données de test:", auc)
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred))
    print_notification('###### FIN: PREDICTION ET EVALUATION DU MODELE FINAL (TEST) ##########', color="blue")



    ###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES ##########
    print_notification('###### DEBUT: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES ##########', color="blue")
    print("\nDistribution des probabilités prédites :")
    print(f"Min : {y_pred_proba.min():.4f}")
    print(f"Max : {y_pred_proba.max():.4f}")
    print(f"Moyenne : {y_pred_proba.mean():.4f}")
    print(f"Médiane : {np.median(y_pred_proba):.4f}")

    # Compter le nombre de prédictions dans différentes plages de probabilité
    # Liste initiale des plages
    ranges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Insérer le seuil optimal
    bisect.insort(ranges, optimal_threshold)
    ranges = sorted(set(ranges))
    hist, _ = np.histogram(y_pred_proba, bins=ranges)
    for i in range(len(ranges) - 1):
        print(f"Probabilité {ranges[i]:.2f} - {ranges[i+1]:.2f} : {hist[i]} prédictions")

    # Ajoutez ceci juste après le calcul de y_pred_proba
    print("Statistiques de y_pred_proba:")
    print(f"Nombre d'éléments: {len(y_pred_proba)}")
    print(f"Min: {np.min(y_pred_proba)}")
    print(f"Max: {np.max(y_pred_proba)}")
    print(f"Valeurs uniques: {np.unique(y_pred_proba)}")
    print(f"Y a-t-il des NaN?: {np.isnan(y_pred_proba).any()}")

    # Plotting the distribution of predicted probabilities with color-coded bins
    plt.figure(figsize=(10, 6))

    # Define bins
    bins = np.linspace(y_pred_proba.min(), y_pred_proba.max(), 100)

    # Histogram for predictions below the threshold
    plt.hist(y_pred_proba[y_pred_proba <= optimal_threshold], bins=bins, color='orange',
             label=f'Prédictions ≤ {optimal_threshold:.4f}', alpha=0.7)

    # Histogram for predictions above the threshold
    plt.hist(y_pred_proba[y_pred_proba > optimal_threshold], bins=bins, color='blue',
             label=f'Prédictions > {optimal_threshold:.4f}', alpha=0.7)

    # Adding threshold line
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Seuil de décision ({optimal_threshold:.4f})')

    # Add title and labels
    plt.title('Distribution des probabilités prédites avec seuil de décision')
    plt.xlabel('Probabilité')
    plt.ylabel('Nombre de prédictions')

    # Adding text annotations for counts
    num_below = np.sum(y_pred_proba <= optimal_threshold)
    num_above = np.sum(y_pred_proba > optimal_threshold)
    plt.text(0.45, 1500, f'Count ≤ {optimal_threshold:.4f}: {num_below}', color='orange')
    plt.text(0.55, 1500, f'Count > {optimal_threshold:.4f}: {num_above}', color='blue')

    # Adding legend
    plt.legend()

    # Saving and showing the plot
    plt.savefig('probability_distribution_colored.png')
    plt.show()
    plt.close()

    print_notification('###### FIN: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES ##########', color="blue")
    ###### FIN: ANALYSE DE LA DISTRIBUTION DES PROBABILITÉS PRÉDITES ##########

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
    # Ensure all figures are closed before generating a new one

    plt.close('all')
    # Courbe ROC des prédictions
    # Calculate the false positive rate and true positive rate
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Create the ROC plot with customizations
    plt.figure(figsize=(8, 6))

    # Plot the ROC curve
    plt.plot(fpr, tpr, color='blue', linestyle='-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.2f})')

    # Add a dashed diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2)

    # Customize the labels and title
    plt.title('ROC Curve with Custom Design', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)

    # Adding gridlines
    plt.grid(True)

    # Add legend to the plot
    plt.legend(loc='lower right', fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()
    plt.close()
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
    best_params_sklearn['scale_pos_weight'] = class_weights[1] / class_weights[0]
    # Instancier un classificateur XGBoost avec les hyperparamètres optimisés.
    xgb_classifier = xgb.XGBClassifier(**best_params_sklearn)
    # Entraîner le modèle sur les données d'entraînement avec des poids pour chaque échantillon.
    xgb_classifier.fit(X_train, y_train, sample_weight=sample_weights)

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

    # Sélection des top N interactions (par exemple, top 10)
    N = 10
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
        print(f"{f1} <-> {f2}: {value:.4f}")

    # Heatmap des interactions pour les top features
    top_features = interaction_df.sum().sort_values(ascending=False).head(10).index
    plt.figure(figsize=(12, 10))
    sns.heatmap(interaction_df.loc[top_features, top_features], annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("SHAP Interaction Values for Top 10 Features", fontsize=16)
    plt.tight_layout()
    plt.savefig('feature_interaction_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(
        "Graphiques d'interaction sauvegardés sous 'top_feature_interactions.png' et 'feature_interaction_heatmap.png'")
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
    FILE_NAME_ = "Step5_Step4_Step3_Step2_MergedAllFile_Step1_2_merged_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
    DIRECTORY_PATH_ = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL\merge13092024"
    FILE_PATH_ = os.path.join(DIRECTORY_PATH_, FILE_NAME_)

    DEVICE_ = 'cuda'
    USE_OPTIMIZED_THRESHOLD_ = False
    FIXED_THRESHOLD_VALUE_ = 0.54
    NUM_BOOST_MIN_ = 400
    NUM_BOOST_MAX_ = 1200
    N_TRIALS_OPTIMIZATION_ =10
    NB_SPLIT_TSCV_ = 10
    NANVALUE_TO_NEWVAL_ = np.nan #900000.123456789
    LEARNING_CURVE_ENABLED = False

    # Définir les paramètres supplémentaires pour combined_metric
    METRIC_DICT = {
        'threshold': FIXED_THRESHOLD_VALUE_,
        'profit_ratio': 1 / 1.1,  # ≈ 0.909 (si profit donne 1 unité monétaire et perte 1.1)
        'tp_weight': 0.4,
        'fp_penalty': 1.1
    }

    print_notification('###### DEBUT: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")
    file_path = FILE_PATH_
    initial_df = load_data(file_path)
    print_notification('###### FIN: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")

    results = train_and_evaluate_XGBOOST_model(
        initial_df=initial_df,
        n_trials_optimization=N_TRIALS_OPTIMIZATION_,
        device=DEVICE_,
        use_optimized_threshold=USE_OPTIMIZED_THRESHOLD_,
        num_boost_min=NUM_BOOST_MIN_,
        num_boost_max=NUM_BOOST_MAX_,
        nb_split_tscv=NB_SPLIT_TSCV_,
        nanvalue_to_newval=NANVALUE_TO_NEWVAL_,
        learning_curve_enabled=LEARNING_CURVE_ENABLED,
        metric_dict=METRIC_DICT  # Passer les paramètres supplémentaires

    )

    if results is not None:
        print("Meilleurs hyperparamètres trouvés:", results['study'].best_params)
        print("Meilleur score:", results['study'].best_value)
        print("Seuil optimal:", results['optimal_threshold'])
    else:
        print("L'entraînement n'a pas produit de résultats.")
