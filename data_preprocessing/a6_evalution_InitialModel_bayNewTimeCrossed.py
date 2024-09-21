"""""
Les hyperparamètres, y compris num_boost_round, ne sont pas appelés num_boost_round fois dans la fonction objective. Voici une explication plus précise :

Sélection des hyperparamètres :

Au début de chaque appel à objective, Optuna sélectionne un ensemble unique d'hyperparamètres, incluant num_boost_round.
Ces hyperparamètres sont choisis une seule fois par essai (trial).


Utilisation de num_boost_round :

num_boost_round détermine le nombre maximal d'itérations (arbres) pour l'entraînement du modèle XGBoost.
Il n'est pas "appelé" mais utilisé comme paramètre pour configurer l'entraînement.


Processus dans la fonction objective :

Les hyperparamètres sont utilisés pour configurer le modèle XGBoost.
Le modèle est ensuite entraîné et évalué NB_SPLIT_TSCV fois (validation croisée temporelle).
Chaque entraînement utilise le même ensemble d'hyperparamètres, y compris num_boost_round.


Correction de la note précédente :

Le nombre total d'entraînements est effectivement nTrials_4optimization * NB_SPLIT_TSCV.
Chaque entraînement implique num_boost_round itérations, mais ce n'est pas un facteur multiplicatif du nombre d'entraînements.
"""
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit,learning_curve

from standardFunc import load_data, split_sessions, print_notification

import optuna
import time
from sklearn.utils.class_weight import compute_class_weight
import os
from numba import njit
from xgboost.callback import TrainingCallback
from sklearn.metrics import precision_recall_curve,log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import bisect
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import numpy as np
import matplotlib.pyplot as plt
import sys




########################################
#########    FUNCTION DEF      #########
########################################
def combined_metric(y_true, y_pred_proba, threshold=0.5, profit_ratio=1.1, tp_weight=0.4, fp_penalty=0.2):
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
    auc_weight = 0.1
    precision_weight = 0.1
    recall_weight = 0.2
    f1_weight = 0.1
    potential_profit_weight = 0.2
    tp_score_weight = tp_weight  # 0.4 par défaut
    fp_score_weight = fp_penalty  # 0.2 par défaut

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
""""
def combined_metric(y_true, y_pred_proba):
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    # Convertir les probabilités en prédictions binaires pour precision, recall et f1
    y_pred = (y_pred_proba > 0.6).astype(int)

    # Calculer l'AUC directement avec les probabilités
    auc = roc_auc_score(y_true, y_pred_proba)

    # Calculer les autres métriques avec les prédictions binaires
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Combiner les métriques (vous pouvez ajuster les poids selon vos besoins)
    combined_score = (auc * 0.4) + (precision * 0.3) + (recall * 0.2) + (f1 * 0.1)

    return combined_score
"""
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

class CustomCallback(TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        train_score = evals_log['train']['aucpr'][-1]
        valid_score = evals_log['eval']['aucpr'][-1]
        if epoch % 10 == 0 and train_score - valid_score > 1: # on le met à 1 pour annuler ce test. On se base sur l'early stopping desormais
            print(f"Arrêt de l'entraînement à l'itération {epoch}. Écart trop important.")
            return True
        return False


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

import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


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
    train_scores_std = np.array(learning_curve_data['train_scores_std'])
    test_scores_mean = np.array(learning_curve_data['test_scores_mean'])
    test_scores_std = np.array(learning_curve_data['test_scores_std'])

    # --- Ajout des prints pour vérifier les écarts types ---
    print(f"train_scores_std: {train_scores_std}")
    print(f"test_scores_std: {test_scores_std}")

    # Tracé des courbes et des bandes représentant l'écart-type avec transparence
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")  # Bande pour entraînement
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")  # Bande pour validation

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score d'entraînement")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validation")

    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()




def calculate_learning_curve_no_cv(model, X_train, y_train, X_val, y_val, weight_dict, combined_metric, threshold):
    # Création des DMatrix pour l'entraînement et la validation
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=[weight_dict[label] for label in y_train])
    dval = xgb.DMatrix(X_val, label=y_val)

    # Prédictions sur l'ensemble d'entraînement et de validation
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)

    # Calcul des scores pour l'ensemble d'entraînement et l'ensemble de validation
    train_score = combined_metric(y_train, train_pred, threshold=threshold)
    val_score = combined_metric(y_val, val_pred, threshold=threshold)

    return {
        'train_size': len(X_train),
        'train_score': train_score,
        'val_score': val_score
    }

def average_learning_curves(learning_curve_data_list):
    if not learning_curve_data_list:
        return None

    # Regrouper les tailles d'entraînement uniques
    train_sizes = sorted(list(set(data['train_size'] for data in learning_curve_data_list)))

    # Créer des listes pour stocker les scores
    train_scores_per_size = {size: [] for size in train_sizes}
    val_scores_per_size = {size: [] for size in train_sizes}

    # Parcourir les données et stocker les scores pour chaque taille d'entraînement
    for data in learning_curve_data_list:
        size = data['train_size']
        train_scores_per_size[size].append(data['train_score'])
        val_scores_per_size[size].append(data['val_score'])

    # Calculer les moyennes et les écarts types des scores pour chaque taille d'entraînement
    avg_train_scores = []
    avg_val_scores = []
    std_train_scores = []
    std_val_scores = []

    for size in train_sizes:
        avg_train_scores.append(np.mean(train_scores_per_size[size]))
        avg_val_scores.append(np.mean(val_scores_per_size[size]))
        std_train_scores.append(np.std(train_scores_per_size[size]))
        std_val_scores.append(np.std(val_scores_per_size[size]))

    return {
        'train_sizes': train_sizes,
        'train_scores_mean': avg_train_scores,
        'test_scores_mean': avg_val_scores,
        'train_scores_std': std_train_scores,
        'test_scores_std': std_val_scores
    }
def objective(trial, class_weights, weight_dict, X_train, y_train, device, num_boost_min, num_boost_max, nb_split_tscv, use_optimized_threshold, fixed_threshold, learning_curve_enabled=False):
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
        'device': device
    }
    num_boost_round = trial.suggest_int('num_boost_round', num_boost_min, num_boost_max)

    tscv = TimeSeriesSplit(n_splits=nb_split_tscv)
    cv_scores = []
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
            else:
                optimal_threshold = fixed_threshold

            val_score = combined_metric(y_val_cv, y_val_pred_proba, threshold=optimal_threshold, profit_ratio=2,
                                        tp_weight=0.5, fp_penalty=0.3)
            cv_scores.append(val_score)

            if learning_curve_enabled:
                # Calculer la courbe d'apprentissage pour ce split CV
                learning_curve_data = calculate_learning_curve_no_cv(
                    model,
                    X_train_cv, y_train_cv, X_val_cv, y_val_cv,
                    weight_dict, combined_metric, optimal_threshold
                )

                # Ajouter les données de la courbe d'apprentissage pour ce split
                learning_curve_data_list.append(learning_curve_data)

            # Enregistrement du modèle dans les attributs utilisateur pour print_callback
            trial.set_user_attr('model', model)

        except Exception as e:
            print(f"Error during training or evaluation: {e}")
            continue

    if not cv_scores:
        return float('-inf')  # Return worst possible score if all folds failed

    mean_cv_score = np.mean(cv_scores)

    if use_optimized_threshold:
        average_optimal_threshold = np.mean(optimal_thresholds)
        trial.set_user_attr('average_optimal_threshold', average_optimal_threshold)
    else:
        trial.set_user_attr('average_optimal_threshold', fixed_threshold)

    if learning_curve_enabled and learning_curve_data_list:
        # Calculer la courbe d'apprentissage moyenne à partir des splits CV
        avg_learning_curve_data = average_learning_curves(learning_curve_data_list)

        if avg_learning_curve_data is not None:
            # Stocker les données de la courbe d'apprentissage dans le trial pour utilisation ultérieure
            trial.set_user_attr('learning_curve_data', avg_learning_curve_data)

            # Optionnel : tracer la courbe si cet essai est le meilleur ou si c'est le premier essai
            if trial.number == 0 or mean_cv_score > trial.study.best_value:
                plot_learning_curve(
                    avg_learning_curve_data,
                    title=f"Courbe d'apprentissage moyenne (Meilleur essai {trial.number})",
                    filename=f'learning_curve_best_trial_{trial.number}.png'
                )

    return mean_cv_score


# Définir un callback pour afficher les progrès
def print_callback(study, trial, X_train, y_train):
    # Extraction des informations supplémentaires depuis les attributs du trial
    learning_curve_data = trial.user_attrs.get('learning_curve_data')
    best_train_score = None
    best_val_score = trial.value
    best_train_size = None
    total_train_size = len(X_train)  # Taille totale de l'ensemble d'entraînement
    # Extraire le modèle entraîné depuis trial
    model = trial.user_attrs.get("model")

    if model is None:
        print("Erreur : Aucun modèle n'a été trouvé pour ce trial.")
        return
    # Si les données de la courbe d'apprentissage sont disponibles
    if learning_curve_data:
        train_scores_mean = learning_curve_data['train_scores_mean']
        train_sizes = learning_curve_data['train_sizes']

        # Trouver la dernière taille d'entraînement et le dernier score d'entraînement
        best_train_size = train_sizes[-1]
        best_train_score = train_scores_mean[-1]

        # Calculer la différence en pourcentage entre l'entraînement et la validation
        diff_percentage = ((best_train_score - best_val_score) / best_val_score) * 100

        # Calculer le pourcentage de l'ensemble d'entraînement utilisé
        train_size_percentage = (best_train_size / total_train_size) * 100

    # --- Calcul des scores sur l'ensemble d'entraînement complet ---

    # Créer un DMatrix pour l'ensemble complet
    dtrain_full = xgb.DMatrix(X_train, label=y_train)

    # Prédire sur l'ensemble d'entraînement complet
    train_predictions_full = model.predict(dtrain_full)

    # Calculer le score d'entraînement sur l'ensemble complet
    full_train_score = combined_metric(y_train, train_predictions_full)

    # --- Analyse du surapprentissage ---
    if full_train_score > best_val_score:
        overfitting_warning = "Attention : Possible surapprentissage"
    else:
        overfitting_warning = "Pas de surapprentissage détecté"

    # --- Affichage des informations ---
    print(f"Essai terminé : {trial.number}")
    print(f"Valeur actuelle (score de validation) : {best_val_score:.4f}")
    if best_train_score is not None:
        print(f"Score sur l'ensemble d'entraînement (cross-validation) : {best_train_score:.4f}")
        print(f"Différence en % entre entraînement (cross-validation) et validation : {diff_percentage:.2f}%")
        print(
            f"Nombre d'échantillons utilisés pour obtenir ces scores : {best_train_size} ({train_size_percentage:.2f}% du total)")

    # Afficher le score sur l'ensemble complet
    print(
        f"Score sur l'ensemble d'entraînement complet : {full_train_score:.4f} (peut être plus élevé si le modèle surapprend)")
    print(f"Analyse du surapprentissage : {overfitting_warning}")

    print(f"Meilleure valeur jusqu'à présent : {study.best_value:.4f}")
    print(f"Meilleurs paramètres jusqu'à présent : {study.best_params}")
    print("------")


# 3. Fonction pour analyser les erreurs confiantes
def analyze_confident_errors(xgb_classifier, confident_errors, X_test, feature_names, important_features, n=10):
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


# Analyse des erreurs
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

    plt.close('all')
    plt.figure('error_distribution')

    # Visualiser la distribution des probabilités de prédiction pour les erreurs
    plt.close('all')
    plt.figure('error_distribution', figsize=(10, 6))
    sns.histplot(data=error_df, x='prediction_probability', hue='true_label', bins=20)
    plt.title('Distribution des probabilités de prédiction pour les erreurs')
    plt.savefig('error_probability_distribution.png')
    plt.close('error_distribution')

    # Si vous voulez afficher le graphique immédiatement :
    img = plt.imread('error_probability_distribution.png')
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # Identifier les cas les plus confiants mais erronés
    most_confident_errors = error_df.sort_values('prediction_probability', ascending=False).head(5)
    print("\nLes 30 erreurs les plus confiantes:")
    print(most_confident_errors[['true_label', 'predicted_label', 'prediction_probability']])

    return results_df, error_df

# 4. Fonction pour visualiser les erreurs confiantes
def plot_confident_errors(xgb_classifier, confident_errors, X_test, feature_names, n=5):
    explainer = shap.TreeExplainer(xgb_classifier)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n))
    for i, idx in enumerate(confident_errors.index[:n]):
        plt.figure(figsize=(10, 6))  # Create a new figure for each plot
        shap_values = explainer.shap_values(X_test.loc[idx:idx])
        shap.summary_plot(shap_values, X_test.loc[idx:idx], plot_type="bar", feature_names=feature_names, show=False)
        plt.title(f"Erreur {i + 1}: Vrai {confident_errors.loc[idx, 'true_label']}, "
                  f"Prédit {confident_errors.loc[idx, 'predicted_label']} "
                  f"(Prob: {confident_errors.loc[idx, 'prediction_probability']:.4f})")
        plt.tight_layout()
        plt.savefig(f'confident_error_shap_{i + 1}.png')
        plt.close()

    # Create a summary image combining all individual plots
    from PIL import Image
    import os

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





# 5. Fonction pour comparer les erreurs vs les prédictions correctes
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


########################################
#########   END FUNCTION DEF   #########
########################################


""""
# Sélectionner uniquement les colonnes de type int et float
df_numeric = df.select_dtypes(include=[np.number])

# Sélectionner uniquement les colonnes de type int et float
df_numeric = df.select_dtypes(include=[np.number])

# Compter les valeurs égales à NANVALUE_TO_NEWVAL dans le DataFrame d'entrée
count_nanvalue_newval_before = (df_numeric == NANVALUE_TO_NEWVAL).sum().sum()
print(f"Nombre de valeurs égales à {NANVALUE_TO_NEWVAL} avant traitement : {count_nanvalue_newval_before}")

# Convertir le DataFrame en tableau NumPy
data_numeric = df_numeric.to_numpy()



# Appliquer la fonction Numba
data_numeric, count_replacements = process_values(data_numeric, NANVALUE_TO_NEWVAL)

# Reconvertir le tableau NumPy en DataFrame avec les mêmes colonnes et index que l'original
df_numeric_processed = pd.DataFrame(data_numeric, columns=df_numeric.columns, index=df_numeric.index)

# Remplacer le DataFrame source par le DataFrame traité
df[df_numeric.columns] = df_numeric_processed

# Afficher le nombre de remplacements pendant le traitement
print(f"Nombre de valeurs correspondant aux critères et remplacées : {count_replacements}")

# Compter les valeurs égales à NANVALUE_TO_NEWVAL dans le DataFrame après traitement
count_nanvalue_newval_after = (df[df_numeric.columns] == NANVALUE_TO_NEWVAL).sum().sum()
print(f"Nombre de valeurs égales à {NANVALUE_TO_NEWVAL} après traitement : {count_nanvalue_newval_after}")
"""

# DEVICE = 'cpu'
# USE_OPTIMIZED_THRESHOLD = False  # Mettre à True pour optimiser le seuil, False pour utiliser un seuil fixe

# FIXED_THRESHOLD = 0.54  # Définissez ici le seuil que vous souhaitez utiliser

# num_boost_round représente le nombre maximal d'itérations (ou arbres) dans le modèle XGBoost.
# NUM_BOOST_MIN=400
# NUM_BOOST_MAX=1000

# Nombre d'itérations de la fonction objective
# Chaque objectif est appelé nTrials_4optimization fois par Optuna
# Cela représente le nombre total d'essais d'hyperparamètres différents qui seront testés
# N_TRIAL_OPTIMIZATION =5


# Nombre de splits pour la validation croisée temporelle
# Ce paramètre détermine combien de fois l'ensemble d'entraînement sera divisé
# Pour rappel, dans TimeSeriesSplit, l'ensemble d'entraînement grandit progressivement :
# - Le premier split utilise une petite partie des données pour l'entraînement
# - Chaque split suivant ajoute plus de données à l'ensemble d'entraînement
# - Le dernier split utilise presque toutes les données disponibles pour l'entraînement
# NB_SPLIT_TSCV = 12

# Pour chaque essai (nTrials_4optimization) :
#   - Un ensemble d'hyperparamètres est choisi, incluant num_boost_round
#   - Le modèle est entraîné et évalué NB_SPLIT_TSCV fois
#   - Chaque entraînement utilise num_boost_round itérations
## Nombre total d'entraînements = nTrials_4optimization * NB_SPLIT_TSCV * num_boost_round

# NANVALUE_TO_NEWVAL=0




def train_and_evaluate_XGBOOST_model(
        initial_df,
        n_trials_optimization=4,
        device='cuda',
        use_optimized_threshold=False,
        fixed_threshold=0.54,
        num_boost_min=400,
        num_boost_max=1000,
        nb_split_tscv=12,
        nanvalue_to_newval=0,
        learning_curve_enabled=False
):
    """
    Fonction principale pour entraîner et évaluer le modèle XGBoost.

    Args:
    file_path (str): Chemin vers le fichier de données.
    n_trials (int): Nombre d'essais pour l'optimisation Bayésienne.
    device (str): Dispositif à utiliser ('cpu' ou 'cuda').
    use_optimized_threshold (bool): Utiliser un seuil optimisé ou fixe.
    fixed_threshold (float): Seuil fixe à utiliser si use_optimized_threshold est False.
    num_boost_min (int): Nombre minimum de boosting rounds.
    num_boost_max (int): Nombre maximum de boosting rounds.
    nb_split_tscv (int): Nombre de splits pour la validation croisée temporelle.
    nanvalue_to_newval (float): Valeur de remplacement pour les NaN.

    Returns:
    dict: Un dictionnaire contenant les résultats et les objets importants.
    """
    # Chargement et préparation des données
   # print("Chargement des données...")
    df=initial_df
    # Remplacement des valeurs NaN
    REPLACEMENT_VALUE = 90000.54789
    df = df.fillna(REPLACEMENT_VALUE)

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

    ###### FIN: CHARGER ET PREPARER LES DONNEES  ##########
    print_notification('###### FIN: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")


    ###### DEBUT: OPTIMISATION BAYESIENNE ##########
    print_notification('###### DEBUT: OPTIMISATION BAYESIENNE ##########', color="blue")

    # Début du chronomètre
    start_time = time.time()

    # Exécution de l'optimisation bayésienne avec le callback

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
            fixed_threshold,
            learning_curve_enabled=LEARNING_CURVE_ENABLED
        ),
        n_trials=n_trials_optimization,
        callbacks=[lambda study, trial: print_callback(study, trial, X_train, y_train)]
    )

    # Fin du chronomètre
    end_time = time.time()

    # Calcul du temps d'exécution
    execution_time = end_time - start_time

    print("Optimisation terminée.")
    print("Meilleurs hyperparamètres trouvés: ", study.best_params)
    print("Meilleur score: ", study.best_value)
    print(f"Temps d'exécution total : {execution_time:.2f} secondes")

    ###### FIN: OPTIMISATION BAYESIENNE ##########

    ###### DEBUT: ENTRAINEMENT DU MODELE FINAL ##########

    # Après l'optimisation
    best_trial = study.best_trial
    optimal_threshold = best_trial.user_attrs['average_optimal_threshold']
    print(f"Seuil utilisé : {optimal_threshold:.4f}")

    # Création et entraînement du modèle final
    best_params = study.best_params.copy()  # Copie pour éviter de modifier l'original
    best_params['objective'] = 'binary:logistic'
    best_params['eval_metric'] = 'auc'
    best_params['tree_method'] = 'hist'  # 'gpu_hist' for GPU
    best_params['device'] = device

    # Extraction de num_boost_round
    num_boost_round = best_params.pop('num_boost_round')

    # Préparation des données avec poids
    sample_weights = np.array([weight_dict[label] for label in y_train])
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Entraînement du modèle final
    final_model = xgb.train(
        best_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train')],
        verbose_eval=False
    )
    print_notification('###### FIN: ENTRAINEMENT DU MODELE FINAL ##########', color="blue")
    ###### FIN: ENTRAINEMENT DU MODELE FINAL ##########

    ###### DEBUT: PREDICTION ET EVALUATION DU MODELE ##########
    print_notification('###### DEBUT: PREDICTION ET EVALUATION DU MODELE ##########', color="blue")
    # Prédiction finale
    y_pred_proba = final_model.predict(dtest)
    y_pred = (y_pred_proba > optimal_threshold).astype(int)

    # Évaluation des performances
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("Accuracy sur les données de test:", accuracy)
    print("AUC-ROC sur les données de test:", auc)
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred))
    print_notification('###### FIN: PREDICTION ET EVALUATION DU MODELE ##########', color="blue")
    ###### FIN: PREDICTION ET EVALUATION DU MODELE ##########

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

    # Visualiser la distribution des probabilités
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba, bins=20, edgecolor='black')
    plt.title('Distribution des probabilités prédites')
    plt.xlabel('Probabilité')
    plt.ylabel('Nombre de prédictions')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Seuil de décision ({optimal_threshold:.4f})')
    plt.legend()
    plt.savefig('probability_distribution.png')
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

    # Courbe ROC des prédictions
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.show()
    print_notification('###### FIN: ANALYSE DES ERREURS ##########', color="blue")
    ###### FIN: ANALYSE DES ERREURS ##########



    ###### DEBUT: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########
    print_notification('###### DEBUT: ANALYSE DES ERREURS LES PLUS CONFIANTES ##########', color="blue")
    # Assurez-vous que ces variables sont déjà définies dans votre code :
    # results_df, X_test, y_test, y_pred, y_pred_proba, xgb_classifier, feature_importance_df

    # 0. Conversion du modèle en XGBClassifier:
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
    explainer = shap.TreeExplainer(xgb_classifier)
    shap_values = explainer.shap_values(X_test)

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
    plot_confident_errors(xgb_classifier, confident_errors, X_test, X_test.columns)

    # Exécution des analyses
    print("\nAnalyse des erreurs confiantes:")
    analyze_confident_errors(xgb_classifier, confident_errors, X_test, X_test.columns, important_features)

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
    # 1. Définition des chemins et fichiers
    FILE_NAME_ = "Step5_Step4_Step3_Step2_MergedAllFile_Step1_2_merged_extractOnlyFullSession_OnlyShort_feat_winsorizedScaledWithNanVal.csv"
    DIRECTORY_PATH_ = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL\merge13092024"
    FILE_PATH_ = os.path.join(DIRECTORY_PATH_, FILE_NAME_)

    # 2. Définition des constantes et paramètres globaux
    DEVICE_ = 'cuda'  # Utiliser 'cuda' pour GPU, 'cpu' sinon
    USE_OPTIMIZED_THRESHOLD_ = False  # True pour optimiser le seuil, False pour utiliser un seuil fixe
    FIXED_THRESHOLD_ = 0.54  # Seuil fixe à utiliser si USE_OPTIMIZED_THRESHOLD est False
    NUM_BOOST_MIN_ = 400  # Nombre minimum de boosting rounds
    NUM_BOOST_MAX_ = 1000  # Nombre maximum de boosting rounds
    N_TRIALS_OPTIMIZATION_ = 7  # Nombre d'essais pour l'optimisation avec Optuna
    NB_SPLIT_TSCV_ = 8  # Nombre de splits pour la validation croisée temporelle
    NANVALUE_TO_NEWVAL_ = 0  # Valeur de remplacement pour les NaN si les NAN oont été remplacés par une valeur dans l'étude de procession des featurs
    LEARNING_CURVE_ENABLED = True # ou False, selon vos besoins


    print_notification('###### DEBUT: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")
    ###### DEBUT: CHARGER ET PREPARER LES DONNEES  ##########
    file_path=FILE_PATH_
    initial_df = load_data(file_path)
    print_notification('###### FIN: CHARGER ET PREPARER LES DONNEES  ##########', color="blue")
    # 3. Exécution de la fonction principale pour entraîner et évaluer le modèle
    results = train_and_evaluate_XGBOOST_model(initial_df=initial_df,n_trials_optimization=N_TRIALS_OPTIMIZATION_,
                                               device=DEVICE_,use_optimized_threshold=USE_OPTIMIZED_THRESHOLD_,fixed_threshold=FIXED_THRESHOLD_,
                                               num_boost_min=NUM_BOOST_MIN_,num_boost_max=NUM_BOOST_MAX_,nb_split_tscv=NB_SPLIT_TSCV_,nanvalue_to_newval=NANVALUE_TO_NEWVAL_,learning_curve_enabled=LEARNING_CURVE_ENABLED)

    # 4. Utilisation des résultats de l'optimisation
    if results is not None:
        print("Meilleurs hyperparamètres trouvés:", results['study'].best_params)
        print("Meilleur score:", results['study'].best_value)
        print("Seuil optimal:", results['optimal_threshold'])
    else:
        print("L'entraînement n'a pas produit de résultats.")



