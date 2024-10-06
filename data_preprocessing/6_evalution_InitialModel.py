import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from standardFunc import load_data, split_sessions, print_notification
import sys
import matplotlib.pyplot as plt


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

# Chemin du fichier
file_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL_30092024\merge\Step5_Step4_Step3_Step2_MergedAllFile_Step1_2_merged_extractOnlyFullSession_feat_winsorized.csv"
method = input("Choisissez la méthode (appuyez sur Entrée pour non ancrée, 'a' pour ancrée): ").lower()

# 1. Chargement des données
df = load_data(file_path)

# 2. Division en ensembles d'entraînement et de test
try:
    train_df, test_df = split_sessions(df, test_size=0.2, min_train_sessions=2, min_test_sessions=2)
except ValueError as e:
    print(f"Erreur lors de la division des sessions : {e}")
    sys.exit(1)  # Sortie du programme avec un code d'erreur

# 3. Préparation des features et de la cible
feature_columns = [col for col in df.columns if
                   col not in ['class', 'candleDir', 'date', 'trade_category', 'SessionStartEnd']]
X_train = train_df[feature_columns]
y_train = train_df['class']
X_test = test_df[feature_columns]
y_test = test_df['class']

# 4. Vérification de l'équilibre des classes (excluant les 99)
mask_train = y_train != 99
X_train = X_train[mask_train]
y_train = y_train[mask_train]

mask_test = y_test != 99
X_test = X_test[mask_test]
y_test = y_test[mask_test]

trades_distribution = y_train.value_counts(normalize=True)

print("Distribution des trades (excluant les 99):")
print(f"Trades échoués [0]: {trades_distribution.get(0, 0) * 100:.2f}%")
print(f"Trades réussis [1]: {trades_distribution.get(1, 0) * 100:.2f}%")

total_trades = y_train.count()
print(f"Nombre total de trades (excluant les 99): {total_trades}")
threshold = 0.06
class_difference = abs(trades_distribution.get(0, 0) - trades_distribution.get(1, 0))
if class_difference >= threshold:
    error_message = "Erreur : Les classes ne sont pas équilibrées. "
    error_message += f"Différence : {class_difference:.2f}"
    print(error_message)
    sys.exit(1)  # Sortie du programme avec un code d'erreur

print(
    f"Les classes sont considérées comme équilibrées car la différence entre les proportions de trades réussis et échoués ({class_difference:.2f}) est inférieure à 0.05 (5%).")

# 5. Création du modèle initial
model = xgb.XGBClassifier(
    n_estimators=900,        # Réduit pour limiter le surapprentissage
    learning_rate=0.001,      # Réduit pour un apprentissage plus lent
    min_child_weight=7,     # Augmenté pour plus de généralisation
    max_depth=8,             # Réduit pour limiter la complexité
    subsample=0.7,           # Légèrement réduit
    gamma=0.2,               # Augmenté pour des divisions plus conservatrices
    colsample_bytree=0.7,    # Réduit pour utiliser moins de features par arbre
    reg_lambda=11,           # Augmenté pour plus de régularisation L2
    alpha=1,                 # Augmenté pour plus de régularisation L1
    objective='binary:logistic',
    random_state=42
)


# 6. Évaluation du modèle initial avec validation croisée temporelle
unique_sessions = train_df[train_df['SessionStartEnd'] == 10].index
n_splits = 1  # Nombre de splits pour la validation croisée

total_sessions = len(unique_sessions)
print(f"Nombre total de sessions dans train_df: {total_sessions}")

# Calculer la taille de chaque bloc en fonction de n_splits
block_size = total_sessions // (n_splits+1)


# Demander à l'utilisateur de choisir la méthode
# Initialiser la liste pour stocker les scores AUC-ROC
cv_scores = []
for i in range(n_splits):
    print(f"\nSplit {i + 1}:")
    if method == 'a':  # Méthode ancrée
        train_start = 0
        train_end = (i + 1) * block_size
    else:  # Méthode non ancrée (par défaut)
        train_start = i * block_size
        train_end = (i + 1) * block_size

    val_end = train_end + block_size

    # Sélectionner les sessions pour ce split
    train_sessions = unique_sessions[train_start:train_end]
    val_sessions = unique_sessions[train_end:val_end]

    # Créer des masques pour sélectionner les données de ces sessions
    train_mask = (train_df.index >= unique_sessions[train_start]) & (train_df.index < unique_sessions[train_end])
    if (i == n_splits - 1):
        val_mask = (train_df.index >= unique_sessions[train_end]) & (train_df.index <= (
            unique_sessions[val_end - 1] if val_end < total_sessions else train_df.index[-1]))
    else:
        val_mask = (train_df.index >= unique_sessions[train_end]) & (
                    train_df.index < (unique_sessions[val_end] if val_end < total_sessions else train_df.index[-1]))

    train_split = train_df[train_mask]
    val_split = train_df[val_mask]

    # Vérifier l'intégrité des sessions pour train et validation
    if not verify_session_integrity(train_split, f"Train Split {i + 1}"):
        print(f"Erreur dans le split d'entraînement {i + 1}. Arrêt du programme.")
        sys.exit(1)

    if not verify_session_integrity(val_split, f"Validation Split {i + 1}"):
        print(val_split.head(5))
        print(val_split.tail(5))
        print(train_df.tail(5))

        print(f"Erreur dans le split de validation {i + 1}. Arrêt du programme.")
        sys.exit(1)

    # Vérification supplémentaire pour la dernière session de validation
    if i == n_splits - 1:
        last_val_session_start = val_split[val_split['SessionStartEnd'] == 10].index[-1]
        last_val_session_end = val_split[val_split['SessionStartEnd'] == 20].index[-1]
        if last_val_session_start >= last_val_session_end:
            print(
                f"Erreur : La dernière session de validation est invalide. Début : {last_val_session_start}, Fin : {last_val_session_end}")
            sys.exit(1)

    X_train_cv = train_split[feature_columns]
    y_train_cv = train_split['class']
    X_val_cv = val_split[feature_columns]
    y_val_cv = val_split['class']

    # Exclure la classe 99
    mask_train = y_train_cv != 99
    X_train_cv, y_train_cv = X_train_cv[mask_train], y_train_cv[mask_train]
    mask_val = y_val_cv != 99
    X_val_cv, y_val_cv = X_val_cv[mask_val], y_val_cv[mask_val]

    # Vérification si les ensembles d'entraînement et de validation ne sont pas vides
    if X_train_cv.empty or X_val_cv.empty or y_train_cv.empty or y_val_cv.empty:
        print("Erreur : Un des ensembles d'entraînement ou de validation est vide après exclusion de la classe 99.")
        continue  # Passer au prochain split

    # Vérification de la distribution des classes après exclusion de la classe 99
    train_class_distribution = y_train_cv.value_counts(normalize=True)
    test_class_distribution = y_val_cv.value_counts(normalize=True)

    print(
        f"Distribution des classes dans l'ensemble d'entraînement (excluant 99) : {train_class_distribution.to_dict()}")
    print(f"Distribution des classes dans l'ensemble de validation (excluant 99) : {test_class_distribution.to_dict()}")

    # Vérifier si les classes sont déséquilibrées avec une tolérance de 5 % (0,05)
    class_diff_train = abs(train_class_distribution.get(0, 0) - train_class_distribution.get(1, 0))
    class_diff_val = abs(test_class_distribution.get(0, 0) - test_class_distribution.get(1, 0))

    # Vérification de l'équilibre des classes avec un seuil de 5 %


    if class_diff_train >= threshold:
        print(f"Alerte : Classes déséquilibrées dans l'ensemble d'entraînement. Différence : {class_diff_train:.4f}")
    else:
        print(f"Classes équilibrées dans l'ensemble d'entraînement. Différence : {class_diff_train:.4f}")

    if class_diff_val >= threshold:
        print(f"Alerte : Classes déséquilibrées dans l'ensemble de validation. Différence : {class_diff_val:.4f}")
    else:
        print(f"Classes équilibrées dans l'ensemble de validation. Différence : {class_diff_val:.4f}")

    # Entraînement et évaluation du modèle
    model.fit(X_train_cv, y_train_cv)

    # Prédictions sur les ensembles d'entraînement et de validation
    y_train_pred_proba = model.predict_proba(X_train_cv)[:, 1]
    y_val_pred_proba = model.predict_proba(X_val_cv)[:, 1]

    # Calcul de l'AUC-ROC sur l'entraînement et la validation
    train_auc = roc_auc_score(y_train_cv, y_train_pred_proba)
    val_auc = roc_auc_score(y_val_cv, y_val_pred_proba)

    # Ajout des scores aux listes
    cv_scores.append(val_auc)

    # Affichage des informations sur le split et la comparaison des scores
    print(f"  Train sessions: {len(train_sessions)} de l'index {train_sessions[0]} à {train_sessions[-1]}")
    print(
        f"  Validation sessions: {len(val_sessions)} de l'index {val_sessions[0]} à {val_sessions[-1] if len(val_sessions) > 0 else 'N/A'}")
    print(f"  Nb trades entraînement (excluant 99): {len(y_train_cv)}")
    print(f"  Nb trades validation (excluant 99): {len(y_val_cv)}")

    num_99_train_cv = (y_train_cv == 99).sum()
    num_99_val_cv = (y_val_cv == 99).sum()
    print(f"  Nombre de trades exclus (classe 99) dans l'ensemble d'entraînement : {num_99_train_cv}")
    print(f"  Nombre de trades exclus (classe 99) dans l'ensemble de validation : {num_99_val_cv}")

    print(f"  AUC-ROC entraînement pour ce split: {train_auc:.4f}")
    print(f"  AUC-ROC validation pour ce split: {val_auc:.4f}\n")

    # Vérification du surapprentissage : différence entre train et val
    overfit_threshold = 0.05  # Seuil de différence pour alerter sur du surapprentissage
    if train_auc - val_auc > overfit_threshold:
        print(
            f"  Alerte : Possible surapprentissage détecté pour ce split ! (Différence AUC: {train_auc - val_auc:.4f})")

print(f"Méthode utilisée : {'Ancrée' if method == 'a' else 'Non ancrée'}")
mean_cv_auc = np.mean(cv_scores)
std_cv_auc = np.std(cv_scores)

print(f"AUC-ROC moyen (validation croisée): {mean_cv_auc:.4f} (+/- {std_cv_auc * 2:.4f})")
"""

# 7. Utilisation de SHAP pour évaluer l'importance des features
model.fit(X_train, y_train)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# 8. Visualisation de l'importance des features
shap.summary_plot(shap_values[1], X_train, plot_type="bar")
plt.tight_layout()
plt.savefig('shap_importance.png')
plt.close()

# 9. Sélection des features basée sur un seuil d'importance
feature_importance = np.abs(shap_values[1]).mean(0)
normalized_importance = feature_importance / np.sum(feature_importance)

threshold = 0.90  # Seuil à 90% de l'importance cumulée

sorted_idx = np.argsort(normalized_importance)[::-1]
sorted_importance = normalized_importance[sorted_idx]
cumulative_importance = np.cumsum(sorted_importance)
n_features = np.where(cumulative_importance > threshold)[0][0] + 1

top_features = X_train.columns[sorted_idx[:n_features]]

print(f"\nNombre de features sélectionnées : {n_features}")
print("Features sélectionnées :")
for i, feature in enumerate(top_features, 1):
    print(f"{i}. {feature}: {normalized_importance[sorted_idx[i - 1]]:.4f}")

X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]

# 10. Création et évaluation du modèle final avec les features sélectionnées
final_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42
)

final_model.fit(X_train_selected, y_train)
y_pred_proba = final_model.predict_proba(X_test_selected)[:, 1]
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nAUC-ROC sur l'ensemble de test (modèle final): {test_auc:.4f}")

# Sauvegarde des features sélectionnées
pd.Series(top_features).to_csv('selected_features.csv', index=False)

print("\nLes features sélectionnées ont été sauvegardées dans 'selected_features.csv'")
print("Le graphique d'importance SHAP a été sauvegardé dans 'shap_importance.png'")
"""

