import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from standardFunc import load_data, split_sessions, print_notification
import sys
import matplotlib.pyplot as plt
import optuna
import time
from sklearn.utils.class_weight import compute_class_weight

device = 'cuda'


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

# Chemin du fichier
file_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL\merge13092024\Step5_Step4_Step3_Step2_MergedAllFile_Step1_2_merged_extractOnlyFullSession_OnlyShort_feat_winsorizedScaledWithNanVal.csv"
method = input("Choisissez la méthode (appuyez sur Entrée pour non ancrée, 'a' pour ancrée): ").lower()

# 1. Chargement des données
df = load_data(file_path)

# 2. Division en ensembles d'entraînement et de test
try:
    train_df, test_df = split_sessions(df, test_size=0.2, min_train_sessions=2, min_test_sessions=2)
except ValueError as e:
    print(f"Erreur lors de la division des sessions : {e}")
    sys.exit(1)

# 3. Préparation des features et de la cible
feature_columns = [col for col in df.columns if
                   col not in ['class_binaire', 'candleDir', 'date', 'trade_category', 'SessionStartEnd']]
X_train = train_df[feature_columns]
y_train = train_df['class_binaire']
X_test = test_df[feature_columns]
y_test = test_df['class_binaire']

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weight_dict = dict(zip(np.unique(y_train), class_weights))

# 4. Vérification de l'équilibre des classes (excluant les 99)
mask_train = y_train != 99
X_train = X_train[mask_train]
y_train = y_train[mask_train]

mask_test = y_test != 99
X_test = X_test[mask_test]
y_test = y_test[mask_test]

trades_distribution = y_train.value_counts(normalize=True)
trades_counts = y_train.value_counts()

print("Distribution des trades (excluant les 99):")
print(f"Trades échoués [0]: {trades_distribution.get(0, 0) * 100:.2f}% ({trades_counts.get(0, 0)} trades)")
print(f"Trades réussis [1]: {trades_distribution.get(1, 0) * 100:.2f}% ({trades_counts.get(1, 0)} trades)")

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


# Fonction objective pour Optuna
# Fonction objective pour Optuna avec early stopping
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.01, log=True),  # Plage élargie
        'min_child_weight': trial.suggest_int('min_child_weight', 30, 50),  # Augmenté à 50
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'gamma': trial.suggest_float('gamma', 100e-3, 1, log=True),  # Augmenté à 10.0
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 30.0, log=True),  # Plage élargie
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 30.0, log=True),  # Plage élargie
        'max_delta_step': trial.suggest_float('max_delta_step', 0, 20),  # Augmenté à 20
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),

        # Paramètres fixes
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'scale_pos_weight': class_weights[1] / class_weights[0],
        'tree_method': 'hist',
        'device': device
    }

    num_boost_round = trial.suggest_int('num_boost_round', 1500, 3500)

    unique_sessions = train_df[train_df['SessionStartEnd'] == 10].index
    n_splits = 6
    total_sessions = len(unique_sessions)
    block_size = total_sessions // (n_splits + 1)

    cv_scores = []
    for i in range(n_splits):
        if method == 'a':
            train_start = 0
            train_end = (i + 1) * block_size
        else:
            train_start = i * block_size
            train_end = (i + 1) * block_size

        val_end = train_end + block_size

        train_mask = (train_df.index >= unique_sessions[train_start]) & (train_df.index < unique_sessions[train_end])
        val_mask = (train_df.index >= unique_sessions[train_end]) & (train_df.index <= (
            unique_sessions[val_end - 1] if val_end < total_sessions else train_df.index[-1]))

        X_train_cv = train_df[train_mask][feature_columns]
        y_train_cv = train_df[train_mask]['class_binaire']
        X_val_cv = train_df[val_mask][feature_columns]
        y_val_cv = train_df[val_mask]['class_binaire']

        mask_train = y_train_cv != 99
        X_train_cv, y_train_cv = X_train_cv[mask_train], y_train_cv[mask_train]
        mask_val = y_val_cv != 99
        X_val_cv, y_val_cv = X_val_cv[mask_val], y_val_cv[mask_val]

        sample_weights = np.array([weight_dict[label] for label in y_train_cv])
        dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv, weight=sample_weights)
        dval = xgb.DMatrix(X_val_cv, label=y_val_cv)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,  # Utilisez la variable num_boost_round ici
            evals=[(dval, 'eval')],
            early_stopping_rounds=10,
            verbose_eval=False,
        )

        y_val_pred_proba = model.predict(dval)
        val_auc = roc_auc_score(y_val_cv, y_val_pred_proba)
        cv_scores.append(val_auc)

    return np.mean(cv_scores)

# Définir un callback pour afficher les progrès
def print_callback(study, trial):
    print(f"Essai terminé : {trial.number}")
    print(f"Valeur actuelle : {trial.value}")
    print(f"Meilleure valeur jusqu'à présent : {study.best_value}")
    print(f"Meilleurs paramètres jusqu'à présent : {study.best_params}")
    print("------")

# Début du chronomètre
start_time = time.time()

# Exécution de l'optimisation bayésienne avec le callback
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10, callbacks=[print_callback])

# Fin du chronomètre
end_time = time.time()

# Calcul du temps d'exécution
execution_time = end_time - start_time

print("Optimisation terminée.")
print("Meilleurs hyperparamètres trouvés: ", study.best_params)
print("Meilleur score: ", study.best_value)
print(f"Temps d'exécution total : {execution_time:.2f} secondes")

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

# Prédictions sur les données de test
y_pred = final_model.predict(dtest)

# Évaluation des performances
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Pour un problème de classification binaire
y_pred_binary = (y_pred > 0.5).astype(int)  # Seuil à ajuster si nécessaire

accuracy = accuracy_score(y_test, y_pred_binary)
auc = roc_auc_score(y_test, y_pred)

print("Accuracy sur les données de test:", accuracy)
print("AUC-ROC sur les données de test:", auc)
print("\nRapport de classification:")
print(classification_report(y_test, y_pred_binary))

# Identification des erreurs
errors = X_test[y_test != y_pred_binary]
print("Nombre d'erreurs:", len(errors))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC')
plt.show()


# Pour SHAP, conversion du modèle en XGBClassifier
best_params_sklearn = best_params.copy()
best_params_sklearn['n_estimators'] = num_boost_round
best_params_sklearn['scale_pos_weight'] = class_weights[1] / class_weights[0]  # Ajoutez ce paramètre
xgb_classifier = xgb.XGBClassifier(**best_params_sklearn)
xgb_classifier.fit(X_train, y_train, sample_weight=sample_weights)

# Utilisation de SHAP pour évaluer l'importance des features
explainer = shap.TreeExplainer(xgb_classifier)
shap_values = explainer.shap_values(X_test)

# Visualisation de l'importance des features
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.tight_layout()
plt.savefig('shap_importance_optimized.png')
plt.close()


