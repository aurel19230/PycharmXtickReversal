import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from standardFunc import load_data, split_sessions, print_notification
import sys
import matplotlib.pyplot as plt
import optuna
import time
from sklearn.utils.class_weight import compute_class_weight
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

device = 'cuda'


def combined_metric(y_true, y_pred_proba):
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

    # Convertir les probabilités en prédictions binaires pour precision, recall et f1
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Calculer l'AUC directement avec les probabilités
    auc = roc_auc_score(y_true, y_pred_proba)

    # Calculer les autres métriques avec les prédictions binaires
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Combiner les métriques (vous pouvez ajuster les poids selon vos besoins)
    combined_score = (auc * 0.4) + (precision * 0.3) + (recall * 0.2) + (f1 * 0.1)

    return combined_score
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
# Nom du fichier
file_name = "Step5_Step4_Step3_Step2_MergedAllFile_Step1_2_merged_extractOnlyFullSession_OnlyLong_feat_winsorizedScaledWithNanVal.csv"

# Chemin du répertoire
directory_path = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\4_0_4TP_1SL\\merge13092024"

# Construction du chemin complet du fichier
file_path = os.path.join(directory_path, file_name)#method = input("Choisissez la méthode (appuyez sur Entrée pour non ancrée, 'a' pour ancrée): ").lower()
method='a'
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
                 #  col not in ['class_binaire', 'candleDir', 'date', 'trade_category', 'SessionStartEnd']]
                col not in ['class_binaire', 'date', 'trade_category', 'SessionStartEnd',
                            'deltaTimestampOpening','deltaTimestampOpeningSection5min','deltaTimestampOpeningSection5index','deltaTimestampOpeningSection30min']]

#'total_count_abv','total_count_blw','meanVolx',
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
# Fonction objective pour Optuna avec validation croisée temporelle améliorée
def objective(trial):
    # Définition des paramètres comme avant
    params = {
        'max_depth': trial.suggest_int('max_depth', 6, 15),  # Augmenté à 30
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.005, log=True),  # Plage élargie
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),  # Augmenté à 50
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'gamma': trial.suggest_float('gamma', 0.1, 5, log=True),  # Augmenté à 10.0
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 30.0, log=True),  # Plage élargie
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 30.0, log=True),  # Plage élargie
        'max_delta_step': trial.suggest_float('max_delta_step', 0, 20),  # Augmenté à 20
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.6, 1.0),

        # Paramètres fixes
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'scale_pos_weight': class_weights[1] / class_weights[0],
        'tree_method': 'hist',
        'device': device
    }

    num_boost_round = trial.suggest_int('num_boost_round', 1000, 3000)

    tscv = TimeSeriesSplit(n_splits=6)
    cv_scores = []

    for train_index, val_index in tscv.split(X_train):
        X_train_cv, X_val_cv = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]

        # Filtrage des valeurs 99
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
            num_boost_round=num_boost_round,
            evals=[(dval, 'eval')],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        y_val_pred_proba = model.predict(dval)

        # Utilisation de la nouvelle métrique combinée
        val_score = combined_metric(y_val_cv, y_val_pred_proba)
        cv_scores.append(val_score)

    # Retourne la moyenne des scores comme avant
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
study.optimize(objective, n_trials=5, callbacks=[print_callback])

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
print("Entraînement du modèle final")
y_pred = final_model.predict(dtest)

# Évaluation des performances
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Pour un problème de classification binaire
y_pred_binary = (y_pred > 0.5).astype(int)  # Seuil à ajuster si nécessaire
print("accuracy_score:")

accuracy = accuracy_score(y_test, y_pred_binary)
auc = roc_auc_score(y_test, y_pred)

print("Accuracy sur les données de test:", accuracy)
print("AUC-ROC sur les données de test:", auc)
print("\nRapport de classification:")
print(classification_report(y_test, y_pred_binary))


def create_xgb_model(random_state):
    return XGBClassifier(
        max_depth=np.random.randint(6, 15),
        learning_rate=np.random.uniform(1e-3, 0.01),
        n_estimators=np.random.randint(1000, 3000),
        subsample=np.random.uniform(0.5, 1.0),
        colsample_bytree=np.random.uniform(0.5, 0.9),
        random_state=random_state,
        tree_method='hist',
        device='cuda'
    )

def create_xgb_model(random_state):
    return xgb.XGBClassifier(
        max_depth=np.random.randint(6, 30),
        learning_rate=np.random.uniform(1e-5, 0.005),
        n_estimators=np.random.randint(1000, 5000),
        subsample=np.random.uniform(0.5, 1.0),
        colsample_bytree=np.random.uniform(0.5, 0.9),
        random_state=random_state,
        tree_method='hist',
        device='cuda'  # Utilisez 'cpu' si vous n'utilisez pas de GPU
    )

def bagging_xgboost(X_train, y_train, X_test, y_test, sample_weights, n_models=5):
    predictions = np.zeros((X_test.shape[0], n_models))

    for i in range(n_models):
        model = create_xgb_model(random_state=i)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        predictions[:, i] = model.predict_proba(X_test)[:, 1]

    final_predictions = np.mean(predictions, axis=1)
    return combined_metric(y_test, final_predictions)

def stacking_model(X_train, y_train, X_test, y_test, sample_weights):
    tscv = TimeSeriesSplit(n_splits=6)
    n_models = 3  # XGBoost, RandomForest, LightGBM
    oof_predictions = np.zeros((X_train.shape[0], n_models))
    test_predictions = np.zeros((X_test.shape[0], n_models))

    for train_idx, val_idx in tscv.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        fold_sample_weights = sample_weights[train_idx]

        # Création des DMatrix pour XGBoost
        dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold, weight=fold_sample_weights)
        dval_fold = xgb.DMatrix(X_val_fold)
        dtest = xgb.DMatrix(X_test)

        models = [
            xgb.XGBClassifier(**best_params_sklearn),
            RandomForestClassifier(n_estimators=100, random_state=42),
            lgb.LGBMClassifier(n_estimators=100, random_state=42)
        ]

        for i, model in enumerate(models):
            if isinstance(model, xgb.XGBClassifier):
                model.fit(dtrain_fold)
                oof_predictions[val_idx, i] = model.predict_proba(dval_fold)[:, 1]
                test_predictions[:, i] += model.predict_proba(dtest)[:, 1] / tscv.n_splits
            else:
                model.fit(X_train_fold, y_train_fold, sample_weight=fold_sample_weights)
                oof_predictions[val_idx, i] = model.predict_proba(X_val_fold)[:, 1]
                test_predictions[:, i] += model.predict_proba(X_test)[:, 1] / tscv.n_splits

    # Ajout de features importantes originales
    important_features = ['feature1', 'feature2', 'feature3']  # À adapter selon vos features importantes
    stacked_features_train = np.column_stack((oof_predictions, X_train[important_features]))
    stacked_features_test = np.column_stack((test_predictions, X_test[important_features]))

    scaler = StandardScaler()
    stacked_features_train = scaler.fit_transform(stacked_features_train)
    stacked_features_test = scaler.transform(stacked_features_test)

    # Optimisation des hyperparamètres du méta-modèle
    param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
    meta_model = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5)
    meta_model.fit(stacked_features_train, y_train)

    final_predictions = meta_model.predict_proba(stacked_features_test)[:, 1]
    return combined_metric(y_test, final_predictions)


# Évaluation des méthodes d'ensemble BAGGING ET STAKING
print("Évaluation des méthodes d'ensemble:")
print("  Bagging:")
bagging_score = bagging_xgboost(X_train, y_train, X_test, y_test, sample_weights)
print("  Staking:")
stacking_score = stacking_model(X_train, y_train, X_test, y_test, sample_weights)


print(f"Score Bagging: {bagging_score}")
print(f"Score Stacking: {stacking_score}")
print(f"Score XGBoost original: {combined_metric(y_test, y_pred)}")

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
print("Pour SHAP, conversion du modèle en XGBClassifier:")

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


