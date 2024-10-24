# Rapport d'Analyse XGBoost

## PHASE 1: OPTIMISATION DES PARAMÈTRES VIA VALIDATION CROISÉE TEMPORELLE (X_train et X_val) sur l'ensemble df_train

### 1. Paramètres initiaux de validations croisées

- **Nombre de splits de la validation croisée temporelle (`nb_split_tscv_`):** 2
- **Méthode de validation croisée (`cv_method`):** cv_config.K_FOLD


#### **Autres paramètres:**

- **early_stopping_rounds:** 80
- **std_penalty_factor_ standard deviation  à soustraire au score:** 1

### 2. Paramètres initiaux XGBoost et Optuna

#### **Paramètres de poids (`weight_param`):**

- **threshold:**
  - min: 0.51
  - max: 0.65
- **w_p:**
  - min: 1
  - max: 2.2
- **profit_per_tp:**
  - min: 1
  - max: 1
- **loss_per_fp:**
  - min: -1.1
  - max: -1.1
- **penalty_per_fn:**
  - min: -0.0
  - max: -0.0

#### **Plages des hyperparamètres XGBoost pour Optuna (`xgb_param_optuna_range`):**

- **num_boost_round:**
  - min: 200
  - max: 700
- **max_depth:**
  - min: 6
  - max: 11
- **learning_rate:**
  - min: 0.01
  - max: 0.2
  - log: True
- **min_child_weight:**
  - min: 3
  - max: 10
- **subsample:**
  - min: 0.7
  - max: 0.9
- **colsample_bytree:**
  - min: 0.55
  - max: 0.8
- **colsample_bylevel:**
  - min: 0.6
  - max: 0.85
- **colsample_bynode:**
  - min: 0.5
  - max: 1.0
- **gamma:**
  - min: 0
  - max: 5
- **reg_alpha:**
  - min: 1
  - max: 15.0
  - log: True
- **reg_lambda:**
  - min: 2
  - max: 20.0
  - log: True
### 3. MEILLEURS PARAMÈTRES TROUVÉS PAR OPTUNA (sur X_TRAIN et son ensemble de validation X_VAL) PAR TYPE DE SCORE

#### a) Paramètres pour les trois meilleurs `adjusted_score`

##### **Essai numéro 1 (Trial 2)**

- **PnL cumulé:** -4981.4000000000015
- **Win Rate (%):** 49.89%
- **Écart-type des scores:** 870.3070262844027
- **Adjusted Score:** -3361.0070262844024
- **Scores par split:** [-3106.1, -1875.3]

##### **Métriques du Meilleur Essai**

- **PnL cumulé:** -4981.4000000000015
- **Win Rate (%):** 49.89%
- **Total Trades:** 95407
- **True Positives (TP):** 47603
- **False Positives (FP):** 47804
- **True Negatives (TN):** 69144
- **False Negatives (FN):** 62342
- **TP Percentage:** 20.98%
- **Adjusted Score:** -3361.0070262844024
- **Écart-type des scores:** 870.3070262844027

##### **Scores de Validation Croisée**

- **Scores par split:** [-3106.1, -1875.3]
- **Score moyen:** -2490.7
- **Écart-type des scores:** 870.3070262844027

##### **Paramètres de Trading Optimisés**

- **max_depth:** 7
- **learning_rate:** 0.0480098837209512
- **min_child_weight:** 8
- **subsample:** 0.798424687583376
- **colsample_bytree:** 0.7831865213896052
- **colsample_bylevel:** 0.6450563820330746
- **colsample_bynode:** 0.6969252883612299
- **gamma:** 1.6055931169664417
- **reg_alpha:** 2.1190488986855973
- **reg_lambda:** 10.895375354869753
- **threshold:** 0.5569851557111134
- **profit_per_tp:** 1.0
- **loss_per_fp:** -1.1
- **penalty_per_fn:** -0.0
- **num_boost_round:** 449
- **w_p:** 1.3153580871421493

##### **Essai numéro 2 (Trial 3)**

- **PnL cumulé:** -7411.0
- **Win Rate (%):** 49.39%
- **Écart-type des scores:** 856.306312016909
- **Adjusted Score:** -4561.806312016909
- **Scores par split:** [-4311.0, -3100.0]

##### **Essai numéro 3 (Trial 4)**

- **PnL cumulé:** -9969.100000000006
- **Win Rate (%):** 49.38%
- **Écart-type des scores:** 1251.9325560907823
- **Adjusted Score:** -6236.482556090783
- **Scores par split:** [-5869.8, -4099.3]

#### b) Résultats des 3 meilleurs `PnL cumulé`

**Essai numéro 1 (Trial 2)**
- **PnL cumulé:** -4981.4000000000015
- **Win Rate (%):** 49.89%
- **Écart-type des scores:** 870.3070262844027
- **Adjusted Score:** -3361.0070262844024
- **Scores par split:** [-3106.1, -1875.3]

**Essai numéro 2 (Trial 3)**
- **PnL cumulé:** -7411.0
- **Win Rate (%):** 49.39%
- **Écart-type des scores:** 856.306312016909
- **Adjusted Score:** -4561.806312016909
- **Scores par split:** [-4311.0, -3100.0]

**Essai numéro 3 (Trial 4)**
- **PnL cumulé:** -9969.100000000006
- **Win Rate (%):** 49.38%
- **Écart-type des scores:** 1251.9325560907823
- **Adjusted Score:** -6236.482556090783
- **Scores par split:** [-5869.8, -4099.3]

#### c) Résultats des 5 meilleurs `PnL cumulé` triés par `Écart-type des scores` croissant

**Essai numéro 1 (Trial 3)**
- **PnL cumulé:** -7411.0
- **Win Rate (%):** 49.39%
- **Écart-type des scores:** 856.306312016909
- **Adjusted Score:** -4561.806312016909
- **Scores par split:** [-4311.0, -3100.0]

**Essai numéro 2 (Trial 2)**
- **PnL cumulé:** -4981.4000000000015
- **Win Rate (%):** 49.89%
- **Écart-type des scores:** 870.3070262844027
- **Adjusted Score:** -3361.0070262844024
- **Scores par split:** [-3106.1, -1875.3]

**Essai numéro 3 (Trial 1)**
- **PnL cumulé:** -12714.300000000003
- **Win Rate (%):** 48.89%
- **Écart-type des scores:** 1182.2118274657894
- **Adjusted Score:** -7539.361827465789
- **Scores par split:** [-7193.1, -5521.2]

**Essai numéro 4 (Trial 4)**
- **PnL cumulé:** -9969.100000000006
- **Win Rate (%):** 49.38%
- **Écart-type des scores:** 1251.9325560907823
- **Adjusted Score:** -6236.482556090783
- **Scores par split:** [-5869.8, -4099.3]

#### d) Résultats des 10 meilleurs `Win Rate (%)`

**Essai numéro 1 (Trial 2)**
- **PnL cumulé:** -4981.4000000000015
- **Win Rate (%):** 49.89%
- **Écart-type des scores:** 870.3070262844027
- **Adjusted Score:** -3361.0070262844024
- **Scores par split:** [-3106.1, -1875.3]

**Essai numéro 2 (Trial 3)**
- **PnL cumulé:** -7411.0
- **Win Rate (%):** 49.39%
- **Écart-type des scores:** 856.306312016909
- **Adjusted Score:** -4561.806312016909
- **Scores par split:** [-4311.0, -3100.0]

**Essai numéro 3 (Trial 4)**
- **PnL cumulé:** -9969.100000000006
- **Win Rate (%):** 49.38%
- **Écart-type des scores:** 1251.9325560907823
- **Adjusted Score:** -6236.482556090783
- **Scores par split:** [-5869.8, -4099.3]

**Essai numéro 4 (Trial 1)**
- **PnL cumulé:** -12714.300000000003
- **Win Rate (%):** 48.89%
- **Écart-type des scores:** 1182.2118274657894
- **Adjusted Score:** -7539.361827465789
- **Scores par split:** [-7193.1, -5521.2]

## PHASE 2: ENTRAÎNEMENT FINAL (X_train complet et ensemble de validation nouveau X_test)

### 5. MÉTRIQUES D'ENTRAÎNEMENT FINAL

Les métriques d'entraînement final ne sont pas disponibles dans les données fournies.
Veuillez les fournir dans le prompt à ChatGPT pour une analyse complète.

### 6. COURBES D'APPRENTISSAGE

La figure sélectionnée représente les courbes d'apprentissage XGBoost sur le modèle final avec les paramètres optimaux trouvés par Optuna.

Cette figure inclut:
1. Résultats sur X_train et X_train normalisé
2. Résultats sur la métrique d'évaluation basée sur X_test avec normalisation de la figure
3. Un zoom sur la partie avant l'early stopping

Veuillez fournir des détails supplémentaires sur ces résultats dans le prompt à ChatGPT pour une analyse approfondie.

## CRITÈRES DE RÉFÉRENCE POUR L'ANALYSE DES COURBES

### Courbe X_train (bleue) - Caractéristiques Optimales
1. Progression régulière mais pas trop rapide
2. Pente modérée (pas verticale)
3. Légère courbure (forme concave) indiquant un apprentissage progressif
4. Potentiellement un léger plateau vers la fin, mais pas plat trop tôt

### Courbe X_test (rouge) - Caractéristiques Optimales
1. Progression similaire à X_train avec :
   - Score plus bas que X_train (écart raisonnable)
   - Oscillations minimales
   - Tendance haussière stable jusqu'à l'early stopping
2. Absence de chute brutale après le pic
3. Convergence visible vers un niveau stable

Veuillez fournir les courbes ou les données correspondantes pour une analyse détaillée.

## STRUCTURE DU RAPPORT D'ANALYSE

1. **Analyse des Paramètres Optimisés**
   - Évaluation des paramètres par rapport aux plages définies
   - Identification des paramètres proches des bornes
   - Analyse de la robustesse des paramètres (via écart-type des scores)

2. **Analyse des Performances de Trading**
   - Évaluation des métriques de trading (validation)
   - Comparaison des performances train/test
   - Analyse du compromis précision/rappel

3. **Analyse des Courbes d'Entraînement Final**
   - Évaluation de la progression sur X_train
   - Analyse de la stabilité sur X_test
   - Évaluation de l'early stopping
   - Diagnostic du surapprentissage

4. **Diagnostic Global**
   - Points forts du modèle
   - Points d'attention
   - Risques identifiés

5. **Recommandations**
   - Ajustements proposés pour les paramètres
   - Modifications suggérées pour les plages Optuna
   - Suggestions pour la stratégie d'entraînement

6. **Plan d'Action**
   - Prochaines étapes prioritaires
   - Tests suggérés
   - Modifications à valider

