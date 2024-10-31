# Rapport d'Analyse XGBoost

## PHASE 1: OPTIMISATION DES PARAMÈTRES VIA VALIDATION CROISÉE TEMPORELLE (X_train et X_val) sur l'ensemble df_train

### 3. MEILLEURS PARAMÈTRES TROUVÉS PAR OPTUNA (sur X_TRAIN et son ensemble de validation X_VAL) PAR TYPE DE SCORE

#### a) Paramètres pour les trois meilleures mises à jour du meilleur essai selon `best_trial_with_2_Obj`

**Essai numéro 1 (Trial 626)**
- **PnL cumulé:** 274.5
- **Win Rate (%):** 53.87%
- **Écart-type des scores:** 23.821067246952317
- **Adjusted Score:** 25.314726917164112
- **Scores par split:** [54.4, 10.3, 37.9, 72.3, 70.3, 29.3]

**Paramètres détaillés:**
- **max_depth:** 11
- **learning_rate:** 0.03450847780313629
- **min_child_weight:** 12
- **subsample:** 0.760175661963354
- **colsample_bytree:** 0.7509180192247786
- **colsample_bylevel:** 0.6466425147215089
- **colsample_bynode:** 0.5808143570473069
- **gamma:** 8.09013362873476
- **reg_alpha:** 8.904782345122019
- **reg_lambda:** 3.5064102137767668
- **threshold:** 0.5492795931445461
- **profit_per_tp:** 1.0
- **loss_per_fp:** -1.1
- **penalty_per_fn:** -0.0
- **num_boost_round:** 1419
- **w_p:** 1.0060739006154624
- **w_n:** 1.0
- **weight_split:** 0.65
- **nb_split_weight:** 2
- **std_penalty_factor:** 0.9321616102834998

**Essai numéro 2 (Trial 521)**
- **PnL cumulé:** 231.4999999999991
- **Win Rate (%):** 53.22%
- **Écart-type des scores:** 20.2521644942588
- **Adjusted Score:** 21.188690865380867
- **Scores par split:** [36.6, 18.1, 44.4, 71.5, 42.4, 18.5]

**Paramètres détaillés:**
- **max_depth:** 11
- **learning_rate:** 0.05096611771196284
- **min_child_weight:** 11
- **subsample:** 0.8559751091715248
- **colsample_bytree:** 0.710507911538572
- **colsample_bylevel:** 0.6210349912487622
- **colsample_bynode:** 0.5808143570473069
- **gamma:** 13.478312827906189
- **reg_alpha:** 5.166731775903051
- **reg_lambda:** 3.5064102137767668
- **threshold:** 0.5492795931445461
- **profit_per_tp:** 1.0
- **loss_per_fp:** -1.1
- **penalty_per_fn:** -0.0
- **num_boost_round:** 1049
- **w_p:** 1.0060739006154624
- **w_n:** 1.0
- **weight_split:** 0.65
- **nb_split_weight:** 2
- **std_penalty_factor:** 0.9321616102834998

**Essai numéro 3 (Trial 268)**
- **PnL cumulé:** 268.39999999999964
- **Win Rate (%):** 54.71%
- **Écart-type des scores:** 20.11853494873682
- **Adjusted Score:** 24.70117704600548
- **Scores par split:** [48.9, 43.7, 34.2, 60.4, 66.8, 14.4]

**Paramètres détaillés:**
- **max_depth:** 11
- **learning_rate:** 0.03450847780313629
- **min_child_weight:** 12
- **subsample:** 0.835820463828898
- **colsample_bytree:** 0.7509180192247786
- **colsample_bylevel:** 0.6466425147215089
- **colsample_bynode:** 0.5434601440437119
- **gamma:** 8.09013362873476
- **reg_alpha:** 8.904782345122019
- **reg_lambda:** 12.166475353872201
- **threshold:** 0.5904206602446541
- **profit_per_tp:** 1.0
- **loss_per_fp:** -1.1
- **penalty_per_fn:** -0.0
- **num_boost_round:** 1419
- **w_p:** 1.27352219505033
- **w_n:** 1.0
- **weight_split:** 0.65
- **nb_split_weight:** 2
- **std_penalty_factor:** 0.9854215577252513

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

