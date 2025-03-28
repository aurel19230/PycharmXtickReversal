import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.metrics import roc_auc_score

import os

from func_standard import (print_notification, load_data, calculate_naked_poc_distances, CUSTOM_SESSIONS, \
    save_features_with_sessions,remplace_0_nan_reg_slope_p_2d,process_reg_slope_replacement,
                           calculate_slopes_and_r2_numba,calculate_atr,calculate_percent_bb,enhanced_close_to_sma_ratio,calculate_imbalance,split_sessions)
file_name = "Step4_version2_170924_110325_bugFixTradeResult1_extractOnlyFullSession_OnlyShort.csv"
#file_name = "Step4_5_0_5TP_1SL_150924_280225_bugFixTradeResult_extractOnlyFullSession_OnlyShort.csv"
# Chemin du répertoire
directory_path =  r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version1"
directory_path =  r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version2\merge"
file_name_test = "Step4_version2_100325_260325_bugFixTradeResult1_extractOnlyFullSession_OnlyShort.csv"
file_path= os.path.join(directory_path, file_name)


df_init = load_data(file_path)

# Variable globale pour contrôler l'arrêt de l'optimisation
DF_TEST_CALCULATION = False
STOP_OPTIMIZATION = False


directory_path_test =  r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version2\merge\extend"
file_path_test = os.path.join(directory_path_test, file_name_test)
df_test = load_data(file_path_test)

test_size_ratio=0.2
df_train, nb_SessionTrain, df_val, nb_SessionVal = split_sessions(df_init, test_size=test_size_ratio,
                                                                            min_train_sessions=2,
                                                                            min_test_sessions=2)


def afficher_stats_dataset(df_,nb_Session ,nom_df="Dataset"):
    df4Stat = df_[df_['class_binaire'].isin([0, 1])].copy()
    target_y = df4Stat['class_binaire']

    print(f"\n📊 Statistiques pour {nom_df}")
    print(f"Nombre total de lignes : {len(df_)}")
    print(f"Nombre total de session : {nb_Session}")
    print(f"Nombre de lignes après filtrage (0 ou 1 uniquement) : {len(df4Stat)}")

    winrate = target_y.mean()
    print(f"✅ Winrate global : {winrate:.4f} ({winrate:.2%})")

    counts = target_y.value_counts()
    value_counts = target_y.value_counts(normalize=True)

    print(f"🔢 Nombre de trades gagnants (1) : {counts.get(1, 0)}")
    print(f"🔢 Nombre de trades perdants (0) : {counts.get(0, 0)}")
    print("📈 Distribution (%) des classes :")
    print(value_counts)

    # print("📋 Aperçu des 50 dernières valeurs :")
    # print(target_y.tail(50))
    # print("-" * 50)

afficher_stats_dataset(df_train,nb_SessionTrain ,nom_df="Train")
afficher_stats_dataset(df_val, nb_SessionVal,nom_df="Val")
#
df_val_filtered = df_val[df_val['class_binaire'].isin([0, 1])].copy()
target_val_y = df_val_filtered['class_binaire']


from stats_sc.standard_stat_sc import *

# Construction du chemin complet du fichier
import keyboard  # Assurez-vous d'installer cette bibliothèque avec: pip install keyboard

import keyboard  # Assurez-vous d'installer cette bibliothèque avec: pip install keyboard
from pynput import keyboard as pynput_keyboard  # Alternative si keyboard pose problème




# Définir le type d'indicateur à optimiser
indicator_type = "stochastic"
# Add these global flags to control which zones to optimize


if indicator_type == "regression_slope":
    OPTIMIZE_OVERSOLD = False  # Set to False to disable oversold zone optimization => => find the worst winrate
    OPTIMIZE_OVERBOUGHT = True  # Set to False to disable overbought zone optimization => optimize de winrate
    SPLIT_SCORE_VAL=0.65 #0.6 represente que le score val compte pour 60% et 40% pour train

    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.0501  # Écart minimum entre bins
    MAX_BIN_0_WIN_RATE = 0.471
    MAX_BIN_1_WIN_RATE = 0.52  # Minimum pour bin1 (doit être > 0.5) => optimize de winrate (work with OPTIMIZE_OVERBOUGHT param)
    MIN_BIN_SIZE_0 = 0.0007  # Taille minimale pour un bin individuel => find the worst winrate (work with OPTIMIZE_OVERSOLD param)
    MIN_BIN_SIZE_1 = 0.07  # Taille minimale pour un bin individuel => optimize de winrate (work with OPTIMIZE_OVERBOUGHT param)
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    PERIOD_VAR_L = 25
    PERIOD_VAR_H = 40

    SLOPE_RANGE_THRESHOLD_L = -0.3
    SLOPE_RANGE_THRESHOLD_H = 0.5

    SLOPE_EXTREM_THRESHOLD_L = -0.3
    SLOPE_EXTREM_THRESHOLD_H = 0.5
if indicator_type == "atr":
    OPTIMIZE_OVERSOLD = False  # Set to False to disable oversold zone optimization => => find the worst winrate
    OPTIMIZE_OVERBOUGHT = True  # Set to False to disable overbought zone optimization => optimize de winrate
    SPLIT_SCORE_VAL=0.1 #0.6 represente que le score val compte pour 60% et 40% pour train

    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.0501  # Écart minimum entre bins
    MAX_BIN_0_WIN_RATE = 0.474  # Maximum pour bin0 (doit être < 0.5) => find the worst winrate (work with OPTIMIZE_OVERSOLD param)
    MAX_BIN_1_WIN_RATE = 0.515# Minimum pour bin1 (doit être > 0.5) => optimize de winrate (work with OPTIMIZE_OVERBOUGHT param)²
    MIN_BIN_SIZE_0 = 0.001  # Taille minimale pour un bin individuel => find the worst winrate (work with OPTIMIZE_OVERSOLD param)
    MIN_BIN_SIZE_1 = 0.07  # Taille minimale pour un bin individuel => optimize de winrate (work with OPTIMIZE_OVERBOUGHT param)
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    PERIOD_VAR_L = 10
    PERIOD_VAR_H = 25

    ATR_LOW_THRESHOLD_L = 1.5
    ATR_LOW_THRESHOLD_H = 6

    ATR_HIGH_THRESHOLD_L = 6
    ATR_HIGH_THRESHOLD_H = 13

if indicator_type == "zscore":
    OPTIMIZE_OVERSOLD = False  # Set to False to disable oversold zone optimization => => find the worst winrate
    OPTIMIZE_OVERBOUGHT = True  # Set to False to disable overbought zone optimization => optimize de winrate
    SPLIT_SCORE_VAL=0.5 #0.6 represente que le score val compte pour 60% et 40% pour train

    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.0501  # Écart minimum entre bins
    MAX_BIN_0_WIN_RATE = 0.544  # Maximum pour bin0 (doit être < 0.5) => find the worst winrate
    MAX_BIN_1_WIN_RATE = 0.52  # Minimum pour bin1 (doit être > 0.5) => optimize de winrate
    MIN_BIN_SIZE_0 = 0.0007 # Taille minimale pour un bin individuel pour le worst winrate
    MIN_BIN_SIZE_1 = 0.12 # Taille minimale pour un bin individuel pour optimize winrate
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    # Paramètres spécifiques à l'indicateur Z-Score
    PERIOD_ZSCORE_ZCORE_L = 25
    PERIOD_ZSCORE_ZCORE_H = 55

    ZSCORE_LOW_THRESHOLD_L = -1
    ZSCORE_LOW_THRESHOLD_H = 0

    ZSCORE_HIGH_THRESHOLD_L = 0
    ZSCORE_HIGH_THRESHOLD_H = 3

if indicator_type == "vwap":
    OPTIMIZE_OVERSOLD = True  # Set to False to disable oversold zone optimization => => find the worst winrate
    OPTIMIZE_OVERBOUGHT = False  # Set to False to disable overbought zone optimization => optimize de winrate
    SPLIT_SCORE_VAL=0.85 #0.6 represente que le score val compte pour 60% et 40% pour train

    # Définir des constantes globales pour les contraintes
    MAX_BIN_0_WIN_RATE = 0.475 # Maximum pour bin0 (doit être < 0.5) => find the worst winrate (work with OPTIMIZE_OVERSOLD param)
    MAX_BIN_1_WIN_RATE = 0.55  # Minimum pour bin1 (doit être > 0.5) => optimize de winrate (work with OPTIMIZE_OVERBOUGHT param)
    MIN_BIN_SIZE_0 = 0.125  # Taille minimale pour bin0 => find the worst winrate (work with OPTIMIZE_OVERSOLD param)
    MIN_BIN_SIZE_1 = 0.00001125  # Taille minimale pour bin1 => optimize de winrate (work with OPTIMIZE_OVERBOUGHT param)
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    # Seuils pour la différence entre prix et VWAP
    VWAP_LOW_L = -50 # Limite inférieure pour le seuil bas
    VWAP_LOW_H = 0  # Limite supérieure pour le seuil bas

    VWAP_HIGH_L = 0  # Limite inférieure pour le seuil haut
    VWAP_HIGH_H = 50  # Limite supérieure pour le seuil haut



if indicator_type == "regression_std":
    OPTIMIZE_OVERSOLD = False  # Set to False to disable oversold zone optimization => => find the worst winrate
    OPTIMIZE_OVERBOUGHT = True  # Set to False to disable overbought zone optimization => optimize de winrate
    SPLIT_SCORE_VAL=0.65 #0.6 represente que le score val compte pour 60% et 40% pour train

    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.0501  # Écart minimum entre bins
    MAX_BIN_0_WIN_RATE = 0.47  # Maximum pour bin0 (doit être < 0.5)
    MAX_BIN_1_WIN_RATE = 0.535  # Minimum pour bin1 (doit être > 0.5)
    MIN_BIN_SIZE_0 = 0.0000010  # Taille minimale pour un bin individuel
    MIN_BIN_SIZE_1 = 0.06 # Taille minimale pour un bin individuel
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    PERIOD_VAR_L=10
    PERIOD_VAR_H=35

    STD_LOW_THRESHOLD_L=0
    STD_LOW_THRESHOLD_H=2

    STD_HIGH_THRESHOLD_L=1
    STD_HIGH_THRESHOLD_H=3

if indicator_type == "regression_r2":
    OPTIMIZE_OVERSOLD = False  # Set to False to disable oversold zone optimization => => find the worst winrate
    OPTIMIZE_OVERBOUGHT = True  # Set to False to disable overbought zone optimization => optimize de winrate
    SPLIT_SCORE_VAL=0.65 #0.6 represente que le score val compte pour 60% et 40% pour train

    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.0501  # Écart minimum entre bins
    MAX_BIN_0_WIN_RATE = 0.48  # Maximum pour bin0 (doit être < 0.5)
    MAX_BIN_1_WIN_RATE = 0.515  # Minimum pour bin1 (doit être > 0.5)
    MIN_BIN_SIZE_0 = 0.0001  # Taille minimale pour un bin individuel
    MIN_BIN_SIZE_1 = 0.9  # Taille minimale pour un bin individuel
    COEFF_SPREAD = 0.7
    COEFF_BIN_SIZE = 0.3

    # Paramètres pour l'optimisation Optuna
    PERIOD_VAR_L = 12
    PERIOD_VAR_H = 30

    R2_LOW_THRESHOLD_L = 0.0#très volatile
    R2_LOW_THRESHOLD_H = 0.3

    R2_HIGH_THRESHOLD_L = 0
    R2_HIGH_THRESHOLD_H = 0.3
if indicator_type == "stochastic":
    OPTIMIZE_OVERSOLD = False  # Set to False to disable oversold zone optimization => => find the worst winrate
    OPTIMIZE_OVERBOUGHT = True  # Set to False to disable overbought zone optimization => optimize de winrate
    SPLIT_SCORE_VAL=0.15 #0.6 represente que le score val compte pour 60% et 40% pour train

    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.0501  # Écart minimum entre bins
    MAX_BIN_0_WIN_RATE = 0.48  # Maximum pour bin0 (doit être < 0.5)
    MAX_BIN_1_WIN_RATE = 0.529  # Minimum pour bin1 (doit être > 0.5)
    MIN_BIN_SIZE_0 = 0.0001# Taille minimale pour un bin individuel
    MIN_BIN_SIZE_1 = 0.107  # Taille minimale pour un bin individuel
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    # Paramètres pour l'optimisation Optuna
    K_PERIOD_L = 30
    K_PERIOD_H = 85
    D_PERIOD_L = 30  # Doit être >= K_PERIOD dans la fonction
    D_PERIOD_H = 85

    OS_LIMIT_L = 10
    OS_LIMIT_H = 38

    OB_LIMIT_L = 55
    OB_LIMIT_H = 97
if indicator_type == "williams_r":
    OPTIMIZE_OVERSOLD = True  # Set to False to disable oversold zone optimization => => find the worst winrate
    OPTIMIZE_OVERBOUGHT = False  # Set to False to disable overbought zone optimization => optimize de winrate
    SPLIT_SCORE_VAL=0.65 #0.6 represente que le score val compte pour 60% et 40% pour train

    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.03  # Écart minimum entre bins
    MAX_BIN_0_WIN_RATE = 0.445  # Maximum pour bin0 (doit être < 0.5)
    MAX_BIN_1_WIN_RATE = 0.529  # Minimum pour bin1 (doit être > 0.5)
    MIN_BIN_SIZE_0 = 0.075  # Taille minimale pour un bin individuel
    MIN_BIN_SIZE_1 = 0.0000001  # Taille minimale pour un bin individuel
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    # Paramètres pour l'optimisation Optuna
    PERIOD_L = 90
    PERIOD_H = 120

    OS_LIMIT_L = -90
    OS_LIMIT_H = -65

    OB_LIMIT_L = -40
    OB_LIMIT_H = -3

if indicator_type == "mfi":
    OPTIMIZE_OVERSOLD = True  # Set to False to disable oversold zone optimization => => find the worst winrate
    OPTIMIZE_OVERBOUGHT = False  # Set to False to disable overbought zone optimization => optimize de winrate
    SPLIT_SCORE_VAL=0.65 #0.6 represente que le score val compte pour 60% et 40% pour train

    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.02  # Écart minimum entre bins
    MAX_BIN_0_WIN_RATE = 0.46  # Maximum pour bin0 (doit être < 0.5)
    MAX_BIN_1_WIN_RATE = 0.52  # Minimum pour bin1 (doit être > 0.5)
    MIN_BIN_SIZE_0 = 0.00001  # Taille minimale pour un bin individuel
    MIN_BIN_SIZE_1 = 0.07  # Taille minimale pour un bin individuel
    COEFF_SPREAD = 0.6
    COEFF_BIN_SIZE = 0.4

    # Paramètres pour l'optimisation Optuna
    PERIOD_L = 23
    PERIOD_H = 60

    OS_LIMIT_L = 25
    OS_LIMIT_H = 50

    OB_LIMIT_L = 55
    OB_LIMIT_H = 85
if indicator_type == "mfi_divergence":
    OPTIMIZE_OVERSOLD = True  # Set to False to disable oversold zone optimization => => find the worst winrate
    OPTIMIZE_OVERBOUGHT = False  # Set to False to disable overbought zone optimization => optimize de winrate
    SPLIT_SCORE_VAL=0.65 #0.6 represente que le score val compte pour 60% et 40% pour train

    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.01  # Écart minimum entre bins
    MAX_BIN_0_WIN_RATE = 0.542  # Maximum pour bin0 (doit être < 0.5)
    MAX_BIN_1_WIN_RATE = 0.5358# Minimum pour bin1 (doit être > 0.5)
    MIN_BIN_SIZE_0 = 0.085  # Taille minimale pour un bin individuel
    MIN_BIN_SIZE_1 = 0.08  # Taille minimale pour un bin individuel
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    # Paramètres pour l'optimisation Optuna
    MFI_PERIOD_L = 3
    MFI_PERIOD_H = 15

    DIV_LOOKBACK_L = 3
    DIV_LOOKBACK_H = 15

    MIN_PRICE_INCREASE_L = 0.0008
    MIN_PRICE_INCREASE_H = 0.008

    MIN_MFI_DECREASE_L = 0.0008
    MIN_MFI_DECREASE_H = 0.007

if indicator_type == "percent_bb_simu":
    OPTIMIZE_OVERSOLD = True  # Set to False to disable oversold zone optimization => => find the worst winrate
    OPTIMIZE_OVERBOUGHT = False  # Set to False to disable overbought zone optimization => optimize de winrate
    SPLIT_SCORE_VAL=0.65 #0.6 represente que le score val compte pour 60% et 40% pour train

    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.0501  # Écart minimum entre bins
    MAX_BIN_0_WIN_RATE = 0.467  # Maximum pour bin0
    MAX_BIN_1_WIN_RATE = 0.53  # Minimum pour bin1
    MIN_BIN_SIZE_0 = 0.000009  # Taille minimale pour bin0
    MIN_BIN_SIZE_1 = 0.1  # Taille minimale pour bin1
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    # Paramètres spécifiques à l'indicateur
    PERCTBB_PERIOD_L = 95
    PERCTBB_PERIOD_H = 115

    PERCTBB_STD_DEV_L = 1.4
    PERCTBB_STD_DEV_H = 2.2

    PERCTBB_LOW_THRESHOLD_L = 0
    PERCTBB_LOW_THRESHOLD_H = 0.7

    PERCTBB_HIGH_THRESHOLD_L = 0.8
    PERCTBB_HIGH_THRESHOLD_H = 1.2

else:
    print(f"Type d'indicateur non reconnu: {indicator_type}")




# Function to display current optimization mode
def check_bin_constraints(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate,params_check_constraints=None,optimize_oversold=None,optimize_overbought=None):
    """
    Vérifie les contraintes pour les bins selon le mode d'optimisation choisi.
    """

    # Special handling when only one mode is enabled
    if optimize_oversold:
        # Only check oversold constraints
        if bin_0_win_rate > params_check_constraints['max_bin_0_win_rate']:
            #print("❌ Rejected: bin_0_win_rate too low")
            return False
        if bin_0_pct < params_check_constraints['min_bin_size_0']:
            #print("❌ Rejected: bin_0_pct too small")
            return False
        # No need to check bin spread or bin 1 constraints
        return True

    elif optimize_overbought:
        # Only check overbought constraints
        # print(
        #     f"Checking overbought constraints: win_rate={bin_1_win_rate}, min={MAX_BIN_1_WIN_RATE}, pct={bin_1_pct}, min_size={MIN_BIN_SIZE_1}")
        if bin_1_win_rate < params_check_constraints['max_bin_1_win_rate']:
            # print("❌ Rejected: bin_1_win_rate too low")
            return False
        if bin_1_pct < params_check_constraints['min_bin_size_1']:
            # print("❌ Rejected: bin_1_pct too small")
            return False
        print("✅ Constraints satisfied!")
        return True

    else:
        print("Both modes enabled, check all constraints")
        exit(10)
    print("-" * 60)
# Then modify the constraint checking in your objective functions like this:


def get_bin_name(indicator_type, bin_index):
    """Retourne le nom approprié pour un bin en fonction du type d'indicateur."""
    if bin_index == 0:  # Bin 0 (généralement pour minimiser le win rate)
        if indicator_type == "stochastic" or indicator_type == "williams_r" or indicator_type == "mfi":
            return "Survente"
        elif indicator_type == "mfi_divergence":
            return "Divergence Haussière"
        elif indicator_type == "regression_r2":
            return "Volatilité Basse"
        elif indicator_type == "regression_std":
            return "Écart-type Faible"
        elif indicator_type == "regression_slope":
            return "Volatilité Extrême"
        elif indicator_type == "atr":
            return "ATR Faible"
        elif indicator_type == "vwap":
            return "Distance VWAP Extrême"
        elif indicator_type == "percent_bb_simu":
            return "%B Extrême"
        elif indicator_type == "zscore":
            return "Z-Score Extrême"

        else:
            return "Bin 0"
    else:  # Bin 1 (généralement pour maximiser le win rate)
        if indicator_type == "stochastic" or indicator_type == "williams_r" or indicator_type == "mfi":
            return "Surachat"
        elif indicator_type == "mfi_divergence":
            return "Divergence Baissière"
        elif indicator_type == "regression_r2":
            return "Volatilité Haute"
        elif indicator_type == "regression_std":
            return "Écart-type Élevé"
        elif indicator_type == "regression_slope":
            return "Volatilité Modérée"
        elif indicator_type == "atr":
            return "ATR Modéré"
        elif indicator_type == "vwap":
            return "Distance VWAP Modérée"
        elif indicator_type == "percent_bb_simu":
            return "%B Modéré"
        elif indicator_type == "zscore":
            return "Z-Score Modéré"

        else:
            return "Bin 1"
def print_test_results(results, params, indicator_type):
    """
    Affiche les résultats de l'évaluation sur les données de test.

    Parameters:
    -----------
    results : dict
        Dictionnaire contenant les métriques d'évaluation
    params : dict
        Dictionnaire contenant les paramètres utilisés
    indicator_type : str
        Type d'indicateur évalué
    """
    # Récupérer les métriques principales
    bin_0_win_rate = results.get('bin_0_win_rate', 0)
    bin_1_win_rate = results.get('bin_1_win_rate', 0)
    bin_0_pct = results.get('bin_0_pct', 0)
    bin_1_pct = results.get('bin_1_pct', 0)
    bin_spread = results.get('bin_spread', 0)
    bin_0_samples = results.get('bin_0_samples', 0)
    bin_1_samples = results.get('bin_1_samples', 0)

    # Afficher les paramètres
    print("\n📊 Paramètres utilisés:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Adapter le nom des bins selon l'indicateur
    bin0_name = get_bin_name(indicator_type, 0)
    bin1_name = get_bin_name(indicator_type, 1)

    # Afficher les métriques de bin 0 si pertinent
    if OPTIMIZE_OVERSOLD and bin_0_samples > 0:
        print(f"\n📉 Résultats pour Bin 0 ({bin0_name}):")
        print(f"  Win Rate: {bin_0_win_rate:.4f}")
        print(f"  Nombre d'échantillons: {bin_0_samples}")
        print(f"  Trades gagnants: {results.get('oversold_success_count', 0)}")
        print(f"  Pourcentage des données: {bin_0_pct:.2%}")

    # Afficher les métriques de bin 1 si pertinent
    if OPTIMIZE_OVERBOUGHT and bin_1_samples > 0:
        print(f"\n📈 Résultats pour Bin 1 ({bin1_name}):")
        print(f"  Win Rate: {bin_1_win_rate:.4f}")
        print(f"  Nombre d'échantillons: {bin_1_samples}")
        print(f"  Trades gagnants: {results.get('overbought_success_count', 0)}")
        print(f"  Pourcentage des données: {bin_1_pct:.2%}")

    # Afficher le spread si pertinent
    if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT and bin_0_samples > 0 and bin_1_samples > 0:
        print(f"\n⚖️ Spread: {bin_spread:.4f}")

    # Afficher des statistiques de généralisation
    print("\n🔄 Généralisation:")
    global_winrate = results.get('test_winrate_global', 0.5)

    if OPTIMIZE_OVERSOLD and bin_0_samples > 0:
        expected_bin_0_wr = MAX_BIN_0_WIN_RATE if 'MAX_BIN_0_WIN_RATE' in globals() else 0.47
        if bin_0_win_rate <= expected_bin_0_wr:
            print(f"  ✅ Bin 0: Win rate ({bin_0_win_rate:.4f}) ≤ {expected_bin_0_wr:.4f}")
        else:
            print(f"  ❌ Bin 0: Win rate ({bin_0_win_rate:.4f}) > {expected_bin_0_wr:.4f}")

    if OPTIMIZE_OVERBOUGHT and bin_1_samples > 0:
        expected_bin_1_wr = MAX_BIN_1_WIN_RATE if 'MAX_BIN_1_WIN_RATE' in globals() else 0.53
        if bin_1_win_rate >= expected_bin_1_wr:
            print(f"  ✅ Bin 1: Win rate ({bin_1_win_rate:.4f}) ≥ {expected_bin_1_wr:.4f}")
        else:
            print(f"  ❌ Bin 1: Win rate ({bin_1_win_rate:.4f}) < {expected_bin_1_wr:.4f}")

    # Conclusion
    print(f"\n📝 Conclusion pour {indicator_type}:")
    if (OPTIMIZE_OVERSOLD and bin_0_win_rate <= expected_bin_0_wr) or \
            (OPTIMIZE_OVERBOUGHT and bin_1_win_rate >= expected_bin_1_wr):
        print("  L'indicateur généralise bien sur les données de test.")
    else:
        print("  L'indicateur montre des signes de surajustement (overfitting).")

    print(f"{'-' * 80}\n")
def evaluate_best_params_on_test_data(best_params, indicator_type, df_test):
    """
    Évalue les meilleurs paramètres d'un indicateur sur un jeu de données de test.

    Parameters:
    -----------
    best_params : dict
        Dictionnaire contenant les meilleurs paramètres trouvés
    indicator_type : str
        Type d'indicateur ('stochastic', 'williams_r', 'mfi', etc.)
    df_test : pandas.DataFrame
        DataFrame contenant les données de test

    Returns:
    --------
    dict
        Dictionnaire contenant les métriques d'évaluation
    """
    print(f"\n{'=' * 80}")
    print(f"ÉVALUATION DES MEILLEURS PARAMÈTRES POUR '{indicator_type}' SUR LES DONNÉES DE TEST")
    print(f"{'=' * 80}")



    # Initialiser les résultats
    test_results = {}

    # Pour chaque type d'indicateur, appliquer la logique d'évaluation appropriée
    if indicator_type == "stochastic":
        test_results,df_test_filtered,target_y_test =  evaluate_stochastic(best_params, df_test, optimize_oversold=OPTIMIZE_OVERSOLD, optimize_overbought=OPTIMIZE_OVERBOUGHT)
    elif indicator_type == "williams_r":
        test_results,df_test_filtered,target_y_test =  evaluate_williams_r(best_params, df_test, optimize_oversold=OPTIMIZE_OVERSOLD, optimize_overbought=OPTIMIZE_OVERBOUGHT)
    elif indicator_type == "mfi":
        test_results,df_test_filtered,target_y_test =  evaluate_mfi(best_params, df_test, optimize_oversold=OPTIMIZE_OVERSOLD, optimize_overbought=OPTIMIZE_OVERBOUGHT)
    elif indicator_type == "mfi_divergence":
        test_results,df_test_filtered,target_y_test =  evaluate_mfi_divergence(best_params,df_test, optimize_oversold=OPTIMIZE_OVERSOLD, optimize_overbought=OPTIMIZE_OVERBOUGHT)
    elif indicator_type == "regression_r2":
        test_results,df_test_filtered,target_y_test =  evaluate_regression_r2(best_params, df_test, optimize_oversold=OPTIMIZE_OVERSOLD, optimize_overbought=OPTIMIZE_OVERBOUGHT)
    elif indicator_type == "regression_std":
        test_results,df_test_filtered,target_y_test =  evaluate_regression_std(best_params,df_test, optimize_oversold=OPTIMIZE_OVERSOLD, optimize_overbought=OPTIMIZE_OVERBOUGHT)
    elif indicator_type == "regression_slope":
        test_results,df_test_filtered,target_y_test = evaluate_regression_slope(best_params, df_test,optimize_oversold=OPTIMIZE_OVERSOLD, optimize_overbought=OPTIMIZE_OVERBOUGHT)
    elif indicator_type == "atr":
        test_results,df_test_filtered,target_y_test =  evaluate_atr(best_params, df_test, optimize_oversold=OPTIMIZE_OVERSOLD, optimize_overbought=OPTIMIZE_OVERBOUGHT)
    elif indicator_type == "vwap":
        test_results,df_test_filtered,target_y_test =  evaluate_vwap(best_params, df_test, optimize_oversold=OPTIMIZE_OVERSOLD, optimize_overbought=OPTIMIZE_OVERBOUGHT)
    elif indicator_type == "percent_bb_simu":
        test_results,df_test_filtered,target_y_test =  evaluate_percent_bb(best_params, df_test, optimize_oversold=OPTIMIZE_OVERSOLD, optimize_overbought=OPTIMIZE_OVERBOUGHT)
    elif indicator_type == "zscore":
        test_results,df_test_filtered,target_y_test =  evaluate_zscore(best_params, df_test, optimize_oversold=OPTIMIZE_OVERSOLD, optimize_overbought=OPTIMIZE_OVERBOUGHT)
    else:
        print(f"Type d'indicateur '{indicator_type}' non reconnu.")
        return {}

    # Ajouter quelques métadonnées
    test_results['indicator_type'] = indicator_type
    test_results['total_test_samples'] = len(df_test_filtered)
    test_results['test_winrate_global'] = target_y_test.mean()

    # Afficher les résultats
    print_test_results(test_results, best_params, indicator_type)

    return test_results


# Function to display current optimization mode
def show_optimization_mode():
    print("\n🔄 MODE D'OPTIMISATION ACTUEL:")
    if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
        print("  ✅ Optimisation COMPLÈTE (zones de survente ET de surachat)")
    elif OPTIMIZE_OVERSOLD:
        print("  ✅ Optimisation des zones de SURVENTE uniquement")
        print("  ❌ Zones de surachat ignorées")
    elif OPTIMIZE_OVERBOUGHT:
        print("  ❌ Zones de survente ignorées")
        print("  ✅ Optimisation des zones de SURACHAT uniquement")
    else:
        print("  ⚠️ ATTENTION: Aucune zone activée pour l'optimisation!")
    print("-" * 60)

# Modified score calculation function
def calculate_optimization_score(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread):
    """
    Calcule le score d'optimisation en fonction du mode choisi.
    """
    # Base scores
    normalized_spread = bin_spread * 100  # Convert to percentage
    bin_size_score = 0



    if OPTIMIZE_OVERSOLD:
        # Calculer la distance avec 0.5 (équilibre) - plus le winrate est bas, plus cette valeur est grande
        normalized_win_rate = (0.5 - bin_0_win_rate) * 100  # Déjà présent
        # Ajouter un facteur exponentiel pour favoriser fortement les win rates très bas
        # if bin_0_win_rate < 0.45:
        #     normalized_win_rate = normalized_win_rate * 1.5  # Augmenter encore davantage l'impact des win rates très bas
        bin_size_score = bin_0_pct * 15
        combined_score = (COEFF_SPREAD * normalized_win_rate) + (COEFF_BIN_SIZE * bin_size_score)
        print(
            f"📊 Score calculation: win_rate={bin_0_win_rate}, normalized={normalized_win_rate}, bin_0_pct={bin_0_pct}, bin_size_score={bin_size_score}, combined_score={combined_score}")

    elif OPTIMIZE_OVERBOUGHT:
        # Only optimize overbought, focus on high win rate in bin 1
        normalized_win_rate = (bin_1_win_rate - 0.5) * 100  # Normalize around 0.5, scale up
        # Boost pour les win rates très élevés
        # if bin_1_win_rate > 0.54:  # Vérifier bin_1_win_rate (pas bin_0_win_rate)
        #     normalized_win_rate = normalized_win_rate * 1.5  # Augmenter l'impact des win rates très élevés
        bin_size_score = bin_1_pct * 15
        combined_score = (COEFF_SPREAD * normalized_win_rate) + (COEFF_BIN_SIZE * bin_size_score)
        print(
            f"📊 Score calculation: win_rate={bin_1_win_rate}, normalized={normalized_win_rate}, bin_1_pct={bin_1_pct}, bin_size_score={bin_size_score}, combined_score={combined_score}")
    else:
        # Should never happen, but just in case
        combined_score = 0

    # # Add bonus for exceptional values based on mode
    # if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
    #     # Bonus for high spread
    #     if bin_spread >= 0.15:
    #         combined_score *= 1.15
    #     elif bin_spread >= 0.12:
    #         combined_score *= 1.1
    # elif OPTIMIZE_OVERSOLD:
    #     # Bonus for very low bin_0_win_rate
    #     if bin_0_win_rate <= 0.4:
    #         combined_score *= 1.15
    #     elif bin_0_win_rate <= 0.43:
    #         combined_score *= 1.1
    # elif OPTIMIZE_OVERBOUGHT:
    #     # Bonus for very high bin_1_win_rate
    #     if bin_1_win_rate >= 0.6:
    #         combined_score *= 1.15
    #     elif bin_1_win_rate >= 0.57:
    #         combined_score *= 1.1

    return combined_score
def callback_optuna_stop(study, trial):
    global STOP_OPTIMIZATION
    if STOP_OPTIMIZATION:
        print("Callback triggered: stopping the study.")
        study.stop()

REPLACE_NAN = False
REPLACED_NANVALUE_BY = 90000.54789
REPLACED_NANVALUE_BY_INDEX = 1
if REPLACE_NAN:
    print(
        f"\nINFO : Implémenter dans le code => les valeurs NaN seront remplacées par {REPLACED_NANVALUE_BY} et un index")
else:
    print(
        f"\nINFO : Implémenter dans le code => les valeurs NaN ne seront pas remplacées par une valeur choisie par l'utilisateur mais laissé à NAN")






# Variable globale pour le contrôle d'arrêt
should_stop = False


# Fonction pour afficher les contraintes actuelles
def show_constraints():
    print("\n📋 Contraintes actuelles:")
    #print(f"  MAX_BIN_0_WIN_RATE = {MAX_BIN_0_WIN_RATE}")
    print(f"  MAX_BIN_1_WIN_RATE = {MAX_BIN_1_WIN_RATE}")
    #print(f"  MIN_BIN_SIZE_0 = {MIN_BIN_SIZE_0}")
    print(f"  MIN_BIN_SIZE_1 = {MIN_BIN_SIZE_1}")



# Define a function to modify constraints
def modify_constraints(min_spread=None, max_bin0=None, min_bin1=None, min_size_0=None,min_size_1=None):
    global MIN_BIN_SPREAD, MAX_BIN_0_WIN_RATE, MAX_BIN_1_WIN_RATE, MIN_BIN_SIZE_0,MIN_BIN_SIZE_1

    if min_spread is not None:
        MIN_BIN_SPREAD = min_spread
    if max_bin0 is not None:
        MAX_BIN_0_WIN_RATE = max_bin0
    if min_bin1 is not None:
        MAX_BIN_1_WIN_RATE = min_bin1
    if min_size_0 is not None:
        MIN_BIN_SIZE_0 = min_size_0
    if min_size_1 is not None:
        MIN_BIN_SIZE_1 = min_size_1

    show_constraints()
    print("Contraintes modifiées!")


# Fonction pour calculer les contraintes
def calculate_constraints_optuna(trial):
    bin_0_wr = trial.user_attrs.get('bin_0_win_rate_val', 0)
    bin_1_wr = trial.user_attrs.get('bin_1_win_rate_val', 0)
    bin_spread = trial.user_attrs.get('bin_spread_val', 0)
    bin_0_pct = trial.user_attrs.get('bin_0_pct_val', 0)
    bin_1_pct = trial.user_attrs.get('bin_1_pct_val', 0)

    constraints = []

    if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
        # Même logique que check_bin_constraints pour "both" :
        c1 = MIN_BIN_SPREAD - bin_spread
        c2 = bin_0_wr - MAX_BIN_0_WIN_RATE
        c3 = MAX_BIN_1_WIN_RATE - bin_1_wr
        c4 = MIN_BIN_SIZE_0 - bin_0_pct
        c5 = MIN_BIN_SIZE_1 - bin_1_pct
        constraints = [c1, c2, c3, c4, c5]

    elif OPTIMIZE_OVERSOLD and not OPTIMIZE_OVERBOUGHT:
        # Oversold only => On ignore le spread et bin_1
        # On vérifie seulement MAX_BIN_0_WIN_RATE et MIN_BIN_SIZE_0
        c2 = bin_0_wr - MAX_BIN_0_WIN_RATE       # si c2 > 0 => violation
        c4 = MIN_BIN_SIZE_0 - bin_0_pct          # si c4 > 0 => violation
        constraints = [c2, c4]

    elif not OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
        # Overbought only => On ignore le spread et bin_0
        c3 = MAX_BIN_1_WIN_RATE - bin_1_wr       # si c3 > 0 => violation
        c5 = MIN_BIN_SIZE_1 - bin_1_pct          # si c5 > 0 => violation
        constraints = [c3, c5]

    else:
        # Aucune zone => Aucune contrainte => on met 0
        constraints = [0]

    return constraints


# Fonction pour créer et retourner la fonction de contraintes
def create_constraints_func():
    return calculate_constraints_optuna


# Gestionnaire d'événement clavier pour arrêter l'optimisation
def on_press(key):
    global STOP_OPTIMIZATION
    global DF_TEST_CALCULATION
    global should_stop

    # print(f"[DEBUG] key = {key}, type(key) = {type(key)}")
    if hasattr(key, 'char'):
        # print(f"       key.char = {repr(key.char)}")
        if key.char == '²':
            print("🛑 Stop signal received: stopping the study.")
            STOP_OPTIMIZATION = True
            should_stop = True

        elif key.char == '&':
            print("🧪& Test signal received: calculating on test dataset.")
            DF_TEST_CALCULATION = True


# Callback pour vérifier si l'utilisateur a demandé l'arrêt
def callback_optuna_stop(study, trial):
    global should_stop
    if should_stop:
        print("\n🛑 Optimisation arrêtée par l'utilisateur.")
        # Envoyer un signal pour arrêter l'optimisation
        raise optuna.exceptions.TrialPruned("Arrêté par l'utilisateur")

BEST_STUDY_SCORE = -np.inf


def is_df_test_respect_constraint(study,best_params_, indicator_type, df_test):
    """
    Évalue les performances sur le jeu de TEST avec les meilleurs paramètres.
    Retourne True si les contraintes sont respectées selon check_bin_constraints().
    """
    import sys
    best_trial = study.best_trial
    # print("1")
    try:
        test_results = evaluate_best_params_on_test_data(
            best_params=best_params_,
            indicator_type=indicator_type,
            df_test=df_test
        )
    except Exception as e:
        print(f"❌ Une erreur est survenue lors de l'exécution de `evaluate_best_params_on_test_data`: {e}")
        sys.exit(1)  # Quitte le script avec un code d'erreur
    # print("2")

    bin_0_wr_test = test_results.get("bin_0_win_rate", 0)
    bin_1_wr_test = test_results.get("bin_1_win_rate", 0)
    bin_0_pct_test = test_results.get("bin_0_pct", 0)
    bin_1_pct_test = test_results.get("bin_1_pct", 0)

    bin_0_win_rate_val = best_trial.user_attrs.get('bin_0_win_rate_val', 0)
    bin_1_win_rate_val = best_trial.user_attrs.get('bin_1_win_rate_val', 0)
    # bin_0_pct_val = best_trial.user_attrs.get('bin_0_pct_val', 0)
    # bin_1_pct_val = best_trial.user_attrs.get('bin_1_pct_val', 0)

    bin_0_win_rate_train = best_trial.user_attrs.get('bin_0_win_rate_train', 0)
    bin_1_win_rate_train = best_trial.user_attrs.get('bin_1_win_rate_train', 0)
    # bin_0_pct_train = best_trial.user_attrs.get('bin_0_pct_train', 0)
    # bin_1_pct_train = best_trial.user_attrs.get('bin_1_pct_train', 0)

    params_check_constraints_val = {
    'max_bin_0_win_rate':bin_0_win_rate_val,
    'max_bin_1_win_rate':bin_1_win_rate_val,
    'min_bin_size_0':MIN_BIN_SIZE_0, #pour la taille on prend le hypothe d'optimation
    'min_bin_size_1':MIN_BIN_SIZE_1,
    }


    is_testVSval=check_bin_constraints(
        bin_0_pct_test,
        bin_1_pct_test,
        bin_0_wr_test,
        bin_1_wr_test,params_check_constraints_val,
        optimize_oversold=OPTIMIZE_OVERSOLD,
        optimize_overbought=OPTIMIZE_OVERBOUGHT
    )
    params_check_constraints_train = {
    'max_bin_0_win_rate':bin_0_win_rate_train,
    'max_bin_1_win_rate':bin_1_win_rate_train,
    'min_bin_size_0':MIN_BIN_SIZE_0,#pour la taille on prend le hypothe d'optimation
    'min_bin_size_1':MIN_BIN_SIZE_1,
    }


    is_testVStrain = check_bin_constraints(
        bin_0_pct_test,
        bin_1_pct_test,
        bin_0_wr_test,
        bin_1_wr_test, params_check_constraints_train,
        optimize_oversold=OPTIMIZE_OVERSOLD,
        optimize_overbought=OPTIMIZE_OVERBOUGHT
    )
    print("is_testVSval:", is_testVSval)
    print("is_testVStrain:", is_testVStrain)
    print("is_testVSval & is_testVStrain:", is_testVSval & is_testVStrain)

    return is_testVSval or is_testVStrain,test_results





def print_best_trial_callback(study, trial):
    # N'afficher que périodiquement
    global BEST_STUDY_SCORE
    global DF_TEST_CALCULATION
    global STOP_OPTIMIZATION
    # print(DF_TEST_CALCULATION)
    # print(STOP_OPTIMIZATION)
    test_result_={}
    is_tresult=False
    try:
        if study.best_value >= BEST_STUDY_SCORE and DF_TEST_CALCULATION==True:
            print("study.best_value > BEST_STUDY_SCORE and DF_TEST_CALCULATION==True")
            BEST_STUDY_SCORE = study.best_value
            best_trial = study.best_trial
            best_params_ = best_trial.params
            is_tresult, test_result_=is_df_test_respect_constraint(study, best_params_, indicator_type, df_test)
            if is_tresult:
                print(
                    "✅ Contraintes respectées sur le jeu de test par rapport à val et train. On stoppe l'optimisation.")
                study.stop()
            else:
                print("❌ Contraintes non respectées sur le jeu de test par raport à VAL et/ou TRAIN:  TEST doit avoir WR < à au moins a VAL ou TEST 2\n Pour la taille on prend les données cible d'optim")

    except ValueError:
        BEST_STUDY_SCORE = -np.inf



    if trial.number % 50 == 0 or is_tresult:
        try:
            print(f"\n----⚠️ trial: {trial.number}----")
            best_trial = study.best_trial
            print(f"\n📊 Meilleur Trial jusqu'à présent pour {indicator_type} en mode " +
                  ("OPTIMIZE_OVERBOUGH" if OPTIMIZE_OVERBOUGHT else "OPTIMIZE_OVERSOLD" if OPTIMIZE_OVERSOLD else "INCONNU") + " :")
            print(f"  Trial ID: {best_trial.number}")
            print(f"  Score: {best_trial.value:.4f}")

            # Récupération des métriques utilisateur
            bin_0_wr_val = best_trial.user_attrs.get('bin_0_win_rate_val', None)
            bin_1_wr_val = best_trial.user_attrs.get('bin_1_win_rate_val', None)
            bin_0_pct_val = best_trial.user_attrs.get('bin_0_pct_val', None)
            bin_1_pct_val = best_trial.user_attrs.get('bin_1_pct_val', None)
            bin_spread_val = best_trial.user_attrs.get('bin_spread_val', None)
            column_name_val = best_trial.user_attrs.get('column_name_val', None)

            bin_0_wr_train = best_trial.user_attrs.get('bin_0_win_rate_train', None)
            bin_1_wr_train = best_trial.user_attrs.get('bin_1_win_rate_train', None)
            bin_0_pct_train = best_trial.user_attrs.get('bin_0_pct_train', None)
            bin_1_pct_train = best_trial.user_attrs.get('bin_1_pct_train', None)
            bin_spread_train = best_trial.user_attrs.get('bin_spread_train', None)
            column_name_train = best_trial.user_attrs.get('column_name_train', None)


            # Vérification des paramètres selon l'indicateur
            if 'vwap_low_threshold' in best_trial.params or 'vwap_high_threshold' in best_trial.params:
                vwap_low_threshold = best_trial.params.get('vwap_low_threshold', 'N/A')
                vwap_high_threshold = best_trial.params.get('vwap_high_threshold', 'N/A')

                # Affichage des seuils VWAP
                params_str = f"  📌 Paramètres VWAP: vwap_low_threshold={vwap_low_threshold:.4f}, vwap_high_threshold={vwap_high_threshold:.4f}"
                print(params_str)

            elif 'mfi_period' in best_trial.params:
                print(f"  📌 Paramètres MFI: mfi_period={best_trial.params.get('mfi_period', 'N/A')}, "
                      f"div_lookback={best_trial.params.get('div_lookback', 'N/A')}")

            elif 'k_period' in best_trial.params:
                k_period = best_trial.params.get('k_period', 'N/A')
                d_period = best_trial.params.get('d_period', 'N/A')

                # Construction conditionnelle des paramètres affichés
                params_str = f"  📌 Paramètres Stochastique: k_period={k_period}, d_period={d_period}"
                if OPTIMIZE_OVERSOLD:
                    params_str += f", OS_limit={best_trial.params.get('OS_limit', 'N/A')}"
                if OPTIMIZE_OVERBOUGHT:
                    params_str += f", OB_limit={best_trial.params.get('OB_limit', 'N/A')}"
                print(params_str)

            elif 'r2_low_threshold' in best_trial.params or 'r2_high_threshold' in best_trial.params:
                period_var = best_trial.params.get('period_var_r2', best_trial.params.get('slope', 'N/A'))
                r2_low_threshold = best_trial.params.get('r2_low_threshold', 'N/A')
                r2_high_threshold = best_trial.params.get('r2_high_threshold', 'N/A')

                # Afficher les deux seuils comme pour std, quel que soit le mode d'optimisation
                params_str = f"  📌 Paramètres Régression R²: période={period_var}, r2_low_threshold={r2_low_threshold}, r2_high_threshold={r2_high_threshold}"
                print(params_str)

            elif 'period' in best_trial.params:
                period = best_trial.params.get('period', 'N/A')

                # Affichage conditionnel des paramètres Williams %R / MFI
                params_str = f"  📌 Paramètres: period={period}"
                if OPTIMIZE_OVERSOLD:
                    params_str += f", OS_limit={best_trial.params.get('OS_limit', 'N/A')}"
                if OPTIMIZE_OVERBOUGHT:
                    params_str += f", OB_limit={best_trial.params.get('OB_limit', 'N/A')}"
                print(params_str)

            elif 'std_low_threshold' in best_trial.params or 'std_high_threshold' in best_trial.params:
                period_var_std = best_trial.params.get('period_var_std', 'N/A')
                std_low_threshold = best_trial.params.get('std_low_threshold', 'N/A')
                std_high_threshold = best_trial.params.get('std_high_threshold', 'N/A')

                # Construction conditionnelle des paramètres de régression par écart-type
                params_str = f"  📌 Paramètres Régression Écart-type: période={period_var_std}, std_low_threshold={std_low_threshold}, std_high_threshold={std_high_threshold}"
                print(params_str)

            elif 'slope_range_threshold' in best_trial.params or 'slope_extrem_threshold' in best_trial.params:
                period_var_slope = best_trial.params.get('period_var_slope', 'N/A')
                slope_range_threshold = best_trial.params.get('slope_range_threshold', 'N/A')
                slope_extrem_threshold = best_trial.params.get('slope_extrem_threshold', 'N/A')

                # Construction conditionnelle des paramètres de régression par pente
                params_str = f"  📌 Paramètres Régression Pente: période={period_var_slope}"
                if OPTIMIZE_OVERSOLD:
                    params_str += f", slope_range_threshold={slope_range_threshold} et slope_extrem_threshold={slope_extrem_threshold}"
                if OPTIMIZE_OVERBOUGHT:
                    params_str += f", slope_range_threshold={slope_range_threshold} et slope_extrem_threshold={slope_extrem_threshold}"
                print(params_str)

            elif 'period_var_atr' in best_trial.params or 'atr_low_threshold' in best_trial.params or 'atr_high_threshold' in best_trial.params:
                period_var_atr = best_trial.params.get('period_var_atr', 'N/A')
                atr_low_threshold = best_trial.params.get('atr_low_threshold', 'N/A')
                atr_high_threshold = best_trial.params.get('atr_high_threshold', 'N/A')

                # Construction conditionnelle des paramètres ATR
                params_str = f"  📌 Paramètres ATR: période={period_var_atr}"
                if OPTIMIZE_OVERSOLD:
                    params_str += (f", atr_low_threshold={atr_low_threshold:.4f},"
                                   # f" atr_high_threshold={atr_high_threshold:.4f}"
                                   )
                if OPTIMIZE_OVERBOUGHT:
                    params_str += f", atr_low_threshold={atr_low_threshold:.4f}, atr_high_threshold={atr_high_threshold:.4f}"
                print(params_str)
            elif 'period_var_zscore' in best_trial.params or 'zscore_low_threshold' in best_trial.params or 'zscore_high_threshold' in best_trial.params:
                period_var_zscore = best_trial.params.get('period_var_zscore', 'N/A')
                zscore_low_threshold = best_trial.params.get('zscore_low_threshold', 'N/A')
                zscore_high_threshold = best_trial.params.get('zscore_high_threshold', 'N/A')

                # Construction conditionnelle des paramètres Z-Score
                params_str = f"  📌 Paramètres Z-Score: période={period_var_zscore}"
                if OPTIMIZE_OVERSOLD:
                    params_str += f", zscore_low_threshold={zscore_low_threshold:.4f} | zscore_high_threshold={zscore_high_threshold:.4f}"

                if OPTIMIZE_OVERBOUGHT:
                    params_str += f", zscore_low_threshold={zscore_low_threshold:.4f} | zscore_high_threshold={zscore_high_threshold:.4f}"
                print(params_str)
            else:
                print(f"  📌 Paramètres optimisés: {best_trial.params}")

            # Affichage conditionnel des résultats des bins
            if OPTIMIZE_OVERSOLD and bin_0_wr_val is not None and bin_0_pct_val is not None:
                # Adapte le nom du bin selon l'indicateur
                bin0_name = "Survente"
                if 'vwap_low_threshold' in best_trial.params:
                    bin0_name = "Distance VWAP Extrême"
                elif 'r2_low_threshold' in best_trial.params:
                    bin0_name = "Volatilité Extrême"
                elif 'slope' in best_trial.params:
                    bin0_name = "Volatilité Basse"
                elif 'slope_range_threshold' in best_trial.params:
                    bin0_name = "Pente Faible"
                elif 'period_var_atr' in best_trial.params:
                    bin0_name = "ATR Extrême"
                elif 'mfi_period' in best_trial.params:
                    bin0_name = "Divergence Haussière"

                # Affichage du bin 0 avec le nombre de trades réussis
                oversold_success_count_val = best_trial.user_attrs.get('oversold_success_count_val', 0)
                oversold_success_count_train = best_trial.user_attrs.get('oversold_success_count_train', 0)

                if oversold_success_count_val > 0 or oversold_success_count_train > 0:
                    print(
                        f"  🔻 Train: Bin 0 ({bin0_name}) : Win Rate={bin_0_wr_train:.4f}, Couverture={bin_0_pct_train:.2%}, Trades réussis={oversold_success_count_train}")
                    print(
                        f"  🔻 Val : Bin 0 ({bin0_name}) : Win Rate={bin_0_wr_val:.4f}, Couverture={bin_0_pct_val:.2%}, Trades réussis={oversold_success_count_val}")
                    if test_result_:
                        print(
                            f"  🔻 Test : Bin 0 ({bin0_name}) : Win Rate={test_result_['bin_0_win_rate']:.4f}, Couverture={test_result_['bin_0_pct']:.2%}, Trades réussis={test_result_['oversold_success_count']}")
                else:
                    print(
                        f"  🔻 Train: Bin 0 ({bin0_name}) : Win Rate={bin_0_wr_train:.4f}, Couverture={bin_0_pct_train:.2%}")
                    print(
                        f"  🔻 Val : Bin 0 ({bin0_name}) : Win Rate={bin_0_wr_val:.4f}, Couverture={bin_0_pct_val:.2%}")
                    if test_result_:
                        print(
                            f"  🔻 Test : Bin 0 ({bin0_name}) : Win Rate={test_result_['bin_0_win_rate']:.4f}, Couverture={test_result_['bin_0_pct']:.2%}")

            if OPTIMIZE_OVERBOUGHT and bin_1_wr_val is not None and bin_1_pct_val is not None:
                # Adapte le nom du bin selon l'indicateur
                bin1_name = "Surachat"
                if 'vwap_low_threshold' in best_trial.params:
                    bin1_name = "Distance VWAP Modérée"
                elif 'r2_low_threshold' in best_trial.params:
                    bin1_name = "Volatilité Modérée"
                elif 'slope' in best_trial.params:
                    bin1_name = "Volatilité Haute"
                elif 'slope_extrem_threshold' in best_trial.params:
                    bin1_name = "Pente Forte"
                elif 'period_var_atr' in best_trial.params:
                    bin1_name = "ATR Modéré"
                elif 'mfi_period' in best_trial.params:
                    bin1_name = "Divergence Baissière"

                # Affichage du bin 1 avec le nombre de trades réussis
                overbought_success_count_val = best_trial.user_attrs.get('overbought_success_count_val', 0)
                overbought_success_count_train = best_trial.user_attrs.get('overbought_success_count_train', 0)

                if overbought_success_count_val > 0 or overbought_success_count_train > 0:
                    print(
                        f"  🔺 Train: Bin 1 ({bin1_name}) : Win Rate={bin_1_wr_train:.4f}, Couverture={bin_1_pct_train:.2%}, Trades réussis={overbought_success_count_train}")
                    print(
                        f"  🔺 Val : Bin 1 ({bin1_name}) : Win Rate={bin_1_wr_val:.4f}, Couverture={bin_1_pct_val:.2%}, Trades réussis={overbought_success_count_val}")
                    if test_result_:
                        print(
                            f"  🔺 Test : Bin 1 ({bin1_name}) : Win Rate={test_result_['bin_1_win_rate']:.4f}, Couverture={test_result_['bin_1_pct']:.2%}, Trades réussis={test_result_['overbought_success_count']}")
                else:
                    print(
                        f"  🔺 Train: Bin 1 ({bin1_name}) : Win Rate={bin_1_wr_train:.4f}, Couverture={bin_1_pct_train:.2%}")
                    print(
                        f"  🔺 Val : Bin 1 ({bin1_name}) : Win Rate={bin_1_wr_val:.4f}, Couverture={bin_1_pct_val:.2%}")
                    if test_result_:
                        print(
                            f"  🔺 Test : Bin 1 ({bin1_name}) : Win Rate={test_result_['bin_1_win_rate']:.4f}, Couverture={test_result_['bin_1_pct']:.2%}")

                # Ajout des paramètres spécifiques pour MFI divergence si applicable
                if 'mfi_period' in best_trial.params:
                    min_mfi_decrease = best_trial.params.get('min_mfi_decrease', 'N/A')
                    min_price_increase = best_trial.params.get('min_price_increase', 'N/A')
                    print(
                        f"  📊 Paramètres MFI: min_mfi_decrease={min_mfi_decrease}, min_price_increase={min_price_increase}")
            # Affichage du spread uniquement si les deux zones sont actives
            if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT and bin_spread_val is not None:
                print(f"  ⚖️ Écart (Bin Spread) : {bin_spread_val:.4f}")

        except ValueError:
            print(f" ❌ Pas encore de meilleur essai trouvé... pour {indicator_type}")

        print("\n")

def objective_regressionSlope_modified(trial, df_train_, df_val_):
    """
    Fonction objective pour Optuna, qui ajuste les paramètres de l'indicateur
    de régression basé sur la pente. On calcule et vérifie les métriques sur
    le TRAIN, puis on fait la même chose sur la VALIDATION.
    """

    # ==============================
    # 1. Paramètres à optimiser
    # ==============================
    period_var = trial.suggest_int('period_var_slope', PERIOD_VAR_L, PERIOD_VAR_H)
    slope_range_threshold = trial.suggest_float('slope_range_threshold',
                                                SLOPE_RANGE_THRESHOLD_L,
                                                SLOPE_RANGE_THRESHOLD_H)
    # On définit le low selon OPTIMIZE_OVERBOUGHT
    if OPTIMIZE_OVERBOUGHT:
        low = slope_range_threshold
    else:
        low = SLOPE_EXTREM_THRESHOLD_L

    slope_extrem_threshold = trial.suggest_float('slope_extrem_threshold',
                                                 low,  # Commence à la valeur du seuil bas
                                                 SLOPE_EXTREM_THRESHOLD_H)

    # ==============================
    # 2. Calculer signaux + métriques sur TRAIN
    # ==============================

    # --- 2.1 Calculer slopes sur le TRAIN
    close_train = pd.to_numeric(df_train_['close'], errors='coerce').values
    session_starts_train = (df_train_['SessionStartEnd'] == 10).values
    slopes_train, _, _ = calculate_slopes_and_r2_numba(close_train, session_starts_train, period_var)

    # --- 2.2 Créer les colonnes de signaux sur le TRAIN
    if OPTIMIZE_OVERBOUGHT:
        df_train_['is_low_slope'] = np.where(
            (slopes_train > slope_range_threshold) & (slopes_train < slope_extrem_threshold),
            1, 0
        )
    else:
        df_train_['is_high_slope'] = np.where(
            (slopes_train < slope_range_threshold) | (slopes_train > slope_extrem_threshold),
            1, 0
        )

    # --- 2.3 Filtrer les lignes binaires sur le TRAIN
    df_train_filtered = df_train_[df_train_['class_binaire'].isin([0, 1])].copy()

    # --- 2.4 Calculer les métriques sur TRAIN (winrate, occurrence, etc.)
    bin_0_win_rate_train = 0.5
    bin_1_win_rate_train = 0.5
    bin_0_pct_train = 0
    bin_1_pct_train = 0
    oversold_success_count_train = 0
    overbought_success_count_train = 0

    # Surligne les exceptions (signal vide, etc.)
    try:
        # Si on optimise la "survente" (pente faible)
        if OPTIMIZE_OVERSOLD:
            oversold_df_train = df_train_filtered[df_train_filtered['is_high_slope'] == 1]
            if len(oversold_df_train) == 0:
                return -np.inf  # Pas de trades, on pénalise
            bin_0_win_rate_train = oversold_df_train['class_binaire'].mean()
            bin_0_pct_train = len(oversold_df_train) / len(df_train_filtered)
            oversold_success_count_train = oversold_df_train['class_binaire'].sum()

        # Si on optimise le "surachat" (pente élevée)
        if OPTIMIZE_OVERBOUGHT:
            overbought_df_train = df_train_filtered[df_train_filtered['is_low_slope'] == 1]
            if len(overbought_df_train) == 0:
                return -np.inf
            bin_1_win_rate_train = overbought_df_train['class_binaire'].mean()
            bin_1_pct_train = len(overbought_df_train) / len(df_train_filtered)
            overbought_success_count_train = overbought_df_train['class_binaire'].sum()

        # Calcul spread si on optimise les deux en même temps
        bin_spread_train = (bin_1_win_rate_train - bin_0_win_rate_train) if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        params_check_constraints = {
            "max_bin_0_win_rate": MAX_BIN_0_WIN_RATE,
            "max_bin_1_win_rate": MAX_BIN_1_WIN_RATE,
            "min_bin_size_0": MIN_BIN_SIZE_0,
            "min_bin_size_1": MIN_BIN_SIZE_1,
        }

        # Vérifier les contraintes sur TRAIN
        if not check_bin_constraints(
                bin_0_pct_train,
                bin_1_pct_train,
                bin_0_win_rate_train,
                bin_1_win_rate_train,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du "score" TRAIN (optionnel, on peut s’en servir si on veut un multi-critère)
        combined_score_train = calculate_optimization_score(
            bin_0_pct_train,
            bin_1_pct_train,
            bin_0_win_rate_train,
            bin_1_win_rate_train,
            bin_spread_train
        )

    except Exception as e:
        print(f"[TRAIN] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 3. Calculer signaux + métriques sur VALIDATION
    # ==============================

    # --- 3.1 Calculer slopes sur VALIDATION
    close_val = pd.to_numeric(df_val_['close'], errors='coerce').values
    session_starts_val = (df_val_['SessionStartEnd'] == 10).values
    slopes_val, _, _ = calculate_slopes_and_r2_numba(close_val, session_starts_val, period_var)

    # --- 3.2 Créer les colonnes de signaux sur la VAL
    # On répète la même logique
    if OPTIMIZE_OVERBOUGHT:
        df_val_['is_low_slope'] = np.where(
            (slopes_val > slope_range_threshold) & (slopes_val < slope_extrem_threshold),
            1, 0
        )
    else:
        df_val_['is_high_slope'] = np.where(
            (slopes_val < slope_range_threshold) | (slopes_val > slope_extrem_threshold),
            1, 0
        )

    # --- 3.3 Filtrer les lignes binaires sur la VAL
    df_val_filtered = df_val_[df_val_['class_binaire'].isin([0, 1])].copy()

    print(df_val_filtered.shape)

    # --- 3.4 Calculer les métriques sur VAL
    bin_0_win_rate_val = 0.5
    bin_1_win_rate_val = 0.5
    bin_0_pct_val = 0
    bin_1_pct_val = 0
    oversold_success_count_val = 0
    overbought_success_count_val = 0

    try:
        if OPTIMIZE_OVERSOLD:
            oversold_df_val = df_val_filtered[df_val_filtered['is_high_slope'] == 1]
            if len(oversold_df_val) == 0:
                return -np.inf
            bin_0_win_rate_val = oversold_df_val['class_binaire'].mean()
            bin_0_pct_val = len(oversold_df_val) / len(df_val_filtered)
            oversold_success_count_val = oversold_df_val['class_binaire'].sum()

        if OPTIMIZE_OVERBOUGHT:
            overbought_df_val = df_val_filtered[df_val_filtered['is_low_slope'] == 1]
            if len(overbought_df_val) == 0:
                return -np.inf
            bin_1_win_rate_val = overbought_df_val['class_binaire'].mean()
            bin_1_pct_val = len(overbought_df_val) / len(df_val_filtered)
            overbought_success_count_val = overbought_df_val['class_binaire'].sum()

        bin_spread_val = (bin_1_win_rate_val - bin_0_win_rate_val) if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        print(f"{bin_0_pct_val} {bin_1_pct_val} {bin_0_win_rate_val} {bin_1_win_rate_val} {bin_spread_val}")
        # Vérifier les contraintes sur VAL
        if not check_bin_constraints(
                bin_0_pct_val,
                bin_1_pct_val,
                bin_0_win_rate_val,
                bin_1_win_rate_val,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du "score" VAL (celui qui va réellement guider l’optimiseur)
        combined_score_val = calculate_optimization_score(
            bin_0_pct_val,
            bin_1_pct_val,
            bin_0_win_rate_val,
            bin_1_win_rate_val,
            bin_spread_val
        )

    except Exception as e:
        print(f"[VAL] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 4. Stocker toutes les métriques (TRAIN + VAL)
    # ==============================
    metrics = {
        # --- Métriques TRAIN
        'bin_0_win_rate_train': float(bin_0_win_rate_train),
        'bin_1_win_rate_train': float(bin_1_win_rate_train),
        'bin_0_pct_train': float(bin_0_pct_train),
        'bin_1_pct_train': float(bin_1_pct_train),
        'bin_spread_train': float(bin_spread_train),
        'combined_score_train': float(combined_score_train),

        # --- Métriques VAL
        'bin_0_win_rate_val': float(bin_0_win_rate_val),
        'bin_1_win_rate_val': float(bin_1_win_rate_val),
        'bin_0_pct_val': float(bin_0_pct_val),
        'bin_1_pct_val': float(bin_1_pct_val),
        'bin_spread_val': float(bin_spread_val),
        'combined_score_val': float(combined_score_val),

        # --- Hyperparamètres + slopes
        'period_var': period_var,
        'slope_range_threshold': slope_range_threshold,
        'slope_extrem_threshold': slope_extrem_threshold,

        # --- Comptes bruts
        'oversold_success_count_train': int(oversold_success_count_train),
        'overbought_success_count_train': int(overbought_success_count_train),
        'oversold_success_count_val': int(oversold_success_count_val),
        'overbought_success_count_val': int(overbought_success_count_val),
    }

    for key, value in metrics.items():
        trial.set_user_attr(key, value)

    # ==============================
    # 5. Retourner le score final (Mix)
    # ==============================
    score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
    return score_mix


def objective_regressionATR_modified(trial, df_train_, df_val_):
    """
    Fonction objective pour Optuna, qui ajuste les paramètres de l'indicateur
    ATR (Average True Range). On calcule et vérifie les métriques sur
    le TRAIN, puis on fait la même chose sur la VALIDATION.
    """

    # ==============================
    # 1. Paramètres à optimiser
    # ==============================
    period_var = trial.suggest_int('period_var_atr', PERIOD_VAR_L, PERIOD_VAR_H)

    # Suggérer le seuil bas dans sa plage complète
    atr_low_threshold = trial.suggest_float('atr_low_threshold', ATR_LOW_THRESHOLD_L, ATR_LOW_THRESHOLD_H)

    # Déterminer la valeur minimale pour atr_high_threshold en fonction du mode d'optimisation
    if OPTIMIZE_OVERBOUGHT:
        low = atr_low_threshold
        atr_high_threshold = trial.suggest_float('atr_high_threshold',
                                                 low,  # Commence à la valeur du seuil bas
                                                 ATR_HIGH_THRESHOLD_H)
    else:
        low = ATR_HIGH_THRESHOLD_L
        # Suggérer le seuil haut en commençant à partir du seuil bas



    # ==============================
    # 2. Calculer signaux + métriques sur TRAIN
    # ==============================

    # --- 2.1 Calculer ATR sur le TRAIN
    atr_train = calculate_atr(
        df_train_,
        period=period_var,
        avg_type='sma',  # ou 'ema' / 'wma' selon ton besoin
        fill_value=0.0  # ou une autre valeur selon ton système
    )

    # --- 2.2 Créer les colonnes de signaux sur le TRAIN
    if OPTIMIZE_OVERBOUGHT:
        # Pour maximiser le win rate, on cherche les régions où l'ATR est dans une plage modérée
        df_train_['atr_range'] = np.where(
            (atr_train > atr_low_threshold) & (atr_train < atr_high_threshold),
            1, 0
        )
    else:
        # Pour minimiser le win rate, on cherche les régions où l'ATR est soit très bas
        df_train_['atr_extrem'] = np.where(
            (atr_train < atr_low_threshold),
            1, 0
        )

    # --- 2.3 Filtrer les lignes binaires sur le TRAIN
    df_train_filtered = df_train_[df_train_['class_binaire'].isin([0, 1])].copy()

    # --- 2.4 Calculer les métriques sur TRAIN (winrate, occurrence, etc.)
    bin_0_win_rate_train = 0.5
    bin_1_win_rate_train = 0.5
    bin_0_pct_train = 0
    bin_1_pct_train = 0
    oversold_success_count_train = 0
    overbought_success_count_train = 0

    # Surligne les exceptions (signal vide, etc.)
    try:
        # Si on optimise la "survente" (ATR extrême)
        if OPTIMIZE_OVERSOLD:
            oversold_df_train = df_train_filtered[df_train_filtered['atr_extrem'] == 1]
            if len(oversold_df_train) == 0:
                return -np.inf  # Pas de trades, on pénalise
            bin_0_win_rate_train = oversold_df_train['class_binaire'].mean()
            bin_0_pct_train = len(oversold_df_train) / len(df_train_filtered)
            oversold_success_count_train = oversold_df_train['class_binaire'].sum()

        # Si on optimise le "surachat" (ATR modéré)
        if OPTIMIZE_OVERBOUGHT:
            overbought_df_train = df_train_filtered[df_train_filtered['atr_range'] == 1]
            if len(overbought_df_train) == 0:
                return -np.inf
            bin_1_win_rate_train = overbought_df_train['class_binaire'].mean()
            bin_1_pct_train = len(overbought_df_train) / len(df_train_filtered)
            overbought_success_count_train = overbought_df_train['class_binaire'].sum()

        # Calcul spread si on optimise les deux en même temps
        bin_spread_train = (bin_1_win_rate_train - bin_0_win_rate_train) if (
                    OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        params_check_constraints = {
            "max_bin_0_win_rate": MAX_BIN_0_WIN_RATE,
            "max_bin_1_win_rate": MAX_BIN_1_WIN_RATE,
            "min_bin_size_0": MIN_BIN_SIZE_0,
            "min_bin_size_1": MIN_BIN_SIZE_1,
        }

        # Vérifier les contraintes sur TRAIN
        if not check_bin_constraints(
                bin_0_pct_train,
                bin_1_pct_train,
                bin_0_win_rate_train,
                bin_1_win_rate_train,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du "score" TRAIN
        combined_score_train = calculate_optimization_score(
            bin_0_pct_train,
            bin_1_pct_train,
            bin_0_win_rate_train,
            bin_1_win_rate_train,
            bin_spread_train
        )

    except Exception as e:
        print(f"[TRAIN] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 3. Calculer signaux + métriques sur VALIDATION
    # ==============================

    # --- 3.1 Calculer ATR sur VALIDATION
    atr_val = calculate_atr(df_val_, period_var)

    # --- 3.2 Créer les colonnes de signaux sur la VAL
    if OPTIMIZE_OVERBOUGHT:
        df_val_['atr_range'] = np.where(
            (atr_val > atr_low_threshold) & (atr_val < atr_high_threshold),
            1, 0
        )
    else:
        df_val_['atr_extrem'] = np.where(
            (atr_val < atr_low_threshold),
            1, 0
        )

    # --- 3.3 Filtrer les lignes binaires sur la VAL
    df_val_filtered = df_val_[df_val_['class_binaire'].isin([0, 1])].copy()

    # --- 3.4 Calculer les métriques sur VAL
    bin_0_win_rate_val = 0.5
    bin_1_win_rate_val = 0.5
    bin_0_pct_val = 0
    bin_1_pct_val = 0
    oversold_success_count_val = 0
    overbought_success_count_val = 0

    try:
        if OPTIMIZE_OVERSOLD:
            oversold_df_val = df_val_filtered[df_val_filtered['atr_extrem'] == 1]
            if len(oversold_df_val) == 0:
                return -np.inf
            bin_0_win_rate_val = oversold_df_val['class_binaire'].mean()
            bin_0_pct_val = len(oversold_df_val) / len(df_val_filtered)
            oversold_success_count_val = oversold_df_val['class_binaire'].sum()
            print("Train - Win Rate (signal=1) :", bin_0_win_rate_train)
            print("Train - % de signaux à 1     :", bin_0_pct_train)
            print("Val  - Win Rate (signal=1) :", bin_0_win_rate_val)
            print("Val  - % de signaux à 1     :", bin_0_pct_val)

        if OPTIMIZE_OVERBOUGHT:
            overbought_df_val = df_val_filtered[df_val_filtered['atr_range'] == 1]
            if len(overbought_df_val) == 0:
                return -np.inf
            bin_1_win_rate_val = overbought_df_val['class_binaire'].mean()
            bin_1_pct_val = len(overbought_df_val) / len(df_val_filtered)
            overbought_success_count_val = overbought_df_val['class_binaire'].sum()

        bin_spread_val = (bin_1_win_rate_val - bin_0_win_rate_val) if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes sur VAL
        if not check_bin_constraints(
                bin_0_pct_val,
                bin_1_pct_val,
                bin_0_win_rate_val,
                bin_1_win_rate_val,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du "score" VAL
        combined_score_val = calculate_optimization_score(
            bin_0_pct_val,
            bin_1_pct_val,
            bin_0_win_rate_val,
            bin_1_win_rate_val,
            bin_spread_val
        )

    except Exception as e:
        print(f"[VAL] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 4. Stocker toutes les métriques (TRAIN + VAL)
    # ==============================
    metrics = {
        # --- Métriques TRAIN
        'bin_0_win_rate_train': float(bin_0_win_rate_train),
        'bin_1_win_rate_train': float(bin_1_win_rate_train),
        'bin_0_pct_train': float(bin_0_pct_train),
        'bin_1_pct_train': float(bin_1_pct_train),
        'bin_spread_train': float(bin_spread_train),
        'combined_score_train': float(combined_score_train),

        # --- Métriques VAL
        'bin_0_win_rate_val': float(bin_0_win_rate_val),
        'bin_1_win_rate_val': float(bin_1_win_rate_val),
        'bin_0_pct_val': float(bin_0_pct_val),
        'bin_1_pct_val': float(bin_1_pct_val),
        'bin_spread_val': float(bin_spread_val),
        'combined_score_val': float(combined_score_val),

        # --- Hyperparamètres + ATR
        'period_var': period_var,
        'atr_low_threshold': atr_low_threshold,
        'atr_high_threshold': atr_high_threshold if OPTIMIZE_OVERBOUGHT else "Non utilisé",

        # --- Comptes bruts
        'oversold_success_count_train': int(oversold_success_count_train),
        'overbought_success_count_train': int(overbought_success_count_train),
        'oversold_success_count_val': int(oversold_success_count_val),
        'overbought_success_count_val': int(overbought_success_count_val),
    }

    for key, value in metrics.items():
        trial.set_user_attr(key, value)


    # ==============================
    # 6. Retourner le score final (Mix)
    # ==============================
    score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
    return score_mix


def objective_zscore_modified(trial, df_train_, df_val_):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres de l'indicateur
    Z-Score selon le mode d'optimisation (survente, surachat ou les deux).
    Calcule et vérifie les métriques sur le TRAIN, puis sur la VALIDATION.
    """
    # ==============================
    # 1. Paramètres à optimiser
    # ==============================
    period_var_zscore = trial.suggest_int('period_var_zscore', PERIOD_ZSCORE_ZCORE_L, PERIOD_ZSCORE_ZCORE_H)


    # Approche modifi   ée pour traiter correctement les deux modes d'optimisation
    if OPTIMIZE_OVERSOLD and not OPTIMIZE_OVERBOUGHT:
        # En mode survente uniquement, nous voulons des valeurs extrêmes (très négatives ou très positives)
        # Donc nous suggérons deux valeurs indépendantes pour les seuils
        zscore_low_threshold = trial.suggest_float('zscore_low_threshold', ZSCORE_LOW_THRESHOLD_L,
                                                   ZSCORE_LOW_THRESHOLD_H)
        zscore_high_threshold = trial.suggest_float('zscore_high_threshold', ZSCORE_HIGH_THRESHOLD_L,
                                                    ZSCORE_HIGH_THRESHOLD_H)

        # Ici on s'assure que les valeurs extrêmes sont bien définies (inversion possible)
        if zscore_low_threshold > zscore_high_threshold:
            # Nous n'avons pas besoin d'échanger ces valeurs car nous les utilisons pour définir
            # des conditions de "soit < low SOIT > high" dans la logique plus bas
            pass
    else:
        # En mode normal ou surachat uniquement, nous voulons des valeurs dans un intervalle
        zscore_low_threshold = trial.suggest_float('zscore_low_threshold', ZSCORE_LOW_THRESHOLD_L,
                                                   ZSCORE_LOW_THRESHOLD_H)
        # S'assurer que high_threshold > low_threshold
        zscore_high_threshold = trial.suggest_float('zscore_high_threshold',
                                                    zscore_low_threshold,  # Commence à la valeur du seuil bas
                                                    ZSCORE_HIGH_THRESHOLD_H)

    # ==============================
    # 2. Calculer signaux + métriques sur TRAIN
    # ==============================

    # --- 2.1 Calcul du Z-Score sur TRAIN
    _, zscores_train = enhanced_close_to_sma_ratio(df_train_, period_var_zscore)
    # --- 2.2 Créer les indicateurs avec une logique adaptée au mode d'optimisation
    if OPTIMIZE_OVERBOUGHT:
        # Zone modérée du Z-Score (intervalle entre low et high)
        df_train_['is_zscore_range'] = np.where((zscores_train > zscore_low_threshold) &
                                                (zscores_train < zscore_high_threshold), 1, 0)

    if OPTIMIZE_OVERSOLD:
        # Zone extrême du Z-Score (en dehors de l'intervalle)
        df_train_['is_zscore_extrem'] = np.where((zscores_train < zscore_low_threshold) |
                                                 (zscores_train > zscore_high_threshold), 1, 0)

    # --- 2.3 Filtrer les lignes binaires sur le TRAIN
    df_train_filtered = df_train_[df_train_['class_binaire'].isin([0, 1])].copy()

    # --- 2.4 Calculer les métriques sur TRAIN
    bin_0_win_rate_train = 0.5  # Valeur par défaut
    bin_1_win_rate_train = 0.5  # Valeur par défaut
    bin_0_pct_train = 0
    bin_1_pct_train = 0
    oversold_success_count_train = 0
    overbought_success_count_train = 0

    try:
        # Si on optimise la "survente" (Z-Score extrême)
        if OPTIMIZE_OVERSOLD:
            oversold_df_train = df_train_filtered[df_train_filtered['is_zscore_extrem'] == 1]
            if len(oversold_df_train) == 0:
                return -np.inf  # Pas de trades, on pénalise
            bin_0_win_rate_train = oversold_df_train['class_binaire'].mean()
            bin_0_pct_train = len(oversold_df_train) / len(df_train_filtered)
            oversold_success_count_train = oversold_df_train['class_binaire'].sum()

        # Si on optimise le "surachat" (Z-Score modéré)
        if OPTIMIZE_OVERBOUGHT:
            overbought_df_train = df_train_filtered[df_train_filtered['is_zscore_range'] == 1]
            if len(overbought_df_train) == 0:
                return -np.inf
            bin_1_win_rate_train = overbought_df_train['class_binaire'].mean()
            bin_1_pct_train = len(overbought_df_train) / len(df_train_filtered)
            overbought_success_count_train = overbought_df_train['class_binaire'].sum()

        # Calcul spread si on optimise les deux en même temps
        bin_spread_train = (bin_1_win_rate_train - bin_0_win_rate_train) if (
                    OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        params_check_constraints = {
            "max_bin_0_win_rate": MAX_BIN_0_WIN_RATE,
            "max_bin_1_win_rate": MAX_BIN_1_WIN_RATE,
            "min_bin_size_0": MIN_BIN_SIZE_0,
            "min_bin_size_1": MIN_BIN_SIZE_1,
        }

        # Vérifier les contraintes sur TRAIN
        if not check_bin_constraints(
                bin_0_pct_train,
                bin_1_pct_train,
                bin_0_win_rate_train,
                bin_1_win_rate_train,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):


            return -np.inf

        # Calcul du score TRAIN
        combined_score_train = calculate_optimization_score(
            bin_0_pct_train,
            bin_1_pct_train,
            bin_0_win_rate_train,
            bin_1_win_rate_train,
            bin_spread_train
        )


    except Exception as e:
        print(f"[TRAIN] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 3. Calculer signaux + métriques sur VALIDATION
    # ==============================

    # --- 3.1 Calcul du Z-Score sur VALIDATION
    _, zscores_val = enhanced_close_to_sma_ratio(df_val_, period_var_zscore)

    # --- 3.2 Créer les indicateurs pour VALIDATION
    if OPTIMIZE_OVERBOUGHT:
        df_val_['is_zscore_range'] = np.where((zscores_val > zscore_low_threshold) &
                                              (zscores_val < zscore_high_threshold), 1, 0)

    if OPTIMIZE_OVERSOLD:
        df_val_['is_zscore_extrem'] = np.where((zscores_val < zscore_low_threshold) |
                                               (zscores_val > zscore_high_threshold), 1, 0)

    # --- 3.3 Filtrer les lignes binaires sur la VAL
    df_val_filtered = df_val_[df_val_['class_binaire'].isin([0, 1])].copy()



    # --- 3.4 Calculer les métriques sur VAL
    bin_0_win_rate_val = 0.5
    bin_1_win_rate_val = 0.5
    bin_0_pct_val = 0
    bin_1_pct_val = 0
    oversold_success_count_val = 0
    overbought_success_count_val = 0

    try:


        if OPTIMIZE_OVERSOLD:
            oversold_df_val = df_val_filtered[df_val_filtered['is_zscore_extrem'] == 1]
            if len(oversold_df_val) == 0:
                return -np.inf
            bin_0_win_rate_val = oversold_df_val['class_binaire'].mean()
            bin_0_pct_val = len(oversold_df_val) / len(df_val_filtered)
            oversold_success_count_val = oversold_df_val['class_binaire'].sum()

        if OPTIMIZE_OVERBOUGHT:
            overbought_df_val = df_val_filtered[df_val_filtered['is_zscore_range'] == 1]
            if len(overbought_df_val) == 0:
                return -np.inf
            bin_1_win_rate_val = overbought_df_val['class_binaire'].mean()
            bin_1_pct_val = len(overbought_df_val) / len(df_val_filtered)
            overbought_success_count_val = overbought_df_val['class_binaire'].sum()



        bin_spread_val = (bin_1_win_rate_val - bin_0_win_rate_val) if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0



        # Vérifier les contraintes sur VAL
        if not check_bin_constraints(
                bin_0_pct_val,
                bin_1_pct_val,
                bin_0_win_rate_val,
                bin_1_win_rate_val,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score VAL
        combined_score_val = calculate_optimization_score(
            bin_0_pct_val,
            bin_1_pct_val,
            bin_0_win_rate_val,
            bin_1_win_rate_val,
            bin_spread_val
        )

    except Exception as e:
        print(f"[VAL] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 4. Stocker toutes les métriques (TRAIN + VAL)
    # ==============================
    metrics = {
        # --- Métriques TRAIN
        'bin_0_win_rate_train': float(bin_0_win_rate_train),
        'bin_1_win_rate_train': float(bin_1_win_rate_train),
        'bin_0_pct_train': float(bin_0_pct_train),
        'bin_1_pct_train': float(bin_1_pct_train),
        'bin_spread_train': float(bin_spread_train),
        'combined_score_train': float(combined_score_train),

        # --- Métriques VAL
        'bin_0_win_rate_val': float(bin_0_win_rate_val),
        'bin_1_win_rate_val': float(bin_1_win_rate_val),
        'bin_0_pct_val': float(bin_0_pct_val),
        'bin_1_pct_val': float(bin_1_pct_val),
        'bin_spread_val': float(bin_spread_val),
        'combined_score_val': float(combined_score_val),

        # --- Hyperparamètres + zscores
        'period_var_zscore': period_var_zscore,
        'zscore_low_threshold': zscore_low_threshold,
        'zscore_high_threshold': zscore_high_threshold,

        # --- Comptes bruts
        'oversold_success_count_train': int(oversold_success_count_train),
        'overbought_success_count_train': int(overbought_success_count_train),
        'oversold_success_count_val': int(oversold_success_count_val),
        'overbought_success_count_val': int(overbought_success_count_val),
    }

    # Stocker toutes les métriques d'un coup
    for key, value in metrics.items():
        trial.set_user_attr(key, value)

    # ==============================
    # 5. Logs périodiques
    # ==============================
    if trial.number % 50 == 0:
        mode_str = ("COMPLET" if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else
                    "SURVENTE uniquement" if OPTIMIZE_OVERSOLD else
                    "SURACHAT uniquement")

        print(f"Trial {trial.number} [Mode: {mode_str}]:")

        print(f"  --- TRAIN ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_train:.4f}, Bin0={bin_0_pct_train:.2%}, Bin1={bin_1_pct_train:.2%}")
            print(
                f"  Win rates: Bin0(Z-Score extrême)={bin_0_win_rate_train:.4f}, Bin1(Z-Score modéré)={bin_1_win_rate_train:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_train:.2%}, Win rate: {bin_0_win_rate_train:.4f}, Trades réussis: {oversold_success_count_train}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_train:.2%}, Win rate: {bin_1_win_rate_train:.4f}, Trades réussis: {overbought_success_count_train}")
        print(f"  Score TRAIN: {combined_score_train:.2f}")

        print(f"  --- VALIDATION ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_val:.4f}, Bin0={bin_0_pct_val:.2%}, Bin1={bin_1_pct_val:.2%}")
            print(
                f"  Win rates: Bin0(Z-Score extrême)={bin_0_win_rate_val:.4f}, Bin1(Z-Score modéré)={bin_1_win_rate_val:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_val:.2%}, Win rate: {bin_0_win_rate_val:.4f}, Trades réussis: {oversold_success_count_val}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_val:.2%}, Win rate: {bin_1_win_rate_val:.4f}, Trades réussis: {overbought_success_count_val}")
        print(f"  Score VAL: {combined_score_val:.2f}")

        params_str = f"  Paramètres: period_var_zscore={period_var_zscore}"
        params_str += f", zscore_low_threshold={zscore_low_threshold:.4f}"
        params_str += f", zscore_high_threshold={zscore_high_threshold:.4f}"
        print(params_str)

        # Score mixte final
        score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
        print(f"  Score MIX: {score_mix:.2f}")

    # ==============================
    # 6. Retourner le score final (Mix)
    # ==============================
    score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
    return score_mix
def objective_vwap_modified(trial, df_train_, df_val_):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres des seuils
    de distance au VWAP sans imposer de contrainte sur la direction.
    Calcule et vérifie les métriques sur le TRAIN, puis sur la VALIDATION.
    """
    # ==============================
    # 1. Paramètres à optimiser
    # ==============================
    vwap_low_threshold = trial.suggest_float('vwap_low_threshold', VWAP_LOW_L, VWAP_LOW_H)

    if OPTIMIZE_OVERBOUGHT:
        low = vwap_low_threshold
    else:
        low = VWAP_HIGH_L

    # Ensuite, suggérer le seuil haut en commençant à partir du seuil bas
    vwap_high_threshold = trial.suggest_float('vwap_high_threshold',
                                              low,  # Commence à la valeur du seuil bas
                                              VWAP_HIGH_H)  # Jusqu'à la limite supérieure

    # ==============================
    # 2. Calculer signaux + métriques sur TRAIN
    # ==============================

    # --- 2.1 Calcul de la différence avec VWAP sur TRAIN
    df_train_['diffPriceCloseVWAP'] = df_train_['close'] - df_train_['VWAP']
    diff_vwap_train = pd.to_numeric(df_train_['diffPriceCloseVWAP'], errors='coerce')

    # --- 2.2 Créer les indicateurs avec une logique adaptée au mode d'optimisation
    if OPTIMIZE_OVERBOUGHT:
        # Optimiser pour maximiser le win rate dans une plage spécifique de distance au VWAP
        df_train_['is_vwap_range'] = np.where(
            (diff_vwap_train > vwap_low_threshold) & (diff_vwap_train < vwap_high_threshold),
            1, 0
        )

    if OPTIMIZE_OVERSOLD:
        # Optimiser pour minimiser le win rate en dehors d'une plage spécifique
        df_train_['is_vwap_extrem'] = np.where(
            (diff_vwap_train < vwap_low_threshold) | (diff_vwap_train > vwap_high_threshold),
            1, 0
        )

    # --- 2.3 Filtrer les lignes binaires sur le TRAIN
    df_train_filtered = df_train_[df_train_['class_binaire'].isin([0, 1])].copy()

    # --- 2.4 Calculer les métriques sur TRAIN
    bin_0_win_rate_train = 0.5  # Valeur par défaut
    bin_1_win_rate_train = 0.5  # Valeur par défaut
    bin_0_pct_train = 0
    bin_1_pct_train = 0
    oversold_success_count_train = 0
    overbought_success_count_train = 0

    try:
        # Si on optimise la "survente" (VWAP extrême)
        if OPTIMIZE_OVERSOLD:
            oversold_df_train = df_train_filtered[df_train_filtered['is_vwap_extrem'] == 1]
            if len(oversold_df_train) == 0:
                return -np.inf  # Pas de trades, on pénalise
            bin_0_win_rate_train = oversold_df_train['class_binaire'].mean()
            bin_0_pct_train = len(oversold_df_train) / len(df_train_filtered)
            oversold_success_count_train = oversold_df_train['class_binaire'].sum()

        # Si on optimise le "surachat" (VWAP modéré)
        if OPTIMIZE_OVERBOUGHT:
            overbought_df_train = df_train_filtered[df_train_filtered['is_vwap_range'] == 1]
            if len(overbought_df_train) == 0:
                return -np.inf
            bin_1_win_rate_train = overbought_df_train['class_binaire'].mean()
            bin_1_pct_train = len(overbought_df_train) / len(df_train_filtered)
            overbought_success_count_train = overbought_df_train['class_binaire'].sum()

        # Calcul spread si on optimise les deux en même temps
        bin_spread_train = (bin_1_win_rate_train - bin_0_win_rate_train) if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        params_check_constraints = {
            "max_bin_0_win_rate": MAX_BIN_0_WIN_RATE,
            "max_bin_1_win_rate": MAX_BIN_1_WIN_RATE,
            "min_bin_size_0": MIN_BIN_SIZE_0,
            "min_bin_size_1": MIN_BIN_SIZE_1,
        }

        # Vérifier les contraintes sur TRAIN
        if not check_bin_constraints(
                bin_0_pct_train,
                bin_1_pct_train,
                bin_0_win_rate_train,
                bin_1_win_rate_train,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score TRAIN
        combined_score_train = calculate_optimization_score(
            bin_0_pct_train,
            bin_1_pct_train,
            bin_0_win_rate_train,
            bin_1_win_rate_train,
            bin_spread_train
        )

    except Exception as e:
        print(f"[TRAIN] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 3. Calculer signaux + métriques sur VALIDATION
    # ==============================

    # --- 3.1 Calcul de la différence avec VWAP sur VALIDATION
    df_val_['diffPriceCloseVWAP'] = df_val_['close'] - df_val_['VWAP']
    diff_vwap_val = pd.to_numeric(df_val_['diffPriceCloseVWAP'], errors='coerce')

    # --- 3.2 Créer les indicateurs pour VALIDATION
    if OPTIMIZE_OVERBOUGHT:
        df_val_['is_vwap_range'] = np.where(
            (diff_vwap_val > vwap_low_threshold) & (diff_vwap_val < vwap_high_threshold),
            1, 0
        )

    if OPTIMIZE_OVERSOLD:
        df_val_['is_vwap_extrem'] = np.where(
            (diff_vwap_val < vwap_low_threshold) | (diff_vwap_val > vwap_high_threshold),
            1, 0
        )

    # --- 3.3 Filtrer les lignes binaires sur la VAL
    df_val_filtered = df_val_[df_val_['class_binaire'].isin([0, 1])].copy()

    # --- 3.4 Calculer les métriques sur VAL
    bin_0_win_rate_val = 0.5
    bin_1_win_rate_val = 0.5
    bin_0_pct_val = 0
    bin_1_pct_val = 0
    oversold_success_count_val = 0
    overbought_success_count_val = 0

    try:
        if OPTIMIZE_OVERSOLD:
            oversold_df_val = df_val_filtered[df_val_filtered['is_vwap_extrem'] == 1]
            if len(oversold_df_val) == 0:
                return -np.inf
            bin_0_win_rate_val = oversold_df_val['class_binaire'].mean()
            bin_0_pct_val = len(oversold_df_val) / len(df_val_filtered)
            oversold_success_count_val = oversold_df_val['class_binaire'].sum()

        if OPTIMIZE_OVERBOUGHT:
            overbought_df_val = df_val_filtered[df_val_filtered['is_vwap_range'] == 1]
            if len(overbought_df_val) == 0:
                return -np.inf
            bin_1_win_rate_val = overbought_df_val['class_binaire'].mean()
            bin_1_pct_val = len(overbought_df_val) / len(df_val_filtered)
            overbought_success_count_val = overbought_df_val['class_binaire'].sum()

        bin_spread_val = (bin_1_win_rate_val - bin_0_win_rate_val) if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes sur VAL
        if not check_bin_constraints(
                bin_0_pct_val,
                bin_1_pct_val,
                bin_0_win_rate_val,
                bin_1_win_rate_val,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score VAL
        combined_score_val = calculate_optimization_score(
            bin_0_pct_val,
            bin_1_pct_val,
            bin_0_win_rate_val,
            bin_1_win_rate_val,
            bin_spread_val
        )

    except Exception as e:
        print(f"[VAL] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 4. Stocker toutes les métriques (TRAIN + VAL)
    # ==============================
    metrics = {
        # --- Métriques TRAIN
        'bin_0_win_rate_train': float(bin_0_win_rate_train),
        'bin_1_win_rate_train': float(bin_1_win_rate_train),
        'bin_0_pct_train': float(bin_0_pct_train),
        'bin_1_pct_train': float(bin_1_pct_train),
        'bin_spread_train': float(bin_spread_train),
        'combined_score_train': float(combined_score_train),

        # --- Métriques VAL
        'bin_0_win_rate_val': float(bin_0_win_rate_val),
        'bin_1_win_rate_val': float(bin_1_win_rate_val),
        'bin_0_pct_val': float(bin_0_pct_val),
        'bin_1_pct_val': float(bin_1_pct_val),
        'bin_spread_val': float(bin_spread_val),
        'combined_score_val': float(combined_score_val),

        # --- Hyperparamètres
        'vwap_low_threshold': vwap_low_threshold,
        'vwap_high_threshold': vwap_high_threshold,

        # --- Comptes bruts
        'oversold_success_count_train': int(oversold_success_count_train),
        'overbought_success_count_train': int(overbought_success_count_train),
        'oversold_success_count_val': int(oversold_success_count_val),
        'overbought_success_count_val': int(overbought_success_count_val),
    }

    # Stocker toutes les métriques d'un coup
    for key, value in metrics.items():
        trial.set_user_attr(key, value)

    # ==============================
    # 5. Logs périodiques
    # ==============================
    if trial.number % 50 == 0:
        mode_str = ("COMPLET" if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else
                   "DISTANCE EXTRÊME uniquement" if OPTIMIZE_OVERSOLD else
                   "DISTANCE MODÉRÉE uniquement")

        print(f"Trial {trial.number} [Mode: {mode_str}]:")

        print(f"  --- TRAIN ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_train:.4f}, Bin0={bin_0_pct_train:.2%}, Bin1={bin_1_pct_train:.2%}")
            print(
                f"  Win rates: Bin0(Distance extrême)={bin_0_win_rate_train:.4f}, Bin1(Distance modérée)={bin_1_win_rate_train:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_train:.2%}, Win rate: {bin_0_win_rate_train:.4f}, Trades réussis: {oversold_success_count_train}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_train:.2%}, Win rate: {bin_1_win_rate_train:.4f}, Trades réussis: {overbought_success_count_train}")
        print(f"  Score TRAIN: {combined_score_train:.2f}")

        print(f"  --- VALIDATION ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_val:.4f}, Bin0={bin_0_pct_val:.2%}, Bin1={bin_1_pct_val:.2%}")
            print(
                f"  Win rates: Bin0(Distance extrême)={bin_0_win_rate_val:.4f}, Bin1(Distance modérée)={bin_1_win_rate_val:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_val:.2%}, Win rate: {bin_0_win_rate_val:.4f}, Trades réussis: {oversold_success_count_val}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_val:.2%}, Win rate: {bin_1_win_rate_val:.4f}, Trades réussis: {overbought_success_count_val}")
        print(f"  Score VAL: {combined_score_val:.2f}")

        params_str = f"  Paramètres:"
        params_str += f" vwap_low_threshold={vwap_low_threshold:.4f}, vwap_high_threshold={vwap_high_threshold:.4f}"
        print(params_str)

        # Score mixte final
        score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
        print(f"  Score MIX: {score_mix:.2f}")

    # ==============================
    # 6. Retourner le score final (Mix)
    # ==============================
    score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
    return score_mix
def objective_regressionStd_modified(trial, df_train_, df_val_):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres de l'indicateur
    de régression basé sur l'écart-type selon le mode d'optimisation (survente, surachat ou les deux).
    Calcule et vérifie les métriques sur le TRAIN, puis sur la VALIDATION.
    """
    # ==============================
    # 1. Paramètres à optimiser
    # ==============================
    period_var = trial.suggest_int('period_var_std', PERIOD_VAR_L, PERIOD_VAR_H)
    std_low_threshold = trial.suggest_float('std_low_threshold', STD_LOW_THRESHOLD_L, STD_LOW_THRESHOLD_H)

    if OPTIMIZE_OVERBOUGHT:
        low = std_low_threshold
    else:
        low = STD_HIGH_THRESHOLD_L

    # Ensuite, suggérer le seuil haut en commençant à partir du seuil bas
    # Cela garantit que high_threshold > low_threshold
    std_high_threshold = trial.suggest_float('std_high_threshold',
                                             low,  # Commence à la valeur du seuil bas
                                             STD_HIGH_THRESHOLD_H)  # Jusqu'à la limite supérieure

    # ==============================
    # 2. Calculer signaux + métriques sur TRAIN
    # ==============================

    # --- 2.1 Calcul des pentes, R² et écarts-types sur TRAIN
    close_train = pd.to_numeric(df_train_['close'], errors='coerce').values
    session_starts_train = (df_train_['SessionStartEnd'] == 10).values
    _, _, stds_train = calculate_slopes_and_r2_numba(close_train, session_starts_train, period_var)

    # Filtrer les valeurs NaN pour les écarts-types (uniquement pour le logging)
    valid_std_train = stds_train[~np.isnan(stds_train)]

    if len(valid_std_train) > 0 and trial.number % 50 == 0:
        print(f"[TRAIN] Std max (excluding NaN): {valid_std_train.max()}")
        print(f"[TRAIN] Std min (excluding NaN): {valid_std_train.min()}")
        print(f"[TRAIN] Number of valid Std values: {len(valid_std_train)} out of {len(stds_train)} "
              f"({len(valid_std_train) / len(stds_train) * 100:.2f}%)")

    # --- 2.2 Créer les indicateurs avec une logique adaptée au mode d'optimisation
    if OPTIMIZE_OVERBOUGHT:
        # Zone modérée de volatilité (dans l'intervalle)
        df_train_['range_volatility'] = np.where(
            (stds_train > std_low_threshold) & (stds_train < std_high_threshold),
            1, 0
        )

    if OPTIMIZE_OVERSOLD:
        # Zone extrême de volatilité (en dehors de l'intervalle)
        df_train_['extrem_volatility'] = np.where(
            (stds_train < std_low_threshold) | (stds_train > std_high_threshold),
            1, 0
        )

    # --- 2.3 Filtrer les lignes binaires sur le TRAIN
    df_train_filtered = df_train_[df_train_['class_binaire'].isin([0, 1])].copy()

    # --- 2.4 Calculer les métriques sur TRAIN
    bin_0_win_rate_train = 0.5  # Valeur par défaut
    bin_1_win_rate_train = 0.5  # Valeur par défaut
    bin_0_pct_train = 0
    bin_1_pct_train = 0
    oversold_success_count_train = 0
    overbought_success_count_train = 0

    try:
        # Si on optimise la "survente" (volatilité extrême)
        if OPTIMIZE_OVERSOLD:
            oversold_df_train = df_train_filtered[df_train_filtered['extrem_volatility'] == 1]
            if len(oversold_df_train) == 0:
                return -np.inf  # Pas de trades, on pénalise
            bin_0_win_rate_train = oversold_df_train['class_binaire'].mean()
            bin_0_pct_train = len(oversold_df_train) / len(df_train_filtered)
            oversold_success_count_train = oversold_df_train['class_binaire'].sum()

        # Si on optimise le "surachat" (volatilité modérée)
        if OPTIMIZE_OVERBOUGHT:
            overbought_df_train = df_train_filtered[df_train_filtered['range_volatility'] == 1]
            if len(overbought_df_train) == 0:
                return -np.inf
            bin_1_win_rate_train = overbought_df_train['class_binaire'].mean()
            bin_1_pct_train = len(overbought_df_train) / len(df_train_filtered)
            overbought_success_count_train = overbought_df_train['class_binaire'].sum()

        # Calcul spread si on optimise les deux en même temps
        bin_spread_train = (bin_1_win_rate_train - bin_0_win_rate_train) if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        params_check_constraints = {
            "max_bin_0_win_rate": MAX_BIN_0_WIN_RATE,
            "max_bin_1_win_rate": MAX_BIN_1_WIN_RATE,
            "min_bin_size_0": MIN_BIN_SIZE_0,
            "min_bin_size_1": MIN_BIN_SIZE_1,
        }

        # Vérifier les contraintes sur TRAIN
        if not check_bin_constraints(
                bin_0_pct_train,
                bin_1_pct_train,
                bin_0_win_rate_train,
                bin_1_win_rate_train,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score TRAIN
        combined_score_train = calculate_optimization_score(
            bin_0_pct_train,
            bin_1_pct_train,
            bin_0_win_rate_train,
            bin_1_win_rate_train,
            bin_spread_train
        )

    except Exception as e:
        print(f"[TRAIN] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 3. Calculer signaux + métriques sur VALIDATION
    # ==============================

    # --- 3.1 Calcul des pentes, R² et écarts-types sur VALIDATION
    close_val = pd.to_numeric(df_val_['close'], errors='coerce').values
    session_starts_val = (df_val_['SessionStartEnd'] == 10).values
    _, _, stds_val = calculate_slopes_and_r2_numba(close_val, session_starts_val, period_var)

    # Filtrer les valeurs NaN pour les écarts-types (uniquement pour le logging)
    valid_std_val = stds_val[~np.isnan(stds_val)]

    if len(valid_std_val) > 0 and trial.number % 50 == 0:
        print(f"[VAL] Std max (excluding NaN): {valid_std_val.max()}")
        print(f"[VAL] Std min (excluding NaN): {valid_std_val.min()}")
        print(f"[VAL] Number of valid Std values: {len(valid_std_val)} out of {len(stds_val)} "
              f"({len(valid_std_val) / len(stds_val) * 100:.2f}%)")

    # --- 3.2 Créer les indicateurs pour VALIDATION
    if OPTIMIZE_OVERBOUGHT:
        df_val_['range_volatility'] = np.where(
            (stds_val > std_low_threshold) & (stds_val < std_high_threshold),
            1, 0
        )

    if OPTIMIZE_OVERSOLD:
        df_val_['extrem_volatility'] = np.where(
            (stds_val < std_low_threshold) | (stds_val > std_high_threshold),
            1, 0
        )

    # --- 3.3 Filtrer les lignes binaires sur la VAL
    df_val_filtered = df_val_[df_val_['class_binaire'].isin([0, 1])].copy()

    # --- 3.4 Calculer les métriques sur VAL
    bin_0_win_rate_val = 0.5
    bin_1_win_rate_val = 0.5
    bin_0_pct_val = 0
    bin_1_pct_val = 0
    oversold_success_count_val = 0
    overbought_success_count_val = 0

    try:
        if OPTIMIZE_OVERSOLD:
            oversold_df_val = df_val_filtered[df_val_filtered['extrem_volatility'] == 1]
            if len(oversold_df_val) == 0:
                return -np.inf
            bin_0_win_rate_val = oversold_df_val['class_binaire'].mean()
            bin_0_pct_val = len(oversold_df_val) / len(df_val_filtered)
            oversold_success_count_val = oversold_df_val['class_binaire'].sum()

        if OPTIMIZE_OVERBOUGHT:
            overbought_df_val = df_val_filtered[df_val_filtered['range_volatility'] == 1]
            if len(overbought_df_val) == 0:
                return -np.inf
            bin_1_win_rate_val = overbought_df_val['class_binaire'].mean()
            bin_1_pct_val = len(overbought_df_val) / len(df_val_filtered)
            overbought_success_count_val = overbought_df_val['class_binaire'].sum()

        bin_spread_val = (bin_1_win_rate_val - bin_0_win_rate_val) if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes sur VAL
        if not check_bin_constraints(
                bin_0_pct_val,
                bin_1_pct_val,
                bin_0_win_rate_val,
                bin_1_win_rate_val,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score VAL
        combined_score_val = calculate_optimization_score(
            bin_0_pct_val,
            bin_1_pct_val,
            bin_0_win_rate_val,
            bin_1_win_rate_val,
            bin_spread_val
        )

    except Exception as e:
        print(f"[VAL] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 4. Stocker toutes les métriques (TRAIN + VAL)
    # ==============================
    metrics = {
        # --- Métriques TRAIN
        'bin_0_win_rate_train': float(bin_0_win_rate_train),
        'bin_1_win_rate_train': float(bin_1_win_rate_train),
        'bin_0_pct_train': float(bin_0_pct_train),
        'bin_1_pct_train': float(bin_1_pct_train),
        'bin_spread_train': float(bin_spread_train),
        'combined_score_train': float(combined_score_train),

        # --- Métriques VAL
        'bin_0_win_rate_val': float(bin_0_win_rate_val),
        'bin_1_win_rate_val': float(bin_1_win_rate_val),
        'bin_0_pct_val': float(bin_0_pct_val),
        'bin_1_pct_val': float(bin_1_pct_val),
        'bin_spread_val': float(bin_spread_val),
        'combined_score_val': float(combined_score_val),

        # --- Hyperparamètres
        'period_var': period_var,
        'std_low_threshold': std_low_threshold,
        'std_high_threshold': std_high_threshold,

        # --- Comptes bruts
        'oversold_success_count_train': int(oversold_success_count_train),
        'overbought_success_count_train': int(overbought_success_count_train),
        'oversold_success_count_val': int(oversold_success_count_val),
        'overbought_success_count_val': int(overbought_success_count_val),
    }

    # Stocker toutes les métriques d'un coup
    for key, value in metrics.items():
        trial.set_user_attr(key, value)

    # ==============================
    # 5. Logs périodiques
    # ==============================
    if trial.number % 50 == 0:
        mode_str = ("COMPLET" if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else
                    "VOLATILITÉ EXTRÊME uniquement" if OPTIMIZE_OVERSOLD else
                    "VOLATILITÉ MODÉRÉE uniquement")

        print(f"Trial {trial.number} [Mode: {mode_str}]:")

        print(f"  --- TRAIN ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_train:.4f}, Bin0={bin_0_pct_train:.2%}, Bin1={bin_1_pct_train:.2%}")
            print(
                f"  Win rates: Bin0(Volatilité extrême)={bin_0_win_rate_train:.4f}, Bin1(Volatilité modérée)={bin_1_win_rate_train:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_train:.2%}, Win rate: {bin_0_win_rate_train:.4f}, Trades réussis: {oversold_success_count_train}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_train:.2%}, Win rate: {bin_1_win_rate_train:.4f}, Trades réussis: {overbought_success_count_train}")
        print(f"  Score TRAIN: {combined_score_train:.2f}")

        print(f"  --- VALIDATION ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_val:.4f}, Bin0={bin_0_pct_val:.2%}, Bin1={bin_1_pct_val:.2%}")
            print(
                f"  Win rates: Bin0(Volatilité extrême)={bin_0_win_rate_val:.4f}, Bin1(Volatilité modérée)={bin_1_win_rate_val:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_val:.2%}, Win rate: {bin_0_win_rate_val:.4f}, Trades réussis: {oversold_success_count_val}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_val:.2%}, Win rate: {bin_1_win_rate_val:.4f}, Trades réussis: {overbought_success_count_val}")
        print(f"  Score VAL: {combined_score_val:.2f}")

        params_str = f"  Paramètres: period={period_var}"
        params_str += f", std_low_threshold={std_low_threshold:.4f}, std_high_threshold={std_high_threshold:.4f}"
        print(params_str)

        # Score mixte final
        score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
        print(f"  Score MIX: {score_mix:.2f}")

    # ==============================
    # 6. Retourner le score final (Mix)
    # ==============================
    score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
    return score_mix
def objective_perctBB_simu_modified(trial, df):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres de l'indicateur
    Percent B (%B) des bandes de Bollinger selon le mode d'optimisation (survente, surachat ou les deux).
    Version optimisée utilisant directement les fonctions Numba.
    """
    # 1. Paramètres à optimiser
    period = trial.suggest_int('period_var_bb', PERCTBB_PERIOD_L, PERCTBB_PERIOD_H)
    std_dev = trial.suggest_float('std_dev', PERCTBB_STD_DEV_L, PERCTBB_STD_DEV_H)

    # Définir les seuils selon le mode d'optimisation
    bb_high_threshold = None
    bb_low_threshold = None

    if OPTIMIZE_OVERBOUGHT:
        bb_high_threshold = trial.suggest_float('bb_high_threshold', PERCTBB_HIGH_THRESHOLD_L, PERCTBB_HIGH_THRESHOLD_H)

    if OPTIMIZE_OVERSOLD:
        bb_low_threshold = trial.suggest_float('bb_low_threshold', PERCTBB_LOW_THRESHOLD_L, PERCTBB_LOW_THRESHOLD_H)


    # 3. Calcul du %B directement avec Numba
    # 3. Calcul du %B directement avec Numba (retourne directement le tableau NumPy)
    percent_b_values = calculate_percent_bb(
        df=df, period=period, std_dev=std_dev, fill_value=0, return_array=True
    )

    # Affichez quelques valeurs pour déboguer (optionnel)
    # print(pd.Series(percent_b_values[:50]))  # Affiche les 50 premières valeurs

    # 4. Créer les indicateurs en fonction du mode d'optimisation
    if OPTIMIZE_OVERBOUGHT:
        df['is_bb_range'] = np.where(
            (percent_b_values >= bb_high_threshold),
            1, 0
        )

    if OPTIMIZE_OVERSOLD:
        df['is_bb_extrem'] = np.where(
            (percent_b_values <= bb_low_threshold),
            1, 0
        )

    # 5. Filtrer df pour ne garder que les entrées avec trade (0 ou 1)
    df_filtered = df[df['class_binaire'].isin([0, 1])].copy()

    # 6. Calculer les métriques
    bin_0_win_rate = 0.5  # Valeur par défaut
    bin_1_win_rate = 0.5  # Valeur par défaut
    bin_0_pct = 0
    bin_1_pct = 0
    oversold_success_count = 0
    overbought_success_count = 0
    bin_spread = 0

    try:
        # Calcul pour le signal de volatilité extrême (minimiser win rate)
        if OPTIMIZE_OVERSOLD:
            oversold_df = df_filtered[df_filtered['is_bb_extrem'] == 1]
            if len(oversold_df) == 0:
                return -np.inf
            bin_0_win_rate = oversold_df['class_binaire'].mean()
            bin_0_pct = len(oversold_df) / len(df_filtered)
            oversold_success_count = oversold_df['class_binaire'].sum()

        # Calcul pour le signal de volatilité modérée (maximiser win rate)
        if OPTIMIZE_OVERBOUGHT:
            overbought_df = df_filtered[df_filtered['is_bb_range'] == 1]
            if len(overbought_df) == 0:
                return -np.inf
            bin_1_win_rate = overbought_df['class_binaire'].mean()
            bin_1_pct = len(overbought_df) / len(df_filtered)
            overbought_success_count = overbought_df['class_binaire'].sum()

        # Calcul de l'écart (spread) si les deux modes sont actifs
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            bin_spread = bin_1_win_rate - bin_0_win_rate

        # Vérifier les contraintes selon le mode d'optimisation
        min_bin_pct = 0.05  # Minimum 5% des échantillons

        # Vérification des contraintes
        if OPTIMIZE_OVERSOLD and bin_0_pct < min_bin_pct:
            return -np.inf
        if OPTIMIZE_OVERBOUGHT and bin_1_pct < min_bin_pct:
            return -np.inf
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT and bin_spread <= 0:
            return -np.inf

        # Calculer le score final
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            combined_score = bin_spread * 100  # Maximiser l'écart
        elif OPTIMIZE_OVERSOLD:
            combined_score = (1 - bin_0_win_rate) * 100  # Minimiser le win rate
        elif OPTIMIZE_OVERBOUGHT:
            combined_score = bin_1_win_rate * 100  # Maximiser le win rate
        else:
            return -np.inf

    except Exception as e:
        print(f"Erreur lors du calcul: {e}")
        return -np.inf

    # 7. Stocker les métriques
    metrics = {
        'bin_0_win_rate': float(bin_0_win_rate),
        'bin_1_win_rate': float(bin_1_win_rate),
        'bin_0_pct': float(bin_0_pct),
        'bin_1_pct': float(bin_1_pct),
        'bin_spread': float(bin_spread),
        'combined_score': float(combined_score),
        'period_var_bb': period,
        'std_dev': std_dev,
        'bb_low_threshold': bb_low_threshold if OPTIMIZE_OVERSOLD else "n.a",
        'bb_high_threshold': bb_high_threshold if OPTIMIZE_OVERBOUGHT else "n.a",
        'oversold_success_count': int(oversold_success_count),
        'overbought_success_count': int(overbought_success_count)
    }

    # Stocker toutes les métriques d'un coup
    for key, value in metrics.items():
        trial.set_user_attr(key, value)

    # 8. Logs périodiques
    if trial.number % 50 == 0:
        mode_str = ("COMPLET" if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else
                    "SURVENTE uniquement" if OPTIMIZE_OVERSOLD else
                    "SURACHAT uniquement")

        print(f"Trial {trial.number} [Mode: {mode_str}]:")

        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread:.4f}, Bin0={bin_0_pct:.2%}, Bin1={bin_1_pct:.2%}")
            print(f"  Win rates: Bin0(Oversold)={bin_0_win_rate:.4f}, Bin1(Overbought)={bin_1_win_rate:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(f"  Bin0={bin_0_pct:.2%}, Win rate: {bin_0_win_rate:.4f}, Trades réussis: {oversold_success_count}")
        elif OPTIMIZE_OVERBOUGHT:
            print(f"  Bin1={bin_1_pct:.2%}, Win rate: {bin_1_win_rate:.4f}, Trades réussis: {overbought_success_count}")

        params_str = f"  Paramètres: period={period}, std_dev={std_dev}"
        if OPTIMIZE_OVERSOLD:
            params_str += f", bb_low_threshold={bb_low_threshold:.4f}"
        if OPTIMIZE_OVERBOUGHT:
            params_str += f", bb_high_threshold={bb_high_threshold:.4f}"
        print(params_str)
        print(f"  Score: {combined_score:.2f}")

    return combined_score


def objective_regressionR2_modified(trial, df_train_, df_val_):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres de l'indicateur
    de régression selon le mode d'optimisation (survente, surachat ou les deux).
    Calcule et vérifie les métriques sur le TRAIN, puis sur la VALIDATION.
    """
    # ==============================
    # 1. Paramètres à optimiser
    # ==============================
    period_var = trial.suggest_int('period_var_r2', PERIOD_VAR_L, PERIOD_VAR_H)

    # Seuil bas pour R²
    r2_low_threshold = trial.suggest_float('r2_low_threshold', R2_LOW_THRESHOLD_L, R2_LOW_THRESHOLD_H)

    if OPTIMIZE_OVERBOUGHT:
        low = r2_low_threshold
    else:
        low = R2_HIGH_THRESHOLD_L

    # Seuil haut pour R² en commençant à partir du seuil bas (garantit que high > low)
    r2_high_threshold = trial.suggest_float('r2_high_threshold',
                                            low,  # Commence à la valeur du seuil bas
                                            R2_HIGH_THRESHOLD_H)  # Jusqu'à la limite supérieure

    # ==============================
    # 2. Calculer signaux + métriques sur TRAIN
    # ==============================

    # --- 2.1 Calcul des pentes et R² sur TRAIN
    close_train = pd.to_numeric(df_train_['close'], errors='coerce').values
    session_starts_train = (df_train_['SessionStartEnd'] == 10).values
    slopes_r2_train, r2s_train, _ = calculate_slopes_and_r2_numba(close_train, session_starts_train, period_var)

    # Filtrer les valeurs NaN pour logs
    valid_r2s_train = r2s_train[~np.isnan(r2s_train)]

    if len(valid_r2s_train) > 0 and trial.number % 50 == 0:
        print(f"[TRAIN] R² max (excluding NaN): {valid_r2s_train.max()}")
        print(f"[TRAIN] R² min (excluding NaN): {valid_r2s_train.min()}")
        print(f"[TRAIN] Number of valid R² values: {len(valid_r2s_train)} out of {len(r2s_train)} "
              f"({len(valid_r2s_train) / len(r2s_train) * 100:.2f}%)")

    # --- 2.2 Créer les indicateurs avec une logique adaptée au mode d'optimisation
    if OPTIMIZE_OVERBOUGHT:
        # Zone modérée de volatilité (dans l'intervalle)
        df_train_['range_volatility'] = np.where(
            (r2s_train > r2_low_threshold) & (r2s_train < r2_high_threshold),
            1, 0
        )

    if OPTIMIZE_OVERSOLD:
        # Zone extrême de volatilité (en dehors de l'intervalle)
        df_train_['extrem_volatility'] = np.where(
            (r2s_train < r2_low_threshold) | (r2s_train > r2_high_threshold),
            1, 0
        )

    # --- 2.3 Filtrer les lignes binaires sur le TRAIN
    df_train_filtered = df_train_[df_train_['class_binaire'].isin([0, 1])].copy()

    # --- 2.4 Calculer les métriques sur TRAIN
    bin_0_win_rate_train = 0.5  # Valeur par défaut
    bin_1_win_rate_train = 0.5  # Valeur par défaut
    bin_0_pct_train = 0
    bin_1_pct_train = 0
    oversold_success_count_train = 0
    overbought_success_count_train = 0

    try:
        # Si on optimise la "survente" (volatilité extrême)
        if OPTIMIZE_OVERSOLD:
            oversold_df_train = df_train_filtered[df_train_filtered['extrem_volatility'] == 1]
            if len(oversold_df_train) == 0:
                return -np.inf  # Pas de trades, on pénalise
            bin_0_win_rate_train = oversold_df_train['class_binaire'].mean()
            bin_0_pct_train = len(oversold_df_train) / len(df_train_filtered)
            oversold_success_count_train = oversold_df_train['class_binaire'].sum()

        # Si on optimise le "surachat" (volatilité modérée)
        if OPTIMIZE_OVERBOUGHT:
            overbought_df_train = df_train_filtered[df_train_filtered['range_volatility'] == 1]
            if len(overbought_df_train) == 0:
                return -np.inf
            bin_1_win_rate_train = overbought_df_train['class_binaire'].mean()
            bin_1_pct_train = len(overbought_df_train) / len(df_train_filtered)
            overbought_success_count_train = overbought_df_train['class_binaire'].sum()

        # Calcul spread si on optimise les deux en même temps
        bin_spread_train = (bin_1_win_rate_train - bin_0_win_rate_train) if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        params_check_constraints = {
            "max_bin_0_win_rate": MAX_BIN_0_WIN_RATE,
            "max_bin_1_win_rate": MAX_BIN_1_WIN_RATE,
            "min_bin_size_0": MIN_BIN_SIZE_0,
            "min_bin_size_1": MIN_BIN_SIZE_1,
        }

        # Vérifier les contraintes sur TRAIN
        if not check_bin_constraints(
                bin_0_pct_train,
                bin_1_pct_train,
                bin_0_win_rate_train,
                bin_1_win_rate_train,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score TRAIN
        combined_score_train = calculate_optimization_score(
            bin_0_pct_train,
            bin_1_pct_train,
            bin_0_win_rate_train,
            bin_1_win_rate_train,
            bin_spread_train
        )

    except Exception as e:
        print(f"[TRAIN] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 3. Calculer signaux + métriques sur VALIDATION
    # ==============================

    # --- 3.1 Calcul des pentes et R² sur VALIDATION
    close_val = pd.to_numeric(df_val_['close'], errors='coerce').values
    session_starts_val = (df_val_['SessionStartEnd'] == 10).values
    slopes_r2_val, r2s_val, _ = calculate_slopes_and_r2_numba(close_val, session_starts_val, period_var)

    # Filtrer les valeurs NaN pour logs
    valid_r2s_val = r2s_val[~np.isnan(r2s_val)]

    if len(valid_r2s_val) > 0 and trial.number % 50 == 0:
        print(f"[VAL] R² max (excluding NaN): {valid_r2s_val.max()}")
        print(f"[VAL] R² min (excluding NaN): {valid_r2s_val.min()}")
        print(f"[VAL] Number of valid R² values: {len(valid_r2s_val)} out of {len(r2s_val)} "
              f"({len(valid_r2s_val) / len(r2s_val) * 100:.2f}%)")

    # --- 3.2 Créer les indicateurs pour VALIDATION
    if OPTIMIZE_OVERBOUGHT:
        df_val_['range_volatility'] = np.where(
            (r2s_val > r2_low_threshold) & (r2s_val < r2_high_threshold),
            1, 0
        )

    if OPTIMIZE_OVERSOLD:
        df_val_['extrem_volatility'] = np.where(
            (r2s_val < r2_low_threshold) | (r2s_val > r2_high_threshold),
            1, 0
        )

    # --- 3.3 Filtrer les lignes binaires sur la VAL
    df_val_filtered = df_val_[df_val_['class_binaire'].isin([0, 1])].copy()

    # --- 3.4 Calculer les métriques sur VAL
    bin_0_win_rate_val = 0.5
    bin_1_win_rate_val = 0.5
    bin_0_pct_val = 0
    bin_1_pct_val = 0
    oversold_success_count_val = 0
    overbought_success_count_val = 0

    try:
        if OPTIMIZE_OVERSOLD:
            oversold_df_val = df_val_filtered[df_val_filtered['extrem_volatility'] == 1]
            if len(oversold_df_val) == 0:
                return -np.inf
            bin_0_win_rate_val = oversold_df_val['class_binaire'].mean()
            bin_0_pct_val = len(oversold_df_val) / len(df_val_filtered)
            oversold_success_count_val = oversold_df_val['class_binaire'].sum()

        if OPTIMIZE_OVERBOUGHT:
            overbought_df_val = df_val_filtered[df_val_filtered['range_volatility'] == 1]
            if len(overbought_df_val) == 0:
                return -np.inf
            bin_1_win_rate_val = overbought_df_val['class_binaire'].mean()
            bin_1_pct_val = len(overbought_df_val) / len(df_val_filtered)
            overbought_success_count_val = overbought_df_val['class_binaire'].sum()

        bin_spread_val = (bin_1_win_rate_val - bin_0_win_rate_val) if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes sur VAL
        if not check_bin_constraints(
                bin_0_pct_val,
                bin_1_pct_val,
                bin_0_win_rate_val,
                bin_1_win_rate_val,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score VAL
        combined_score_val = calculate_optimization_score(
            bin_0_pct_val,
            bin_1_pct_val,
            bin_0_win_rate_val,
            bin_1_win_rate_val,
            bin_spread_val
        )

    except Exception as e:
        print(f"[VAL] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 4. Stocker toutes les métriques (TRAIN + VAL)
    # ==============================
    metrics = {
        # --- Métriques TRAIN
        'bin_0_win_rate_train': float(bin_0_win_rate_train),
        'bin_1_win_rate_train': float(bin_1_win_rate_train),
        'bin_0_pct_train': float(bin_0_pct_train),
        'bin_1_pct_train': float(bin_1_pct_train),
        'bin_spread_train': float(bin_spread_train),
        'combined_score_train': float(combined_score_train),

        # --- Métriques VAL
        'bin_0_win_rate_val': float(bin_0_win_rate_val),
        'bin_1_win_rate_val': float(bin_1_win_rate_val),
        'bin_0_pct_val': float(bin_0_pct_val),
        'bin_1_pct_val': float(bin_1_pct_val),
        'bin_spread_val': float(bin_spread_val),
        'combined_score_val': float(combined_score_val),

        # --- Hyperparamètres
        'period_var': period_var,
        'r2_low_threshold': r2_low_threshold,
        'r2_high_threshold': r2_high_threshold,

        # --- Comptes bruts
        'oversold_success_count_train': int(oversold_success_count_train),
        'overbought_success_count_train': int(overbought_success_count_train),
        'oversold_success_count_val': int(oversold_success_count_val),
        'overbought_success_count_val': int(overbought_success_count_val),
    }

    # Stocker toutes les métriques d'un coup
    for key, value in metrics.items():
        trial.set_user_attr(key, value)

    # ==============================
    # 5. Logs périodiques
    # ==============================
    if trial.number % 50 == 0:
        mode_str = ("COMPLET" if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else
                    "VOLATILITÉ EXTRÊME uniquement" if OPTIMIZE_OVERSOLD else
                    "VOLATILITÉ MODÉRÉE uniquement")

        print(f"Trial {trial.number} [Mode: {mode_str}]:")

        print(f"  --- TRAIN ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_train:.4f}, Bin0={bin_0_pct_train:.2%}, Bin1={bin_1_pct_train:.2%}")
            print(
                f"  Win rates: Bin0(Volatilité extrême)={bin_0_win_rate_train:.4f}, Bin1(Volatilité modérée)={bin_1_win_rate_train:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_train:.2%}, Win rate: {bin_0_win_rate_train:.4f}, Trades réussis: {oversold_success_count_train}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_train:.2%}, Win rate: {bin_1_win_rate_train:.4f}, Trades réussis: {overbought_success_count_train}")
        print(f"  Score TRAIN: {combined_score_train:.2f}")

        print(f"  --- VALIDATION ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_val:.4f}, Bin0={bin_0_pct_val:.2%}, Bin1={bin_1_pct_val:.2%}")
            print(
                f"  Win rates: Bin0(Volatilité extrême)={bin_0_win_rate_val:.4f}, Bin1(Volatilité modérée)={bin_1_win_rate_val:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_val:.2%}, Win rate: {bin_0_win_rate_val:.4f}, Trades réussis: {oversold_success_count_val}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_val:.2%}, Win rate: {bin_1_win_rate_val:.4f}, Trades réussis: {overbought_success_count_val}")
        print(f"  Score VAL: {combined_score_val:.2f}")

        params_str = f"  Paramètres: period={period_var}"
        params_str += f", r2_low_threshold={r2_low_threshold:.4f}, r2_high_threshold={r2_high_threshold:.4f}"
        print(params_str)

        # Score mixte final
        score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
        print(f"  Score MIX: {score_mix:.2f}")

    # ==============================
    # 6. Retourner le score final (Mix)
    # ==============================
    score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
    return score_mix


def objective_stochastic_modified(trial, df_train_, df_val_):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres du Stochastique
    selon le mode d'optimisation (survente, surachat ou les deux).
    Calcule et vérifie les métriques sur le TRAIN, puis sur la VALIDATION.
    """
    # ==============================
    # 1. Paramètres à optimiser
    # ==============================
    d_period = trial.suggest_int('d_period', D_PERIOD_L, D_PERIOD_H)
    k_period = trial.suggest_int('k_period', d_period, K_PERIOD_H)

    OS_limit = trial.suggest_int('OS_limit', OS_LIMIT_L, OS_LIMIT_H) if OPTIMIZE_OVERSOLD else 20  # Seuil de survente
    OB_limit = trial.suggest_int('OB_limit', OB_LIMIT_L, OB_LIMIT_H) if OPTIMIZE_OVERBOUGHT else 80  # Seuil de surachat

    # ==============================
    # 2. Calculer signaux + métriques sur TRAIN
    # ==============================

    # --- 2.1 Calculer le Stochastique sur le dataset TRAIN
    high_train = pd.to_numeric(df_train_['high'], errors='coerce')
    low_train = pd.to_numeric(df_train_['low'], errors='coerce')
    close_train = pd.to_numeric(df_train_['close'], errors='coerce')
    session_starts_train = (df_train_['SessionStartEnd'] == 10).values

    k_values_train, d_values_train = compute_stoch(high_train, low_train, close_train,
                                                   session_starts_train, k_period=k_period, d_period=d_period)

    # --- 2.2 Créer les indicateurs avec une logique adaptée au mode d'optimisation
    if OPTIMIZE_OVERBOUGHT:
        df_train_['stoch_overbought'] = np.where(k_values_train > OB_limit, 1, 0)

    if OPTIMIZE_OVERSOLD:
        df_train_['stoch_oversold'] = np.where(k_values_train < OS_limit, 1, 0)

    # --- 2.3 Filtrer les lignes binaires sur le TRAIN
    df_train_filtered = df_train_[df_train_['class_binaire'].isin([0, 1])].copy()

    # --- 2.4 Calculer les métriques sur TRAIN
    bin_0_win_rate_train = 0.5  # Valeur par défaut
    bin_1_win_rate_train = 0.5  # Valeur par défaut
    bin_0_pct_train = 0
    bin_1_pct_train = 0
    oversold_success_count_train = 0
    overbought_success_count_train = 0

    try:
        # Si on optimise la "survente"
        if OPTIMIZE_OVERSOLD:
            oversold_df_train = df_train_filtered[df_train_filtered['stoch_oversold'] == 1]
            if len(oversold_df_train) == 0:
                return -np.inf  # Pas de trades, on pénalise
            bin_0_win_rate_train = oversold_df_train['class_binaire'].mean()
            bin_0_pct_train = len(oversold_df_train) / len(df_train_filtered)
            oversold_success_count_train = oversold_df_train['class_binaire'].sum()

        # Si on optimise le "surachat"
        if OPTIMIZE_OVERBOUGHT:
            overbought_df_train = df_train_filtered[df_train_filtered['stoch_overbought'] == 1]
            if len(overbought_df_train) == 0:
                return -np.inf
            bin_1_win_rate_train = overbought_df_train['class_binaire'].mean()
            bin_1_pct_train = len(overbought_df_train) / len(df_train_filtered)
            overbought_success_count_train = overbought_df_train['class_binaire'].sum()

        # Calcul spread si on optimise les deux en même temps
        bin_spread_train = (bin_1_win_rate_train - bin_0_win_rate_train) if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        params_check_constraints = {
            "max_bin_0_win_rate": MAX_BIN_0_WIN_RATE,
            "max_bin_1_win_rate": MAX_BIN_1_WIN_RATE,
            "min_bin_size_0": MIN_BIN_SIZE_0,
            "min_bin_size_1": MIN_BIN_SIZE_1,
        }

        # Vérifier les contraintes sur TRAIN
        if not check_bin_constraints(
                bin_0_pct_train,
                bin_1_pct_train,
                bin_0_win_rate_train,
                bin_1_win_rate_train,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score TRAIN
        combined_score_train = calculate_optimization_score(
            bin_0_pct_train,
            bin_1_pct_train,
            bin_0_win_rate_train,
            bin_1_win_rate_train,
            bin_spread_train
        )

    except Exception as e:
        print(f"[TRAIN] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 3. Calculer signaux + métriques sur VALIDATION
    # ==============================

    # --- 3.1 Calculer le Stochastique sur le dataset VALIDATION
    high_val = pd.to_numeric(df_val_['high'], errors='coerce')
    low_val = pd.to_numeric(df_val_['low'], errors='coerce')
    close_val = pd.to_numeric(df_val_['close'], errors='coerce')
    session_starts_val = (df_val_['SessionStartEnd'] == 10).values

    k_values_val, d_values_val = compute_stoch(high_val, low_val, close_val,
                                               session_starts_val, k_period=k_period, d_period=d_period)

    # --- 3.2 Créer les indicateurs pour VALIDATION
    if OPTIMIZE_OVERBOUGHT:
        df_val_['stoch_overbought'] = np.where(k_values_val > OB_limit, 1, 0)

    if OPTIMIZE_OVERSOLD:
        df_val_['stoch_oversold'] = np.where(k_values_val < OS_limit, 1, 0)

    # --- 3.3 Filtrer les lignes binaires sur la VAL
    df_val_filtered = df_val_[df_val_['class_binaire'].isin([0, 1])].copy()

    # --- 3.4 Calculer les métriques sur VAL
    bin_0_win_rate_val = 0.5
    bin_1_win_rate_val = 0.5
    bin_0_pct_val = 0
    bin_1_pct_val = 0
    oversold_success_count_val = 0
    overbought_success_count_val = 0

    try:
        if OPTIMIZE_OVERSOLD:
            oversold_df_val = df_val_filtered[df_val_filtered['stoch_oversold'] == 1]
            if len(oversold_df_val) == 0:
                return -np.inf
            bin_0_win_rate_val = oversold_df_val['class_binaire'].mean()
            bin_0_pct_val = len(oversold_df_val) / len(df_val_filtered)
            oversold_success_count_val = oversold_df_val['class_binaire'].sum()

        if OPTIMIZE_OVERBOUGHT:
            overbought_df_val = df_val_filtered[df_val_filtered['stoch_overbought'] == 1]
            if len(overbought_df_val) == 0:
                return -np.inf
            bin_1_win_rate_val = overbought_df_val['class_binaire'].mean()
            bin_1_pct_val = len(overbought_df_val) / len(df_val_filtered)
            overbought_success_count_val = overbought_df_val['class_binaire'].sum()

        bin_spread_val = (bin_1_win_rate_val - bin_0_win_rate_val) if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes sur VAL
        if not check_bin_constraints(
                bin_0_pct_val,
                bin_1_pct_val,
                bin_0_win_rate_val,
                bin_1_win_rate_val,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score VAL
        combined_score_val = calculate_optimization_score(
            bin_0_pct_val,
            bin_1_pct_val,
            bin_0_win_rate_val,
            bin_1_win_rate_val,
            bin_spread_val
        )

    except Exception as e:
        print(f"[VAL] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 4. Stocker toutes les métriques (TRAIN + VAL)
    # ==============================
    metrics = {
        # --- Métriques TRAIN
        'bin_0_win_rate_train': float(bin_0_win_rate_train),
        'bin_1_win_rate_train': float(bin_1_win_rate_train),
        'bin_0_pct_train': float(bin_0_pct_train),
        'bin_1_pct_train': float(bin_1_pct_train),
        'bin_spread_train': float(bin_spread_train),
        'combined_score_train': float(combined_score_train),

        # --- Métriques VAL
        'bin_0_win_rate_val': float(bin_0_win_rate_val),
        'bin_1_win_rate_val': float(bin_1_win_rate_val),
        'bin_0_pct_val': float(bin_0_pct_val),
        'bin_1_pct_val': float(bin_1_pct_val),
        'bin_spread_val': float(bin_spread_val),
        'combined_score_val': float(combined_score_val),

        # --- Hyperparamètres
        'k_period': k_period,
        'd_period': d_period,
        'OS_limit': OS_limit,
        'OB_limit': OB_limit,

        # --- Comptes bruts
        'oversold_success_count_train': int(oversold_success_count_train),
        'overbought_success_count_train': int(overbought_success_count_train),
        'oversold_success_count_val': int(oversold_success_count_val),
        'overbought_success_count_val': int(overbought_success_count_val),
    }

    # Stocker toutes les métriques d'un coup
    for key, value in metrics.items():
        trial.set_user_attr(key, value)

    # ==============================
    # 5. Logs périodiques
    # ==============================
    if trial.number % 50 == 0:
        mode_str = ("COMPLET" if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else
                    "SURVENTE uniquement" if OPTIMIZE_OVERSOLD else
                    "SURACHAT uniquement")

        print(f"Trial {trial.number} [Mode: {mode_str}]:")

        print(f"  --- TRAIN ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_train:.4f}, Bin0={bin_0_pct_train:.2%}, Bin1={bin_1_pct_train:.2%}")
            print(
                f"  Win rates: Bin0(Oversold)={bin_0_win_rate_train:.4f}, Bin1(Overbought)={bin_1_win_rate_train:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_train:.2%}, Win rate: {bin_0_win_rate_train:.4f}, Trades réussis: {oversold_success_count_train}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_train:.2%}, Win rate: {bin_1_win_rate_train:.4f}, Trades réussis: {overbought_success_count_train}")
        print(f"  Score TRAIN: {combined_score_train:.2f}")

        print(f"  --- VALIDATION ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_val:.4f}, Bin0={bin_0_pct_val:.2%}, Bin1={bin_1_pct_val:.2%}")
            print(
                f"  Win rates: Bin0(Oversold)={bin_0_win_rate_val:.4f}, Bin1(Overbought)={bin_1_win_rate_val:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_val:.2%}, Win rate: {bin_0_win_rate_val:.4f}, Trades réussis: {oversold_success_count_val}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_val:.2%}, Win rate: {bin_1_win_rate_val:.4f}, Trades réussis: {overbought_success_count_val}")
        print(f"  Score VAL: {combined_score_val:.2f}")

        params_str = f"  Paramètres: k_period={k_period}, d_period={d_period}"
        if OPTIMIZE_OVERSOLD:
            params_str += f", OS_limit={OS_limit}"
        if OPTIMIZE_OVERBOUGHT:
            params_str += f", OB_limit={OB_limit}"
        print(params_str)

        # Score mixte final
        score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
        print(f"  Score MIX: {score_mix:.2f}")

    # ==============================
    # 6. Retourner le score final (Mix)
    # ==============================
    score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
    return score_mix


def objective_williams_r_modified(trial, df_train_, df_val_):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres de Williams %R
    selon le mode d'optimisation (survente, surachat ou les deux).
    Calcule et vérifie les métriques sur le TRAIN, puis sur la VALIDATION.
    """
    # ==============================
    # 1. Paramètres à optimiser
    # ==============================
    period = trial.suggest_int('period', PERIOD_L, PERIOD_H)
    OS_limit = trial.suggest_int('OS_limit', OS_LIMIT_L, OS_LIMIT_H) if OPTIMIZE_OVERSOLD else -70  # Seuil de survente
    OB_limit = trial.suggest_int('OB_limit', OB_LIMIT_L,
                                 OB_LIMIT_H) if OPTIMIZE_OVERBOUGHT else -30  # Seuil de surachat

    # ==============================
    # 2. Calculer signaux + métriques sur TRAIN
    # ==============================

    # --- 2.1 Calculer le Williams %R sur le dataset TRAIN
    high_train = pd.to_numeric(df_train_['high'], errors='coerce')
    low_train = pd.to_numeric(df_train_['low'], errors='coerce')
    close_train = pd.to_numeric(df_train_['close'], errors='coerce')
    session_starts_train = (df_train_['SessionStartEnd'] == 10).values

    will_r_values_train = compute_wr(high_train, low_train, close_train, session_starts_train, period=period)

    # --- 2.2 Créer les indicateurs avec une logique adaptée au mode d'optimisation
    if OPTIMIZE_OVERBOUGHT:
        df_train_['wr_overbought'] = np.where(will_r_values_train > OB_limit, 1, 0)

    if OPTIMIZE_OVERSOLD:
        df_train_['wr_oversold'] = np.where(will_r_values_train < OS_limit, 1, 0)

    # --- 2.3 Filtrer les lignes binaires sur le TRAIN
    df_train_filtered = df_train_[df_train_['class_binaire'].isin([0, 1])].copy()

    # --- 2.4 Calculer les métriques sur TRAIN
    bin_0_win_rate_train = 0.5  # Valeur par défaut
    bin_1_win_rate_train = 0.5  # Valeur par défaut
    bin_0_pct_train = 0
    bin_1_pct_train = 0
    oversold_success_count_train = 0
    overbought_success_count_train = 0

    try:
        # Si on optimise la "survente"
        if OPTIMIZE_OVERSOLD:
            oversold_df_train = df_train_filtered[df_train_filtered['wr_oversold'] == 1]
            if len(oversold_df_train) == 0:
                return -np.inf  # Pas de trades, on pénalise
            bin_0_win_rate_train = oversold_df_train['class_binaire'].mean()
            bin_0_pct_train = len(oversold_df_train) / len(df_train_filtered)
            oversold_success_count_train = oversold_df_train['class_binaire'].sum()

        # Si on optimise le "surachat"
        if OPTIMIZE_OVERBOUGHT:
            overbought_df_train = df_train_filtered[df_train_filtered['wr_overbought'] == 1]
            if len(overbought_df_train) == 0:
                return -np.inf
            bin_1_win_rate_train = overbought_df_train['class_binaire'].mean()
            bin_1_pct_train = len(overbought_df_train) / len(df_train_filtered)
            overbought_success_count_train = overbought_df_train['class_binaire'].sum()

        # Calcul spread si on optimise les deux en même temps
        bin_spread_train = (bin_1_win_rate_train - bin_0_win_rate_train) if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        params_check_constraints = {
            "max_bin_0_win_rate": MAX_BIN_0_WIN_RATE,
            "max_bin_1_win_rate": MAX_BIN_1_WIN_RATE,
            "min_bin_size_0": MIN_BIN_SIZE_0,
            "min_bin_size_1": MIN_BIN_SIZE_1,
        }

        # Vérifier les contraintes sur TRAIN
        if not check_bin_constraints(
                bin_0_pct_train,
                bin_1_pct_train,
                bin_0_win_rate_train,
                bin_1_win_rate_train,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score TRAIN
        combined_score_train = calculate_optimization_score(
            bin_0_pct_train,
            bin_1_pct_train,
            bin_0_win_rate_train,
            bin_1_win_rate_train,
            bin_spread_train
        )

    except Exception as e:
        print(f"[TRAIN] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 3. Calculer signaux + métriques sur VALIDATION
    # ==============================

    # --- 3.1 Calculer le Williams %R sur le dataset VALIDATION
    high_val = pd.to_numeric(df_val_['high'], errors='coerce')
    low_val = pd.to_numeric(df_val_['low'], errors='coerce')
    close_val = pd.to_numeric(df_val_['close'], errors='coerce')
    session_starts_val = (df_val_['SessionStartEnd'] == 10).values

    will_r_values_val = compute_wr(high_val, low_val, close_val, session_starts_val, period=period)

    # --- 3.2 Créer les indicateurs pour VALIDATION
    if OPTIMIZE_OVERBOUGHT:
        df_val_['wr_overbought'] = np.where(will_r_values_val > OB_limit, 1, 0)

    if OPTIMIZE_OVERSOLD:
        df_val_['wr_oversold'] = np.where(will_r_values_val < OS_limit, 1, 0)

    # --- 3.3 Filtrer les lignes binaires sur la VAL
    df_val_filtered = df_val_[df_val_['class_binaire'].isin([0, 1])].copy()

    # --- 3.4 Calculer les métriques sur VAL
    bin_0_win_rate_val = 0.5
    bin_1_win_rate_val = 0.5
    bin_0_pct_val = 0
    bin_1_pct_val = 0
    oversold_success_count_val = 0
    overbought_success_count_val = 0

    try:
        if OPTIMIZE_OVERSOLD:
            oversold_df_val = df_val_filtered[df_val_filtered['wr_oversold'] == 1]
            if len(oversold_df_val) == 0:
                return -np.inf
            bin_0_win_rate_val = oversold_df_val['class_binaire'].mean()
            bin_0_pct_val = len(oversold_df_val) / len(df_val_filtered)
            oversold_success_count_val = oversold_df_val['class_binaire'].sum()

        if OPTIMIZE_OVERBOUGHT:
            overbought_df_val = df_val_filtered[df_val_filtered['wr_overbought'] == 1]
            if len(overbought_df_val) == 0:
                return -np.inf
            bin_1_win_rate_val = overbought_df_val['class_binaire'].mean()
            bin_1_pct_val = len(overbought_df_val) / len(df_val_filtered)
            overbought_success_count_val = overbought_df_val['class_binaire'].sum()

        bin_spread_val = (bin_1_win_rate_val - bin_0_win_rate_val) if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes sur VAL
        if not check_bin_constraints(
                bin_0_pct_val,
                bin_1_pct_val,
                bin_0_win_rate_val,
                bin_1_win_rate_val,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score VAL
        combined_score_val = calculate_optimization_score(
            bin_0_pct_val,
            bin_1_pct_val,
            bin_0_win_rate_val,
            bin_1_win_rate_val,
            bin_spread_val
        )

    except Exception as e:
        print(f"[VAL] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 4. Stocker toutes les métriques (TRAIN + VAL)
    # ==============================
    metrics = {
        # --- Métriques TRAIN
        'bin_0_win_rate_train': float(bin_0_win_rate_train),
        'bin_1_win_rate_train': float(bin_1_win_rate_train),
        'bin_0_pct_train': float(bin_0_pct_train),
        'bin_1_pct_train': float(bin_1_pct_train),
        'bin_spread_train': float(bin_spread_train),
        'combined_score_train': float(combined_score_train),

        # --- Métriques VAL
        'bin_0_win_rate_val': float(bin_0_win_rate_val),
        'bin_1_win_rate_val': float(bin_1_win_rate_val),
        'bin_0_pct_val': float(bin_0_pct_val),
        'bin_1_pct_val': float(bin_1_pct_val),
        'bin_spread_val': float(bin_spread_val),
        'combined_score_val': float(combined_score_val),

        # --- Hyperparamètres
        'period': period,
        'OS_limit': OS_limit,
        'OB_limit': OB_limit,

        # --- Comptes bruts
        'oversold_success_count_train': int(oversold_success_count_train),
        'overbought_success_count_train': int(overbought_success_count_train),
        'oversold_success_count_val': int(oversold_success_count_val),
        'overbought_success_count_val': int(overbought_success_count_val),
    }

    # Stocker toutes les métriques d'un coup
    for key, value in metrics.items():
        trial.set_user_attr(key, value)

    # ==============================
    # 5. Logs périodiques
    # ==============================
    if trial.number % 50 == 0:
        mode_str = ("COMPLET" if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else
                    "SURVENTE uniquement" if OPTIMIZE_OVERSOLD else
                    "SURACHAT uniquement")

        print(f"Trial {trial.number} [Mode: {mode_str}]:")

        print(f"  --- TRAIN ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_train:.4f}, Bin0={bin_0_pct_train:.2%}, Bin1={bin_1_pct_train:.2%}")
            print(
                f"  Win rates: Bin0(Oversold)={bin_0_win_rate_train:.4f}, Bin1(Overbought)={bin_1_win_rate_train:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_train:.2%}, Win rate: {bin_0_win_rate_train:.4f}, Trades réussis: {oversold_success_count_train}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_train:.2%}, Win rate: {bin_1_win_rate_train:.4f}, Trades réussis: {overbought_success_count_train}")
        print(f"  Score TRAIN: {combined_score_train:.2f}")

        print(f"  --- VALIDATION ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_val:.4f}, Bin0={bin_0_pct_val:.2%}, Bin1={bin_1_pct_val:.2%}")
            print(
                f"  Win rates: Bin0(Oversold)={bin_0_win_rate_val:.4f}, Bin1(Overbought)={bin_1_win_rate_val:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_val:.2%}, Win rate: {bin_0_win_rate_val:.4f}, Trades réussis: {oversold_success_count_val}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_val:.2%}, Win rate: {bin_1_win_rate_val:.4f}, Trades réussis: {overbought_success_count_val}")
        print(f"  Score VAL: {combined_score_val:.2f}")

        params_str = f"  Paramètres: period={period}"
        if OPTIMIZE_OVERSOLD:
            params_str += f", OS_limit={OS_limit}"
        if OPTIMIZE_OVERBOUGHT:
            params_str += f", OB_limit={OB_limit}"
        print(params_str)

        # Score mixte final
        score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
        print(f"  Score MIX: {score_mix:.2f}")

    # ==============================
    # 6. Retourner le score final (Mix)
    # ==============================
    score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
    return score_mix


def objective_mfi_modified(trial, df_train_, df_val_):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres du Money Flow Index (MFI)
    selon le mode d'optimisation (survente, surachat ou les deux).
    Calcule et vérifie les métriques sur le TRAIN, puis sur la VALIDATION.
    """
    # ==============================
    # 1. Paramètres à optimiser
    # ==============================
    period = trial.suggest_int('period', PERIOD_L, PERIOD_H)
    OS_limit = trial.suggest_int('OS_limit', OS_LIMIT_L, OS_LIMIT_H) if OPTIMIZE_OVERSOLD else 20
    OB_limit = trial.suggest_int('OB_limit', OB_LIMIT_L, OB_LIMIT_H) if OPTIMIZE_OVERBOUGHT else 80

    # ==============================
    # 2. Calculer signaux + métriques sur TRAIN
    # ==============================

    # --- 2.1 Calculer le MFI sur le dataset TRAIN
    high_train = pd.to_numeric(df_train_['high'], errors='coerce')
    low_train = pd.to_numeric(df_train_['low'], errors='coerce')
    close_train = pd.to_numeric(df_train_['close'], errors='coerce')
    volume_train = pd.to_numeric(df_train_['volume'], errors='coerce')
    session_starts_train = (df_train_['SessionStartEnd'] == 10).values

    mfi_values_train = compute_mfi(high_train, low_train, close_train, volume_train,
                                   session_starts_train, period=period)

    # --- 2.2 Créer les indicateurs avec une logique adaptée au mode d'optimisation
    if OPTIMIZE_OVERBOUGHT:
        df_train_['mfi_overbought'] = np.where(mfi_values_train > OB_limit, 1, 0)

    if OPTIMIZE_OVERSOLD:
        df_train_['mfi_oversold'] = np.where(mfi_values_train < OS_limit, 1, 0)

    # --- 2.3 Filtrer les lignes binaires sur le TRAIN
    df_train_filtered = df_train_[df_train_['class_binaire'].isin([0, 1])].copy()

    # --- 2.4 Calculer les métriques sur TRAIN
    bin_0_win_rate_train = 0.5  # Valeur par défaut
    bin_1_win_rate_train = 0.5  # Valeur par défaut
    bin_0_pct_train = 0
    bin_1_pct_train = 0
    oversold_success_count_train = 0
    overbought_success_count_train = 0

    try:
        # Si on optimise la "survente"
        if OPTIMIZE_OVERSOLD:
            oversold_df_train = df_train_filtered[df_train_filtered['mfi_oversold'] == 1]
            if len(oversold_df_train) == 0:
                return -np.inf  # Pas de trades, on pénalise
            bin_0_win_rate_train = oversold_df_train['class_binaire'].mean()
            bin_0_pct_train = len(oversold_df_train) / len(df_train_filtered)
            oversold_success_count_train = oversold_df_train['class_binaire'].sum()

        # Si on optimise le "surachat"
        if OPTIMIZE_OVERBOUGHT:
            overbought_df_train = df_train_filtered[df_train_filtered['mfi_overbought'] == 1]
            if len(overbought_df_train) == 0:
                return -np.inf
            bin_1_win_rate_train = overbought_df_train['class_binaire'].mean()
            bin_1_pct_train = len(overbought_df_train) / len(df_train_filtered)
            overbought_success_count_train = overbought_df_train['class_binaire'].sum()

        # Calcul spread si on optimise les deux en même temps
        bin_spread_train = (bin_1_win_rate_train - bin_0_win_rate_train) if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        params_check_constraints = {
            "max_bin_0_win_rate": MAX_BIN_0_WIN_RATE,
            "max_bin_1_win_rate": MAX_BIN_1_WIN_RATE,
            "min_bin_size_0": MIN_BIN_SIZE_0,
            "min_bin_size_1": MIN_BIN_SIZE_1,
        }

        # Vérifier les contraintes sur TRAIN
        if not check_bin_constraints(
                bin_0_pct_train,
                bin_1_pct_train,
                bin_0_win_rate_train,
                bin_1_win_rate_train,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score TRAIN
        combined_score_train = calculate_optimization_score(
            bin_0_pct_train,
            bin_1_pct_train,
            bin_0_win_rate_train,
            bin_1_win_rate_train,
            bin_spread_train
        )

    except Exception as e:
        print(f"[TRAIN] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 3. Calculer signaux + métriques sur VALIDATION
    # ==============================

    # --- 3.1 Calculer le MFI sur le dataset VALIDATION
    high_val = pd.to_numeric(df_val_['high'], errors='coerce')
    low_val = pd.to_numeric(df_val_['low'], errors='coerce')
    close_val = pd.to_numeric(df_val_['close'], errors='coerce')
    volume_val = pd.to_numeric(df_val_['volume'], errors='coerce')
    session_starts_val = (df_val_['SessionStartEnd'] == 10).values

    mfi_values_val = compute_mfi(high_val, low_val, close_val, volume_val,
                                 session_starts_val, period=period)

    # --- 3.2 Créer les indicateurs pour VALIDATION
    if OPTIMIZE_OVERBOUGHT:
        df_val_['mfi_overbought'] = np.where(mfi_values_val > OB_limit, 1, 0)

    if OPTIMIZE_OVERSOLD:
        df_val_['mfi_oversold'] = np.where(mfi_values_val < OS_limit, 1, 0)

    # --- 3.3 Filtrer les lignes binaires sur la VAL
    df_val_filtered = df_val_[df_val_['class_binaire'].isin([0, 1])].copy()

    # --- 3.4 Calculer les métriques sur VAL
    bin_0_win_rate_val = 0.5
    bin_1_win_rate_val = 0.5
    bin_0_pct_val = 0
    bin_1_pct_val = 0
    oversold_success_count_val = 0
    overbought_success_count_val = 0

    try:
        if OPTIMIZE_OVERSOLD:
            oversold_df_val = df_val_filtered[df_val_filtered['mfi_oversold'] == 1]
            if len(oversold_df_val) == 0:
                return -np.inf
            bin_0_win_rate_val = oversold_df_val['class_binaire'].mean()
            bin_0_pct_val = len(oversold_df_val) / len(df_val_filtered)
            oversold_success_count_val = oversold_df_val['class_binaire'].sum()

        if OPTIMIZE_OVERBOUGHT:
            overbought_df_val = df_val_filtered[df_val_filtered['mfi_overbought'] == 1]
            if len(overbought_df_val) == 0:
                return -np.inf
            bin_1_win_rate_val = overbought_df_val['class_binaire'].mean()
            bin_1_pct_val = len(overbought_df_val) / len(df_val_filtered)
            overbought_success_count_val = overbought_df_val['class_binaire'].sum()

        bin_spread_val = (bin_1_win_rate_val - bin_0_win_rate_val) if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes sur VAL
        if not check_bin_constraints(
                bin_0_pct_val,
                bin_1_pct_val,
                bin_0_win_rate_val,
                bin_1_win_rate_val,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score VAL
        combined_score_val = calculate_optimization_score(
            bin_0_pct_val,
            bin_1_pct_val,
            bin_0_win_rate_val,
            bin_1_win_rate_val,
            bin_spread_val
        )

    except Exception as e:
        print(f"[VAL] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 4. Stocker toutes les métriques (TRAIN + VAL)
    # ==============================
    metrics = {
        # --- Métriques TRAIN
        'bin_0_win_rate_train': float(bin_0_win_rate_train),
        'bin_1_win_rate_train': float(bin_1_win_rate_train),
        'bin_0_pct_train': float(bin_0_pct_train),
        'bin_1_pct_train': float(bin_1_pct_train),
        'bin_spread_train': float(bin_spread_train),
        'combined_score_train': float(combined_score_train),

        # --- Métriques VAL
        'bin_0_win_rate_val': float(bin_0_win_rate_val),
        'bin_1_win_rate_val': float(bin_1_win_rate_val),
        'bin_0_pct_val': float(bin_0_pct_val),
        'bin_1_pct_val': float(bin_1_pct_val),
        'bin_spread_val': float(bin_spread_val),
        'combined_score_val': float(combined_score_val),

        # --- Hyperparamètres
        'period': period,
        'OS_limit': OS_limit,
        'OB_limit': OB_limit,

        # --- Comptes bruts
        'oversold_success_count_train': int(oversold_success_count_train),
        'overbought_success_count_train': int(overbought_success_count_train),
        'oversold_success_count_val': int(oversold_success_count_val),
        'overbought_success_count_val': int(overbought_success_count_val),
    }

    # Stocker toutes les métriques d'un coup
    for key, value in metrics.items():
        trial.set_user_attr(key, value)

    # ==============================
    # 5. Logs périodiques
    # ==============================
    if trial.number % 50 == 0:
        mode_str = ("COMPLET" if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else
                    "SURVENTE uniquement" if OPTIMIZE_OVERSOLD else
                    "SURACHAT uniquement")

        print(f"Trial {trial.number} [Mode: {mode_str}]:")

        print(f"  --- TRAIN ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_train:.4f}, Bin0={bin_0_pct_train:.2%}, Bin1={bin_1_pct_train:.2%}")
            print(
                f"  Win rates: Bin0(Oversold)={bin_0_win_rate_train:.4f}, Bin1(Overbought)={bin_1_win_rate_train:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_train:.2%}, Win rate: {bin_0_win_rate_train:.4f}, Trades réussis: {oversold_success_count_train}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_train:.2%}, Win rate: {bin_1_win_rate_train:.4f}, Trades réussis: {overbought_success_count_train}")
        print(f"  Score TRAIN: {combined_score_train:.2f}")

        print(f"  --- VALIDATION ---")
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread={bin_spread_val:.4f}, Bin0={bin_0_pct_val:.2%}, Bin1={bin_1_pct_val:.2%}")
            print(
                f"  Win rates: Bin0(Oversold)={bin_0_win_rate_val:.4f}, Bin1(Overbought)={bin_1_win_rate_val:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct_val:.2%}, Win rate: {bin_0_win_rate_val:.4f}, Trades réussis: {oversold_success_count_val}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct_val:.2%}, Win rate: {bin_1_win_rate_val:.4f}, Trades réussis: {overbought_success_count_val}")
        print(f"  Score VAL: {combined_score_val:.2f}")

        params_str = f"  Paramètres: period={period}"
        if OPTIMIZE_OVERSOLD:
            params_str += f", OS_limit={OS_limit}"
        if OPTIMIZE_OVERBOUGHT:
            params_str += f", OB_limit={OB_limit}"
        print(params_str)

        # Score mixte final
        score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
        print(f"  Score MIX: {score_mix:.2f}")

    # ==============================
    # 6. Retourner le score final (Mix)
    # ==============================
    score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
    return score_mix


def objective_mfi_divergence_modified(trial, df_train_, df_val_):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres des divergences du MFI
    pour les entrées en position short.
    Calcule et vérifie les métriques sur le TRAIN, puis sur la VALIDATION.
    """
    # ==============================
    # 1. Paramètres à optimiser
    # ==============================
    mfi_period = trial.suggest_int('mfi_period', MFI_PERIOD_L, MFI_PERIOD_H)
    div_lookback = trial.suggest_int('div_lookback', DIV_LOOKBACK_L, DIV_LOOKBACK_H)

    # Paramètres pour les deux modes (overbought et oversold)
    min_price_increase = trial.suggest_float('min_price_increase', MIN_PRICE_INCREASE_L, MIN_PRICE_INCREASE_H)
    min_mfi_decrease = trial.suggest_float('min_mfi_decrease', MIN_MFI_DECREASE_L, MIN_MFI_DECREASE_H)

    # Nouveaux paramètres pour la partie oversold (anti-divergence)
    if OPTIMIZE_OVERSOLD:
        min_price_decrease = trial.suggest_float('min_price_decrease', MIN_PRICE_INCREASE_L, MIN_PRICE_INCREASE_H)
        min_mfi_increase = trial.suggest_float('min_mfi_increase', MIN_MFI_DECREASE_L, MIN_MFI_DECREASE_H)

    # ==============================
    # 2. Calculer signaux + métriques sur TRAIN
    # ==============================

    # --- 2.1 Calcul MFI sur TRAIN
    high_train = pd.to_numeric(df_train_['high'], errors='coerce')
    low_train = pd.to_numeric(df_train_['low'], errors='coerce')
    close_train = pd.to_numeric(df_train_['close'], errors='coerce')
    volume_train = pd.to_numeric(df_train_['volume'], errors='coerce')
    session_starts_train = (df_train_['SessionStartEnd'] == 10).values

    mfi_values_train = compute_mfi(high_train, low_train, close_train, volume_train,
                                   session_starts_train, period=mfi_period)
    mfi_series_train = pd.Series(mfi_values_train, index=df_train_.index)

    # --- 2.2 Initialiser les colonnes de divergence (TRAIN)
    df_train_['bearish_divergence'] = 0
    df_train_['anti_divergence'] = 0  # Nouvelle colonne pour les conditions d'oversold

    # Filtrer pour les trades shorts (TRAIN)
    df_train_mode_filtered = df_train_[df_train_['class_binaire'] != 99].copy()
    all_shorts_train = df_train_mode_filtered['tradeDir'].eq(-1).all() if not df_train_mode_filtered.empty else False

    if trial.number % 50 == 0:
        print(f"[TRAIN] All shorts mode: {all_shorts_train}")
        print(f"[TRAIN] Nombre de trades short: {df_train_['tradeDir'].eq(-1).sum()} / {len(df_train_)}")

    if all_shorts_train:
        # Pour la partie overbought (maximiser le win rate) - TRAIN
        if OPTIMIZE_OVERBOUGHT:
            # Détection des divergences baissières améliorée
            price_pct_change_train = close_train.pct_change(div_lookback).fillna(0)
            mfi_pct_change_train = mfi_series_train.pct_change(div_lookback).fillna(0)

            # Conditions pour une divergence baissière efficace
            price_increase_train = price_pct_change_train > min_price_increase
            mfi_decrease_train = mfi_pct_change_train < -min_mfi_decrease

            # Prix fait un nouveau haut relatif
            price_rolling_max_train = close_train.rolling(window=div_lookback).max().shift(1)
            price_new_high_train = (close_train > price_rolling_max_train).fillna(False)

            # Définir la divergence baissière avec nos critères
            df_train_.loc[df_train_mode_filtered.index, 'bearish_divergence'] = (
                    (
                                price_new_high_train | price_increase_train) &  # Prix fait un nouveau haut ou augmente significativement
                    (mfi_decrease_train)  # MFI diminue
            ).astype(int)

        # Pour la partie oversold (minimiser le win rate) - TRAIN
        if OPTIMIZE_OVERSOLD:
            # Détection des conditions opposées (anti-divergences)
            price_pct_change_train = close_train.pct_change(div_lookback).fillna(0)
            mfi_pct_change_train = mfi_series_train.pct_change(div_lookback).fillna(0)

            # Conditions pour une anti-divergence (mauvais win rate)
            price_decrease_train = price_pct_change_train < -min_price_decrease  # Prix diminue
            mfi_increase_train = mfi_pct_change_train > min_mfi_increase  # MFI augmente

            # Prix fait un nouveau bas relatif
            price_rolling_min_train = close_train.rolling(window=div_lookback).min().shift(1)
            price_new_low_train = (close_train < price_rolling_min_train).fillna(False)

            # Définir l'anti-divergence avec nos critères
            df_train_.loc[df_train_mode_filtered.index, 'anti_divergence'] = (
                    (
                                price_new_low_train | price_decrease_train) &  # Prix fait un nouveau bas ou diminue significativement
                    (mfi_increase_train)  # MFI augmente
            ).astype(int)
    else:
        return -np.inf  # Si ce n'est pas full short, pénaliser

    # --- 2.3 Filtrer pour class_binaire in [0,1] (TRAIN)
    df_train_filtered = df_train_[df_train_['class_binaire'].isin([0, 1])].copy()

    # --- 2.4 Calculer les métriques sur TRAIN
    bin_0_win_rate_train = 0.5  # Valeur par défaut
    bin_1_win_rate_train = 0.5  # Valeur par défaut
    bin_0_pct_train = 0
    bin_1_pct_train = 0
    bearish_div_count_train = 0
    bearish_success_count_train = 0
    anti_div_count_train = 0
    anti_div_success_count_train = 0

    try:
        # Calcul pour oversold (anti-divergences) - TRAIN
        if OPTIMIZE_OVERSOLD:
            oversold_df_train = df_train_filtered[df_train_filtered['anti_divergence'] == 1]
            if len(oversold_df_train) < 10:  # Minimum de 10 échantillons
                return -np.inf

            bin_0_win_rate_train = oversold_df_train['class_binaire'].mean()
            bin_0_pct_train = len(oversold_df_train) / len(df_train_filtered)
            anti_div_count_train = len(oversold_df_train)
            anti_div_success_count_train = oversold_df_train['class_binaire'].sum()

        # Calcul pour overbought (divergences baissières) - TRAIN
        if OPTIMIZE_OVERBOUGHT:
            bearish_df_train = df_train_filtered[df_train_filtered['bearish_divergence'] == 1]
            if len(bearish_df_train) < 10:  # Minimum de 10 échantillons
                return -np.inf

            bin_1_win_rate_train = bearish_df_train['class_binaire'].mean()
            bin_1_pct_train = len(bearish_df_train) / len(df_train_filtered)
            bearish_div_count_train = len(bearish_df_train)
            bearish_success_count_train = bearish_df_train['class_binaire'].sum()

        # Calculer l'écart si les deux modes sont actifs
        bin_spread_train = bin_1_win_rate_train - bin_0_win_rate_train if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        params_check_constraints = {
            "max_bin_0_win_rate": MAX_BIN_0_WIN_RATE,
            "max_bin_1_win_rate": MAX_BIN_1_WIN_RATE,
            "min_bin_size_0": MIN_BIN_SIZE_0,
            "min_bin_size_1": MIN_BIN_SIZE_1,
        }

        # Vérifier les contraintes sur TRAIN
        if not check_bin_constraints(
                bin_0_pct_train,
                bin_1_pct_train,
                bin_0_win_rate_train,
                bin_1_win_rate_train,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score TRAIN
        combined_score_train = calculate_optimization_score(
            bin_0_pct_train,
            bin_1_pct_train,
            bin_0_win_rate_train,
            bin_1_win_rate_train,
            bin_spread_train
        )

    except Exception as e:
        print(f"[TRAIN] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 3. Calculer signaux + métriques sur VALIDATION
    # ==============================

    # --- 3.1 Calcul MFI sur VALIDATION
    high_val = pd.to_numeric(df_val_['high'], errors='coerce')
    low_val = pd.to_numeric(df_val_['low'], errors='coerce')
    close_val = pd.to_numeric(df_val_['close'], errors='coerce')
    volume_val = pd.to_numeric(df_val_['volume'], errors='coerce')
    session_starts_val = (df_val_['SessionStartEnd'] == 10).values

    mfi_values_val = compute_mfi(high_val, low_val, close_val, volume_val,
                                 session_starts_val, period=mfi_period)
    mfi_series_val = pd.Series(mfi_values_val, index=df_val_.index)

    # --- 3.2 Initialiser les colonnes de divergence (VALIDATION)
    df_val_['bearish_divergence'] = 0
    df_val_['anti_divergence'] = 0  # Nouvelle colonne pour les conditions d'oversold

    # Filtrer pour les trades shorts (VALIDATION)
    df_val_mode_filtered = df_val_[df_val_['class_binaire'] != 99].copy()
    all_shorts_val = df_val_mode_filtered['tradeDir'].eq(-1).all() if not df_val_mode_filtered.empty else False

    if trial.number % 50 == 0:
        print(f"[VAL] All shorts mode: {all_shorts_val}")
        print(f"[VAL] Nombre de trades short: {df_val_['tradeDir'].eq(-1).sum()} / {len(df_val_)}")

    if all_shorts_val:
        # Pour la partie overbought (maximiser le win rate) - VALIDATION
        if OPTIMIZE_OVERBOUGHT:
            # Détection des divergences baissières améliorée
            price_pct_change_val = close_val.pct_change(div_lookback).fillna(0)
            mfi_pct_change_val = mfi_series_val.pct_change(div_lookback).fillna(0)

            # Conditions pour une divergence baissière efficace
            price_increase_val = price_pct_change_val > min_price_increase
            mfi_decrease_val = mfi_pct_change_val < -min_mfi_decrease

            # Prix fait un nouveau haut relatif
            price_rolling_max_val = close_val.rolling(window=div_lookback).max().shift(1)
            price_new_high_val = (close_val > price_rolling_max_val).fillna(False)

            # Définir la divergence baissière avec nos critères
            df_val_.loc[df_val_mode_filtered.index, 'bearish_divergence'] = (
                    (
                                price_new_high_val | price_increase_val) &  # Prix fait un nouveau haut ou augmente significativement
                    (mfi_decrease_val)  # MFI diminue
            ).astype(int)

        # Pour la partie oversold (minimiser le win rate) - VALIDATION
        if OPTIMIZE_OVERSOLD:
            # Détection des conditions opposées (anti-divergences)
            price_pct_change_val = close_val.pct_change(div_lookback).fillna(0)
            mfi_pct_change_val = mfi_series_val.pct_change(div_lookback).fillna(0)

            # Conditions pour une anti-divergence (mauvais win rate)
            price_decrease_val = price_pct_change_val < -min_price_decrease  # Prix diminue
            mfi_increase_val = mfi_pct_change_val > min_mfi_increase  # MFI augmente

            # Prix fait un nouveau bas relatif
            price_rolling_min_val = close_val.rolling(window=div_lookback).min().shift(1)
            price_new_low_val = (close_val < price_rolling_min_val).fillna(False)

            # Définir l'anti-divergence avec nos critères
            df_val_.loc[df_val_mode_filtered.index, 'anti_divergence'] = (
                    (price_new_low_val | price_decrease_val) &  # Prix fait un nouveau bas ou diminue significativement
                    (mfi_increase_val)  # MFI augmente
            ).astype(int)
    else:
        return -np.inf  # Si ce n'est pas full short, pénaliser

    # --- 3.3 Filtrer pour class_binaire in [0,1] (VALIDATION)
    df_val_filtered = df_val_[df_val_['class_binaire'].isin([0, 1])].copy()

    # --- 3.4 Calculer les métriques sur VALIDATION
    bin_0_win_rate_val = 0.5  # Valeur par défaut
    bin_1_win_rate_val = 0.5  # Valeur par défaut
    bin_0_pct_val = 0
    bin_1_pct_val = 0
    bearish_div_count_val = 0
    bearish_success_count_val = 0
    anti_div_count_val = 0
    anti_div_success_count_val = 0

    try:
        # Calcul pour oversold (anti-divergences) - VALIDATION
        if OPTIMIZE_OVERSOLD:
            oversold_df_val = df_val_filtered[df_val_filtered['anti_divergence'] == 1]
            if len(oversold_df_val) < 10:  # Minimum de 10 échantillons
                return -np.inf

            bin_0_win_rate_val = oversold_df_val['class_binaire'].mean()
            bin_0_pct_val = len(oversold_df_val) / len(df_val_filtered)
            anti_div_count_val = len(oversold_df_val)
            anti_div_success_count_val = oversold_df_val['class_binaire'].sum()

        # Calcul pour overbought (divergences baissières) - VALIDATION
        if OPTIMIZE_OVERBOUGHT:
            bearish_df_val = df_val_filtered[df_val_filtered['bearish_divergence'] == 1]
            if len(bearish_df_val) < 10:  # Minimum de 10 échantillons
                return -np.inf

            bin_1_win_rate_val = bearish_df_val['class_binaire'].mean()
            bin_1_pct_val = len(bearish_df_val) / len(df_val_filtered)
            bearish_div_count_val = len(bearish_df_val)
            bearish_success_count_val = bearish_df_val['class_binaire'].sum()

        # Calculer l'écart si les deux modes sont actifs
        bin_spread_val = bin_1_win_rate_val - bin_0_win_rate_val if (
                OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes sur VAL
        if not check_bin_constraints(
                bin_0_pct_val,
                bin_1_pct_val,
                bin_0_win_rate_val,
                bin_1_win_rate_val,
                params_check_constraints,
                optimize_oversold=OPTIMIZE_OVERSOLD,
                optimize_overbought=OPTIMIZE_OVERBOUGHT):
            return -np.inf

        # Calcul du score VAL
        combined_score_val = calculate_optimization_score(
            bin_0_pct_val,
            bin_1_pct_val,
            bin_0_win_rate_val,
            bin_1_win_rate_val,
            bin_spread_val
        )

    except Exception as e:
        print(f"[VAL] Erreur lors du calcul: {e}")
        return -np.inf

    # ==============================
    # 4. Stocker toutes les métriques (TRAIN + VAL)
    # ==============================
    metrics = {
        # --- Métriques TRAIN
        'bin_0_win_rate_train': float(bin_0_win_rate_train),
        'bin_1_win_rate_train': float(bin_1_win_rate_train),
        'bin_0_pct_train': float(bin_0_pct_train),
        'bin_1_pct_train': float(bin_1_pct_train),
        'bin_spread_train': float(bin_spread_train),
        'combined_score_train': float(combined_score_train),
        'bearish_div_count_train': int(bearish_div_count_train),
        'bearish_success_count_train': int(bearish_success_count_train),
        'anti_div_count_train': int(anti_div_count_train),
        'anti_div_success_count_train': int(anti_div_success_count_train),

        # --- Métriques VAL
        'bin_0_win_rate_val': float(bin_0_win_rate_val),
        'bin_1_win_rate_val': float(bin_1_win_rate_val),
        'bin_0_pct_val': float(bin_0_pct_val),
        'bin_1_pct_val': float(bin_1_pct_val),
        'bin_spread_val': float(bin_spread_val),
        'combined_score_val': float(combined_score_val),
        'bearish_div_count_val': int(bearish_div_count_val),
        'bearish_success_count_val': int(bearish_success_count_val),
        'anti_div_count_val': int(anti_div_count_val),
        'anti_div_success_count_val': int(anti_div_success_count_val),

        # --- Hyperparamètres
        'mfi_period': mfi_period,
        'div_lookback': div_lookback,
        'min_price_increase': min_price_increase,
        'min_mfi_decrease': min_mfi_decrease,
    }

    if OPTIMIZE_OVERSOLD:
        metrics['min_price_decrease'] = min_price_decrease
        metrics['min_mfi_increase'] = min_mfi_increase

    # Stocker toutes les métriques d'un coup
    for key, value in metrics.items():
        trial.set_user_attr(key, value)

    # ==============================
    # 5. Logs périodiques
    # ==============================
    if trial.number % 20 == 0:
        print(f"\nTrial {trial.number}")
        print(f"Paramètres: mfi_period={mfi_period}, div_lookback={div_lookback}")

        print(f"  --- TRAIN ---")
        if OPTIMIZE_OVERSOLD:
            print(f"  Anti-divergences: {anti_div_count_train} signaux, win rate: {bin_0_win_rate_train:.4f}")
            print(f"  Trades gagnants anti-div: {anti_div_success_count_train}/{anti_div_count_train}")
            print(f"  Couverture anti-div: {bin_0_pct_train:.2%}")

        if OPTIMIZE_OVERBOUGHT:
            print(f"  Divergences baissières: {bearish_div_count_train} signaux, win rate: {bin_1_win_rate_train:.4f}")
            print(f"  Trades gagnants div: {bearish_success_count_train}/{bearish_div_count_train}")
            print(f"  Couverture div: {bin_1_pct_train:.2%}")

        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread (différence de win rate): {bin_spread_train:.4f}")

        print(f"  Score TRAIN: {combined_score_train:.2f}")

        print(f"  --- VALIDATION ---")
        if OPTIMIZE_OVERSOLD:
            print(f"  Anti-divergences: {anti_div_count_val} signaux, win rate: {bin_0_win_rate_val:.4f}")
            print(f"  Trades gagnants anti-div: {anti_div_success_count_val}/{anti_div_count_val}")
            print(f"  Couverture anti-div: {bin_0_pct_val:.2%}")

        if OPTIMIZE_OVERBOUGHT:
            print(f"  Divergences baissières: {bearish_div_count_val} signaux, win rate: {bin_1_win_rate_val:.4f}")
            print(f"  Trades gagnants div: {bearish_success_count_val}/{bearish_div_count_val}")
            print(f"  Couverture div: {bin_1_pct_val:.2%}")

        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread (différence de win rate): {bin_spread_val:.4f}")

        print(f"  Score VAL: {combined_score_val:.2f}")

        # Score mixte final
        score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
        print(f"  Score MIX: {score_mix:.2f}")

    # ==============================
    # 6. Retourner le score final (Mix)
    # ==============================
    score_mix = SPLIT_SCORE_VAL * combined_score_val + (1 - SPLIT_SCORE_VAL) * combined_score_train
    return score_mix
def run_indicator_optimization(df_train_,df_val_ ,df_val_filtered_, indicator_type="stochastic", n_trials=20000):
    """
    Exécute l'optimisation Optuna pour l'indicateur choisi avec support pour l'optimisation sélective.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame complet contenant toutes les données
    df_filtered : pandas.DataFrame
        DataFrame contenant les données filtrées (high, low, close, volume)
    target_y : pandas.Series
        Série contenant les résultats des trades (0/1)
    indicator_type : str, default="stochastic"
        Type d'indicateur à optimiser ('stochastic', 'williams_r', 'mfi', 'mfi_divergence',
        'regression_r2', 'regression_std' ou 'regression_slope')
    n_trials : int, default=20000
        Nombre d'essais à effectuer

    Returns:
    --------
    optuna.study.Study
        L'étude Optuna avec les résultats
    """
    # Afficher les contraintes initiales et le mode d'optimisation
    show_constraints()
    show_optimization_mode()

    if indicator_type in ["stochastic", "regression_slope", "regression_std", "regression_r2", "atr", "vwap",
                          "percent_bb_simu", "zscore","imbalance"]:

        # Pour le stochastique, désactiver l'échantillonnage multivarié
        sampler = optuna.samplers.TPESampler(
            seed=41,
            constraints_func=create_constraints_func(),
            multivariate=False  # Désactive l'échantillonnage multivarié
        )
    else:
        # Pour les autres indicateurs, garder l'échantillonnage multivarié
        sampler = optuna.samplers.TPESampler(
            seed=41,
            constraints_func=create_constraints_func(),
            multivariate=True  # Considère les dépendances entre paramètres
        )

    # Créer l'étude
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner()
    )

    # Démarrer le listener de clavier avant l'optimisation
    global should_stop
    should_stop = False
    listener = pynput_keyboard.Listener(on_press=on_press)
    listener.start()

    try:
        # Correction de la vérification de type d'indicateur pour être compatible avec le code existant
        # Cette version permet "stochastic" comme valeur par défaut

        # Dans la fonction run_indicator_optimization, remplacer la vérification stricte par cette version
        # qui accepte des types non reconnus mais affiche un avertissement

        # Vérifier si le type d'indicateur est reconnu, sinon utiliser stochastic comme valeur par défaut
        valid_indicator_types = ["stochastic", "williams_r", "mfi", "mfi_divergence",
                                 "regression_r2", "regression_std", "regression_slope", "atr", "vwap",
                                 "percent_bb_simu", "zscore", "imbalance"]


        if indicator_type.lower() not in valid_indicator_types:
            print(f"\n⚠️ AVERTISSEMENT: Type d'indicateur '{indicator_type}' non reconnu.")
            print(f"Types d'indicateurs valides: {', '.join(valid_indicator_types)}")
            print(f"L'indicateur 'stochastic' sera utilisé par défaut.\n")
            # On ne lève pas d'exception, on laisse le code utiliser stochastic par défaut

        # Sélectionner la fonction objective selon le type d'indicateur et le mode d'optimisation
        if indicator_type.lower() == "williams_r":
            objective_func = lambda trial: objective_williams_r_modified(trial, df_train_,df_val_)
        elif indicator_type.lower() == "mfi":
            objective_func = lambda trial: objective_mfi_modified(trial, df_train_,df_val_)
        elif indicator_type.lower() == "mfi_divergence":
            objective_func = lambda trial: objective_mfi_divergence_modified(trial, df_train_,df_val_)
        elif indicator_type.lower() == "regression_r2":
            objective_func = lambda trial: objective_regressionR2_modified(trial, df_train_,df_val_)
        elif indicator_type.lower() == "regression_std":
            objective_func = lambda trial: objective_regressionStd_modified(trial, df_train_,df_val_)
        elif indicator_type.lower() == "regression_slope":
            objective_func = lambda trial: objective_regressionSlope_modified(trial, df_train_,df_val_)
        elif indicator_type.lower() == "atr":
            objective_func = lambda trial: objective_regressionATR_modified(trial, df_train_,df_val_)
        elif indicator_type.lower() == "vwap":
            objective_func = lambda trial: objective_vwap_modified(trial, df_train_,df_val_)
        elif indicator_type.lower() == "percent_bb_simu":
            objective_func = lambda trial: objective_perctBB_simu_modified(trial, df_train_,df_val_)
        elif indicator_type.lower() == "zscore":
            objective_func = lambda trial: objective_zscore_modified(trial, df_train_,df_val_)
        elif indicator_type.lower() == "stochastic":
            objective_func = lambda trial: objective_stochastic_modified(trial, df_train_,df_val_)

        else:  # Default to stochastic
            exit(77)
            objective_func = lambda trial: objective_stochastic_modified(trial, df_train_)
            # Si l'indicateur n'était pas reconnu, on force la valeur à "stochastic" pour la cohérence
            if indicator_type.lower() not in valid_indicator_types:
                print("indicator_type.lower() not in valid_indicator_types:")
                exit(63)

        # Afficher les informations d'optimisation
        mode_text = ""
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            mode_text = "COMPLÈTE (SURVENTE + SURACHAT)"
        elif OPTIMIZE_OVERSOLD:
            mode_text = "SURVENTE UNIQUEMENT"
        elif OPTIMIZE_OVERBOUGHT:
            mode_text = "SURACHAT UNIQUEMENT"

        print(f"  Mode d'optimisation: {mode_text}")
        print(f"  Indicateur: {indicator_type}")
        print(f"  Nombre maximal d'essais: {n_trials}")
        print("  Appuyez sur '²' pour arrêter prématurément l'optimisation")
        print("-" * 60)
        #optuna.logging.set_verbosity(optuna.logging.WARNING)  # ou ERROR pour encore moins de messages

        study.optimize(
            objective_func,
            n_trials=n_trials,
            callbacks=[callback_optuna_stop, print_best_trial_callback]
            , show_progress_bar=False)
    except KeyboardInterrupt:
        print("\nOptimisation interrompue par l'utilisateur (Ctrl+C)")
    except Exception as e:
        print(f"\nErreur lors de l'optimisation: {e}")
    finally:
        # Arrêter proprement le listener après l'optimisation
        listener.stop()
        if listener.is_alive():
            listener.join()

    # Afficher les résultats
    print("\n" + "=" * 80)

    # Détermine le texte du mode
    # Correction pour la section d'affichage des résultats
    # Cette version est cohérente avec le traitement des types d'indicateur par défaut

    # Détermine le texte du mode
    mode_text = ""
    if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
        mode_text = "COMPLÈTE (SURVENTE + SURACHAT)"
    elif OPTIMIZE_OVERSOLD:
        mode_text = "SURVENTE UNIQUEMENT"
    elif OPTIMIZE_OVERBOUGHT:
        mode_text = "SURACHAT UNIQUEMENT"

    # Afficher l'en-tête de résultats
    if indicator_type.lower() == "williams_r":
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION WILLIAMS %R - MODE {mode_text}")
    elif indicator_type.lower() == "mfi":
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION MFI - MODE {mode_text}")
    elif indicator_type.lower() == "mfi_divergence":
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION DIVERGENCES MFI - MODE {mode_text}")
    elif indicator_type.lower() == "regression_r2":
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION RÉGRESSION R² - MODE {mode_text}")
    elif indicator_type.lower() == "regression_std":
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION RÉGRESSION ÉCART-TYPE - MODE {mode_text}")
    elif indicator_type.lower() == "regression_slope":
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION RÉGRESSION PENTE - MODE {mode_text}")
    elif indicator_type.lower() == "atr":
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION ATR - MODE {mode_text}")
    elif indicator_type.lower() == "vwap":
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION VWAP - MODE {mode_text}")
    elif indicator_type.lower() == "percent_bb_simu":
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION BANDES DE BOLLINGER %B - MODE {mode_text}")
    elif indicator_type.lower() == "zscore":
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION Z-SCORE - MODE {mode_text}")
    elif indicator_type.lower() == "imbalance":
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION IMBALANCE - MODE {mode_text}")
    else:  # Default to stochastic (cohérent avec la section d'attribution de fonction objective)
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION STOCHASTIQUE - MODE {mode_text}")

    # Afficher les meilleurs résultats avec statistiques détaillées
    if len(study.trials) > 0:
        best_trial = study.best_trial
        bin_0_pct = best_trial.user_attrs.get('bin_0_pct_val', 0)
        bin_1_pct = best_trial.user_attrs.get('bin_1_pct_val', 0)
        bin_0_wr = best_trial.user_attrs.get('bin_0_win_rate_val', 0)
        bin_1_wr = best_trial.user_attrs.get('bin_1_win_rate_val', 0)
        bin_1_win_count = best_trial.user_attrs.get('bin_1_win_count_val', 0)

        # Calculer le nombre approximatif d'échantillons dans chaque bin
        total_samples = len(df_val_filtered_)
        valid_samples = total_samples * (1 - df_val_filtered_.isna().mean().mean())  # Estimation des échantillons valides
        bin_0_samples = int(valid_samples * bin_0_pct)
        bin_1_samples = int(valid_samples * bin_1_pct)

        # Rappel des paramètres d'optimisation
        print("\n🔧 PARAMÈTRES D'OPTIMISATION sur VAL:")
        print(f"  Mode d'optimisation: {mode_text}")

        # Afficher uniquement les paramètres pertinents selon le mode
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  MIN_BIN_SPREAD: {MIN_BIN_SPREAD}")  # Écart minimum entre bins
            print(f"  MAX_BIN_0_WIN_RATE: {MAX_BIN_0_WIN_RATE}")  # Maximum pour bin0
            print(f"  MAX_BIN_1_WIN_RATE: {MAX_BIN_1_WIN_RATE}")  # Minimum pour bin1
            print(f"  MIN_BIN_SIZE_0: {MIN_BIN_SIZE_0}")  # Taille minimale pour un bin individuel
            print(f"  MIN_BIN_SIZE_1: {MIN_BIN_SIZE_1}")  # Taille minimale pour un bin individuel
        elif OPTIMIZE_OVERSOLD:
            print(f"  MAX_BIN_0_WIN_RATE: {MAX_BIN_0_WIN_RATE}")  # Maximum pour bin0
            print(f"  MIN_BIN_SIZE_0: {MIN_BIN_SIZE_0}")  # Taille minimale pour un bin individuel
        elif OPTIMIZE_OVERBOUGHT:
            print(f"  MAX_BIN_1_WIN_RATE: {MAX_BIN_1_WIN_RATE}")  # Minimum pour bin1
            print(f"  MIN_BIN_SIZE_1: {MIN_BIN_SIZE_1}")  # Taille minimale pour un bin individuel

        print(f"  COEFF_SPREAD: {COEFF_SPREAD}")
        print(f"  COEFF_BIN_SIZE: {COEFF_BIN_SIZE}")

        # 1. Paramètres optimaux trouvés
        print(f"\n🏆 MEILLEURS PARAMÈTRES TROUVÉS POUR '{indicator_type}'")
        if indicator_type.lower() == "williams_r":
            print(f"  Période: {best_trial.params.get('period', 'N/A')}")
            if OPTIMIZE_OVERSOLD:
                print(f"  Seuil de survente (OS): {best_trial.params.get('OS_limit', 'N/A')}")
            if OPTIMIZE_OVERBOUGHT:
                print(f"  Seuil de surachat (OB): {best_trial.params.get('OB_limit', 'N/A')}")
        elif indicator_type.lower() == "mfi":
            print(f"  Période: {best_trial.params.get('period', 'N/A')}")
            if OPTIMIZE_OVERSOLD:
                print(f"  Seuil de survente (OS): {best_trial.params.get('OS_limit', 'N/A')}")
            if OPTIMIZE_OVERBOUGHT:
                print(f"  Seuil de surachat (OB): {best_trial.params.get('OB_limit', 'N/A')}")
        elif indicator_type.lower() == "mfi_divergence":
            print(f"  Période MFI: {best_trial.params.get('mfi_period', 'N/A')}")
            print(f"  Période de recherche des divergences: {best_trial.params.get('div_lookback', 'N/A')}")
            print(f"  Seuil minimal d'augmentation de prix: {best_trial.params.get('min_price_increase', 'N/A')}")
            print(f"  Seuil minimal de diminution du MFI: {best_trial.params.get('min_mfi_decrease', 'N/A')}")

            # Afficher ces paramètres uniquement s'ils sont présents (mode OPTIMIZE_OVERSOLD)
            if 'min_price_decrease' in best_trial.params:
                print(f"  Seuil minimal de diminution de prix: {best_trial.params.get('min_price_decrease', 'N/A')}")
            if 'min_mfi_increase' in best_trial.params:
                print(f"  Seuil minimal d'augmentation du MFI: {best_trial.params.get('min_mfi_increase', 'N/A')}")
        elif indicator_type.lower() == "regression_r2":
            period_var_r2 = best_trial.params.get('period_var_r2', best_trial.params.get('slope', 'N/A'))
            r2_low_threshold = best_trial.params.get('r2_low_threshold', 'N/A')
            r2_high_threshold = best_trial.params.get('r2_high_threshold', 'N/A')

            # Afficher toujours les deux seuils comme pour std
            print(f"  Période de la pente: {period_var_r2}")
            print(f"  Seuil bas R² (r2_low_threshold): {r2_low_threshold}")
            print(f"  Seuil haut R² (r2_high_threshold): {r2_high_threshold}")
            if OPTIMIZE_OVERSOLD:
                print(
                    f"  Seuil de volatilité basse (r2_high_threshold): {best_trial.params.get('r2_high_threshold', 'N/A')}")
            if OPTIMIZE_OVERBOUGHT:
                print(
                    f"  Seuil de volatilité haute (r2_low_threshold): {best_trial.params.get('r2_low_threshold', 'N/A')}")
        elif indicator_type.lower() == "regression_std":
            print(f"  Période de la pente: {best_trial.params.get('period_var_std', 'N/A')}")
            if OPTIMIZE_OVERSOLD:
                print(
                    f"  Seuil de volatilité basse (std_low_threshold): {best_trial.params.get('std_low_threshold', 'N/A')} et (std_high_threshold): {best_trial.params.get('std_high_threshold', 'N/A')}")
            if OPTIMIZE_OVERBOUGHT:
                print(
                    f"  Seuil de volatilité haute (std_low_threshold): {best_trial.params.get('std_low_threshold', 'N/A')}  et  (std_high_threshold): {best_trial.params.get('std_high_threshold', 'N/A')}")
        elif indicator_type.lower() == "regression_slope":
            print(f"  Période de la pente: {best_trial.params.get('period_var_slope', 'N/A')}")
            if OPTIMIZE_OVERSOLD:
                print(
                    f"  Seuil de pente faible slope_range_threshold: {best_trial.params.get('slope_range_threshold', 'N/A')}, slope_extrem_threshold {best_trial.params.get('slope_extrem_threshold', 'N/A')} ")
            if OPTIMIZE_OVERBOUGHT:
                print(
                    f"  Seuil de pente forte slope_range_threshold: {best_trial.params.get('slope_range_threshold', 'N/A')}, slope_extrem_threshold {best_trial.params.get('slope_extrem_threshold', 'N/A')} ")
        elif indicator_type.lower() == "atr":
            print(f"  Période de l'ATR: {best_trial.params.get('period_var_atr', 'N/A')}")
            if OPTIMIZE_OVERSOLD:
                print(f"  Seuil bas de l'ATR (atr_low_threshold): {best_trial.params.get('atr_low_threshold', 'N/A')}")
                # print(
                #     f"  Seuil haut de l'ATR (atr_high_threshold): {best_trial.params.get('atr_high_threshold', 'N/A')}")
            if OPTIMIZE_OVERBOUGHT:
                print(f"  Seuil bas de l'ATR (atr_low_threshold): {best_trial.params.get('atr_low_threshold', 'N/A')}")
                print(
                    f"  Seuil haut de l'ATR (atr_high_threshold): {best_trial.params.get('atr_high_threshold', 'N/A')}")
        elif indicator_type.lower() == "vwap":
            vwap_low_threshold = best_trial.params.get('vwap_low_threshold', 'N/A')
            vwap_high_threshold = best_trial.params.get('vwap_high_threshold', 'N/A')
            print(f"  Seuil bas VWAP (vwap_low_threshold): {vwap_low_threshold}")
            print(f"  Seuil haut VWAP (vwap_high_threshold): {vwap_high_threshold}")
        elif indicator_type.lower() == "Stochastique":
            print(f"  Période K: {best_trial.params.get('k_period', 'N/A')}")
            print(f"  Période D: {best_trial.params.get('d_period', 'N/A')}")
            if OPTIMIZE_OVERSOLD:
                print(f"  Seuil de survente (OS): {best_trial.params.get('OS_limit', 'N/A')}")
            if OPTIMIZE_OVERBOUGHT:
                print(f"  Seuil de surachat (OB): {best_trial.params.get('OB_limit', 'N/A')}")
        elif indicator_type.lower() == "zscore":
            print(f"  Période du Z-Score: {best_trial.params.get('period_var_zscore', 'N/A')}")
            if OPTIMIZE_OVERSOLD:
                print(
                    f"  Seuil du Z-Score bas (zscore_low_threshold): {best_trial.params.get('zscore_low_threshold', 'N/A')}, haut (zscore_high_threshold): {best_trial.params.get('zscore_high_threshold', 'N/A')}")
            if OPTIMIZE_OVERBOUGHT:
                print(
                    f"  Seuil du Z-Score bas (zscore_low_threshold): {best_trial.params.get('zscore_low_threshold', 'N/A')}, haut (zscore_high_threshold): {best_trial.params.get('zscore_high_threshold', 'N/A')}")
        elif indicator_type.lower() == "percent_bb_simu":
            print(f"  Période des bandes de Bollinger: {best_trial.params.get('period_var_bb', 'N/A')}")
            print(f"  Écart-type: {best_trial.params.get('std_dev', 'N/A')}")
            if OPTIMIZE_OVERSOLD:
                print(f"  Seuil bas %B (bb_low_threshold): {best_trial.params.get('bb_low_threshold', 'N/A')}")
            if OPTIMIZE_OVERBOUGHT:
                print(f"  Seuil haut %B (bb_high_threshold): {best_trial.params.get('bb_high_threshold', 'N/A')}")
        elif indicator_type.lower() == "percent_bb_simu":
            print(f"  Période des bandes de Bollinger: {best_trial.params.get('period_var_bb', 'N/A')}")
            print(f"  Écart-type: {best_trial.params.get('std_dev', 'N/A')}")
            if OPTIMIZE_OVERSOLD:
                print(f"  Seuil bas %B (bb_low_threshold): {best_trial.params.get('bb_low_threshold', 'N/A')}")
            if OPTIMIZE_OVERBOUGHT:
                print(f"  Seuil haut %B (bb_high_threshold): {best_trial.params.get('bb_high_threshold', 'N/A')}")

        else:
            exit(456)
        print(f"  Score d'optimisation: {study.best_value:.4f}")

        # 2. Statistiques de performance - ajusté selon le mode
        print("\n📊 STATISTIQUES DE PERFORMANCE:")
        print(f"  Nombre total d'échantillons: {total_samples}")
        print(f"  Échantillons valides estimés: {valid_samples:.0f}")

        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  Spread global (différence de winrate): {best_trial.user_attrs.get('bin_spread', 'N/A'):.4f}")

        # Adapter le nom des bins selon l'indicateur
        bin0_name = "Survente"
        bin1_name = "Surachat"

        if indicator_type.lower() == "mfi_divergence":
            bin0_name = "Divergence Haussière"
            bin1_name = "Divergence Baissière"
        elif indicator_type.lower() == "regression_r2":
            bin0_name = "Volatilité Basse"
            bin1_name = "Volatilité Haute"
        elif indicator_type.lower() == "regression_std":
            bin0_name = "Écart-type Faible"
            bin1_name = "Écart-type Élevé"
        elif indicator_type.lower() == "regression_slope":
            bin0_name = "Volatilité dans les Extrême"
            bin1_name = "Volatilité Modérée"
        elif indicator_type.lower() == "atr":
            bin0_name = "ATR Faible"
            bin1_name = "ATR Modéré"
        elif indicator_type.lower() == "vwap":
            bin0_name = "Distance VWAP Extrême"
            bin1_name = "Distance VWAP Modérée"
        elif indicator_type.lower() == "percent_bb_simu":
            bin0_name = "%B Extrême"
            bin1_name = "%B Modéré"
        elif indicator_type.lower() == "zscore":
            bin0_name = "Z-Score Extrême"
            bin1_name = "Z-Score Modéré"

        # Afficher uniquement les statistiques pertinentes selon le mode
        if OPTIMIZE_OVERSOLD:
            oversold_success_count = best_trial.user_attrs.get('oversold_success_count', 0)
            print(f"\n  Statistiques du bin 0 ({bin0_name}):")
            print(f"    • Win Rate: {bin_0_wr:.4f}")
            print(f"    • Nombre d'échantillons: ~{bin_0_samples}")
            print(f"    • Nombre de trades gagnants: {oversold_success_count}")
            print(f"    • Pourcentage des données: {bin_0_pct:.2%}")

        if OPTIMIZE_OVERBOUGHT:
            overbought_success_count = best_trial.user_attrs.get('overbought_success_count', 0)
            print(f"\n  Statistiques du bin 1 ({bin1_name}):")
            print(f"    • Win Rate: {bin_1_wr:.4f}")
            print(f"    • Nombre d'échantillons: ~{bin_1_samples}")
            print(f"    • Nombre de trades gagnants: {overbought_success_count}")
            print(f"    • Pourcentage des données: {bin_1_pct:.2%}")

        # 3. Évaluation de la robustesse statistique - ajustée selon le mode
        print("\n🔍 ANALYSE DE ROBUSTESSE STATISTIQUE:")

        # Taille des échantillons - uniquement pour les bins pertinents
        print("  1. Taille des échantillons:")
        if OPTIMIZE_OVERSOLD:
            if bin_0_samples < 30:
                print(f"    ❌ Bin 0: Trop peu d'échantillons ({bin_0_samples}), résultats non fiables")
            elif bin_0_samples < 100:
                print(f"    ⚠️ Bin 0: Nombre d'échantillons limité ({bin_0_samples}), fiabilité modérée")
            else:
                print(f"    ✅ Bin 0: Nombre d'échantillons suffisant ({bin_0_samples})")

        if OPTIMIZE_OVERBOUGHT:
            if bin_1_samples < 30:
                print(f"    ❌ Bin 1: Trop peu d'échantillons ({bin_1_samples}), résultats non fiables")
            elif bin_1_samples < 100:
                print(f"    ⚠️ Bin 1: Nombre d'échantillons limité ({bin_1_samples}), fiabilité modérée")
            else:
                print(f"    ✅ Bin 1: Nombre d'échantillons suffisant ({bin_1_samples})")

        # Pourcentage des données - uniquement pour les bins pertinents
        print("\n  2. Couverture des données:")
        if OPTIMIZE_OVERSOLD:
            if bin_0_pct < 0.05:
                print(
                    f"    ❌ Bin 0: Représente moins de 5% des données ({bin_0_pct:.2%}), risque élevé de surajustement")
            elif bin_0_pct < 0.10:
                print(
                    f"    ⚠️ Bin 0: Représente moins de 10% des données ({bin_0_pct:.2%}), risque modéré de surajustement")
            else:
                print(f"    ✅ Bin 0: Bonne couverture des données ({bin_0_pct:.2%})")

        if OPTIMIZE_OVERBOUGHT:
            if bin_1_pct < 0.05:
                print(
                    f"    ❌ Bin 1: Représente moins de 5% des données ({bin_1_pct:.2%}), risque élevé de surajustement")
            elif bin_1_pct < 0.10:
                print(
                    f"    ⚠️ Bin 1: Représente moins de 10% des données ({bin_1_pct:.2%}), risque modéré de surajustement")
            else:
                print(f"    ✅ Bin 1: Bonne couverture des données ({bin_1_pct:.2%})")

        # Écart par rapport à 50% - uniquement pour les bins pertinents
        print("\n  3. Écart par rapport à 50% (neutralité):")
        if OPTIMIZE_OVERSOLD:
            if bin_0_wr > 0.48:
                print(f"    ❌ Bin 0: Win rate trop proche de 50% ({bin_0_wr:.2%}), signal faible")
            elif bin_0_wr > 0.45:
                print(f"    ⚠️ Bin 0: Win rate modérément inférieur à 50% ({bin_0_wr:.2%})")
            else:
                print(f"    ✅ Bin 0: Win rate significativement inférieur à 50% ({bin_0_wr:.2%})")

        if OPTIMIZE_OVERBOUGHT:
            if bin_1_wr < 0.52:
                print(f"    ❌ Bin 1: Win rate trop proche de 50% ({bin_1_wr:.2%}), signal faible")
            elif bin_1_wr < 0.55:
                print(f"    ⚠️ Bin 1: Win rate modérément supérieur à 50% ({bin_1_wr:.2%})")
            else:
                print(f"    ✅ Bin 1: Win rate significativement supérieur à 50% ({bin_1_wr:.2%})")

        # 4. Conclusion et recommandations - ajustée selon le mode
        print("\n💡 CONCLUSION:")

        indicator_name = ""
        indicator_name = ""
        if indicator_type.lower() == "williams_r":
            indicator_name = "Williams %R"
        elif indicator_type.lower() == "mfi":
            indicator_name = "Money Flow Index"
        elif indicator_type.lower() == "mfi_divergence":
            indicator_name = "Divergences MFI"
        elif indicator_type.lower() == "regression_r2":
            indicator_name = "Régression R²"
        elif indicator_type.lower() == "regression_std":
            indicator_name = "Régression Écart-type"
        elif indicator_type.lower() == "regression_slope":
            indicator_name = "Régression Pente"
        elif indicator_type.lower() == "atr":
            indicator_name = "ATR"
        elif indicator_type.lower() == "percent_bb_simu":
            indicator_name = "Bandes de Bollinger %B"
        elif indicator_type.lower() == "zscore":
            indicator_name = "Z-Score"
        elif indicator_type.lower() == "imbalance":
            indicator_name = "Imbalance"
        else:
            exit(78)

        # Évaluation de qualité adaptée selon le mode
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            # Critères complets pour le mode double
            spread_quality = "faible" if best_trial.user_attrs.get('bin_spread',
                                                                   0) < 0.08 else "modéré" if best_trial.user_attrs.get(
                'bin_spread', 0) < 0.12 else "significatif"
            sample_quality = "insuffisante" if bin_0_samples < 30 or bin_1_samples < 30 else "limitée" if bin_0_samples < 100 or bin_1_samples < 100 else "suffisante"
            coverage_quality = "faible" if bin_0_pct < 0.05 or bin_1_pct < 0.05 else "modérée" if bin_0_pct < 0.10 or bin_1_pct < 0.10 else "bonne"

            print(
                f"  L'indicateur {indicator_name} optimisé présente un écart de win rate {spread_quality} ({best_trial.user_attrs.get('bin_spread', 0):.4f})")
            print(f"  avec une taille d'échantillon {sample_quality} et une couverture {coverage_quality} des données.")

            quality_issues = (
                        spread_quality == "faible" or sample_quality == "insuffisante" or coverage_quality == "faible")

        elif OPTIMIZE_OVERSOLD:
            # Critères pour le mode survente uniquement
            wr_quality = "faible" if bin_0_wr > 0.48 else "modéré" if bin_0_wr > 0.45 else "significatif"
            sample_quality = "insuffisante" if bin_0_samples < 30 else "limitée" if bin_0_samples < 100 else "suffisante"
            coverage_quality = "faible" if bin_0_pct < 0.05 else "modérée" if bin_0_pct < 0.10 else "bonne"

            print(
                f"  L'indicateur {indicator_name} optimisé présente une zone de {bin0_name.lower()} avec un win rate de {bin_0_wr:.4f}")
            print(f"  avec une taille d'échantillon {sample_quality} et une couverture {coverage_quality} des données.")

            quality_issues = (
                        wr_quality == "faible" or sample_quality == "insuffisante" or coverage_quality == "faible")

        elif OPTIMIZE_OVERBOUGHT:
            # Critères pour le mode surachat uniquement
            wr_quality = "faible" if bin_1_wr < 0.52 else "modéré" if bin_1_wr < 0.55 else "significatif"
            sample_quality = "insuffisante" if bin_1_samples < 30 else "limitée" if bin_1_samples < 100 else "suffisante"
            coverage_quality = "faible" if bin_1_pct < 0.05 else "modérée" if bin_1_pct < 0.10 else "bonne"

            print(
                f"  L'indicateur {indicator_name} optimisé présente une zone de {bin1_name.lower()} avec un win rate de {bin_1_wr:.4f}")
            print(f"  avec une taille d'échantillon {sample_quality} et une couverture {coverage_quality} des données.")

            quality_issues = (
                        wr_quality == "faible" or sample_quality == "insuffisante" or coverage_quality == "faible")

        # Recommandations finales
        if quality_issues:
            print("\n  ⚠️ RECOMMANDATION: Considérez cet indicateur avec prudence. Il est préférable de:")
            print("   - L'utiliser en complément d'autres indicateurs")
            print("   - Tester sa robustesse sur d'autres périodes")
            print("   - Ajuster les contraintes pour trouver un meilleur équilibre")
        else:
            print("\n  ✅ RECOMMANDATION: Cet indicateur montre des caractéristiques prometteuses.")
            print("   - Utilisable pour filtrer les trades")
            print("   - Vérifiez sa stabilité sur différentes périodes de temps")
            print("   - Envisagez de combiner avec d'autres indicateurs pour renforcer le signal")
    else:
        print("\n⚠️ Aucun essai valide n'a été trouvé.")
        print("Essayez d'ajuster les contraintes avec modify_constraints().")

    # Afficher des informations supplémentaires pour les divergences MFI
    if indicator_type.lower() == "mfi_divergence" and len(study.trials) > 0:
        bearish_success_count = best_trial.user_attrs.get('bearish_success_count', 0)
        bullish_success_count = best_trial.user_attrs.get('bullish_success_count', 0)
        bearish_div_count = best_trial.user_attrs.get('bearish_div_count', 0)
        bullish_div_count = best_trial.user_attrs.get('bullish_div_count', 0)

        print(f"\n  Statistiques des divergences:")

        if OPTIMIZE_OVERSOLD:
            print(f"    • Divergences baissières: {bearish_div_count}")
            print(f"    • Trades réussis avec divergence baissière: {bearish_success_count}")
            print(f"    • Win rate: {bin_0_wr:.4f}")

        if OPTIMIZE_OVERBOUGHT:
            print(f"    • Divergences haussières: {bullish_div_count}")
            print(f"    • Trades réussis avec divergence haussière: {bullish_success_count}")
            print(f"    • Win rate: {bin_1_wr:.4f}")

    return study
if indicator_type == "stochastic":
    study_stoch = run_indicator_optimization(df_train_=df_train, df_val_=df_val, df_val_filtered_=df_val_filtered,indicator_type="stochastic", n_trials=20000)
    study = study_stoch
elif indicator_type == "williams_r":
    study_williams = run_indicator_optimization(df_train_=df_train, df_val_=df_val, df_val_filtered_=df_val_filtered, indicator_type="williams_r", n_trials=20000)
    study = study_williams
elif indicator_type == "mfi":
    study_mfi = run_indicator_optimization(df_train_=df_train, df_val_=df_val, df_val_filtered_=df_val_filtered, indicator_type="mfi", n_trials=20000)
    study = study_mfi
elif indicator_type == "mfi_divergence":
    study_mfi_div = run_indicator_optimization(df_train_=df_train, df_val_=df_val, df_val_filtered_=df_val_filtered, indicator_type="mfi_divergence", n_trials=20000)
    study = study_mfi_div
elif indicator_type == "regression_r2":
    study_regression = run_indicator_optimization(df_train_=df_train, df_val_=df_val, df_val_filtered_=df_val_filtered,indicator_type="regression_r2", n_trials=20000)
    study = study_regression
elif indicator_type == "regression_std":
    study_regression_std = run_indicator_optimization(df_train_=df_train, df_val_=df_val,df_val_filtered_=df_val_filtered,  indicator_type="regression_std", n_trials=200000)
    study = study_regression_std
elif indicator_type == "regression_slope":
    study_regression_slope = run_indicator_optimization(df_train_=df_train, df_val_=df_val,df_val_filtered_=df_val_filtered, indicator_type="regression_slope", n_trials=20000)
    study = study_regression_slope
elif indicator_type == "atr":
    study_atr = run_indicator_optimization(df_train_=df_train, df_val_=df_val,df_val_filtered_=df_val_filtered,  indicator_type="atr", n_trials=20000)
    study = study_atr
elif indicator_type == "vwap":
    study_vwap = run_indicator_optimization(df_train_=df_train, df_val_=df_val,df_val_filtered_=df_val_filtered,  indicator_type="vwap", n_trials=20000)
    study = study_vwap
elif indicator_type == "percent_bb_simu":
    study_bb = run_indicator_optimization(df_train_=df_train, df_val_=df_val,df_val_filtered_=df_val_filtered, indicator_type="percent_bb_simu", n_trials=20000)
    study = study_bb
elif indicator_type == "zscore":
    study_zscore = run_indicator_optimization(df_train_=df_train, df_val_=df_val,df_val_filtered_=df_val_filtered,  indicator_type="zscore", n_trials=20000)
    study = study_zscore
elif indicator_type == "imbalance":
    study_imbalance = run_indicator_optimization(df_train_=df_train, df_val_=df_val,df_val_filtered_=df_val_filtered, indicator_type="imbalance", n_trials=20000)
    study = study_imbalance
else:
    print(f"Type d'indicateur non reconnu: {indicator_type}")
    exit(78)








# Récupérer les meilleurs paramètres de l'étude
best_params = study.best_trial.params
print(f"Meilleurs paramètres trouvés: {best_params}")


# Évaluer ces paramètres sur le jeu de test
test_results = evaluate_best_params_on_test_data(best_params, indicator_type, df_test)


def print_comparison(best_trial, test_results, indicator_type):
    """
    Affiche une comparaison entre les résultats d'entraînement et de test,
    en tenant compte des modes d'optimisation activés.
    """
    # Récupérer les métriques d'entraînement
    train_bin_0_wr = best_trial.user_attrs.get('bin_0_win_rate_val', 0)
    train_bin_1_wr = best_trial.user_attrs.get('bin_1_win_rate_val', 0)
    train_bin_0_pct = best_trial.user_attrs.get('bin_0_pct_val', 0)
    train_bin_1_pct = best_trial.user_attrs.get('bin_1_pct_val', 0)
    train_bin_spread = best_trial.user_attrs.get('bin_spread_val', 0)

    # Récupérer les métriques de test
    test_bin_0_wr = test_results.get('bin_0_win_rate', 0)
    test_bin_1_wr = test_results.get('bin_1_win_rate', 0)
    test_bin_0_pct = test_results.get('bin_0_pct', 0)
    test_bin_1_pct = test_results.get('bin_1_pct', 0)
    test_bin_spread = test_results.get('bin_spread', 0)

    # Noms des bins selon l'indicateur
    bin0_name = get_bin_name(indicator_type, 0)
    bin1_name = get_bin_name(indicator_type, 1)

    print(f"\n{'=' * 80}")
    print(f"COMPARAISON DES RÉSULTATS D'ENTRAÎNEMENT ET DE TEST POUR {indicator_type.upper()}")
    print(f"{'=' * 80}")

    # Tableau comparatif
    print(f"\n{'Métrique':<38} {'Entraînement':<15} {'Test':<15} {'Différence':<15} {'Variation %':<15}")
    print(f"{'-' * 25} {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15}")

    # Afficher les métriques de bin 0 uniquement si OPTIMIZE_OVERSOLD est activé
    if OPTIMIZE_OVERSOLD and train_bin_0_wr > 0 and test_bin_0_wr > 0:
        print(
            f"Win Rate Bin 0 ({bin0_name}){' ' * (4)} {train_bin_0_wr:<15.4f} {test_bin_0_wr:<15.4f} {test_bin_0_wr - train_bin_0_wr:<15.4f} {((test_bin_0_wr / train_bin_0_wr) - 1) * 100:<15.2f}%")
        print(
            f"Couverture Bin 0{' ' * (13)} {train_bin_0_pct:<15.2%} {test_bin_0_pct:<15.2%} {test_bin_0_pct - train_bin_0_pct:<15.2%} {((test_bin_0_pct / train_bin_0_pct) - 1) * 100:<15.2f}%")

    # Afficher les métriques de bin 1 uniquement si OPTIMIZE_OVERBOUGHT est activé
    if OPTIMIZE_OVERBOUGHT and train_bin_1_wr > 0 and test_bin_1_wr > 0:
        print(
            f"Win Rate Bin 1 ({bin1_name}){' ' * (4)} {train_bin_1_wr:<15.4f} {test_bin_1_wr:<15.4f} {test_bin_1_wr - train_bin_1_wr:<15.4f} {((test_bin_1_wr / train_bin_1_wr) - 1) * 100:<15.2f}%")
        print(
            f"Couverture Bin 1{' ' * (13)} {train_bin_1_pct:<15.2%} {test_bin_1_pct:<15.2%} {test_bin_1_pct - train_bin_1_pct:<15.2%} {((test_bin_1_pct / train_bin_1_pct) - 1) * 100:<15.2f}%")

    # Afficher le spread uniquement si les deux modes sont activés
    if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT and train_bin_spread > 0 and test_bin_spread > 0:
        print(
            f"Spread{' ' * (19)} {train_bin_spread:<15.4f} {test_bin_spread:<15.4f} {test_bin_spread - train_bin_spread:<15.4f} {((test_bin_spread / train_bin_spread) - 1) * 100:<15.2f}%")

    # Analyse des variations
    print(f"\n🔍 ANALYSE DE LA GÉNÉRALISATION:")

    # Vérifier la stabilité du win rate seulement pour les modes activés
    if OPTIMIZE_OVERSOLD and train_bin_0_wr > 0 and test_bin_0_wr > 0:
        wr_diff = test_bin_0_wr - train_bin_0_wr
        if abs(wr_diff) > 0.05:
            print(
                f"  ⚠️ Variation importante du win rate Bin 0: {wr_diff:.4f} ({((test_bin_0_wr / train_bin_0_wr) - 1) * 100:.2f}%)")
        else:
            print(
                f"  ✅ Win rate Bin 0 stable entre entraînement et test: {wr_diff:.4f} ({((test_bin_0_wr / train_bin_0_wr) - 1) * 100:.2f}%)")

    if OPTIMIZE_OVERBOUGHT and train_bin_1_wr > 0 and test_bin_1_wr > 0:
        wr_diff = test_bin_1_wr - train_bin_1_wr
        if abs(wr_diff) > 0.05:
            print(
                f"  ⚠️ Variation importante du win rate Bin 1: {wr_diff:.4f} ({((test_bin_1_wr / train_bin_1_wr) - 1) * 100:.2f}%)")
        else:
            print(
                f"  ✅ Win rate Bin 1 stable entre entraînement et test: {wr_diff:.4f} ({((test_bin_1_wr / train_bin_1_wr) - 1) * 100:.2f}%)")

    # Vérifier la stabilité du spread seulement si les deux modes sont activés
    if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT and train_bin_spread > 0 and test_bin_spread > 0:
        spread_diff = test_bin_spread - train_bin_spread
        if abs(spread_diff) > 0.05:
            print(
                f"  ⚠️ Variation importante du spread: {spread_diff:.4f} ({((test_bin_spread / train_bin_spread) - 1) * 100:.2f}%)")
        else:
            print(
                f"  ✅ Spread stable entre entraînement et test: {spread_diff:.4f} ({((test_bin_spread / train_bin_spread) - 1) * 100:.2f}%)")

    # Conclusion
    print(f"\n📝 CONCLUSION:")
    overfitting = False

    # Vérifier les signes de surapprentissage seulement pour les modes activés
    if OPTIMIZE_OVERSOLD and train_bin_0_wr > 0 and test_bin_0_wr > 0 and (
            test_bin_0_wr > train_bin_0_wr + 0.05 or test_bin_0_wr < train_bin_0_wr - 0.05):
        overfitting = True

    if OPTIMIZE_OVERBOUGHT and train_bin_1_wr > 0 and test_bin_1_wr > 0 and (
            test_bin_1_wr > train_bin_1_wr + 0.05 or test_bin_1_wr < train_bin_1_wr - 0.05):
        overfitting = True

    if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT and train_bin_spread > 0 and test_bin_spread > 0 and test_bin_spread < train_bin_spread * 0.8:
        overfitting = True

    if overfitting:
        print(f"  ⚠️ L'indicateur {indicator_type} présente des signes de surapprentissage (overfitting).")
        print(
            f"  Les performances sur les données de test diffèrent significativement des performances d'entraînement.")
        print(f"  Il est recommandé de revoir les contraintes ou d'essayer une optimisation avec plus de données.")
    else:
        print(f"  ✅ L'indicateur {indicator_type} généralise bien sur les données de test.")
        print(f"  Les performances restent stables entre l'entraînement et le test, ce qui suggère que")
        print(f"  l'indicateur capte des motifs réels et non des fluctuations aléatoires.")


# Afficher une comparaison des résultats d'entraînement et de test
print_comparison(study.best_trial, test_results, indicator_type)