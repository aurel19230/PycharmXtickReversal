import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.metrics import roc_auc_score

import os

from func_standard import print_notification, load_data, calculate_naked_poc_distances, CUSTOM_SESSIONS, \
    save_features_with_sessions,remplace_0_nan_reg_slope_p_2d,process_reg_slope_replacement, calculate_slopes_and_r2_numba,calculate_atr
file_name = "Step4_version2_170924_110325_bugFixTradeResult1_extractOnlyFullSession_OnlyShort.csv"
#file_name = "Step4_5_0_5TP_1SL_150924_280225_bugFixTradeResult_extractOnlyFullSession_OnlyShort.csv"
# Chemin du répertoire
directory_path =  r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version1"
directory_path =  r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version2\merge"
from stats_sc.standard_stat_sc import *

# Construction du chemin complet du fichier
file_path = os.path.join(directory_path, file_name)
import keyboard  # Assurez-vous d'installer cette bibliothèque avec: pip install keyboard

import keyboard  # Assurez-vous d'installer cette bibliothèque avec: pip install keyboard
from pynput import keyboard as pynput_keyboard  # Alternative si keyboard pose problème



# Définir le type d'indicateur à optimiser
indicator_type = "regression_std"
# Add these global flags to control which zones to optimize
OPTIMIZE_OVERSOLD = True  # Set to False to disable oversold zone optimization => => find the worst winrate
OPTIMIZE_OVERBOUGHT = False  # Set to False to disable overbought zone optimization => optimize de winrate

if indicator_type == "atr":
    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.0501  # Écart minimum entre bins
    MAX_bin_0_win_rate = 0.465  # Maximum pour bin0 (doit être < 0.5) => find the worst winrate (work with OPTIMIZE_OVERSOLD param)
    MIN_bin_1_win_rate = 0.052  # Minimum pour bin1 (doit être > 0.5) => optimize de winrate (work with OPTIMIZE_OVERBOUGHT param)²
    MIN_BIN_SIZE_0 = 0.08  # Taille minimale pour un bin individuel => find the worst winrate (work with OPTIMIZE_OVERSOLD param)
    MIN_BIN_SIZE_1 = 0.0000095  # Taille minimale pour un bin individuel => optimize de winrate (work with OPTIMIZE_OVERBOUGHT param)
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    PERIOD_VAR_L = 18
    PERIOD_VAR_H = 32

    ATR_LOW_THRESHOLD_L = 0
    ATR_LOW_THRESHOLD_H = 2

    ATR_HIGH_THRESHOLD_L = 0
    ATR_HIGH_THRESHOLD_H = 23

if indicator_type == "regression_slope":
    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.0501  # Écart minimum entre bins
    MAX_bin_0_win_rate = 0.46
    # Maximum pour bin0 (doit être < 0.5) => find the worst winrate (work with OPTIMIZE_OVERSOLD param)
    MIN_bin_1_win_rate = 0.54  # Minimum pour bin1 (doit être > 0.5) => optimize de winrate (work with OPTIMIZE_OVERBOUGHT param)
    MIN_BIN_SIZE_0 = 0.0001  # Taille minimale pour un bin individuel => find the worst winrate (work with OPTIMIZE_OVERSOLD param)
    MIN_BIN_SIZE_1 = 0.1  # Taille minimale pour un bin individuel => optimize de winrate (work with OPTIMIZE_OVERBOUGHT param)
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    PERIOD_VAR_L = 10
    PERIOD_VAR_H = 22

    SLOPE_RANGE_THRESHOLD_L = -0.4
    SLOPE_RANGE_THRESHOLD_H = -0.2

    SLOPE_EXTREM_THRESHOLD_L = -0.3
    SLOPE_EXTREM_THRESHOLD_H = -0.1

if indicator_type == "regression_std":
    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.0501  # Écart minimum entre bins
    MAX_bin_0_win_rate = 0.455  # Maximum pour bin0 (doit être < 0.5)
    MIN_bin_1_win_rate = 0.53  # Minimum pour bin1 (doit être > 0.5)
    MIN_BIN_SIZE_0 = 0.12  # Taille minimale pour un bin individuel
    MIN_BIN_SIZE_1 = 0.0000105  # Taille minimale pour un bin individuel
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    PERIOD_VAR_L=40
    PERIOD_VAR_H=50

    STD_LOW_THRESHOLD_L=1.3
    STD_LOW_THRESHOLD_H=1.4

    STD_HIGH_THRESHOLD_L=4.7
    STD_HIGH_THRESHOLD_H=5

if indicator_type == "regression_r2":
    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.0501  # Écart minimum entre bins
    MAX_bin_0_win_rate = 0.48  # Maximum pour bin0 (doit être < 0.5)
    MIN_bin_1_win_rate = 0.52  # Minimum pour bin1 (doit être > 0.5)
    MIN_BIN_SIZE_0 = 0.05  # Taille minimale pour un bin individuel
    MIN_BIN_SIZE_1 = 0.07  # Taille minimale pour un bin individuel
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    # Paramètres pour l'optimisation Optuna
    PERIOD_VAR_L = 4
    PERIOD_VAR_H = 40

    R2_LOW_THRESHOLD_L = -1.0
    R2_LOW_THRESHOLD_H = 0.5

    R2_HIGH_THRESHOLD_L = -0.2
    R2_HIGH_THRESHOLD_H = 0.9

if indicator_type == "stochastic":
    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.0501  # Écart minimum entre bins
    MAX_bin_0_win_rate = 0.455  # Maximum pour bin0 (doit être < 0.5)
    MIN_bin_1_win_rate = 0.535  # Minimum pour bin1 (doit être > 0.5)
    MIN_BIN_SIZE_0 = 0.08  # Taille minimale pour un bin individuel
    MIN_BIN_SIZE_1 = 0.10  # Taille minimale pour un bin individuel
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    # Paramètres pour l'optimisation Optuna
    K_PERIOD_L = 40
    K_PERIOD_H = 90

    D_PERIOD_L = 40  # Doit être >= K_PERIOD dans la fonction
    D_PERIOD_H = 90

    OS_LIMIT_L = 5
    OS_LIMIT_H = 45

    OB_LIMIT_L = 55
    OB_LIMIT_H = 95
if indicator_type == "williams_r":
    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.03  # Écart minimum entre bins
    MAX_bin_0_win_rate = 0.46  # Maximum pour bin0 (doit être < 0.5)
    MIN_bin_1_win_rate = 0.529  # Minimum pour bin1 (doit être > 0.5)
    MIN_BIN_SIZE_0 = 0.00000095  # Taille minimale pour un bin individuel
    MIN_BIN_SIZE_1 = 0.1  # Taille minimale pour un bin individuel
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    # Paramètres pour l'optimisation Optuna
    PERIOD_L = 55
    PERIOD_H = 100

    OS_LIMIT_L = -85
    OS_LIMIT_H = -70

    OB_LIMIT_L = -40
    OB_LIMIT_H = -3

if indicator_type == "mfi":
    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.02  # Écart minimum entre bins
    MAX_bin_0_win_rate = 0.46  # Maximum pour bin0 (doit être < 0.5)
    MIN_bin_1_win_rate = 0.48  # Minimum pour bin1 (doit être > 0.5)
    MIN_BIN_SIZE_0 = 0.1  # Taille minimale pour un bin individuel
    MIN_BIN_SIZE_1 = 0.0001  # Taille minimale pour un bin individuel
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    # Paramètres pour l'optimisation Optuna
    PERIOD_L = 35
    PERIOD_H = 65

    OS_LIMIT_L = 5
    OS_LIMIT_H = 45

    OB_LIMIT_L = 60
    OB_LIMIT_H = 80
if indicator_type == "mfi_divergence":
    # Définir des constantes globales pour les contraintes
    MIN_BIN_SPREAD = 0.01  # Écart minimum entre bins
    MAX_bin_0_win_rate = 0.48  # Maximum pour bin0 (doit être < 0.5)
    MIN_bin_1_win_rate = 0.5358# Minimum pour bin1 (doit être > 0.5)
    MIN_BIN_SIZE_0 = 0.0003  # Taille minimale pour un bin individuel
    MIN_BIN_SIZE_1 = 0.095  # Taille minimale pour un bin individuel
    COEFF_SPREAD = 1
    COEFF_BIN_SIZE = 0

    # Paramètres pour l'optimisation Optuna
    MFI_PERIOD_L = 7
    MFI_PERIOD_H = 16

    DIV_LOOKBACK_L = 4
    DIV_LOOKBACK_H = 15

    MIN_PRICE_INCREASE_L = 0.00008
    MIN_PRICE_INCREASE_H = 0.001

    MIN_MFI_DECREASE_L = 0.00005
    MIN_MFI_DECREASE_H = 0.005
else:
    print(f"Type d'indicateur non reconnu: {indicator_type}")




# Function to display current optimization mode
def check_bin_constraints(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread):
    """
    Vérifie les contraintes pour les bins selon le mode d'optimisation choisi.
    """
    # Special handling when only one mode is enabled
    if OPTIMIZE_OVERSOLD and not OPTIMIZE_OVERBOUGHT:
        # Only check oversold constraints
        if bin_0_win_rate > MAX_bin_0_win_rate:
            #print("❌ Rejected: bin_0_win_rate too low")


            return False
        if bin_0_pct < MIN_BIN_SIZE_0:
            #print("❌ Rejected: bin_0_pct too small")

            return False
        # No need to check bin spread or bin 1 constraints
        return True


    elif OPTIMIZE_OVERBOUGHT and not OPTIMIZE_OVERSOLD:

        # Only check overbought constraints

        # print(
        #     f"Checking overbought constraints: win_rate={bin_1_win_rate}, min={MIN_bin_1_win_rate}, pct={bin_1_pct}, min_size={MIN_BIN_SIZE_1}")

        if bin_1_win_rate < MIN_bin_1_win_rate:
            # print("❌ Rejected: bin_1_win_rate too low")

            return False

        if bin_1_pct < MIN_BIN_SIZE_1:
            # print("❌ Rejected: bin_1_pct too small")

            return False

        print("✅ Constraints satisfied!")

        return True


    else:
        # Both modes enabled, check all constraints
        # Check spread constraint
        if bin_spread < MIN_BIN_SPREAD:
            return False

        # Check oversold constraints
        if bin_0_win_rate > MAX_bin_0_win_rate:
            return False
        if bin_0_pct < MIN_BIN_SIZE_0:
            return False

        # Check overbought constraints
        if bin_1_win_rate < MIN_bin_1_win_rate:
            return False
        if bin_1_pct < MIN_BIN_SIZE_1:
            return False

        return True
    print("-" * 60)
# Then modify the constraint checking in your objective functions like this:



# Variable globale pour contrôler l'arrêt de l'optimisation
STOP_OPTIMIZATION = False

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

    if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
        # Both modes enabled, use both bin sizes and spread
        bin_size_score = (bin_0_pct + bin_1_pct) * 15
        combined_score = (COEFF_SPREAD * normalized_spread) + (COEFF_BIN_SIZE * bin_size_score)


    elif OPTIMIZE_OVERSOLD:
        # Calculer la distance avec 0.5 (équilibre) - plus le winrate est bas, plus cette valeur est grande
        normalized_win_rate = (0.5 - bin_0_win_rate) * 100  # Déjà présent
        # Ajouter un facteur exponentiel pour favoriser fortement les win rates très bas
        if bin_0_win_rate < 0.45:
            normalized_win_rate = normalized_win_rate * 1.5  # Augmenter encore davantage l'impact des win rates très bas
        bin_size_score = bin_0_pct * 15
        combined_score = (COEFF_SPREAD * normalized_win_rate) + (COEFF_BIN_SIZE * bin_size_score)
        print(
            f"📊 Score calculation: win_rate={bin_0_win_rate}, normalized={normalized_win_rate}, bin_0_pct={bin_0_pct}, bin_size_score={bin_size_score}, combined_score={combined_score}")

    elif OPTIMIZE_OVERBOUGHT:
        # Only optimize overbought, focus on high win rate in bin 1
        normalized_win_rate = (bin_1_win_rate - 0.5) * 100  # Normalize around 0.5, scale up
        # Boost pour les win rates très élevés
        if bin_1_win_rate > 0.54:  # Vérifier bin_1_win_rate (pas bin_0_win_rate)
            normalized_win_rate = normalized_win_rate * 1.5  # Augmenter l'impact des win rates très élevés
        bin_size_score = bin_1_pct * 15
        combined_score = (COEFF_SPREAD * normalized_win_rate) + (COEFF_BIN_SIZE * bin_size_score)
        print(

            f"📊 Score calculation: win_rate={bin_1_win_rate}, normalized={normalized_win_rate}, bin_1_pct={bin_1_pct}, bin_size_score={bin_size_score}, combined_score={combined_score}")
    else:
        # Should never happen, but just in case
        combined_score = 0

    # Add bonus for exceptional values based on mode
    if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
        # Bonus for high spread
        if bin_spread >= 0.15:
            combined_score *= 1.15
        elif bin_spread >= 0.12:
            combined_score *= 1.1
    elif OPTIMIZE_OVERSOLD:
        # Bonus for very low bin_0_win_rate
        if bin_0_win_rate <= 0.4:
            combined_score *= 1.15
        elif bin_0_win_rate <= 0.43:
            combined_score *= 1.1
    elif OPTIMIZE_OVERBOUGHT:
        # Bonus for very high bin_1_win_rate
        if bin_1_win_rate >= 0.6:
            combined_score *= 1.15
        elif bin_1_win_rate >= 0.57:
            combined_score *= 1.1

    return combined_score
def callback_optuna_stop(study, trial):
    global STOP_OPTIMIZATION
    if STOP_OPTIMIZATION:
        print("Callback triggered: stopping the study.")
        study.stop()

def on_press(key):
    global STOP_OPTIMIZATION
    try:
        if key.char == '²':
            print("Stop signal received: stopping the study.")
            STOP_OPTIMIZATION = True
    except AttributeError:
        pass
REPLACE_NAN = False
REPLACED_NANVALUE_BY = 90000.54789
REPLACED_NANVALUE_BY_INDEX = 1
if REPLACE_NAN:
    print(
        f"\nINFO : Implémenter dans le code => les valeurs NaN seront remplacées par {REPLACED_NANVALUE_BY} et un index")
else:
    print(
        f"\nINFO : Implémenter dans le code => les valeurs NaN ne seront pas remplacées par une valeur choisie par l'utilisateur mais laissé à NAN")

# Configuration
CONFIG = {
    'FILE_PATH': file_path,
}

df = load_data(CONFIG['FILE_PATH'])

df_filtered = df[df['class_binaire'].isin([0, 1])].copy()
print(df_filtered.shape)

target_y = df_filtered['class_binaire'].copy()  # Série de la cible


# Afficher le nombre de lignes dans df et df_filtered
print(f"Nombre de lignes dans df: {len(df)}")
print(f"Nombre de lignes dans df_filtered: {len(df_filtered)}")

# Calculer et afficher le winrate global de target_y
winrate = target_y.mean()
print(f"Winrate global de target_y: {winrate:.4f} ({winrate:.2%})")

# Afficher la distribution des valeurs dans target_y
value_counts = target_y.value_counts(normalize=True)
print("Distribution des valeurs dans target_y:")
print(target_y.tail(50))

print(value_counts)

# Afficher le nombre exact de 0 et de 1
counts = target_y.value_counts()
print(f"Nombre de 0 (trades perdants): {counts.get(0, 0)}")
print(f"Nombre de 1 (trades gagnants): {counts.get(1, 0)}")



# Variable globale pour le contrôle d'arrêt
should_stop = False


# Fonction pour afficher les contraintes actuelles
def show_constraints():
    print("\n📋 Contraintes actuelles:")
    print(f"  MIN_BIN_SPREAD = {MIN_BIN_SPREAD}")
    print(f"  MAX_bin_0_win_rate = {MAX_bin_0_win_rate}")
    print(f"  MIN_bin_1_win_rate = {MIN_bin_1_win_rate}")
    print(f"  MIN_BIN_SIZE_0 = {MIN_BIN_SIZE_0}")
    print(f"  MIN_BIN_SIZE_1 = {MIN_BIN_SIZE_1}")



# Define a function to modify constraints
def modify_constraints(min_spread=None, max_bin0=None, min_bin1=None, min_size_0=None,min_size_1=None):
    global MIN_BIN_SPREAD, MAX_bin_0_win_rate, MIN_bin_1_win_rate, MIN_BIN_SIZE_0,MIN_BIN_SIZE_1

    if min_spread is not None:
        MIN_BIN_SPREAD = min_spread
    if max_bin0 is not None:
        MAX_bin_0_win_rate = max_bin0
    if min_bin1 is not None:
        MIN_bin_1_win_rate = min_bin1
    if min_size_0 is not None:
        MIN_BIN_SIZE_0 = min_size_0
    if min_size_1 is not None:
        MIN_BIN_SIZE_1 = min_size_1

    show_constraints()
    print("Contraintes modifiées!")


# Fonction pour calculer les contraintes
def calculate_constraints_optuna(trial):
    bin_0_wr = trial.user_attrs.get('bin_0_win_rate', 0)
    bin_1_wr = trial.user_attrs.get('bin_1_win_rate', 0)
    bin_spread = trial.user_attrs.get('bin_spread', 0)
    bin_0_pct = trial.user_attrs.get('bin_0_pct', 0)
    bin_1_pct = trial.user_attrs.get('bin_1_pct', 0)

    constraints = []

    if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
        # Même logique que check_bin_constraints pour "both" :
        c1 = MIN_BIN_SPREAD - bin_spread
        c2 = bin_0_wr - MAX_bin_0_win_rate
        c3 = MIN_bin_1_win_rate - bin_1_wr
        c4 = MIN_BIN_SIZE_0 - bin_0_pct
        c5 = MIN_BIN_SIZE_1 - bin_1_pct
        constraints = [c1, c2, c3, c4, c5]

    elif OPTIMIZE_OVERSOLD and not OPTIMIZE_OVERBOUGHT:
        # Oversold only => On ignore le spread et bin_1
        # On vérifie seulement MAX_bin_0_win_rate et MIN_BIN_SIZE_0
        c2 = bin_0_wr - MAX_bin_0_win_rate       # si c2 > 0 => violation
        c4 = MIN_BIN_SIZE_0 - bin_0_pct          # si c4 > 0 => violation
        constraints = [c2, c4]

    elif not OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
        # Overbought only => On ignore le spread et bin_0
        c3 = MIN_bin_1_win_rate - bin_1_wr       # si c3 > 0 => violation
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
    global should_stop
    try:
        # Si la touche 'q' est pressée, arrêter l'optimisation
        if key.char.lower() == '²':
            print("\n⚠️ Arrêt de l'optimisation demandé par l'utilisateur...")
            should_stop = True
            return False  # Arrêter l'écoute du clavier
    except AttributeError:
        # Ignorer les touches spéciales
        pass


# Callback pour vérifier si l'utilisateur a demandé l'arrêt
def callback_optuna_stop(study, trial):
    global should_stop
    if should_stop:
        print("\n🛑 Optimisation arrêtée par l'utilisateur.")
        # Envoyer un signal pour arrêter l'optimisation
        raise optuna.exceptions.TrialPruned("Arrêté par l'utilisateur")


# Callback pour suivre le meilleur essai
def print_best_trial_callback(study, trial):
    # N'afficher que périodiquement
    if trial.number % 100 == 0:
        try:
            print(f"\n----⚠️ trial: {trial.number}----")
            best_trial = study.best_trial
            print(f"\n📊 Meilleur Trial jusqu'à présent pour {indicator_type}:")
            print(f"  Trial ID: {best_trial.number}")
            print(f"  Score: {best_trial.value:.4f}")

            # Récupération des métriques utilisateur
            bin_0_wr = best_trial.user_attrs.get('bin_0_win_rate', None)
            bin_1_wr = best_trial.user_attrs.get('bin_1_win_rate', None)
            bin_0_pct = best_trial.user_attrs.get('bin_0_pct', None)
            bin_1_pct = best_trial.user_attrs.get('bin_1_pct', None)
            bin_spread = best_trial.user_attrs.get('bin_spread', None)

            # Vérification des paramètres selon l'indicateur
            if 'mfi_period' in best_trial.params:
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
                slope = best_trial.params.get('slope', 'N/A')
                r2_low_threshold = best_trial.params.get('r2_low_threshold', 'N/A')
                r2_high_threshold = best_trial.params.get('r2_high_threshold', 'N/A')

                # Construction conditionnelle des paramètres de régression
                params_str = f"  📌 Paramètres Régression: période de pente={slope}"
                if OPTIMIZE_OVERSOLD:
                    params_str += f", r2_high_threshold={r2_high_threshold}"
                if OPTIMIZE_OVERBOUGHT:
                    params_str += f", r2_low_threshold={r2_low_threshold}"
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
                params_str = f"  📌 Paramètres Régression Écart-type: période={period_var_std}"
                if OPTIMIZE_OVERSOLD:
                    params_str += f", std_low_threshold={std_low_threshold} et std_high_threshold={std_high_threshold}"
                if OPTIMIZE_OVERBOUGHT:
                    params_str += f", std_low_threshold={std_low_threshold} et std_high_threshold={std_high_threshold}"
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
                                   #f" atr_high_threshold={atr_high_threshold:.4f}"
                                   )
                if OPTIMIZE_OVERBOUGHT:
                    params_str += f", atr_low_threshold={atr_low_threshold:.4f}, atr_high_threshold={atr_high_threshold:.4f}"
                print(params_str)

            else:
                print(f"  📌 Paramètres optimisés: {best_trial.params}")

            # Affichage conditionnel des résultats des bins
            if OPTIMIZE_OVERSOLD and bin_0_wr is not None and bin_0_pct is not None:
                # Adapte le nom du bin selon l'indicateur
                bin0_name = "Survente"
                if 'slope' in best_trial.params:
                    bin0_name = "Volatilité Basse"
                elif 'slope_range_threshold' in best_trial.params:
                    bin0_name = "Pente Faible"
                elif 'period_var_atr' in best_trial.params:
                    bin0_name = "ATR Extrême"
                elif 'mfi_period' in best_trial.params:
                    bin0_name = "Divergence Haussière"

                # Affichage du bin 0 avec le nombre de trades réussis
                oversold_success_count = best_trial.user_attrs.get('oversold_success_count', 0)
                if oversold_success_count > 0:
                    print(
                        f"  🔻 Bin 0 ({bin0_name}) : Win Rate={bin_0_wr:.4f}, Couverture={bin_0_pct:.2%}, Trades réussis={oversold_success_count}")
                else:
                    print(f"  🔻 Bin 0 ({bin0_name}) : Win Rate={bin_0_wr:.4f}, Couverture={bin_0_pct:.2%}")

            if OPTIMIZE_OVERBOUGHT and bin_1_wr is not None and bin_1_pct is not None:
                # Adapte le nom du bin selon l'indicateur
                bin1_name = "Surachat"
                if 'slope' in best_trial.params:
                    bin1_name = "Volatilité Haute"
                elif 'slope_extrem_threshold' in best_trial.params:
                    bin1_name = "Pente Forte"
                elif 'period_var_atr' in best_trial.params:
                    bin1_name = "ATR Modéré"
                elif 'mfi_period' in best_trial.params:
                    bin1_name = "Divergence Baissière"

                # Affichage du bin 1 avec les informations pertinentes
                overbought_success_count = best_trial.user_attrs.get('overbought_success_count', 0)

                # Affichage standard pour la plupart des indicateurs
                bin1_info = f"  🔺 Bin 1 ({bin1_name}) : Win Rate={bin_1_wr:.4f}, Couverture={bin_1_pct:.2%}"

                # Ajout du nombre de trades réussis
                if overbought_success_count > 0:
                    bin1_info += f", Trades réussis={overbought_success_count}"

                # Ajout des paramètres spécifiques pour MFI divergence si applicable
                if 'mfi_period' in best_trial.params:
                    min_mfi_decrease = best_trial.params.get('min_mfi_decrease', 'N/A')
                    min_price_increase = best_trial.params.get('min_price_increase', 'N/A')
                    bin1_info += f", min_mfi_decrease={min_mfi_decrease}, min_price_increase={min_price_increase}"

                print(bin1_info)

            # Affichage du spread uniquement si les deux zones sont actives
            if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT and bin_spread is not None:
                print(f"  ⚖️ Écart (Bin Spread) : {bin_spread:.4f}")

        except ValueError:
            print(f" ❌ Pas encore de meilleur essai trouvé... pour {indicator_type}")

        print("\n")


def objective_regressionATR_modified(trial, df):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres de l'indicateur
    ATR (Average True Range) selon le mode d'optimisation (survente, surachat ou les deux).
    """
    # 1. Paramètres à optimiser
    period_var = trial.suggest_int('period_var_atr', PERIOD_VAR_L, PERIOD_VAR_H)

    # D'abord, suggérer le seuil bas dans sa plage complète
    atr_low_threshold = trial.suggest_float('atr_low_threshold', ATR_LOW_THRESHOLD_L, ATR_LOW_THRESHOLD_H)

    # Déterminer la valeur minimale pour atr_high_threshold en fonction du mode d'optimisation
    # if OPTIMIZE_OVERBOUGHT:
    #     low = atr_low_threshold
    # else:
    #     low = ATR_HIGH_THRESHOLD_L
    low = atr_low_threshold

    # Ensuite, suggérer le seuil haut en commençant à partir du seuil bas
    # Cela garantit que high_threshold > low_threshold
    if OPTIMIZE_OVERBOUGHT:
        atr_high_threshold = trial.suggest_float('atr_high_threshold',
                                                 low,  # Commence à la valeur du seuil bas
                                                 ATR_HIGH_THRESHOLD_H)  # Jusqu'à la limite supérieure

    # 2. Calcul de l'ATR
    close = pd.to_numeric(df['close'], errors='coerce').values
    session_starts = (df['SessionStartEnd'] == 10).values
    atr = calculate_atr(df, period_var)

    # 3. Créer les indicateurs avec une logique cohérente
    if OPTIMIZE_OVERBOUGHT:
        # Pour maximiser le win rate, on cherche les régions où l'ATR est dans une plage modérée
        df['atr_range'] = np.where((atr > atr_low_threshold) & (atr < atr_high_threshold), 1, 0)
    else:
        # Pour minimiser le win rate, on cherche les régions où l'ATR est soit très bas soit très élevé
        #df['atr_extrem'] = np.where((atr < atr_low_threshold) | (atr > atr_high_threshold), 1, 0)
        df['atr_extrem'] = np.where((atr < atr_low_threshold)
                                    #| (atr > atr_high_threshold)
                                    , 1, 0)

    # 4. Filtrer df pour ne garder que les entrées avec trade (0 ou 1)
    df_filtered = df[df['class_binaire'].isin([0, 1])].copy()
    target_y = df_filtered['class_binaire']

    # 5. Calculer les métriques pour chaque signal
    bin_0_win_rate = 0.5  # Valeur par défaut
    bin_1_win_rate = 0.5  # Valeur par défaut
    bin_0_pct = 0
    bin_1_pct = 0
    oversold_success_count = 0  # Initialiser à zéro
    overbought_success_count = 0  # Initialiser à zéro

    try:
        # Calcul pour le signal de survente (oversold) - ATR extrême
        if OPTIMIZE_OVERSOLD:
            oversold_df = df_filtered[df_filtered['atr_extrem'] == 1]
            if len(oversold_df) == 0:
                return -np.inf
            bin_0_win_rate = oversold_df['class_binaire'].mean()
            bin_0_pct = len(oversold_df) / len(df_filtered)
            oversold_success_count = oversold_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul pour le signal de surachat (overbought) - ATR modéré
        if OPTIMIZE_OVERBOUGHT:
            overbought_df = df_filtered[df_filtered['atr_range'] == 1]
            if len(overbought_df) == 0:
                return -np.inf
            bin_1_win_rate = overbought_df['class_binaire'].mean()
            bin_1_pct = len(overbought_df) / len(df_filtered)
            overbought_success_count = overbought_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul de l'écart (spread) si les deux modes sont actifs
        bin_spread = bin_1_win_rate - bin_0_win_rate if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes selon le mode d'optimisation
        if not check_bin_constraints(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread):
            return -np.inf

        # Calculer le score final
        combined_score = calculate_optimization_score(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate,
                                                      bin_spread)

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
        'period_var': period_var,
        'atr_low_threshold': atr_low_threshold,
        'atr_high_threshold': atr_high_threshold if OPTIMIZE_OVERBOUGHT else "Non utilisé",
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
            print(f"  Win rates: Bin0(ATR extrême)={bin_0_win_rate:.4f}, Bin1(ATR modéré)={bin_1_win_rate:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(f"  Bin0={bin_0_pct:.2%}, Win rate: {bin_0_win_rate:.4f}, Trades réussis: {oversold_success_count}")
        elif OPTIMIZE_OVERBOUGHT:
            print(f"  Bin1={bin_1_pct:.2%}, Win rate: {bin_1_win_rate:.4f}, Trades réussis: {overbought_success_count}")

        params_str = f"  Paramètres: période ATR={period_var}"
        if OPTIMIZE_OVERSOLD:
            params_str += f", atr_low_threshold={atr_low_threshold:.4f}"
        if OPTIMIZE_OVERBOUGHT:
            params_str += f", atr_high_threshold={atr_high_threshold:.4f}"
        print(params_str)
        print(f"  Score: {combined_score:.2f}")

    return combined_score

    # 7. Stocker les métriques
    metrics = {
        'bin_0_win_rate': float(bin_0_win_rate),
        'bin_1_win_rate': float(bin_1_win_rate),
        'bin_0_pct': float(bin_0_pct),
        'bin_1_pct': float(bin_1_pct),
        'bin_spread': float(bin_spread),
        'combined_score': float(combined_score),
        'period_var': period_var,
        # 'atrs': atrs,
        'atr_low_threshold': atr_low_threshold,
        'atr_high_threshold': atr_high_threshold,
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
            print(f"  Win rates: Bin0(Pente faible)={bin_0_win_rate:.4f}, Bin1(Pente élevée)={bin_1_win_rate:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(
                f"  Bin0={bin_0_pct:.2%}, Win rate: {bin_0_win_rate:.4f}, Trades réussis: {oversold_success_count}")
        elif OPTIMIZE_OVERBOUGHT:
            print(
                f"  Bin1={bin_1_pct:.2%}, Win rate: {bin_1_win_rate:.4f}, Trades réussis: {overbought_success_count}")

        params_str = f"  Paramètres: period={period_var}"
        if OPTIMIZE_OVERSOLD:
            params_str += f", atr_low_threshold={atr_low_threshold:.4f}"
        if OPTIMIZE_OVERBOUGHT:
            params_str += f", atr_high_threshold={atr_high_threshold:.4f}"
        print(params_str)
        print(f"  Score: {combined_score:.2f}")

    return combined_score
def objective_regressionSlope_modified(trial, df):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres de l'indicateur
    de régression basé sur la pente selon le mode d'optimisation (survente, surachat ou les deux).
    """
    # 1. Paramètres à optimiser
    period_var = trial.suggest_int('period_var_slope', PERIOD_VAR_L, PERIOD_VAR_H)

    # D'abord, suggérer le seuil bas dans sa plage complète
    slope_range_threshold = trial.suggest_float('slope_range_threshold', SLOPE_RANGE_THRESHOLD_L, SLOPE_RANGE_THRESHOLD_H)
    if OPTIMIZE_OVERBOUGHT:
        low=slope_range_threshold
    else:
        low = SLOPE_EXTREM_THRESHOLD_L
    # Ensuite, suggérer le seuil haut en commençant à partir du seuil bas
    # Cela garantit que high_threshold > low_threshold
    slope_extrem_threshold = trial.suggest_float('slope_extrem_threshold',
                                               low,  # Commence à la valeur du seuil bas
                                               SLOPE_EXTREM_THRESHOLD_H)  # Jusqu'à la limite supérieure

    # 2. Calcul des pentes, R² et écarts-types
    close = pd.to_numeric(df['close'], errors='coerce').values
    session_starts = (df['SessionStartEnd'] == 10).values
    slopes, _, _ = calculate_slopes_and_r2_numba(close, session_starts, period_var)

    # Filtrer les valeurs NaN pour les pentes
    valid_slopes = slopes[~np.isnan(slopes)]

    # if len(valid_slopes) > 0:
    #     print(f"Slope max (excluding NaN): {valid_slopes.max()}")
    #     print(f"Slope min (excluding NaN): {valid_slopes.min()}")
    #     print(f"Number of valid Slope values: {len(valid_slopes)} out of {len(slopes)} ({len(valid_slopes) / len(slopes) * 100:.2f}%)")
    # else:
    #     print("No valid Slope values found (all are NaN)")

    # 3. Créer les indicateurs avec une logique cohérente
    if OPTIMIZE_OVERBOUGHT:
        df['low_slope'] = np.where((slopes > slope_range_threshold) & (slopes < slope_extrem_threshold), 1,0)  # Pente modéré = faible tendance maximise le Winrate
    else:
        df['high_slope'] = np.where((slopes < slope_range_threshold) | (slopes > slope_extrem_threshold), 1, 0)  # Pente élevée = forte tendance minimise le Winrate

    # 4. Filtrer df pour ne garder que les entrées avec trade (0 ou 1)
    df_filtered = df[df['class_binaire'].isin([0, 1])].copy()
    target_y = df_filtered['class_binaire']

    # 5. Calculer les métriques pour chaque signal
    bin_0_win_rate = 0.5  # Valeur par défaut
    bin_1_win_rate = 0.5  # Valeur par défaut
    bin_0_pct = 0
    bin_1_pct = 0
    oversold_success_count = 0  # Initialiser à zéro
    overbought_success_count = 0  # Initialiser à zéro

    try:
        # Calcul pour le signal de survente (oversold) - pente faible
        if OPTIMIZE_OVERSOLD:
            oversold_df = df_filtered[df_filtered['high_slope'] == 1]
            if len(oversold_df) == 0:
                return -np.inf
            bin_0_win_rate = oversold_df['class_binaire'].mean()
            bin_0_pct = len(oversold_df) / len(df_filtered)
            oversold_success_count = oversold_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul pour le signal de surachat (overbought) - pente élevée
        if OPTIMIZE_OVERBOUGHT:
            overbought_df = df_filtered[df_filtered['low_slope'] == 1]
            if len(overbought_df) == 0:
                return -np.inf
            bin_1_win_rate = overbought_df['class_binaire'].mean()
            bin_1_pct = len(overbought_df) / len(df_filtered)
            overbought_success_count = overbought_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul de l'écart (spread) si les deux modes sont actifs
        bin_spread = bin_1_win_rate - bin_0_win_rate if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes selon le mode d'optimisation
        if not check_bin_constraints(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread):
            return -np.inf

        # Calculer le score final
        combined_score = calculate_optimization_score(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread)

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
        'period_var': period_var,
        'slopes': slopes,
        'slope_range_threshold': slope_range_threshold,
        'slope_extrem_threshold': slope_extrem_threshold,
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
            print(f"  Win rates: Bin0(Pente faible)={bin_0_win_rate:.4f}, Bin1(Pente élevée)={bin_1_win_rate:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(f"  Bin0={bin_0_pct:.2%}, Win rate: {bin_0_win_rate:.4f}, Trades réussis: {oversold_success_count}")
        elif OPTIMIZE_OVERBOUGHT:
            print(f"  Bin1={bin_1_pct:.2%}, Win rate: {bin_1_win_rate:.4f}, Trades réussis: {overbought_success_count}")

        params_str = f"  Paramètres: period={period_var}"
        if OPTIMIZE_OVERSOLD:
            params_str += f", slope_range_threshold={slope_range_threshold:.4f} et slope_extrem_threshold={slope_extrem_threshold}"
        if OPTIMIZE_OVERBOUGHT:
            params_str += f", slope_range_threshold={slope_range_threshold:.4f} et slope_extrem_threshold={slope_extrem_threshold}"
        print(params_str)
        print(f"  Score: {combined_score:.2f}")

    return combined_score

def objective_regressionStd_modified(trial, df):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres de l'indicateur
    de régression basé sur l'écart-type selon le mode d'optimisation (survente, surachat ou les deux).
    """
    # 1. Paramètres à optimiser
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
    # 2. Calcul des pentes, R² et écarts-types
    close = pd.to_numeric(df['close'], errors='coerce').values
    session_starts = (df['SessionStartEnd'] == 10).values
    _, _, stds = calculate_slopes_and_r2_numba(close, session_starts, period_var)

    # Filtrer les valeurs NaN pour les pentes
    valid_std = stds[~np.isnan(stds)]

    if len(valid_std) > 0:
        print(f"Slope max (excluding NaN): {valid_std.max()}")
        print(f"Slope min (excluding NaN): {valid_std.min()}")
        print(f"Number of valid Slope values: {len(valid_std)} out of {len(stds)} ({len(valid_std) / len(stds) * 100:.2f}%)")
    else:
        print("No valid Slope values found (all are NaN)")

    # 3. Créer les indicateurs avec une logique cohérente
    if OPTIMIZE_OVERBOUGHT:
        df['range_volatility'] = np.where((stds > std_low_threshold) & (stds < std_high_threshold), 1, 0)  # Écart-type modéré
    else:
        df['extrem_volatility'] = np.where((stds < std_low_threshold) | (stds > std_high_threshold), 1, 0)  # Écart-type extrême

    # 4. Filtrer df pour ne garder que les entrées avec trade (0 ou 1)
    df_filtered = df[df['class_binaire'].isin([0, 1])].copy()
    target_y = df_filtered['class_binaire']

    # 5. Calculer les métriques pour chaque signal
    bin_0_win_rate = 0.5  # Valeur par défaut
    bin_1_win_rate = 0.5  # Valeur par défaut
    bin_0_pct = 0
    bin_1_pct = 0
    oversold_success_count = 0  # Initialiser à zéro
    overbought_success_count = 0  # Initialiser à zéro

    try:
        # Calcul pour le signal de survente (oversold) - volatilité basse
        if OPTIMIZE_OVERSOLD:
            oversold_df = df_filtered[df_filtered['extrem_volatility'] == 1]
            if len(oversold_df) == 0:
                return -np.inf
            bin_0_win_rate = oversold_df['class_binaire'].mean()
            bin_0_pct = len(oversold_df) / len(df_filtered)
            oversold_success_count = oversold_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul pour le signal de surachat (overbought) - volatilité haute
        if OPTIMIZE_OVERBOUGHT:
            overbought_df = df_filtered[df_filtered['range_volatility'] == 1]
            if len(overbought_df) == 0:
                return -np.inf
            bin_1_win_rate = overbought_df['class_binaire'].mean()
            bin_1_pct = len(overbought_df) / len(df_filtered)
            overbought_success_count = overbought_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul de l'écart (spread) si les deux modes sont actifs
        bin_spread = bin_1_win_rate - bin_0_win_rate if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes selon le mode d'optimisation
        if not check_bin_constraints(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread):
            return -np.inf

        # Calculer le score final
        combined_score = calculate_optimization_score(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread)

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
        'period_var': period_var,
        'slopes_std': stds,
        'std_low_threshold': std_low_threshold,
        'std_high_threshold': std_high_threshold,
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
            print(f"  Win rates: Bin0(Volatilité basse)={bin_0_win_rate:.4f}, Bin1(Volatilité haute)={bin_1_win_rate:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(f"  Bin0={bin_0_pct:.2%}, Win rate: {bin_0_win_rate:.4f}, Trades réussis: {oversold_success_count}")
        elif OPTIMIZE_OVERBOUGHT:
            print(f"  Bin1={bin_1_pct:.2%}, Win rate: {bin_1_win_rate:.4f}, Trades réussis: {overbought_success_count}")

        params_str = f"  Paramètres: period={period_var}"
        if OPTIMIZE_OVERSOLD:
            params_str += f", std_low_threshold={std_low_threshold:.4f} et std_high_threshold={std_high_threshold:.4f}"
        if OPTIMIZE_OVERBOUGHT:
            params_str += f", std_low_threshold={std_low_threshold:.4f} et std_high_threshold={std_high_threshold:.4f}"
        print(params_str)
        print(f"  Score: {combined_score:.2f}")

    return combined_score
def objective_regressionR2_modified(trial, df):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres de l'indicateur
    de régression selon le mode d'optimisation (survente, surachat ou les deux).
    """
    # 1. Paramètres à optimiser
    period_var = trial.suggest_int('period_var_std', PERIOD_VAR_L, PERIOD_VAR_H)
    r2_low_threshold = trial.suggest_float('r2_low_threshold', R2_LOW_THRESHOLD_L, R2_LOW_THRESHOLD_H)  # Seuil pour R² faible
    r2_high_threshold = trial.suggest_float('r2_high_threshold', R2_HIGH_THRESHOLD_L, R2_HIGH_THRESHOLD_H)  # Seuil pour R² élevé

    # 2. Calcul des pentes et R²
    close = pd.to_numeric(df['close'], errors='coerce').values
    session_starts = (df['SessionStartEnd'] == 10).values
    slopes_r2, r2s,stds = calculate_slopes_and_r2_numba(close, session_starts, period_var)



    # Filtrer les valeurs NaN
    valid_r2s = r2s[~np.isnan(r2s)]

    if len(valid_r2s) > 0:
        print(f"R² max (excluding NaN): {valid_r2s.max()}")
        print(f"R² min (excluding NaN): {valid_r2s.min()}")
        print(f"Number of valid R² values: {len(valid_r2s)} out of {len(r2s)} ({len(valid_r2s) / len(r2s) * 100:.2f}%)")
    else:
        print("No valid R² values found (all are NaN)")

    # 3. Créer les indicateurs avec une logique cohérente
    df['extrem_volatility'] = np.where(r2s < r2_low_threshold, 1, 0)  # R² faible = haute volatilité
    df['range_volatility'] = np.where(r2s > r2_high_threshold, 1, 0)  # R² élevé = basse volatilité
    # print(
    #     f"Sum of range_volatility: {df['range_volatility'].sum()} out of {len(df)} ({df['range_volatility'].sum() / len(df) * 100:.2f}%)")
    # print(
    #     f"Sum of extrem_volatility: {df['extrem_volatility'].sum()} out of {len(df)} ({df['extrem_volatility'].sum() / len(df) * 100:.2f}%)")
    # 4. Filtrer df pour ne garder que les entrées avec trade (0 ou 1)
    df_filtered = df[df['class_binaire'].isin([0, 1])].copy()
    target_y = df_filtered['class_binaire']

    # 5. Calculer les métriques pour chaque signal
    bin_0_win_rate = 0.5  # Valeur par défaut
    bin_1_win_rate = 0.5  # Valeur par défaut
    bin_0_pct = 0
    bin_1_pct = 0
    oversold_success_count = 0  # Initialiser à zéro
    overbought_success_count = 0  # Initialiser à zéro

    try:
        # Calcul pour le signal de survente (oversold)
        if OPTIMIZE_OVERSOLD:
            oversold_df = df_filtered[df_filtered['range_volatility'] == 1]
            if len(oversold_df) == 0:
                #print("len(oversold_df)")
                return -np.inf
            bin_0_win_rate = oversold_df['class_binaire'].mean()
            bin_0_pct = len(oversold_df) / len(df_filtered)
            oversold_success_count = oversold_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul pour le signal de surachat (overbought)
        if OPTIMIZE_OVERBOUGHT:
            overbought_df = df_filtered[df_filtered['extrem_volatility'] == 1]
            if len(overbought_df) == 0:
                #print("len(overbought_df)")

                return -np.inf
            bin_1_win_rate = overbought_df['class_binaire'].mean()
            bin_1_pct = len(overbought_df) / len(df_filtered)
            overbought_success_count = overbought_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul de l'écart (spread) si les deux modes sont actifs
        bin_spread = bin_1_win_rate - bin_0_win_rate if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes selon le mode d'optimisation
        if not check_bin_constraints(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread):
            return -np.inf

        # Calculer le score final
        combined_score = calculate_optimization_score(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread)

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
        'period_var': period_var,
        'slopes_r2': slopes_r2,
        'r2_low_threshold': r2_low_threshold,
        'r2_high_threshold': r2_high_threshold,
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
            print(f"  Win rates: Bin0(Volatilité basse)={bin_0_win_rate:.4f}, Bin1(Volatilité haute)={bin_1_win_rate:.4f}")
        elif OPTIMIZE_OVERSOLD:
            print(f"  Bin0={bin_0_pct:.2%}, Win rate: {bin_0_win_rate:.4f}, Trades réussis: {oversold_success_count}")
        elif OPTIMIZE_OVERBOUGHT:
            print(f"  Bin1={bin_1_pct:.2%}, Win rate: {bin_1_win_rate:.4f}, Trades réussis: {overbought_success_count}")

        params_str = f"  Paramètres: period de la pente={period_var}"
        if OPTIMIZE_OVERSOLD:
            params_str += f", r2_high_threshold={r2_high_threshold:.4f}"
        if OPTIMIZE_OVERBOUGHT:
            params_str += f", r2_low_threshold={r2_low_threshold:.4f}"
        print(params_str)
        print(f"  Score: {combined_score:.2f}")

    return combined_score
def objective_stochastic_modified(trial, df):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres du Stochastique
    selon le mode d'optimisation (survente, surachat ou les deux).
    """
    # 1. Paramètres à optimiser
    d_period = trial.suggest_int('d_period', D_PERIOD_L, D_PERIOD_H)
    k_period = trial.suggest_int('k_period', d_period, K_PERIOD_H)

    OS_limit = trial.suggest_int('OS_limit', OS_LIMIT_L, OS_LIMIT_H) if OPTIMIZE_OVERSOLD else 20  # Seuil de survente
    OB_limit = trial.suggest_int('OB_limit', OB_LIMIT_L, OB_LIMIT_H) if OPTIMIZE_OVERBOUGHT else 80  # Seuil de surachat

    # 2. Calculer le Stochastique sur le dataset complet
    high = pd.to_numeric(df['high'], errors='coerce')
    low = pd.to_numeric(df['low'], errors='coerce')
    close = pd.to_numeric(df['close'], errors='coerce')
    session_starts = (df['SessionStartEnd'] == 10).values

    k_values, d_values = compute_stoch(high, low, close,session_starts, k_period=k_period, d_period=d_period)

    # 3. Créer les indicateurs binaires directement dans df
    df['stoch_overbought'] = np.where(k_values > OB_limit, 1, 0)
    df['stoch_oversold'] = np.where(k_values < OS_limit, 1, 0)

    # 4. Filtrer df pour ne garder que les entrées avec trade (0 ou 1)
    df_filtered = df[df['class_binaire'].isin([0, 1])].copy()
    target_y = df_filtered['class_binaire']

    # 5. Calculer les métriques pour chaque signal
    bin_0_win_rate = 0.5  # Valeur par défaut
    bin_1_win_rate = 0.5  # Valeur par défaut
    bin_0_pct = 0
    bin_1_pct = 0
    oversold_success_count = 0  # Initialiser à zéro
    overbought_success_count = 0  # Initialiser à zéro

    try:
        # Calcul pour le signal de survente (oversold)
        if OPTIMIZE_OVERSOLD:
            oversold_df = df_filtered[df_filtered['stoch_oversold'] == 1]
            if len(oversold_df) == 0:
                return -np.inf
            bin_0_win_rate = oversold_df['class_binaire'].mean()
            bin_0_pct = len(oversold_df) / len(df_filtered)
            oversold_success_count = oversold_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul pour le signal de surachat (overbought)
        if OPTIMIZE_OVERBOUGHT:
            overbought_df = df_filtered[df_filtered['stoch_overbought'] == 1]
            if len(overbought_df) == 0:
                return -np.inf
            bin_1_win_rate = overbought_df['class_binaire'].mean()
            bin_1_pct = len(overbought_df) / len(df_filtered)
            overbought_success_count = overbought_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul de l'écart (spread) si les deux modes sont actifs
        bin_spread = bin_1_win_rate - bin_0_win_rate if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes selon le mode d'optimisation
        if not check_bin_constraints(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread):
            return -np.inf

        # Calculer le score final
        combined_score = calculate_optimization_score(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread)

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
        'k_period': k_period,
        'd_period': d_period,
        'OS_limit': OS_limit,
        'OB_limit': OB_limit,
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

        params_str = f"  Paramètres: k_period={k_period}, d_period={d_period}"
        if OPTIMIZE_OVERSOLD:
            params_str += f", OS_limit={OS_limit}"
        if OPTIMIZE_OVERBOUGHT:
            params_str += f", OB_limit={OB_limit}"
        print(params_str)
        print(f"  Score: {combined_score:.2f}")

    return combined_score

import numpy as np
import pandas as pd



# Define your objective function for Williams %R
def objective_williams_r_modified(trial, df):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres de Williams %R
    selon le mode d'optimisation (survente, surachat ou les deux).
    """
    # 1. Paramètres à optimiser
    period = trial.suggest_int('period', PERIOD_L, PERIOD_H)
    OS_limit = trial.suggest_int('OS_limit', OS_LIMIT_L, OS_LIMIT_H) if OPTIMIZE_OVERSOLD else -70  # Seuil de survente
    OB_limit = trial.suggest_int('OB_limit', OB_LIMIT_L, OB_LIMIT_H) if OPTIMIZE_OVERBOUGHT else -30  # Seuil de surachat

    # 2. Calculer le Williams %R sur le dataset complet
    high = pd.to_numeric(df['high'], errors='coerce')
    low = pd.to_numeric(df['low'], errors='coerce')
    close = pd.to_numeric(df['close'], errors='coerce')

    session_starts = (df['SessionStartEnd'] == 10).values

    will_r_values = compute_wr(high, low, close, session_starts,period=period)

    # 3. Créer les indicateurs binaires directement dans df
    df['wr_overbought'] = np.where(will_r_values > OB_limit, 1, 0)
    df['wr_oversold'] = np.where(will_r_values < OS_limit, 1, 0)

    # 4. Filtrer df pour ne garder que les entrées avec trade (0 ou 1)
    df_filtered = df[df['class_binaire'].isin([0, 1])].copy()
    target_y = df_filtered['class_binaire']

    # 5. Calculer les métriques pour chaque signal
    bin_0_win_rate = 0.5  # Valeur par défaut
    bin_1_win_rate = 0.5  # Valeur par défaut
    bin_0_pct = 0
    bin_1_pct = 0
    oversold_success_count = 0  # Initialiser à zéro
    overbought_success_count = 0  # Initialiser à zéro

    try:
        # Calcul pour le signal de survente (oversold)
        if OPTIMIZE_OVERSOLD:
            oversold_df = df_filtered[df_filtered['wr_oversold'] == 1]
            if len(oversold_df) == 0:
                return -np.inf
            bin_0_win_rate = oversold_df['class_binaire'].mean()
            bin_0_pct = len(oversold_df) / len(df_filtered)
            oversold_success_count = oversold_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul pour le signal de surachat (overbought)
        if OPTIMIZE_OVERBOUGHT:
            overbought_df = df_filtered[df_filtered['wr_overbought'] == 1]
            if len(overbought_df) == 0:
                return -np.inf
            bin_1_win_rate = overbought_df['class_binaire'].mean()
            bin_1_pct = len(overbought_df) / len(df_filtered)
            overbought_success_count = overbought_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul de l'écart (spread) si les deux modes sont actifs
        bin_spread = bin_1_win_rate - bin_0_win_rate if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes selon le mode d'optimisation
        if not check_bin_constraints(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread):
            return -np.inf

        # Calculer le score final
        combined_score = calculate_optimization_score(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread)

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
        'period': period,
        'OS_limit': OS_limit,
        'OB_limit': OB_limit,
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

        params_str = f"  Paramètres: period={period}"
        if OPTIMIZE_OVERSOLD:
            params_str += f", OS_limit={OS_limit}"
        if OPTIMIZE_OVERBOUGHT:
            params_str += f", OB_limit={OB_limit}"
        print(params_str)
        print(f"  Score: {combined_score:.2f}")

    return combined_score

def objective_mfi_modified(trial, df):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres du Money Flow Index (MFI)
    selon le mode d'optimisation (survente, surachat ou les deux).
    """
    # 1. Paramètres à optimiser
    period = trial.suggest_int('period', PERIOD_L , PERIOD_H )
    OS_limit = trial.suggest_int('OS_limit', OS_LIMIT_L , OS_LIMIT_H ) if OPTIMIZE_OVERSOLD else 20
    OB_limit = trial.suggest_int('OB_limit', OB_LIMIT_L , OB_LIMIT_H ) if OPTIMIZE_OVERBOUGHT else 80

    # 2. Calculer le MFI sur le dataset complet
    high = pd.to_numeric(df['high'], errors='coerce')
    low = pd.to_numeric(df['low'], errors='coerce')
    close = pd.to_numeric(df['close'], errors='coerce')
    volume = pd.to_numeric(df['volume'], errors='coerce')

    session_starts = (df['SessionStartEnd'] == 10).values
    mfi_values = compute_mfi(high, low, close, volume, session_starts,period=period)

    # 3. Créer les indicateurs binaires directement dans df
    df['mfi_overbought'] = np.where(mfi_values > OB_limit, 1, 0)
    df['mfi_oversold'] = np.where(mfi_values < OS_limit, 1, 0)

    # 4. Filtrer df pour ne garder que les entrées avec trade (0 ou 1)
    df_filtered = df[df['class_binaire'].isin([0, 1])].copy()
    target_y = df_filtered['class_binaire']

    # 5. Calculer les métriques pour chaque signal
    bin_0_win_rate = 0.5  # Valeur par défaut
    bin_1_win_rate = 0.5  # Valeur par défaut
    bin_0_pct = 0
    bin_1_pct = 0
    oversold_success_count = 0  # Initialiser à zéro
    overbought_success_count = 0  # Initialiser à zéro

    try:
        # Calcul pour le signal de survente (oversold)
        if OPTIMIZE_OVERSOLD:
            oversold_df = df_filtered[df_filtered['mfi_oversold'] == 1]
            if len(oversold_df) == 0:
                return -np.inf
            bin_0_win_rate = oversold_df['class_binaire'].mean()
            bin_0_pct = len(oversold_df) / len(df_filtered)
            oversold_success_count = oversold_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul pour le signal de surachat (overbought)
        if OPTIMIZE_OVERBOUGHT:
            overbought_df = df_filtered[df_filtered['mfi_overbought'] == 1]
            if len(overbought_df) == 0:
                return -np.inf
            bin_1_win_rate = overbought_df['class_binaire'].mean()
            bin_1_pct = len(overbought_df) / len(df_filtered)
            overbought_success_count = overbought_df['class_binaire'].sum()  # Nombre de trades réussis

        # Calcul de l'écart (spread) si les deux modes sont actifs
        bin_spread = bin_1_win_rate - bin_0_win_rate if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes selon le mode d'optimisation
        if not check_bin_constraints(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread):
            return -np.inf

        # Calculer le score final
        combined_score = calculate_optimization_score(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread)

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
        'period': period,
        'OS_limit': OS_limit,
        'OB_limit': OB_limit,
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

        params_str = f"  Paramètres: period={period}"
        if OPTIMIZE_OVERSOLD:
            params_str += f", OS_limit={OS_limit}"
        if OPTIMIZE_OVERBOUGHT:
            params_str += f", OB_limit={OB_limit}"
        print(params_str)
        print(f"  Score: {combined_score:.2f}")

    return combined_score

# Fonction principale pour choisir l'indicateur à optimiser
def objective_mfi_divergence_modified(trial, df):
    """
    Fonction objective optimisée pour Optuna qui ajuste les paramètres des divergences du MFI
    pour les entrées en position short.
    """
    # 1. Paramètres à optimiser
    mfi_period = trial.suggest_int('mfi_period', MFI_PERIOD_L, MFI_PERIOD_H)
    div_lookback = trial.suggest_int('div_lookback', DIV_LOOKBACK_L, DIV_LOOKBACK_H)

    # Paramètres pour les deux modes (overbought et oversold)
    min_price_increase = trial.suggest_float('min_price_increase', MIN_PRICE_INCREASE_L, MIN_PRICE_INCREASE_H)
    min_mfi_decrease = trial.suggest_float('min_mfi_decrease', MIN_MFI_DECREASE_L, MIN_MFI_DECREASE_H)

    # Nouveaux paramètres pour la partie oversold (anti-divergence)
    if OPTIMIZE_OVERSOLD:
        min_price_decrease = trial.suggest_float('min_price_decrease', MIN_PRICE_INCREASE_L, MIN_PRICE_INCREASE_H)
        min_mfi_increase = trial.suggest_float('min_mfi_increase', MIN_MFI_DECREASE_L, MIN_MFI_DECREASE_H)

    # 2. Calcul MFI
    high = pd.to_numeric(df['high'], errors='coerce')
    low = pd.to_numeric(df['low'], errors='coerce')
    close = pd.to_numeric(df['close'], errors='coerce')
    volume = pd.to_numeric(df['volume'], errors='coerce')

    session_starts = (df['SessionStartEnd'] == 10).values
    mfi_values = compute_mfi(high, low, close, volume, session_starts,period=mfi_period)
    mfi_series = pd.Series(mfi_values, index=df.index)

    # 3. Initialiser les colonnes de divergence
    df['bearish_divergence'] = 0
    df['anti_divergence'] = 0  # Nouvelle colonne pour les conditions d'oversold

    # Filtrer pour les trades shorts
    df_mode_filtered = df[df['class_binaire'] != 99].copy()
    all_shorts = df_mode_filtered['tradeDir'].eq(-1).all() if not df_mode_filtered.empty else False

    print(f"All shorts mode: {all_shorts}")
    print(f"Nombre de trades short: {df['tradeDir'].eq(-1).sum()} / {len(df)}")

    if all_shorts:
        # Pour la partie overbought (maximiser le win rate)
        if OPTIMIZE_OVERBOUGHT:
            # Détection des divergences baissières améliorée
            price_pct_change = close.pct_change(div_lookback).fillna(0)
            mfi_pct_change = mfi_series.pct_change(div_lookback).fillna(0)

            # Conditions pour une divergence baissière efficace
            price_increase = price_pct_change > min_price_increase
            mfi_decrease = mfi_pct_change < -min_mfi_decrease

            # Prix fait un nouveau haut relatif
            price_rolling_max = close.rolling(window=div_lookback).max().shift(1)
            price_new_high = (close > price_rolling_max).fillna(False)

            # Définir la divergence baissière avec nos critères
            df.loc[df_mode_filtered.index, 'bearish_divergence'] = (
                    (price_new_high | price_increase) &  # Prix fait un nouveau haut ou augmente significativement
                    (mfi_decrease)  # MFI diminue
            ).astype(int)

        # Pour la partie oversold (minimiser le win rate)
        if OPTIMIZE_OVERSOLD:
            # Détection des conditions opposées (anti-divergences)
            price_pct_change = close.pct_change(div_lookback).fillna(0)
            mfi_pct_change = mfi_series.pct_change(div_lookback).fillna(0)

            # Conditions pour une anti-divergence (mauvais win rate)
            price_decrease = price_pct_change < -min_price_decrease  # Prix diminue
            mfi_increase = mfi_pct_change > min_mfi_increase  # MFI augmente

            # Prix fait un nouveau bas relatif
            price_rolling_min = close.rolling(window=div_lookback).min().shift(1)
            price_new_low = (close < price_rolling_min).fillna(False)

            # Définir l'anti-divergence avec nos critères
            df.loc[df_mode_filtered.index, 'anti_divergence'] = (
                    (price_new_low | price_decrease) &  # Prix fait un nouveau bas ou diminue significativement
                    (mfi_increase)  # MFI augmente
            ).astype(int)
    else:
        exit(98)  # Si ce n'est pas full short, exit

    # 4. Filtrer pour class_binaire in [0,1]
    df_filtered = df[df['class_binaire'].isin([0, 1])].copy()

    # 5. Calcul des métriques pour chaque type de signal
    try:
        # Valeurs par défaut
        bin_0_win_rate = 0.5
        bin_1_win_rate = 0.5
        bin_0_pct = 0
        bin_1_pct = 0
        bearish_div_count = 0
        bearish_success_count = 0
        anti_div_count = 0
        anti_div_success_count = 0

        # Calcul pour oversold (anti-divergences)
        if OPTIMIZE_OVERSOLD:
            oversold_df = df_filtered[df_filtered['anti_divergence'] == 1]
            if len(oversold_df) < 10:  # Minimum de 10 échantillons
                return -np.inf

            bin_0_win_rate = oversold_df['class_binaire'].mean()
            bin_0_pct = len(oversold_df) / len(df_filtered)
            anti_div_count = len(oversold_df)
            anti_div_success_count = oversold_df['class_binaire'].sum()

            # Rejeter si le win rate est trop élevé (pour oversold on veut un win rate bas)
            if bin_0_win_rate > MAX_bin_0_win_rate:
                return -np.inf

            # Rejeter si la couverture est trop faible
            if bin_0_pct < MIN_BIN_SIZE_0:
                return -np.inf

        # Calcul pour overbought (divergences baissières)
        if OPTIMIZE_OVERBOUGHT:
            bearish_df = df_filtered[df_filtered['bearish_divergence'] == 1]
            if len(bearish_df) < 10:  # Minimum de 10 échantillons
                return -np.inf

            bin_1_win_rate = bearish_df['class_binaire'].mean()
            bin_1_pct = len(bearish_df) / len(df_filtered)
            bearish_div_count = len(bearish_df)
            bearish_success_count = bearish_df['class_binaire'].sum()

            # Rejeter si le win rate est trop bas
            if bin_1_win_rate < MIN_bin_1_win_rate:
                return -np.inf

            # Rejeter si la couverture est trop faible
            if bin_1_pct < MIN_BIN_SIZE_1:
                return -np.inf

        # Calculer l'écart si les deux modes sont actifs
        bin_spread = bin_1_win_rate - bin_0_win_rate if (OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT) else 0

        # Vérifier les contraintes
        if not check_bin_constraints(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread):
            return -np.inf

        # Calculer le score final
        combined_score = calculate_optimization_score(bin_0_pct, bin_1_pct, bin_0_win_rate, bin_1_win_rate, bin_spread)

    except Exception as e:
        print(f"Erreur lors du calcul: {e}")
        return -np.inf

    # 6. Stocker les métriques
    # Pour oversold
    if OPTIMIZE_OVERSOLD:
        trial.set_user_attr('bin_0_win_rate', bin_0_win_rate)
        trial.set_user_attr('bin_0_pct', bin_0_pct)
        trial.set_user_attr('anti_div_count', anti_div_count)
        trial.set_user_attr('anti_div_success_count', anti_div_success_count)
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            trial.set_user_attr('bin_spread', bin_spread)

    # Pour overbought
    if OPTIMIZE_OVERBOUGHT:
        trial.set_user_attr('bin_1_win_rate', bin_1_win_rate)
        trial.set_user_attr('bin_1_pct', bin_1_pct)
        trial.set_user_attr('bearish_div_count', bearish_div_count)
        trial.set_user_attr('bearish_success_count', bearish_success_count)

    trial.set_user_attr('combined_score', combined_score)
    trial.set_user_attr('mfi_period', mfi_period)
    trial.set_user_attr('div_lookback', div_lookback)
    trial.set_user_attr('min_price_increase', min_price_increase)
    trial.set_user_attr('min_mfi_decrease', min_mfi_decrease)

    if OPTIMIZE_OVERSOLD:
        trial.set_user_attr('min_price_decrease', min_price_decrease)
        trial.set_user_attr('min_mfi_increase', min_mfi_increase)

    # 7. Logs périodiques plus détaillés
    if trial.number % 20 == 0:
        print(f"\nTrial {trial.number}")
        print(f"Paramètres: mfi_period={mfi_period}, div_lookback={div_lookback}")

        if OPTIMIZE_OVERSOLD:
            print(f"Anti-divergences: {anti_div_count} signaux, win rate: {bin_0_win_rate:.4f}")
            print(f"Trades gagnants anti-div: {anti_div_success_count}/{anti_div_count}")
            print(f"Couverture anti-div: {bin_0_pct:.2%}")

        if OPTIMIZE_OVERBOUGHT:
            print(f"Divergences baissières: {bearish_div_count} signaux, win rate: {bin_1_win_rate:.4f}")
            print(f"Trades gagnants div: {bearish_success_count}/{bearish_div_count}")
            print(f"Couverture div: {bin_1_pct:.2%}")

        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"Spread (différence de win rate): {bin_spread:.4f}")

        print(f"Score combiné: {combined_score:.4f}\n")

    return combined_score


def run_indicator_optimization(df, df_filtered, target_y, indicator_type="stochastic", n_trials=20000):
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

    if indicator_type in ["stochastic", "regression_slope","regression_std","atr"]:
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
                                 "regression_r2", "regression_std", "regression_slope", "atr"]

        if indicator_type.lower() not in valid_indicator_types:
            print(f"\n⚠️ AVERTISSEMENT: Type d'indicateur '{indicator_type}' non reconnu.")
            print(f"Types d'indicateurs valides: {', '.join(valid_indicator_types)}")
            print(f"L'indicateur 'stochastic' sera utilisé par défaut.\n")
            # On ne lève pas d'exception, on laisse le code utiliser stochastic par défaut

        # Sélectionner la fonction objective selon le type d'indicateur et le mode d'optimisation
        if indicator_type.lower() == "williams_r":
            objective_func = lambda trial: objective_williams_r_modified(trial, df)
        elif indicator_type.lower() == "mfi":
            objective_func = lambda trial: objective_mfi_modified(trial, df)
        elif indicator_type.lower() == "mfi_divergence":
            objective_func = lambda trial: objective_mfi_divergence_modified(trial, df)
        elif indicator_type.lower() == "regression_r2":
            objective_func = lambda trial: objective_regressionR2_modified(trial, df)
        elif indicator_type.lower() == "regression_std":
            objective_func = lambda trial: objective_regressionStd_modified(trial, df)
        elif indicator_type.lower() == "regression_slope":
            objective_func = lambda trial: objective_regressionSlope_modified(trial, df)
        elif indicator_type.lower() == "atr":
            objective_func = lambda trial: objective_regressionATR_modified(trial, df)
        else:  # Default to stochastic
            objective_func = lambda trial: objective_stochastic_modified(trial, df)
            # Si l'indicateur n'était pas reconnu, on force la valeur à "stochastic" pour la cohérence
            if indicator_type.lower() not in valid_indicator_types:
                indicator_type = "stochastic"

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
    else:  # Default to stochastic (cohérent avec la section d'attribution de fonction objective)
        print(f"RÉSULTATS FINAUX DE L'OPTIMISATION STOCHASTIQUE - MODE {mode_text}")

    # Afficher les meilleurs résultats avec statistiques détaillées
    if len(study.trials) > 0:
        best_trial = study.best_trial
        bin_0_pct = best_trial.user_attrs.get('bin_0_pct', 0)
        bin_1_pct = best_trial.user_attrs.get('bin_1_pct', 0)
        bin_0_wr = best_trial.user_attrs.get('bin_0_win_rate', 0)
        bin_1_wr = best_trial.user_attrs.get('bin_1_win_rate', 0)
        bin_1_win_count = best_trial.user_attrs.get('bin_1_win_count', 0)

        # Calculer le nombre approximatif d'échantillons dans chaque bin
        total_samples = len(df_filtered)
        valid_samples = total_samples * (1 - df_filtered.isna().mean().mean())  # Estimation des échantillons valides
        bin_0_samples = int(valid_samples * bin_0_pct)
        bin_1_samples = int(valid_samples * bin_1_pct)

        # Rappel des paramètres d'optimisation
        print("\n🔧 PARAMÈTRES D'OPTIMISATION:")
        print(f"  Mode d'optimisation: {mode_text}")

        # Afficher uniquement les paramètres pertinents selon le mode
        if OPTIMIZE_OVERSOLD and OPTIMIZE_OVERBOUGHT:
            print(f"  MIN_BIN_SPREAD: {MIN_BIN_SPREAD}")  # Écart minimum entre bins
            print(f"  MAX_bin_0_win_rate: {MAX_bin_0_win_rate}")  # Maximum pour bin0
            print(f"  MIN_bin_1_win_rate: {MIN_bin_1_win_rate}")  # Minimum pour bin1
            print(f"  MIN_BIN_SIZE_0: {MIN_BIN_SIZE_0}")  # Taille minimale pour un bin individuel
            print(f"  MIN_BIN_SIZE_1: {MIN_BIN_SIZE_1}")  # Taille minimale pour un bin individuel
        elif OPTIMIZE_OVERSOLD:
            print(f"  MAX_bin_0_win_rate: {MAX_bin_0_win_rate}")  # Maximum pour bin0
            print(f"  MIN_BIN_SIZE_0: {MIN_BIN_SIZE_0}")  # Taille minimale pour un bin individuel
        elif OPTIMIZE_OVERBOUGHT:
            print(f"  MIN_bin_1_win_rate: {MIN_bin_1_win_rate}")  # Minimum pour bin1
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
            print(f"  Période de la pente: {best_trial.params.get('period_var_r2', 'N/A')}")
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
        else:  # Stochastique
            print(f"  Période K: {best_trial.params.get('k_period', 'N/A')}")
            print(f"  Période D: {best_trial.params.get('d_period', 'N/A')}")
            if OPTIMIZE_OVERSOLD:
                print(f"  Seuil de survente (OS): {best_trial.params.get('OS_limit', 'N/A')}")
            if OPTIMIZE_OVERBOUGHT:
                print(f"  Seuil de surachat (OB): {best_trial.params.get('OB_limit', 'N/A')}")

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
            bin0_name = "Pente Faible"
            bin1_name = "Pente Forte"
        elif indicator_type.lower() == "atr":
            bin0_name = "ATR Faible"
            bin1_name = "ATR Modéré"

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
        else:
            indicator_name = "Stochastique"

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


# Exécuter l'optimisation pour l'indicateur sélectionné
# Exécuter l'optimisation pour l'indicateur sélectionné
if indicator_type == "stochastic":
    study_stoch = run_indicator_optimization(df, df_filtered, target_y, indicator_type="stochastic", n_trials=20000)
elif indicator_type == "williams_r":
    study_williams = run_indicator_optimization(df, df_filtered, target_y, indicator_type="williams_r", n_trials=20000)
elif indicator_type == "mfi":
    study_mfi = run_indicator_optimization(df, df_filtered, target_y, indicator_type="mfi", n_trials=20000)
elif indicator_type == "mfi_divergence":
    study_mfi_div = run_indicator_optimization(df, df_filtered, target_y, indicator_type="mfi_divergence",
                                               n_trials=20000)
elif indicator_type == "regression_r2":
    study_regression = run_indicator_optimization(df, df_filtered, target_y, indicator_type="regression_r2",
                                                  n_trials=20000)
elif indicator_type == "regression_std":
    study_regression_std = run_indicator_optimization(df, df_filtered, target_y, indicator_type="regression_std",
                                                      n_trials=200000)
elif indicator_type == "regression_slope":
    study_regression_slope = run_indicator_optimization(df, df_filtered, target_y, indicator_type="regression_slope",
                                                        n_trials=20000)
elif indicator_type == "atr":
    study_atr = run_indicator_optimization(df, df_filtered, target_y, indicator_type="atr", n_trials=20000)
else:
    print(f"Type d'indicateur non reconnu: {indicator_type}")
