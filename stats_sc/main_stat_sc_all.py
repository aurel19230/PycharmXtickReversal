from standard_stat_sc import *
from definition import *
from func_standard import *

import pandas as pd, numpy as np, os, sys, platform, io
from pathlib import Path
from contextlib import redirect_stdout
from collections import Counter
import matplotlib.pyplot as plt
MIN_COMMON_TRADES=15
# ────────────────────────────────────────────────────────────────────────────────
# Paramètres
# ────────────────────────────────────────────────────────────────────────────────
FILE_NAME_TRAIN = "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split3_01102024_28022025.csv"
FILE_NAME_TEST = "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split4_02032025_14052025.csv"

# FILE_NAME_TRAIN = "Step5__010124_120525_bugFixTradeResult1_extractOnlyFullSession_OnlyShort_feat__split2_02092024_07032025.csv"
# FILE_NAME_TEST = "Step5__010124_120525_bugFixTradeResult1_extractOnlyFullSession_OnlyShort_feat__split3_10032025_09052025.csv"

ENV = detect_environment()

if platform.system() != "Darwin":
    DIRECTORY_PATH = Path(
         r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_6SL\merge")
        #r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\trainTest")

else:
    DIRECTORY_PATH = "/Users/aurelienlachaud/Documents/trading_local/5_0_5TP_1SL_1/merge"

FILE_PATH_TRAIN = os.path.join(DIRECTORY_PATH, FILE_NAME_TRAIN)
FILE_PATH_TEST = os.path.join(DIRECTORY_PATH, FILE_NAME_TEST)


# ────────────────────────────────────────────────────────────────────────────────
# Chargement et pré‑traitement
# ────────────────────────────────────────────────────────────────────────────────
def load_and_process_data(file_path):
    """Charger et prétraiter les données d'un fichier CSV."""
    df_init_features, CUSTOM_SESSIONS = load_features_and_sections(file_path)

    cats = [
        "Trades échoués short", "Trades échoués long",
        "Trades réussis short", "Trades réussis long"
    ]
    df_analysis = df_init_features[df_init_features["trade_category"].isin(cats)].copy()
    df_analysis["class"] = np.where(df_analysis["trade_category"].str.contains("échoués"), 0, 1)
    df_analysis["pos_type"] = np.where(df_analysis["trade_category"].str.contains("short"), "short", "long")

    return df_init_features, df_analysis, CUSTOM_SESSIONS


# Chargement des données d'entraînement et de test
df_init_features_train, df_analysis_train, CUSTOM_SESSIONS_TRAIN = load_and_process_data(FILE_PATH_TRAIN)
df_init_features_test, df_analysis_test, CUSTOM_SESSIONS_TEST = load_and_process_data(FILE_PATH_TEST)
# ─────────────────────────────────────────────────────────────
# SUPPRESSION DES POSITIONS LONGUES (on conserve uniquement les shorts)
# ─────────────────────────────────────────────────────────────
df_analysis_train = df_analysis_train[df_analysis_train["pos_type"] == "short"].copy()
df_analysis_test  = df_analysis_test [df_analysis_test ["pos_type"] == "short"].copy()
print(f"Données d'entraînement: {len(df_analysis_train)} trades")
print(f"Données de test: {len(df_analysis_test)} trades")

# ────────────────────────────────────────────────────────────────────────────────
# Filtres
# ────────────────────────────────────────────────────────────────────────────────
min_candleduration = 1
max_candleduration = 180

# [Garder toutes les définitions des features_algo ici, identiques à votre code original]
features_algo1 = {

    'finished_auction_low': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}  # Correction de la plage
    ],

    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0.2, 'max': 0.8, 'active': True}],

    'diffVolDelta_2_2Ratio': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': True},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': -0.20, 'max': 0.30, 'active': False}],

    'close_sma_zscore_6': [
        {'type': 'greater_than_or_equal', 'threshold': 0.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'not_between', 'min': 0.4, 'max': 2.2, 'active': True}],

    'diffHighPrice_0_1': [
        {'type': 'greater_than_or_equal', 'threshold': 0.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1.25, 'max': 100, 'active': True}],

    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1.5, 'max': 50, 'active': True}],
}

features_algo3 = {
    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': -0.45, 'max': 0.7, 'active': True}],
    'sc_reg_std_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.6, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1.1, 'max': 100, 'active': True}],

    'ratio_delta_vol_VA11P': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0.275, 'max': 200, 'active': True}],
    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],
}

features_algo4 = {
    'finished_auction_low': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}  # Correction de la plage
    ],

    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': -0.45, 'max': 0.65, 'active': True}],

    'is_williams_r_overbought': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],

    'VolPocVolCandleRatio': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0, 'max': 0.2, 'active': True}],

    'is_rangeSlope': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': min_candleduration, 'max': max_candleduration, 'active': True}],
}

features_algo5 = {
    'imbType_contZone': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': False}  # Correction de la plage
    ],

    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'not_between', 'min': -0.5, 'max': 0.225, 'active': True}],

    'close_sma_zscore_6': [
        {'type': 'greater_than_or_equal', 'threshold': 0.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'not_between', 'min': -150, 'max': 2.2, 'active': True}],

    'sc_reg_std_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.6, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0.4, 'max': 2.2, 'active': True}],

    'sc_reg_slope_10P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.6, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': -0.5, 'max': 0.7, 'active': True}],

    'VolPocVolCandleRatio': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0, 'max': 0.2, 'active': True}],

    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 2.5, 'max': 20, 'active': True}],
}

features_algo6 = {
    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0.32, 'max': 0.8, 'active': True}],

    'ratio_volRevMove_volImpulsMove': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0.6, 'max': 0.9, 'active': True}],

    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': min_candleduration, 'max': 50, 'active': True}],
}
features_algo7 = {

 'imbType_contZone': [
         {'type': 'greater_than_or_equal', 'threshold': -5, 'active': False},
         {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
         {'type': 'between', 'min': 0, 'max': 0, 'active': False}],

 'ratio_volRevMoveZone1_volRevMoveExtrem_XRevZone': [
         {'type': 'greater_than_or_equal', 'threshold': 2.5, 'active': False},
         {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
         {'type': 'between', 'min': 2.5, 'max': 100, 'active': True}],

'ratio_deltaRevMove_volRevMove': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': -0.1, 'max': 0.8, 'active': True}],

'bear_imbalance_high_1': [
        {'type': 'greater_than_or_equal', 'threshold': 3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 1, 'active': False},
        {'type': 'between', 'min': 1.2, 'max': 100, 'active': True}],
'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': -0.5, 'max': 0.9, 'active': True}],


'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],
}
features_algo8 = {
    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': -0.1, 'max': 1, 'active': True}],
    'ratio_volRevMove_volImpulsMove': [
        {'type': 'greater_than_or_equal', 'threshold': 4, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1.1, 'max': 30, 'active': True}],
    'ratio_deltaRevMove_volRevMove': [
        {'type': 'greater_than_or_equal', 'threshold': 4, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0.1, 'max': 1, 'active': True}],
    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],

}

features_algo9 = {
    'is_vwap_reversal_pro_short': [
        {'type': 'greater_than_or_equal', 'threshold': 4, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],

    'bear_imbalance_high_1': [
        {'type': 'greater_than_or_equal', 'threshold': 3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 1, 'active': False},
        {'type': 'between', 'min': 1.5, 'max': 1000, 'active': False}],
    'finished_auction_high': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0, 'max': 0, 'active': True}],  # Correction de la plage
    'finished_auction_low': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],  # Correction de la plage
    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 3, 'max': 180, 'active': True}],
}
features_algo10 = {
'delta_revMove_XRevZone_bigStand_extrem': [
        {'type': 'greater_than_or_equal', 'threshold': 50, 'active': True},
        {'type': 'less_than_or_equal', 'threshold': -60, 'active': False},
        {'type': 'between', 'min': 2, 'max':  600, 'active': False}],

'delta_impulsMove_XRevZone_bigStand_extrem': [
        {'type': 'greater_than_or_equal', 'threshold': -40, 'active': True},
        {'type': 'less_than_or_equal', 'threshold': -25, 'active': False},
        {'type': 'between', 'min': 2, 'max':  10, 'active': False}],
    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 4, 'max': 180, 'active': True}],
}
features_algo11 = {
'ratio_volZone1_volExtrem': [
        {'type': 'greater_than_or_equal', 'threshold': 4, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min':0.01, 'max': 0.7, 'active': True}],
'is_rs_range': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 2, 'max': 50, 'active': True}],
}

features_algo12 = {
    # 'ratio_volRevMoveZone1_volRevMoveExtrem_XRevZone': [
    #     {'type': 'greater_than_or_equal', 'threshold': 2.5, 'active': False},
    #     {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
    #     {'type': 'between', 'min': 0.05, 'max': 0.8, 'active': True}],
    'is_mfi_overbought': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
    # 'finished_auction_low': [
    #     {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
    #     {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
    #     {'type': 'between', 'min': 1, 'max': 1, 'active': True}],  # Correction de la plage
    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 2, 'max': 60, 'active': True}],
    # 'ratio_delta_vol_VA11P': [
    #     {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
    #     {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
    #     {'type': 'between', 'min': 0.275, 'max': 200, 'active': True}],
    'is_rs_range': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],

}
features_algo13 = {
    'pocDeltaPocVolRatio': [
        {'type': 'greater_than_or_equal', 'threshold': -0.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0.25, 'max': 1, 'active': True}],
'diffPriceClosePoc_0_0': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': -0.5, 'max': -0, 'active': True}],
'sc_reg_std_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'not_between', 'min': 0, 'max': 2.1, 'active': True}],
'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': -0, 'max': 0.55, 'active': True}],
'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 2, 'max': 50, 'active': True}],

}
features_algo14 = {
    # 'vix_slope_12_up_15': [
    #     {'type': 'greater_than_or_equal', 'threshold': 4, 'active': False},
    #     {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
    #     {'type': 'between', 'min': 1, 'max': 1, 'active': True}],

    'sc_reg_std_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'not_between', 'min': 0, 'max': 2.1, 'active': False}],
    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0.2, 'max': 1, 'active': True}],
'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 2, 'max': 180, 'active': True}],

}
features_algo15 = {
'sc_reg_std_5P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 1, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1.2, 'max': 6.5, 'active': True}], #bien
    'sc_reg_slope_10P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': -0.65, 'max': 0.97, 'active': True}],  # bien

    'sc_reg_slope_15P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.15, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': -0.9, 'max': 0.9, 'active': True}],
    'diffPriceClose_VA6PPoc': [
        {'type': 'greater_than_or_equal', 'threshold': 4.5, 'active': True},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0.01, 'max': 0.7, 'active': False}],
'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1.5, 'max': 180, 'active': True}],
}
features_algo16 = {
'ratio_deltaImpulsMove_volImpulsMove': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0.15, 'max': 1, 'active': True}],

'ratio_deltaRevMove_volRevMove': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0.15, 'max': 0.55, 'active': True}],
'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1.5, 'max': 180, 'active': True}],

}
features_algo17 = {
 'cumDiffVolDeltaRatio': [
         {'type': 'greater_than_or_equal', 'threshold': -5, 'active': False},
         {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
         {'type': 'between', 'min': -0.65, 'max': 0, 'active': True}],

'finished_auction_low': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],

'is_vwap_reversal_pro_short': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],



'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1.5, 'max': 180, 'active': True}],

}

features_algo18 = {
'is_imBullWithPoc_light': [
        {'type': 'greater_than_or_equal', 'threshold': 3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 1, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
'is_vwap_reversal_pro_short': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],

}
features_algo19 = {
'is_imBullWithPoc_agressive': [
        {'type': 'greater_than_or_equal', 'threshold': 3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 1, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],

'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],

}
features_algo20 = {
'is_imbBullLightPoc_Low00': [
        {'type': 'greater_than_or_equal', 'threshold': 3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 1, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],

'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],

}
algorithms = {
     # "features_algo1": features_algo1,
     # "features_algo3": features_algo3,
     # "features_algo4": features_algo4,
     "features_algo5": features_algo5,
#      "features_algo6": features_algo6,
#      "features_algo7": features_algo7,
#
#      "features_algo8": features_algo8,
#      "features_algo9": features_algo9,
#      "features_algo10": features_algo10,
#      "features_algo11": features_algo11,
#      "features_algo12": features_algo12,
#      "features_algo13": features_algo13,
# "features_algo14": features_algo14,
#  "features_algo15": features_algo15,
#  "features_algo16": features_algo16,
#  "features_algo17": features_algo17,
# "features_algo18": features_algo18,
# "features_algo19": features_algo19,
# "features_algo20": features_algo20


}


# ────────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ────────────────────────────────────────────────────────────────────────────────
def save_csv(df: pd.DataFrame, path, sep=";") -> Path:
    path = Path(path) if not isinstance(path, Path) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=sep, index=False)
    print(f"✓ Fichier enregistré: {path}")
    return path


import re
def header_print(before: dict, after: dict, name: str):
    """Affiche le rapport comparatif sans aucune partie liée aux LONGS."""
    with io.StringIO() as buf, redirect_stdout(buf):
        print_comparative_performance(before, after)
        out = buf.getvalue()

    # 1) retirer totalement tout bloc '… LONGS …'
    #    – qu’il s’agisse d’analyse ou de séquences consécutives –
    long_block_pattern = (
        r"(📊\s*(ANALYSE DES TRADES LONGS|SÉQUENCES CONSÉCUTIVES LONGS)"
        r".*?)"                                     # début du bloc
        r"(?:\n📊|\Z)"                              # jusqu’au prochain titre ou fin
    )
    out = re.sub(long_block_pattern, "", out, flags=re.S)

    # 2) ajouter le suffixe « - <algo / dataset> » aux titres conservés
    for h in ("STATISTIQUES GLOBALES",
              "PERFORMANCE GLOBALE",
              "ANALYSE DES TRADES SHORTS",
              "SÉQUENCES CONSÉCUTIVES SHORTS"):
        out = out.replace(h, f"{h} - {name}")

    print(out)



# ────────────────────────────────────────────────────────────────────────────────
# Fonction pour évaluer les algorithmes sur un jeu de données
# ────────────────────────────────────────────────────────────────────────────────
def evaluate_algorithms(df_analysis, df_init_features, algorithms, dataset_name="Train"):
    """Évalue les algorithmes sur un jeu de données et renvoie les résultats."""
    print(f"\n{'=' * 80}\nÉVALUATION SUR DATASET {dataset_name}\n{'=' * 80}")

    results = {}
    to_save = []
    metrics_before = calculate_performance_metrics(df_analysis)

    for algo_name, cond in algorithms.items():
        print(f"\n{'=' * 80}\nÉVALUATION DE {algo_name} - {dataset_name}\n{'=' * 80}")
        df_filt = apply_feature_conditions(df_analysis, cond)

        # CORRECTION: Vérifier que df_filt contient bien des données
        if len(df_filt) == 0:
            print(f"ATTENTION: Aucun trade ne satisfait les conditions pour {algo_name}")
            continue

        # Ajouter un affichage détaillé du PnL pour debug
        pnl_sum = df_filt["trade_pnl"].sum()
        print(f"Debug - Somme directe des trade_pnl: {pnl_sum}")

        # Compter les trades par type (long/short)
        trade_types = df_filt["pos_type"].value_counts()
        print(f"Debug - Répartition des types de trades: {trade_types}")

        # Calculer le PnL par type de position
        pnl_by_type = df_filt.groupby("pos_type")["trade_pnl"].sum()
        print(f"Debug - PnL par type de position: {pnl_by_type}")

        # Créer le dataframe avec PnL filtré
        df_full = preprocess_sessions_with_date(
            create_full_dataframe_with_filtered_pnl(df_init_features, df_filt)
        )

        # CORRECTION: Vérifier et afficher la somme des PnL après filtrage
        pnl_after_filtering = df_full["PnlAfterFiltering"].sum()
        print(f"Debug - Somme des PnlAfterFiltering: {pnl_after_filtering}")

        # Comparaison pour détecter les incohérences
        if abs(pnl_sum - pnl_after_filtering) > 1.0:  # Tolérance pour erreurs d'arrondi
            print(
                f"ATTENTION: Incohérence détectée entre la somme des trade_pnl ({pnl_sum}) et PnlAfterFiltering ({pnl_after_filtering})")
            # CORRECTION: Utiliser le PnL original plutôt que celui après filtrage
            use_original_pnl = True
        else:
            use_original_pnl = False

        metrics_after = calculate_performance_metrics(df_filt)
        header_print(metrics_before, metrics_after, f"{algo_name} - {dataset_name}")

        wins_a = (df_filt["trade_pnl"] > 0).sum()
        fails_a = (df_filt["trade_pnl"] <= 0).sum()
        win_rate_a = wins_a / (wins_a + fails_a) * 100 if (wins_a + fails_a) > 0 else 0

        # CORRECTION: Utiliser la somme directe des trade_pnl si une incohérence a été détectée
        if use_original_pnl:
            pnl_a = pnl_sum
            print(f"CORRECTION: Utilisation de la somme directe des trade_pnl comme Net PnL")
        else:
            pnl_a = pnl_after_filtering

        profits_a = df_filt.loc[df_filt["trade_pnl"] > 0, "trade_pnl"].sum()
        losses_a = abs(df_filt.loc[df_filt["trade_pnl"] < 0, "trade_pnl"].sum())
        pf_a = profits_a / losses_a if losses_a else 0

        print(f"Win Rate après : {win_rate_a:.2f}%  |  Net PnL : {pnl_a:.2f}  |  Profit Factor : {pf_a:.2f}")

        results[algo_name] = {
            "Nombre de trades": len(df_filt),
            "Net PnL": pnl_a,
            "Win Rate (%)": win_rate_a,
            "Profit Factor": pf_a,
        }

        # on conserve UNIQUEMENT les trades sélectionnés par l'algo
        df_selected = df_full[df_full["PnlAfterFiltering"] != 0].copy()

        # CORRECTION: Si nécessaire, recalculer la colonne PnlAfterFiltering
        if use_original_pnl and 'trade_pnl' in df_selected.columns:
            df_selected["PnlAfterFiltering"] = df_selected["trade_pnl"]
            print(f"CORRECTION: La colonne PnlAfterFiltering a été remplacée par trade_pnl")

        to_save.append((algo_name, df_selected))

    return results, to_save


# ────────────────────────────────────────────────────────────────────────────────
# Évaluation sur les jeux de données d'entraînement et de test
# ────────────────────────────────────────────────────────────────────────────────
results_train, to_save_train = evaluate_algorithms(df_analysis_train, df_init_features_train, algorithms, "Train")
results_test, to_save_test = evaluate_algorithms(df_analysis_test, df_init_features_test, algorithms, "Test")

# ────────────────────────────────────────────────────────────────────────────────
# Création du tableau comparatif avec les deux jeux de données
# ────────────────────────────────────────────────────────────────────────────────
# Créer les DataFrames individuels pour l'entraînement et le test
comparison_train = pd.DataFrame(results_train).T.reset_index().rename(columns={"index": "Algorithme"})
comparison_test = pd.DataFrame(results_test).T.reset_index().rename(columns={"index": "Algorithme"})

# Ajouter les colonnes pour Exp PnL
comparison_train["Exp PnL"] = (comparison_train["Net PnL"] / comparison_train["Nombre de trades"]).round(2)
comparison_test["Exp PnL"] = (comparison_test["Net PnL"] / comparison_test["Nombre de trades"]).round(2)

# Renommer les colonnes pour le jeu de test (pour les distinguer)
comparison_test_renamed = comparison_test.rename(columns={
    "Nombre de trades": "Nombre de trades (Test)",
    "Net PnL": "Net PnL (Test)",
    "Exp PnL": "Exp PnL (Test)",
    "Win Rate (%)": "Win Rate (%) (Test)",
    "Profit Factor": "Profit Factor (Test)",
})

# Fusionner les DataFrames d'entraînement et de test
comparison_merged = pd.merge(comparison_train, comparison_test_renamed[
    ["Algorithme", "Nombre de trades (Test)", "Net PnL (Test)",
     "Exp PnL (Test)", "Win Rate (%) (Test)", "Profit Factor (Test)"]
], on="Algorithme")

# Créer la liste des features utilisées (comme dans le code original)
all_features = sorted({feat for d in algorithms.values() for feat in d})

# Pour chaque feature, inscrire "x" pour les algos qui l'emploient
for feat in all_features:
    comparison_merged[feat] = np.where(
        comparison_merged["Algorithme"].map(lambda a: feat in algorithms[a]), "x", ""
    )

# Ordonner les colonnes
cols_metrics = [
    "Algorithme",
    "Nombre de trades", "Nombre de trades (Test)",
    "Net PnL", "Net PnL (Test)",
    "Exp PnL", "Exp PnL (Test)",
    "Win Rate (%)", "Win Rate (%) (Test)",
    "Profit Factor", "Profit Factor (Test)",
]
comparison_merged = comparison_merged[cols_metrics + all_features]

# Afficher le tableau comparatif
print("\nTABLEAU COMPARATIF DES ALGORITHMES (Entraînement vs Test)\n",
      comparison_merged.to_string(index=False))

# Sauvegarder le tableau comparatif
save_csv(comparison_merged, DIRECTORY_PATH / "algo_comparison_train_test.csv")


# ────────────────────────────────────────────────────────────────────────────────
# Création d'une visualisation de comparaison Train vs Test
# ────────────────────────────────────────────────────────────────────────────────
def create_comparison_chart(comparison_merged, metric_train, metric_test, title):
    """Crée un graphique à barres comparant les métriques entre train et test."""
    algos = comparison_merged["Algorithme"]
    train_values = comparison_merged[metric_train]
    test_values = comparison_merged[metric_test]

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    x = np.arange(len(algos))

    train_bars = ax.bar(x - bar_width / 2, train_values, bar_width, label='Entraînement', color='royalblue')
    test_bars = ax.bar(x + bar_width / 2, test_values, bar_width, label='Test', color='darkorange')

    ax.set_xlabel('Algorithmes')
    ax.set_ylabel(title)
    ax.set_title(f'Comparaison {title} - Entraînement vs Test')
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=45, ha='right')
    ax.legend()

    # Ajouter des valeurs au-dessus des barres
    for i, v in enumerate(train_values):
        ax.text(i - bar_width / 2, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    for i, v in enumerate(test_values):
        ax.text(i + bar_width / 2, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    return fig


# Créer des graphiques pour les principales métriques
win_rate_chart = create_comparison_chart(comparison_merged, "Win Rate (%)", "Win Rate (%) (Test)", "Win Rate (%)")
profit_factor_chart = create_comparison_chart(comparison_merged, "Profit Factor", "Profit Factor (Test)",
                                              "Profit Factor")
exp_pnl_chart = create_comparison_chart(comparison_merged, "Exp PnL", "Exp PnL (Test)", "Espérance par Trade")

# Sauvegarder les graphiques
win_rate_chart.savefig(os.path.join(DIRECTORY_PATH, "win_rate_comparison.png"))
profit_factor_chart.savefig(os.path.join(DIRECTORY_PATH, "profit_factor_comparison.png"))
exp_pnl_chart.savefig(os.path.join(DIRECTORY_PATH, "exp_pnl_comparison.png"))

print("\nGraphiques de comparaison sauvegardés dans le répertoire de sortie.")

# ────────────────────────────────────────────────────────────────────────────────
# Calculs de robustesse (ratio test/train pour les métriques clés)
# ────────────────────────────────────────────────────────────────────────────────
# Ajouter des métriques de robustesse
comparison_merged["Robustesse Win Rate"] = (comparison_merged["Win Rate (%) (Test)"] /
                                            comparison_merged["Win Rate (%)"]).round(2)
comparison_merged["Robustesse Profit Factor"] = (comparison_merged["Profit Factor (Test)"] /
                                                 comparison_merged["Profit Factor"]).round(2)
comparison_merged["Robustesse Exp PnL"] = (comparison_merged["Exp PnL (Test)"] /
                                           comparison_merged["Exp PnL"]).round(2)

# Score de robustesse global (moyenne des 3 métriques)
comparison_merged["Score Robustesse"] = (
        (comparison_merged["Robustesse Win Rate"] +
         comparison_merged["Robustesse Profit Factor"] +
         comparison_merged["Robustesse Exp PnL"]) / 3
).round(2)

# Afficher les métriques de robustesse
print("\nMÉTRIQUES DE ROBUSTESSE DES ALGORITHMES\n")
robustness_cols = ["Algorithme", "Robustesse Win Rate", "Robustesse Profit Factor",
                   "Robustesse Exp PnL", "Score Robustesse"]
print(comparison_merged[robustness_cols].to_string(index=False))

# Sauvegarder le tableau de robustesse
save_csv(comparison_merged[robustness_cols], DIRECTORY_PATH / "algo_robustness.csv")

# ────────────────────────────────────────────────────────────────────────────────
# Sauvegardes différées pour les dataframes
# ────────────────────────────────────────────────────────────────────────────────
# Sauvegarder les DataFrames pour les données d'entraînement
for algo_name, df_full in to_save_train:
    save_csv(df_full, DIRECTORY_PATH / f"df_with_sessions_{algo_name}_train.csv")
    save_csv(df_full[df_full["PnlAfterFiltering"] != 0],
             DIRECTORY_PATH / f"trades_{algo_name}_train.csv")

# Sauvegarder les DataFrames pour les données de test
for algo_name, df_full in to_save_test:
    save_csv(df_full, DIRECTORY_PATH / f"df_with_sessions_{algo_name}_test.csv")
    save_csv(df_full[df_full["PnlAfterFiltering"] != 0],
             DIRECTORY_PATH / f"trades_{algo_name}_test.csv")


# ────────────────────────────────────────────────────────────────────────────────
# Analyse des doublons / Combinaisons entre algos
# ────────────────────────────────────────────────────────────────────────────────
def analyse_doublons_algos(
        algo_dfs: dict[str, pd.DataFrame],
        indicator_columns: list[str] | None = None,
        min_common_trades: int = 20,
        directory_path: str | Path = None  # Ajout du paramètre pour les sauvegardes
) -> None:
    """
    Analyse les doublons entre différents algorithmes de trading.
    Travaille sur un dict {nom_algo: dataframe} au lieu de lire des CSV.

    Args:
        algo_dfs: Dictionnaire {nom_algo: DataFrame}
        indicator_columns: Colonnes à utiliser pour détecter les doublons
        min_common_trades: Nombre min de trades communs pour analyse des paires
        directory_path: Répertoire pour sauvegarder les résultats (optionnel)
    """
    if directory_path is not None:
        directory_path = Path(directory_path) if not isinstance(directory_path, Path) else directory_path
        directory_path.mkdir(parents=True, exist_ok=True)

    if indicator_columns is None:
        indicator_columns = [
            'rsi_', 'macd', 'macd_signal', 'macd_hist',
            'timeElapsed2LastBar', 'timeStampOpening',
            'ratio_deltaRevZone_VolCandle'
        ]

    # 1) Stats par algo + stockage des ensembles uniques
    uniq_sets = {}
    trade_results = {}
    trade_data = {}
    total_rows_before = 0
    file_stats = {}  # Pour statistiques individuelles par algo

    algos = list(algo_dfs.keys())
    for algo, df in algo_dfs.items():
        df = df.copy()
        file_stats[algo] = {'total_rows': len(df)}

        # colonne PnL (on prend la 1ʳᵉ dispo)
        pnl_col = next((c for c in ("PnlAfterFiltering", "trade_pnl") if c in df.columns), None)
        if pnl_col is None:
            print(f"[warn]  Colonne PnL absente pour {algo} – ignoré")
            continue

        total_rows_before += len(df)

        valid_cols = [c for c in indicator_columns if c in df.columns]
        if not valid_cols:
            print(f"Aucune colonne d'indicateur valide trouvée pour {algo}")
            continue

        duplicate_mask = df.duplicated(subset=valid_cols, keep='first')
        dups_int = duplicate_mask.sum()
        file_stats[algo]['duplicates_internal'] = dups_int
        file_stats[algo]['unique_rows'] = len(df) - dups_int

        print(f"Algorithme {algo}: {len(df)} lignes, {dups_int} doublons internes, {len(df) - dups_int} uniques")

        uniq_sets[algo] = set()
        for _, row in df.drop_duplicates(subset=valid_cols).iterrows():
            key = tuple(row[col] for col in valid_cols)
            uniq_sets[algo].add(key)

            # enregistrement global
            if key not in trade_results:
                trade_results[key] = {}
                trade_data[key] = row.to_dict()

            trade_results[key][algo] = row[pnl_col] > 0

    # 2) matrice inter‑algos
    dup_matrix = pd.DataFrame(0, index=algos, columns=algos, dtype=int)

    for i, a1 in enumerate(algos):
        for a2 in algos[i + 1:]:
            common = len(uniq_sets[a1].intersection(uniq_sets[a2]))
            dup_matrix.loc[a1, a2] = dup_matrix.loc[a2, a1] = common

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n=== MATRICE DES DOUBLONS ENTRE ALGOS ===")
    print(dup_matrix)

    # Version simplifiée avec noms courts pour meilleure lisibilité
    short_names = {name: f"algo{i}" for i, name in enumerate(algos)}
    readable_matrix = dup_matrix.copy()
    readable_matrix.index = [short_names[name] for name in readable_matrix.index]
    readable_matrix.columns = [short_names[name] for name in readable_matrix.columns]

    # 3) distribution des occurrences
    occ_counts = Counter(len(v) for v in trade_results.values())
    print("\nDistribution des occurrences (nb d'algos dans lesquels apparaît chaque trade) :")
    for k in sorted(occ_counts):
        print(f"  {k} algo(s) : {occ_counts[k]} trades")

    # 4) consolidation globale dé‑dupliquée
    all_keys = set.union(*uniq_sets.values()) if uniq_sets else set()
    global_rows = []
    for key in all_keys:
        if key in trade_data:
            row = trade_data[key]
            row["is_winning_any"] = any(trade_results[key].values())
            row["is_winning_all"] = all(trade_results[key].get(a, False) for a in algos)
            global_rows.append(row)

    global_df = pd.DataFrame(global_rows) if global_rows else pd.DataFrame()

    if not global_df.empty:
        pnl_col = next((c for c in ("PnlAfterFiltering", "trade_pnl") if c in global_df.columns), None)
        if pnl_col:
            wins = (global_df[pnl_col] > 0).sum()
            total = len(global_df)

            # Calculs des métriques globales
            winrate = wins / total * 100 if total > 0 else 0
            total_pnl = global_df[pnl_col].sum()
            total_gains = global_df.loc[global_df[pnl_col] > 0, pnl_col].sum()
            total_losses = global_df.loc[global_df[pnl_col] <= 0, pnl_col].sum()

            # Moyennes et ratios
            avg_win = global_df.loc[global_df[pnl_col] > 0, pnl_col].mean() if wins > 0 else 0
            avg_loss = global_df.loc[global_df[pnl_col] <= 0, pnl_col].mean() if (total - wins) > 0 else 0
            reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            expectancy = (winrate / 100 * avg_win) + ((100 - winrate) / 100 * avg_loss)

            print(f"\n=== RÉSUMÉ GLOBAL APRÈS DÉDUPLICATION ===")
            print(f"Trades totaux avant déduplication: {total_rows_before}")
            print(f"Trades conservés après déduplication: {total} ({total / total_rows_before * 100:.2f}%)")
            print(f"Trades réussis: {wins}")
            print(f"Trades échoués: {total - wins}")
            print(f"Winrate global: {winrate:.2f}%")
            print(f"PnL total: {total_pnl:.2f}")
            print(f"Gains totaux: {total_gains:.2f}")
            print(f"Pertes totales: {total_losses:.2f}")
            print(f"Gain moyen: {avg_win:.2f}")
            print(f"Perte moyenne: {avg_loss:.2f}")
            print(f"Ratio risque/récompense: {reward_risk_ratio:.2f}")
            print(f"Expectancy par trade: {expectancy:.2f}")

    # 5) Analyse par nombre d'occurrences
    occurrences_stats = {}
    max_occ = max([len(v) for v in trade_results.values()]) if trade_results else 0

    for occ_count in range(2, max_occ + 1):
        trades_with_occ = {k: v for k, v in trade_results.items() if len(v) == occ_count}

        if not trades_with_occ:
            continue

        # Analyse des trades qui apparaissent dans exactement occ_count algos
        winning_trades = []  # Liste des trades gagnants (unanimement)
        losing_trades = []  # Liste des trades non unanimes

        for key, algos_present in trades_with_occ.items():
            # Un trade est considéré comme valide uniquement si TOUS les algos le marquent comme gagnant
            if all(trade_results.get(key, {}).get(algo, False) for algo in algos_present):
                winning_trades.append(key)
            else:
                losing_trades.append(key)

        total_trades = len(winning_trades) + len(losing_trades)
        winrate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        # Calculer le PnL total pour les trades gagnants et perdants
        if pnl_col:
            winning_pnl = sum(trade_data.get(key, {}).get(pnl_col, 0) for key in winning_trades)
            losing_pnl = sum(trade_data.get(key, {}).get(pnl_col, 0) for key in losing_trades)
            total_pnl = winning_pnl + losing_pnl

            occurrences_stats[occ_count] = {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'winrate': winrate,
                'winning_pnl': winning_pnl,
                'losing_pnl': losing_pnl,
                'total_pnl': total_pnl
            }

            print(f"\n=== ANALYSE DES TRADES APPARAISSANT DANS {occ_count} ALGOS ===")
            print(f"Nombre total de trades: {total_trades}")
            print(f"Trades unanimement gagnants: {len(winning_trades)}")
            print(f"Trades non unanimes (au moins un échec): {len(losing_trades)}")
            print(f"Winrate (trades unanimement gagnants): {winrate:.2f}%")
            print(f"PnL total: {total_pnl:.2f}")
            print(f"PnL des trades gagnants: {winning_pnl:.2f}")
            print(f"PnL des trades perdants: {losing_pnl:.2f}")

    # 6) analyse des paires d'algos avec > min_common_trades trades communs
    print(f"\n=== ANALYSE DES PAIRES (> {min_common_trades} trades communs) ===")

    # Éviter les paires redondantes
    analyzed_pairs = set()
    significant_pairs = []

    for i, a1 in enumerate(algos):
        for j, a2 in enumerate(algos[i + 1:], i + 1):
            if dup_matrix.loc[a1, a2] >= min_common_trades:
                significant_pairs.append((a1, a2))
                analyzed_pairs.add((a1, a2))

    pairs_stats = {}

    for a1, a2 in significant_pairs:
        common = uniq_sets[a1].intersection(uniq_sets[a2])
        if len(common) < min_common_trades:
            continue

        # Stats détaillées
        winning_both = 0
        winning_a1_only = 0
        winning_a2_only = 0
        losing_both = 0

        total_pnl = 0
        unanimous_pnl = 0

        for key in common:
            result_a1 = trade_results.get(key, {}).get(a1, False)
            result_a2 = trade_results.get(key, {}).get(a2, False)

            # Compter les cas selon le résultat
            if result_a1 and result_a2:
                winning_both += 1
            elif result_a1 and not result_a2:
                winning_a1_only += 1
            elif not result_a1 and result_a2:
                winning_a2_only += 1
            else:
                losing_both += 1

            # Calculer le PnL si disponible
            if key in trade_data and pnl_col in trade_data[key]:
                pnl = trade_data[key][pnl_col]
                total_pnl += pnl

                # Ajouter au PnL unanime si les deux algos sont d'accord
                if result_a1 == result_a2:
                    unanimous_pnl += pnl

        # Calculer l'accord entre les algos
        agreement_rate = (winning_both + losing_both) / len(common) * 100 if common else 0

        # Stocker les statistiques
        pairs_stats[(a1, a2)] = {
            'common_trades': len(common),
            'winning_both': winning_both,
            'winning_a1_only': winning_a1_only,
            'winning_a2_only': winning_a2_only,
            'losing_both': losing_both,
            'agreement_rate': agreement_rate,
            'total_pnl': total_pnl,
            'unanimous_pnl': unanimous_pnl
        }

        print(f"\n>> Analyse de la paire {a1} / {a2}:")
        print(f"  Trades communs: {len(common)}")
        print(f"  Gagnants pour les deux: {winning_both}")
        print(f"  Gagnants uniquement pour {a1}: {winning_a1_only}")
        print(f"  Gagnants uniquement pour {a2}: {winning_a2_only}")
        print(f"  Perdants pour les deux: {losing_both}")
        print(f"  Taux d'accord: {agreement_rate:.2f}%")
        print(f"  PnL total: {total_pnl:.2f}")
        print(f"  PnL des trades unanimes: {unanimous_pnl:.2f}")

    return pairs_stats, occurrences_stats


# Analyse des doublons pour les données d'entraînement et de test
print("\n\n=== ANALYSE DES DOUBLONS SUR LES DONNÉES D'ENTRAÎNEMENT ===")
algo_dfs_train = {name: df for name, df in to_save_train}
pairs_stats_train, occurrences_stats_train = analyse_doublons_algos(algo_dfs_train, min_common_trades=MIN_COMMON_TRADES)

print("\n\n=== ANALYSE DES DOUBLONS SUR LES DONNÉES DE TEST ===")
algo_dfs_test = {name: df for name, df in to_save_test}
pairs_stats_test, occurrences_stats_test = analyse_doublons_algos(algo_dfs_test, min_common_trades=MIN_COMMON_TRADES)

# ────────────────────────────────────────────────────────────────────────────────
# Comparaison spécifique Train vs Test
# ────────────────────────────────────────────────────────────────────────────────
# Créer des visualisations pour comparer les performances des algorithmes entre Train et Test
print("\n\n=== COMPARAISON SPÉCIFIQUE TRAIN VS TEST ===")

# Identifier les algorithmes les plus robustes (scores de robustesse les plus élevés)
top_robust_algos = comparison_merged.sort_values("Score Robustesse", ascending=False)["Algorithme"].iloc[:3].tolist()
print(f"Les algorithmes les plus robustes sont: {', '.join(top_robust_algos)}")

# Identifier les algorithmes ayant la meilleure performance sur l'échantillon de test
top_test_algos = comparison_merged.sort_values("Net PnL (Test)", ascending=False)["Algorithme"].iloc[:3].tolist()
print(f"Les algorithmes avec le meilleur PnL sur l'échantillon de test sont: {', '.join(top_test_algos)}")

# Créer un graphique de dispersion pour visualiser la relation entre performance train et test
plt.figure(figsize=(10, 8))
plt.scatter(comparison_merged["Net PnL"], comparison_merged["Net PnL (Test)"], alpha=0.7)

# Ajouter des étiquettes pour chaque point
for i, algo in enumerate(comparison_merged["Algorithme"]):
    plt.annotate(algo,
                 (comparison_merged["Net PnL"].iloc[i],
                  comparison_merged["Net PnL (Test)"].iloc[i]),
                 textcoords="offset points",
                 xytext=(0, 10),
                 ha='center')

# Ajouter une ligne diagonale (y=x) qui représenterait une performance identique
max_val = max(comparison_merged["Net PnL"].max(), comparison_merged["Net PnL (Test)"].max())
min_val = min(comparison_merged["Net PnL"].min(), comparison_merged["Net PnL (Test)"].min())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

plt.xlabel('Net PnL (Entraînement)')
plt.ylabel('Net PnL (Test)')
plt.title('Comparaison de la performance entre Entraînement et Test')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(DIRECTORY_PATH, "train_vs_test_scatter.png"))

# Résumé final
print("\n" + "=" * 80 + "\nRÉSUMÉ FINAL DE L'ANALYSE\n" + "=" * 80)
print("1. Analyse effectuée sur deux périodes:")
print(f"   - Entraînement: {FILE_NAME_TRAIN}")
print(f"   - Test: {FILE_NAME_TEST}")
print(f"2. {len(algorithms)} algorithmes évalués sur les deux périodes")
print("3. Métriques clés comparées: Win Rate, Profit Factor, Exp PnL")
print(f"4. Score de robustesse calculé pour identifier les algorithmes les plus stables")
print(f"5. Les algorithmes les plus robustes: {', '.join(top_robust_algos)}")
print(f"6. Les algorithmes avec la meilleure performance sur l'échantillon de test: {', '.join(top_test_algos)}")
print("\nL'analyse complète a été sauvegardée dans le répertoire:")
print(DIRECTORY_PATH)

# Afficher le tableau comparatif
print("\n (RAPPEL) TABLEAU COMPARATIF DES ALGORITHMES (Entraînement vs Test)\n",
      comparison_merged.to_string(index=False))

print("\n" + "=" * 80 + "\nANALYSE TERMINÉE AVEC SUCCÈS\n" + "=" * 80)