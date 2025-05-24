from standard_stat_sc import *
from func_standard import *

import pandas as pd, numpy as np, os, sys, platform, io
from pathlib import Path
from contextlib import redirect_stdout
from collections import Counter
MIN_COMMON_TRADES=10 # nombre de trade en common pour utiliser une paire
JACCARD_THRESHOLD = 0.50  # Seuil de similarité Jaccard

# ────────────────────────────────────────────────────────────────────────────────
# Paramètres
# ────────────────────────────────────────────────────────────────────────────────
FILE_NAME_TRAIN = "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split1_01012024_01052024.csv"
FILE_NAME_TEST = "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split2_01052024_01102024.csv"
FILE_NAME_VAL1 = "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split3_01102024_28022025.csv"
FILE_NAME_VAL = "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split4_02032025_14052025.csv"

ENV = detect_environment()

if platform.system() != "Darwin":
    DIRECTORY_PATH = Path(
         r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_6SL\merge")
else:
    DIRECTORY_PATH = "/Users/aurelienlachaud/Documents/trading_local/5_0_5TP_1SL_1/merge"

FILE_PATH_TRAIN = os.path.join(DIRECTORY_PATH, FILE_NAME_TRAIN)
FILE_PATH_TEST = os.path.join(DIRECTORY_PATH, FILE_NAME_TEST)
FILE_PATH_VAL1 = os.path.join(DIRECTORY_PATH, FILE_NAME_VAL1)
FILE_PATH_VAL = os.path.join(DIRECTORY_PATH, FILE_NAME_VAL)


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
# Chargement des données de test
df_init_features_test, df_analysis_test, CUSTOM_SESSIONS_TEST = load_and_process_data(FILE_PATH_TEST)
# Chargement des données de validation 1
df_init_features_val1, df_analysis_val1, CUSTOM_SESSIONS_VAL1 = load_and_process_data(FILE_PATH_VAL1)
# Chargement des données de validation 2
df_init_features_val, df_analysis_val, CUSTOM_SESSIONS_VAL = load_and_process_data(FILE_PATH_VAL)

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Améliorations suggérées pour le script d'analyse trading

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime





import numpy as np
import pandas as pd

PERDIOD_ATR_SESSION_ANALYSE = 15

# Variables globales pour les groupes de sessions
GROUPE_SESSION_1 = [0, 1]
GROUPE_SESSION_2 = [2, 3, 4, 5, 6]




# ✅ CORRECT (nouvelle version avec sessions intraday)
results = run_enhanced_trading_analysis_with_sessions(
    df_init_features_train=df_init_features_train,
    df_init_features_test=df_init_features_test,
    df_init_features_val1=df_init_features_val1,
    df_init_features_val=df_init_features_val,
    groupe1=GROUPE_SESSION_1,
    groupe2=GROUPE_SESSION_2,
xtickReversalTickPrice=XTICKREVERAL_TICKPRICE,period_atr_stat_session=PERDIOD_ATR_SESSION_ANALYSE
)
# Export optionnel vers Excel (maintenant avec ATR et contrats extrêmes)
export_results_to_excel(results, "analyse_trading_complete.xlsx", directory_path=DIRECTORY_PATH)


# ─────────────────────────────────────────────────────────────
# SUPPRESSION DES POSITIONS LONGUES (on conserve uniquement les shorts)
# ─────────────────────────────────────────────────────────────
df_analysis_train = df_analysis_train[df_analysis_train["pos_type"] == "short"].copy()
df_analysis_test = df_analysis_test[df_analysis_test["pos_type"] == "short"].copy()
df_analysis_val1 = df_analysis_val1[df_analysis_val1["pos_type"] == "short"].copy()
df_analysis_val = df_analysis_val[df_analysis_val["pos_type"] == "short"].copy()

print(f"Données d'entraînement: {len(df_analysis_train)} trades")
print(f"Données de test: {len(df_analysis_test)} trades")
print(f"Données de validation 1: {len(df_analysis_val1)} trades")
print(f"Données de validation 2: {len(df_analysis_val)} trades")

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
features_algo2 = {
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

features_algo3 = {
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
features_algo4 = {
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
features_algo5 = {
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
features_algo6 = {
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
features_algo7 = {
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

features_algo8 = {
    'is_vwap_reversal_pro_short': [
        {'type': 'greater_than_or_equal', 'threshold': 4, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
    'diffVolDelta_2_2Ratio': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': -0.20, 'max': 0.30, 'active': False}],

    'close_sma_zscore_6': [
        {'type': 'greater_than_or_equal', 'threshold': 0.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'not_between', 'min': 100, 'max': 2.5, 'active': True}],
    'diffHighPrice_0_1': [
    ],
    'finished_auction_high': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0, 'max': 0, 'active': False}],  # Correction de la plage
    'finished_auction_low': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': False}],  # Correction de la plage
    'diffPriceClose_VA6PPoc': [
        {'type': 'greater_than_or_equal', 'threshold': 4.5, 'active': True},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0.01, 'max': 0.7, 'active': False}],
    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 3, 'max': 60, 'active': True}],
}
features_algo9 = {
'delta_revMove_XRevZone_bigStand_extrem': [
        {'type': 'greater_than_or_equal', 'threshold': 35, 'active': True},
        {'type': 'less_than_or_equal', 'threshold': -60, 'active': False},
        {'type': 'between', 'min': 2, 'max':  600, 'active': False}],
'delta_impulsMove_XRevZone_bigStand_extrem': [
        {'type': 'greater_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': -25, 'active': False},
        {'type': 'between', 'min': 2, 'max':  10, 'active': False}],
    'finished_auction_high': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0, 'max': 0, 'active': False}],  # Correction de la plage
    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': -0.9, 'max': 0.1, 'active': False}],

    'sc_reg_std_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.6, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0.4, 'max': 2.2, 'active': False}],
    'diffHighPrice_0_1': [
        {'type': 'greater_than_or_equal', 'threshold': 0.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 3, 'max': 100, 'active': True}],
    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 5, 'max': 60, 'active': True}],
}
features_algo10 = {
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

features_algo11 = {
    'is_mfi_overbought': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 2, 'max': 60, 'active': True}],
    'is_rs_range': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
}
features_algo12 = {
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
features_algo13 = {
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
features_algo14 = {
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
features_algo15 = {
 'cumDiffVolDeltaRatio': [
         {'type': 'greater_than_or_equal', 'threshold': -5, 'active': False},
         {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
         {'type': 'between', 'min': -0.6, 'max': -0, 'active': True}],
'finished_auction_low': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0, 'max': 0, 'active': False}],
'is_vwap_reversal_pro_short': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 17.5, 'max': 60, 'active': True}],
}
features_algo16 = {
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
        {'type': 'between', 'min': 2, 'max': 50, 'active': True}],
}
features_algo17 = {
'is_imBullWithPoc_agressive': [
        {'type': 'greater_than_or_equal', 'threshold': 3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 1, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],
}
features_algo18 = {
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
  #  "features_algo1": features_algo1,
    "features_algo2": features_algo2,
    #"features_algo3": features_algo3,
    "features_algo4": features_algo4,
    "features_algo5": features_algo5,
   # "features_algo6": features_algo6,
    "features_algo7": features_algo7,
     # "features_algo8": features_algo8,
    "features_algo9": features_algo9,
    "features_algo10": features_algo10,
    #"features_algo11": features_algo11,
    "features_algo12": features_algo12,
    "features_algo13": features_algo13,
    "features_algo14": features_algo14,
    "features_algo15": features_algo15,
     #"features_algo16": features_algo16,
    "features_algo17": features_algo17,
     "features_algo18": features_algo18,
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
import io
import re
from contextlib import redirect_stdout


import io
import re
from contextlib import redirect_stdout


def header_print(before: dict, after: dict, name: str) -> None:
    """
    Imprime le rapport comparatif en gardant seulement la section
    « => Performance globale - <name> ».
    Toutes les autres sections sont retirées :
        • 📊 STATISTIQUES GLOBALES
        • 📊 ANALYSE DES TRADES LONGS / SHORTS
        • 🎯 TRADES EXTRÊMES LONGS / SHORTS
        • 📑 RÉSUMÉ DE L'IMPACT DU FILTRAGE
        • 📊 SÉQUENCES CONSÉCUTIVES LONGS / SHORTS
    """
    # 1) Générer le rapport complet
    with io.StringIO() as buf, redirect_stdout(buf):
        print_comparative_performance(before, after)
        report = buf.getvalue()

    # 2) En-têtes à supprimer (tout le bloc jusqu’au prochain en-tête ou la fin)
    headers_to_remove = [
        r"📊 STATISTIQUES GLOBALES",
        r"📊 ANALYSE DES TRADES LONGS",
        r"📊 ANALYSE DES TRADES SHORTS",
        r"🎯 TRADES EXTRÊMES LONGS",
        r"🎯 TRADES EXTRÊMES SHORTS",
        r"📑 RÉSUMÉ DE L'IMPACT DU FILTRAGE",
        r"📊 SÉQUENCES CONSÉCUTIVES LONGS",
        r"📊 SÉQUENCES CONSÉCUTIVES SHORTS",
    ]

    pattern = (
        rf"^({'|'.join(headers_to_remove)})[^\n]*\n"   # ligne-titre complète
        rf"(.*?\n)*?"                                  # contenu éventuel
        rf"(?=^📊|^📈|^🎯|^📑|^Win Rate après|$\Z)"      # borne suivante
    )

    report = re.sub(pattern, "", report, flags=re.DOTALL | re.MULTILINE)

    # 3) Suffixe « - <name> » uniquement sur PERFORMANCE GLOBALE
    report = report.replace("PERFORMANCE GLOBALE",
                            f"PERFORMANCE GLOBALE - {name}")

    # 4) Affichage final
    print(report)


# ────────────────────────────────────────────────────────────────────────────────
# Fonction pour évaluer les algorithmes sur un jeu de données
# ────────────────────────────────────────────────────────────────────────────────
def evaluate_algorithms(df_analysis, df_init_features, algorithms, dataset_name="Train"):
    """Évalue les algorithmes sur un jeu de données et renvoie les résultats."""
    print(f"\033[94m\n{'=' * 80}\nÉVALUATION SUR DATASET {dataset_name}\n{'=' * 80}\033[0m")

    results = {}
    to_save = []
    metrics_before = calculate_performance_metrics(df_analysis)

    for algo_name, cond in algorithms.items():
        #print(f"\n{'=' * 80}\nÉVALUATION DE {algo_name} - {dataset_name}\n{'=' * 80}")
        print(f"🎯{'-' * 4}ÉVALUATION DE {algo_name} - {dataset_name}{'-' * 4}")
        df_filt = apply_feature_conditions(df_analysis, cond)

        # CORRECTION: Vérifier que df_filt contient bien des données
        if len(df_filt) == 0:
            print(f"ATTENTION: Aucun trade ne satisfait les conditions pour {algo_name}")
            continue

        # Ajouter un affichage détaillé du PnL pour debug
        pnl_sum = df_filt["trade_pnl"].sum()
        #print(f"Debug - Somme directe des trade_pnl: {pnl_sum}")

        # Compter les trades par type (long/short)
        trade_types = df_filt["pos_type"].value_counts()
        #print(f"Debug - Répartition des types de trades: {trade_types}")

        # Calculer le PnL par type de position
        pnl_by_type = df_filt.groupby("pos_type")["trade_pnl"].sum()
        #print(f"Debug - PnL par type de position: {pnl_by_type}")

        # Créer le dataframe avec PnL filtré
        df_full = preprocess_sessions_with_date(
            create_full_dataframe_with_filtered_pnl(df_init_features, df_filt)
        )

        # CORRECTION: Vérifier et afficher la somme des PnL après filtrage
        pnl_after_filtering = df_full["PnlAfterFiltering"].sum()
        #print(f"Debug - Somme des PnlAfterFiltering: {pnl_after_filtering}")

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
results_val1, to_save_val1 = evaluate_algorithms(df_analysis_val1, df_init_features_val1, algorithms, "Val1")
results_val, to_save_val = evaluate_algorithms(df_analysis_val, df_init_features_val, algorithms, "Val")

# ────────────────────────────────────────────────────────────────────────────────
# Création du tableau comparatif avec les deux jeux de données
# ────────────────────────────────────────────────────────────────────────────────
# Créer les DataFrames individuels pour l'entraînement et le test
comparison_train = pd.DataFrame(results_train).T.reset_index().rename(columns={"index": "Algorithme"})
comparison_test = pd.DataFrame(results_test).T.reset_index().rename(columns={"index": "Algorithme"})
comparison_val1 = pd.DataFrame(results_val1).T.reset_index().rename(columns={"index": "Algorithme"})
comparison_val = pd.DataFrame(results_val).T.reset_index().rename(columns={"index": "Algorithme"})


# Ajouter les colonnes pour Exp PnL pour tous les datasets
for df in [comparison_train, comparison_test, comparison_val1, comparison_val]:
    df["Exp PnL"] = (df["Net PnL"] / df["Nombre de trades"]).round(2)

# Renommer les colonnes pour distinguer les datasets
comparison_test_renamed = comparison_test.rename(columns={
    "Nombre de trades": "Nombre de trades (Test)",
    "Net PnL": "Net PnL (Test)",
    "Exp PnL": "Exp PnL (Test)",
    "Win Rate (%)": "Win Rate (%) (Test)",
    "Profit Factor": "Profit Factor (Test)",
})

comparison_val1_renamed = comparison_val1.rename(columns={
    "Nombre de trades": "Nombre de trades (Val1)",
    "Net PnL": "Net PnL (Val1)",
    "Exp PnL": "Exp PnL (Val1)",
    "Win Rate (%)": "Win Rate (%) (Val1)",
    "Profit Factor": "Profit Factor (Val1)",
})

comparison_val_renamed = comparison_val.rename(columns={
    "Nombre de trades": "Nombre de trades (Val)",
    "Net PnL": "Net PnL (Val)",
    "Exp PnL": "Exp PnL (Val)",
    "Win Rate (%)": "Win Rate (%) (Val)",
    "Profit Factor": "Profit Factor (Val)",
})

# Fusionner les 4 DataFrames
comparison_merged = comparison_train.copy()

# Merge successifs
comparison_merged = pd.merge(comparison_merged, comparison_test_renamed[
    ["Algorithme", "Nombre de trades (Test)", "Net PnL (Test)",
     "Exp PnL (Test)", "Win Rate (%) (Test)", "Profit Factor (Test)"]
], on="Algorithme")

comparison_merged = pd.merge(comparison_merged, comparison_val1_renamed[
    ["Algorithme", "Nombre de trades (Val1)", "Net PnL (Val1)",
     "Exp PnL (Val1)", "Win Rate (%) (Val1)", "Profit Factor (Val1)"]
], on="Algorithme")

comparison_merged = pd.merge(comparison_merged, comparison_val_renamed[
    ["Algorithme", "Nombre de trades (Val)", "Net PnL (Val)",
     "Exp PnL (Val)", "Win Rate (%) (Val)", "Profit Factor (Val)"]
], on="Algorithme")

# Créer la liste des features utilisées (comme dans le code original)
all_features = sorted({feat for d in algorithms.values() for feat in d})

# Pour chaque feature, inscrire "x" pour les algos qui l'emploient
for feat in all_features:
    comparison_merged[feat] = np.where(
        comparison_merged["Algorithme"].map(lambda a: feat in algorithms[a]), "x", ""
    )

# Ordonner les colonnes pour les 4 datasets
cols_metrics = [
    "Algorithme",
    "Nombre de trades", "Nombre de trades (Test)", "Nombre de trades (Val1)", "Nombre de trades (Val)",
    "Net PnL", "Net PnL (Test)", "Net PnL (Val1)", "Net PnL (Val)",
    "Exp PnL", "Exp PnL (Test)", "Exp PnL (Val1)", "Exp PnL (Val)",
    "Win Rate (%)", "Win Rate (%) (Test)", "Win Rate (%) (Val1)", "Win Rate (%) (Val)",
    "Profit Factor", "Profit Factor (Test)", "Profit Factor (Val1)", "Profit Factor (Val)",
]


# Créer deux tableaux séparés
comparison_metrics = comparison_merged[cols_metrics].copy()
comparison_features = comparison_merged[["Algorithme"] + all_features].copy()

# # Afficher le tableau comparatif
# print("\nTABLEAU COMPARATIF DES ALGORITHMES (Entraînement vs Test)\n",
#       comparison_merged.to_string(index=False))

# # Sauvegarder le tableau comparatif
# save_csv(comparison_merged, DIRECTORY_PATH / "algo_comparison_train_test.csv")
#
# # ────────────────────────────────────────────────────────────────────────────────
# # Calculs de robustesse (ratio test/train pour les métriques clés)
# # ────────────────────────────────────────────────────────────────────────────────
# # Ajouter des métriques de robustesse
# comparison_merged["Robustesse Win Rate"] = (comparison_merged["Win Rate (%) (Test)"] /
#                                             comparison_merged["Win Rate (%)"]).round(2)
# comparison_merged["Robustesse Profit Factor"] = (comparison_merged["Profit Factor (Test)"] /
#                                                  comparison_merged["Profit Factor"]).round(2)
# comparison_merged["Robustesse Exp PnL"] = (comparison_merged["Exp PnL (Test)"] /
#                                            comparison_merged["Exp PnL"]).round(2)
#
# # Score de robustesse global (moyenne des 3 métriques)
# comparison_merged["Score Robustesse"] = (
#         (comparison_merged["Robustesse Win Rate"] +
#          comparison_merged["Robustesse Profit Factor"] +
#          comparison_merged["Robustesse Exp PnL"]) / 3
# ).round(2)

# # ────────────────────────────────────────────────────────────────────────────────
# # Sauvegardes différées pour les dataframes
# # ────────────────────────────────────────────────────────────────────────────────
# # Sauvegarder les DataFrames pour les données d'entraînement
# for algo_name, df_full in to_save_train:
#     save_csv(df_full, DIRECTORY_PATH / f"df_with_sessions_{algo_name}_train.csv")
#     save_csv(df_full[df_full["PnlAfterFiltering"] != 0],
#              DIRECTORY_PATH / f"trades_{algo_name}_train.csv")
#
# # Sauvegarder les DataFrames pour les données de test
# for algo_name, df_full in to_save_test:
#     save_csv(df_full, DIRECTORY_PATH / f"df_with_sessions_{algo_name}_test.csv")
#     save_csv(df_full[df_full["PnlAfterFiltering"] != 0],
#              DIRECTORY_PATH / f"trades_{algo_name}_test.csv")


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

        #print(f"Algorithme {algo}: {len(df)} lignes, {dups_int} doublons internes, {len(df) - dups_int} uniques")

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
            print(f"  Trades totaux avant déduplication: {total_rows_before}")
            print(f"  Trades conservés après déduplication: {total} ({total / total_rows_before * 100:.2f}%)")
            print(f"  Trades réussis: {wins}")
            print(f"  Trades échoués: {total - wins}")
            print(f"  Winrate global: {winrate:.2f}%")
            print(f"  PnL total: {total_pnl:.2f}")
            print(f"  Gains totaux: {total_gains:.2f}")
            print(f"  Pertes totales: {total_losses:.2f}")
            print(f"  Gain moyen: {avg_win:.2f}")
            print(f"  Perte moyenne: {avg_loss:.2f}")
            print(f"  Ratio risque/récompense: {reward_risk_ratio:.2f}")
            print(f" Expectancy par trade: {expectancy:.2f}")

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
            print(f"  Nombre total de trades: {total_trades}")
            print(f"  Trades unanimement gagnants: {len(winning_trades)}")
            print(f"  Trades non unanimes (au moins un échec): {len(losing_trades)}")
            print(f"  Winrate (trades unanimement gagnants): {winrate:.2f}%")
            print(f"  PnL total: {total_pnl:.2f}")
            print(f"  PnL des trades gagnants: {winning_pnl:.2f}")
            print(f"  PnL des trades perdants: {losing_pnl:.2f}")

    # 6) analyse des paires d'algos avec > min_common_trades trades communs
    print(f"\n=== ANALYSE DES PAIRES === (> {min_common_trades} trades communs) ===")

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

        # Calculer la similarité Jaccard pour cette paire
        set1 = uniq_sets[a1]
        set2 = uniq_sets[a2]
        jaccard_sim = calculate_jaccard_similarity(set1, set2)

        # Calculer les win rates
        total_wins_a1 = winning_both + winning_a1_only
        total_wins_a2 = winning_both + winning_a2_only
        winrate_a1_common = (total_wins_a1 / len(common) * 100) if len(common) > 0 else 0
        winrate_a2_common = (total_wins_a2 / len(common) * 100) if len(common) > 0 else 0

        # Win rates globaux
        global_wr_a1 = get_algo_winrate(a1, algo_dfs)
        global_wr_a2 = get_algo_winrate(a2, algo_dfs)

        # Déterminer le statut de diversification (utiliser vos variables locales)
        jaccard_threshold = 0.5  # Ou récupérer depuis vos paramètres
        if jaccard_sim < jaccard_threshold:
            jaccard_color = f"{Fore.GREEN}{jaccard_sim:.1%}{Style.RESET_ALL}"
            diversification_status = "DIVERSIFIÉS"
        else:
            jaccard_color = f"{Fore.RED}{jaccard_sim:.1%}{Style.RESET_ALL}"
            diversification_status = "REDONDANTS"

        # Mettre à jour pairs_stats pour inclure les nouvelles métriques
        pairs_stats[(a1, a2)].update({
            'jaccard_similarity': jaccard_sim,
            'winrate_a1_common': winrate_a1_common,
            'winrate_a2_common': winrate_a2_common,
            'global_wr_a1': global_wr_a1,
            'global_wr_a2': global_wr_a2
        })

        # Modifier l'affichage existant pour ajouter les nouvelles lignes
        print(f"\n>> Analyse de la paire {a1} / {a2} ({diversification_status}):")
        print(f"  Trades communs: {len(common)}")
        print(f"  Gagnants pour les deux: {winning_both}")
        print(f"  Gagnants uniquement pour {a1}: {winning_a1_only}")
        print(f"  Gagnants uniquement pour {a2}: {winning_a2_only}")
        print(f"  Perdants pour les deux: {losing_both}")
        print(f"  Taux d'accord: {agreement_rate:.2f}%")
        print(f"  Win Rate {a1} (trades communs): {winrate_a1_common:.1f}%")
        print(f"  Win Rate {a2} (trades communs): {winrate_a2_common:.1f}%")
        print(f"  Win Rate {a1} (global): {global_wr_a1:.1f}%")
        print(f"  Win Rate {a2} (global): {global_wr_a2:.1f}%")
        print(f"  PnL total: {total_pnl:.2f}")
        print(f"  PnL des trades unanimes: {unanimous_pnl:.2f}")
        print(f"  Taux de Jaccard: {jaccard_color}")

    return pairs_stats, occurrences_stats

# ────────────────────────────────────────────────────────────────────────────────
# CRÉATION DU TABLEAU DE RÉPARTITION PAR SESSIONS INTRADAY
# ────────────────────────────────────────────────────────────────────────────────

def create_sessions_analysis_table(datasets_info_with_results):
    """
    Crée un tableau d'analyse par sessions intraday pour tous les algorithmes et datasets

    Parameters:
    -----------
    datasets_info_with_results : list
        Liste de tuples (dataset_name, algo_dfs, results_dict)

    Returns:
    --------
    pd.DataFrame : Tableau avec répartition par sessions intraday
    """

    # Récupérer tous les algorithmes
    all_algos = set()
    for dataset_name, algo_dfs, results_dict in datasets_info_with_results:
        all_algos.update(algo_dfs.keys())

    all_algos = sorted(list(all_algos))

    # Initialiser le DataFrame
    sessions_analysis = pd.DataFrame({'Algorithme': all_algos})

    # Pour chaque dataset
    for dataset_name, algo_dfs, results_dict in datasets_info_with_results:
        print(f"🔄 Analyse des sessions pour {dataset_name}...")

        # Colonnes pour ce dataset
        col_total = f"{dataset_name}_Total"
        col_total_wr = f"{dataset_name}_Total_WR"
        col_pct_g1 = f"{dataset_name}_%G1"
        col_wr_g1 = f"{dataset_name}_WR_G1"
        col_pct_g2 = f"{dataset_name}_%G2"
        col_wr_g2 = f"{dataset_name}_WR_G2"

        # Initialiser les colonnes
        sessions_analysis[col_total] = 0
        sessions_analysis[col_total_wr] = 0.0
        sessions_analysis[col_pct_g1] = 0.0
        sessions_analysis[col_wr_g1] = 0.0
        sessions_analysis[col_pct_g2] = 0.0
        sessions_analysis[col_wr_g2] = 0.0

        # Analyser chaque algorithme
        for algo_name in all_algos:
            if algo_name not in algo_dfs:
                # Algorithme absent de ce dataset
                continue

            df_algo = algo_dfs[algo_name]

            # Vérifier la présence de la colonne deltaCustomSessionIndex
            if 'deltaCustomSessionIndex' not in df_algo.columns:
                print(f"⚠️ Colonne 'deltaCustomSessionIndex' manquante pour {algo_name} dans {dataset_name}")
                continue

            # Déterminer la colonne PnL
            pnl_col = None
            for col in ['PnlAfterFiltering', 'trade_pnl']:
                if col in df_algo.columns:
                    pnl_col = col
                    break

            if pnl_col is None:
                print(f"⚠️ Colonne PnL manquante pour {algo_name} dans {dataset_name}")
                continue

            # Filtrer les trades avec PnL non nul
            df_trades = df_algo[df_algo[pnl_col] != 0].copy()

            if len(df_trades) == 0:
                continue

            # Calculs globaux
            total_trades = len(df_trades)
            total_wins = (df_trades[pnl_col] > 0).sum()
            total_wr = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

            # Analyser les groupes de sessions
            df_groupe1 = df_trades[df_trades['deltaCustomSessionIndex'].isin(GROUPE_SESSION_1)]
            df_groupe2 = df_trades[df_trades['deltaCustomSessionIndex'].isin(GROUPE_SESSION_2)]

            # Calculs pour groupe 1
            count_g1 = len(df_groupe1)
            pct_g1 = (count_g1 / total_trades * 100) if total_trades > 0 else 0.0
            wins_g1 = (df_groupe1[pnl_col] > 0).sum() if count_g1 > 0 else 0
            wr_g1 = (wins_g1 / count_g1 * 100) if count_g1 > 0 else 0.0

            # Calculs pour groupe 2
            count_g2 = len(df_groupe2)
            pct_g2 = (count_g2 / total_trades * 100) if total_trades > 0 else 0.0
            wins_g2 = (df_groupe2[pnl_col] > 0).sum() if count_g2 > 0 else 0
            wr_g2 = (wins_g2 / count_g2 * 100) if count_g2 > 0 else 0.0

            # Remplir le DataFrame
            mask = sessions_analysis['Algorithme'] == algo_name
            sessions_analysis.loc[mask, col_total] = total_trades
            sessions_analysis.loc[mask, col_total_wr] = round(total_wr, 1)
            sessions_analysis.loc[mask, col_pct_g1] = round(pct_g1, 1)
            sessions_analysis.loc[mask, col_wr_g1] = round(wr_g1, 1)
            sessions_analysis.loc[mask, col_pct_g2] = round(pct_g2, 1)
            sessions_analysis.loc[mask, col_wr_g2] = round(wr_g2, 1)

    return sessions_analysis


def format_sessions_table_for_display(sessions_df):
    """
    Formate le tableau des sessions pour un affichage optimal
    """
    # Créer une copie pour le formatage
    display_df = sessions_df.copy()

    # Formater les colonnes de pourcentage et win rates avec le symbole %
    for col in display_df.columns:
        if col != 'Algorithme':
            if '%' in col or 'WR' in col:
                # Ajouter le symbole % pour les pourcentages et win rates
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) and x > 0 else "0.0%")
            elif 'Total' in col and 'WR' not in col:
                # Garder les totaux en nombres entiers
                display_df[col] = display_df[col].astype(int)

    return display_df




# ────────────────────────────────────────────────────────────────────────────────
# ANALYSE SUPPLÉMENTAIRE - INSIGHTS PAR SESSIONS
# ────────────────────────────────────────────────────────────────────────────────

def analyze_session_insights(sessions_df):
    """
    Génère des insights sur les performances par sessions
    """
    print(f"\n{Fore.YELLOW}💡 INSIGHTS PAR SESSIONS INTRADAY{Style.RESET_ALL}")
    print("=" * 80)

    datasets = ["Train", "Test", "Val1", "Val"]

    for dataset in datasets:
        print(f"\n🔍 Dataset {dataset}:")

        # Colonnes pour ce dataset
        col_total = f"{dataset}_Total"
        col_total_wr = f"{dataset}_Total_WR"
        col_wr_g1 = f"{dataset}_WR_G1"
        col_wr_g2 = f"{dataset}_WR_G2"
        col_pct_g1 = f"{dataset}_%G1"
        col_pct_g2 = f"{dataset}_%G2"

        if col_total not in sessions_df.columns:
            continue

        # Filtrer les algorithmes actifs sur ce dataset
        active_algos = sessions_df[sessions_df[col_total] > 0]

        if len(active_algos) == 0:
            print(f"   Aucun algorithme actif")
            continue

        # Moyennes
        avg_total_wr = active_algos[col_total_wr].mean()
        avg_wr_g1 = active_algos[col_wr_g1].mean()
        avg_wr_g2 = active_algos[col_wr_g2].mean()
        avg_pct_g1 = active_algos[col_pct_g1].mean()
        avg_pct_g2 = active_algos[col_pct_g2].mean()

        # Algorithme avec meilleur WR sur groupe 1
        best_g1_idx = active_algos[col_wr_g1].idxmax()
        best_g1_algo = active_algos.loc[best_g1_idx, 'Algorithme']
        best_g1_wr = active_algos.loc[best_g1_idx, col_wr_g1]

        # Algorithme avec meilleur WR sur groupe 2
        best_g2_idx = active_algos[col_wr_g2].idxmax()
        best_g2_algo = active_algos.loc[best_g2_idx, 'Algorithme']
        best_g2_wr = active_algos.loc[best_g2_idx, col_wr_g2]

        print(f"   📊 WR moyen global: {avg_total_wr:.1f}%")
        print(f"   📊 WR moyen Groupe 1: {avg_wr_g1:.1f}% | Groupe 2: {avg_wr_g2:.1f}%")
        print(f"   📊 Répartition moyenne: G1 {avg_pct_g1:.1f}% | G2 {avg_pct_g2:.1f}%")
        print(f"   🏆 Meilleur sur G1: {best_g1_algo} ({best_g1_wr:.1f}%)")
        print(f"   🏆 Meilleur sur G2: {best_g2_algo} ({best_g2_wr:.1f}%)")

        # Identifier les algorithmes avec biais temporel
        bias_threshold = 10.0  # Différence de 10% entre groupes
        biased_algos = []

        for _, row in active_algos.iterrows():
            diff_wr = abs(row[col_wr_g1] - row[col_wr_g2])
            if diff_wr > bias_threshold:
                biased_algos.append({
                    'algo': row['Algorithme'],
                    'diff': diff_wr,
                    'better_group': 'G1' if row[col_wr_g1] > row[col_wr_g2] else 'G2'
                })

        if biased_algos:
            print(f"   ⚠️  Algorithmes avec biais de winrate (>±{bias_threshold}%):")
            for bias in sorted(biased_algos, key=lambda x: x['diff'], reverse=True):
                print(f"      {bias['algo']}: {bias['diff']:.1f}% (meilleur sur {bias['better_group']})")


# ────────────────────────────────────────────────────────────────────────────────
# EXÉCUTION DE L'ANALYSE PAR SESSIONS
# ────────────────────────────────────────────────────────────────────────────────



# Analyse des doublons pour les données d'entraînement et de test
# ────────────────────────────────────────────────────────────────────────────────
# Analyse des doublons pour les 4 datasets (MODIFIÉ)
# ────────────────────────────────────────────────────────────────────────────────
print(f"{Fore.BLUE}\n{'=' * 80}\nANALYSE DES DOUBLONS SUR LES DONNÉES D'ENTRAÎNEMENT\n{'=' * 80}{Style.RESET_ALL}")
algo_dfs_train = {name: df for name, df in to_save_train}
pairs_stats_train, occurrences_stats_train = analyse_doublons_algos(algo_dfs_train, min_common_trades=MIN_COMMON_TRADES)

print(f"{Fore.BLUE}\n{'=' * 80}\nANALYSE DES DOUBLONS SUR LES DONNÉES DE TEST\n{'=' * 80}{Style.RESET_ALL}")
algo_dfs_test = {name: df for name, df in to_save_test}
pairs_stats_test, occurrences_stats_test = analyse_doublons_algos(algo_dfs_test, min_common_trades=MIN_COMMON_TRADES)

print(f"{Fore.BLUE}\n{'=' * 80}\nANALYSE DES DOUBLONS SUR LES DONNÉES DE VALIDATION 1\n{'=' * 80}{Style.RESET_ALL}")
algo_dfs_val1 = {name: df for name, df in to_save_val1}
pairs_stats_val1, occurrences_stats_val1 = analyse_doublons_algos(algo_dfs_val1, min_common_trades=MIN_COMMON_TRADES)

print(f"{Fore.BLUE}\n{'=' * 80}\nANALYSE DES DOUBLONS SUR LES DONNÉES DE VALIDATION 2\n{'=' * 80}{Style.RESET_ALL}")
algo_dfs_val = {name: df for name, df in to_save_val}
pairs_stats_val, occurrences_stats_val = analyse_doublons_algos(algo_dfs_val, min_common_trades=MIN_COMMON_TRADES)



print(DIRECTORY_PATH)
print(f"{Fore.GREEN}\n{'=' * 80}\nSYNTHESE GLOBALE (4 DATASETS)\n{'=' * 80}{Style.RESET_ALL}")

# Afficher le tableau des métriques pour les 4 datasets
print("\n📊 TABLEAU DES MÉTRIQUES (Train / Test / Val1 / Val)")
print("=" * 120)
print(comparison_metrics.to_string(index=False))

# Afficher le tableau des features
print("\n🔧 TABLEAU DES FEATURES UTILISÉES PAR ALGORITHME")
print("=" * 80)
print(comparison_features.to_string(index=False))

# Préparer les données pour l'analyse
datasets_info_with_results = [
    ("Train", algo_dfs_train, results_train),
    ("Test", algo_dfs_test, results_test),
    ("Val1", algo_dfs_val1, results_val1),
    ("Val", algo_dfs_val, results_val)
]

# Créer le tableau d'analyse par sessions
print(f"\n{Fore.CYAN}🔄 Création du tableau d'analyse par sessions intraday...{Style.RESET_ALL}")
sessions_analysis_table = create_sessions_analysis_table(datasets_info_with_results)

# Formater pour l'affichage
sessions_display_table = format_sessions_table_for_display(sessions_analysis_table)

# Afficher le tableau
print(f"\n{Fore.GREEN}📊 RÉPARTITION PAR SESSIONS INTRADAY{Style.RESET_ALL}")
print("=" * 150)
print(f"Groupe 1: Sessions {GROUPE_SESSION_1} | Groupe 2: Sessions {GROUPE_SESSION_2}")
print("=" * 150)
print(sessions_display_table.to_string(index=False))
# Exécuter l'analyse des insights
analyze_session_insights(sessions_analysis_table)

# ────────────────────────────────────────────────────────────────────────────────
# MATRICE GLOBALE DE JACARD
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# MATRICES GLOBALES DE JACCARD pour les 4 datasets
# ────────────────────────────────────────────────────────────────────────────────
print(f"{Fore.YELLOW}\n{'=' * 120}")
print("ANALYSE GLOBALE DE SIMILARITÉ JACCARD (4 DATASETS)")
print(f"{'=' * 120}{Style.RESET_ALL}")

# Définir les datasets avec leurs noms et DataFrames
datasets_info = [
    ("ENTRAINEMENT", algo_dfs_train),
    ("TEST", algo_dfs_test),
    ("VALIDATION 1", algo_dfs_val1),
    ("VALIDATION 2", algo_dfs_val)
]

# Stocker les matrices pour analyse ultérieure
jaccard_matrices = {}
redundant_pairs_by_dataset = {}

# Analyser chaque dataset
for dataset_name, algo_dfs in datasets_info:
    print(f"\n{Fore.BLUE}{'=' * 120}")
    print(f"MATRICE JACCARD - DONNÉES {dataset_name}")
    print(f"{'=' * 120}{Style.RESET_ALL}")

    # Créer et afficher la matrice Jaccard
    jaccard_matrix = create_full_jaccard_matrix(algo_dfs)
    jaccard_matrices[dataset_name] = jaccard_matrix

    # Afficher la matrice avec analyse
    display_jaccard_matrix(jaccard_matrix=jaccard_matrix,
                           threshold=JACCARD_THRESHOLD,
                           algo_dfs=algo_dfs,
                           min_common_trades=MIN_COMMON_TRADES)

    # Analyser la redondance globale
    redundant_pairs = analyze_global_redundancy(jaccard_matrix, JACCARD_THRESHOLD)
    redundant_pairs_by_dataset[dataset_name] = redundant_pairs

# ────────────────────────────────────────────────────────────────────────────────
# ANALYSE DE CONSISTANCE INTER-DATASETS (NOUVEAU)
# ────────────────────────────────────────────────────────────────────────────────
print(f"{Fore.CYAN}\n{'=' * 120}")
print("ANALYSE DE CONSISTANCE DES REDONDANCES INTER-DATASETS")
print(f"{'=' * 120}{Style.RESET_ALL}")

# Analyser la consistance des paires redondantes
all_redundant_pairs = set()
for dataset_name, pairs in redundant_pairs_by_dataset.items():
    all_redundant_pairs.update(pairs)

if all_redundant_pairs:
    print("\n📊 CONSISTANCE DES PAIRES REDONDANTES ENTRE DATASETS:")

    # Pour chaque paire redondante trouvée, vérifier sur combien de datasets elle apparaît
    consistency_analysis = {}

    for pair in all_redundant_pairs:
        datasets_with_pair = []
        for dataset_name, pairs in redundant_pairs_by_dataset.items():
            if pair in pairs:
                datasets_with_pair.append(dataset_name)

        consistency_analysis[pair] = {
            'datasets': datasets_with_pair,
            'count': len(datasets_with_pair),
            'consistency_rate': len(datasets_with_pair) / len(datasets_info) * 100
        }

    # Trier par niveau de consistance
    sorted_pairs = sorted(consistency_analysis.items(),
                          key=lambda x: x[1]['consistency_rate'],
                          reverse=True)

    print(f"\n🔴 PAIRES CONSTAMMENT REDONDANTES (présentes sur plusieurs datasets):")
    for pair, info in sorted_pairs:
        if info['count'] > 1:  # Présent sur plus d'un dataset
            datasets_str = ", ".join(info['datasets'])
            print(
                f"  {pair[0]} ↔ {pair[1]}: {info['consistency_rate']:.0f}% ({info['count']}/{len(datasets_info)} datasets)")
            print(f"    Présent sur: {datasets_str}")

    # Identifier les paires redondantes uniquement sur un dataset (potentiels faux positifs)
    print(f"\n🟡 PAIRES REDONDANTES SUR UN SEUL DATASET (potentiels faux positifs):")
    for pair, info in sorted_pairs:
        if info['count'] == 1:
            dataset_str = info['datasets'][0]
            print(f"  {pair[0]} ↔ {pair[1]}: Uniquement sur {dataset_str}")

else:
    print("✅ Aucune paire redondante détectée sur l'ensemble des datasets.")

# ────────────────────────────────────────────────────────────────────────────────
# SYNTHÈSE GLOBALE DES REDONDANCES (NOUVEAU)
# ────────────────────────────────────────────────────────────────────────────────
print(f"\n{Fore.GREEN}{'=' * 120}")
print("SYNTHÈSE GLOBALE DES REDONDANCES")
print(f"{'=' * 120}{Style.RESET_ALL}")

print("\n📈 RÉSUMÉ PAR DATASET:")
for dataset_name, pairs in redundant_pairs_by_dataset.items():
    print(f"  {dataset_name}: {len(pairs)} paires redondantes")

if all_redundant_pairs:
    # Calculer le score de redondance global pour chaque algorithme
    algo_redundancy_scores = {}
    all_algos = set()

    # Collecter tous les algorithmes
    for dataset_name, algo_dfs in datasets_info:
        all_algos.update(algo_dfs.keys())

    # Calculer le score de redondance pour chaque algorithme
    for algo in all_algos:
        redundancy_count = 0
        total_possible_pairs = 0

        for pair in all_redundant_pairs:
            if algo in pair:
                redundancy_count += 1

        # Le nombre total de paires possibles pour cet algo
        total_possible_pairs = len(all_algos) - 1

        if total_possible_pairs > 0:
            redundancy_rate = redundancy_count / total_possible_pairs * 100
            algo_redundancy_scores[algo] = redundancy_rate

    # Afficher les algorithmes les plus redondants
    if algo_redundancy_scores:
        sorted_algos = sorted(algo_redundancy_scores.items(),
                              key=lambda x: x[1],
                              reverse=True)

        print(f"\n🔴 ALGORITHMES LES PLUS REDONDANTS:")
        for algo, score in sorted_algos[:5]:  # Top 5
            if score > 0:
                print(f"  {algo}: {score:.1f}% de redondance")

        print(f"\n✅ ALGORITHMES LES MOINS REDONDANTS:")
        for algo, score in sorted_algos[-5:]:  # Bottom 5
            print(f"  {algo}: {score:.1f}% de redondance")

print(f"\n💡 RECOMMANDATIONS:")
if all_redundant_pairs:
    print("  • Considérer la suppression des algorithmes constamment redondants")
    print("  • Prioriser les algorithmes avec faible score de redondance")
    print("  • Vérifier manuellement les paires redondantes sur un seul dataset")
else:
    print("  • Excellente diversification des algorithmes détectée")
    print("  • Aucune optimisation de redondance nécessaire")




# ────────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE OPTIONNELLE
# ────────────────────────────────────────────────────────────────────────────────

# Sauvegarder le tableau des sessions (optionnel)
# save_csv(sessions_analysis_table, DIRECTORY_PATH / "sessions_analysis_table.csv")
# print(f"✅ Tableau des sessions sauvegardé")