import pandas as pd
import numpy as np
from standardFunc import print_notification
from standardFunc import load_data

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
diffDivBy0 = np.nan
addDivBy0 = np.nan
from sklearn.preprocessing import MinMaxScaler

user_choice = input("Appuyez sur Entrée pour calculer les features sans la afficher. \n"
                    "Appuyez sur 'd' puis Entrée pour les calculer et les afficher : \n"
                    "Appuyez sur 's' puis Entrée pour les calculer et les afficher :")

# Nom du fichier
file_name = "Step4_Step3_Step2_MergedAllFile_Step1_2_merged_extractOnlyFullSession_OnlyShort.csv"

# Chemin du répertoire
directory_path = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\4_0_4TP_1SL\\merge13092024"

# Construction du chemin complet du fichier
file_path = os.path.join(directory_path, file_name)

REPLACE_NAN=False
REPLACED_NANVALUE_BY=90000.54789
REPLACED_NANVALUE_BY_INDEX=1
if REPLACE_NAN:
    print(f"\nINFO : Implémenter dans le code => les valeurs NaN seront remplacées par {REPLACED_NANVALUE_BY} et un index")
else:
    print(f"\nINFO : Implémenter dans le code => les valeurs NaN ne seront pas remplacées par une valeur choisie par l'utilisateur mais laissé à NAN")

# Configuration
CONFIG = {
    'NUM_GROUPS': 9,
    'MIN_RANGE': 30,  # en minutes
    'FILE_PATH': file_path,
    'TRADING_START_TIME': "22:00",
    'FIGURE_SIZE': (20, 10),
    'GRID_ALPHA': 0.7,
}

# Définition des sections personnalisées
CUSTOM_SECTIONS = [
    {"name": "preAsian", "start": 0, "end": 240, "index": 0},
    {"name": "asianAndPreEurop", "start": 240, "end": 540, "index": 1},
    {"name": "europMorning", "start": 540, "end": 810, "index": 2},
    {"name": "europLunch", "start": 810, "end": 870, "index": 3},
    {"name": "preUS", "start": 870, "end": 930, "index": 4},
    {"name": "usMoning", "start": 930, "end": 1065, "index": 5},
    {"name": "usAfternoon", "start": 1065, "end": 1200, "index": 6},
    {"name": "usEvening", "start": 1200, "end": 1290, "index": 7},
    {"name": "usEnd", "start": 1290, "end": 1335, "index": 8},
    {"name": "closing", "start": 1335, "end": 1380, "index": 9},
]



def get_custom_section(minutes: int) -> dict:
    for section in CUSTOM_SECTIONS:
        if section['start'] <= minutes < section['end']:
            return section
    return CUSTOM_SECTIONS[-1]  # Retourne la dernière section si aucune correspondance n'est trouvée

# Chargement des données
df = load_data(CONFIG['FILE_PATH'])

print_notification("Début du calcul des features")

# Calcul des features
features_df = pd.DataFrame()
features_df['deltaTimestampOpening'] = df['deltaTimestampOpening']

features_df['deltaTimestampOpeningSection5min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x/5))*5, 1350))

unique_sections = sorted(features_df['deltaTimestampOpeningSection5min'].unique())
section_to_index = {section: index for index, section in enumerate(unique_sections)}
features_df['deltaTimestampOpeningSection5index'] = features_df['deltaTimestampOpeningSection5min'].map(section_to_index)

features_df['deltaTimestampOpeningSection30min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x/30))*30, 1350))

unique_sections = sorted(features_df['deltaTimestampOpeningSection30min'].unique())
section_to_index = {section: index for index, section in enumerate(unique_sections)}
features_df['deltaTimestampOpeningSection30index'] = features_df['deltaTimestampOpeningSection30min'].map(section_to_index)

features_df['deltaCustomSectionMin'] = df['deltaTimestampOpening'].apply(
    lambda x: get_custom_section(x)['start'])

unique_custom_sections = sorted(features_df['deltaCustomSectionMin'].unique())
custom_section_to_index = {section: index for index, section in enumerate(unique_custom_sections)}
features_df['deltaCustomSectionIndex'] = features_df['deltaCustomSectionMin'].map(custom_section_to_index)

# Features précédentes
features_df['candleSizeTicks'] = df['candleSizeTicks']
features_df['diffPriceClosePoc_0_0'] = df['close'] - df['pocPrice']
features_df['diffPriceClosePoc_0_1'] = df['close'] - df['pocPrice'].shift(1)
features_df['diffPriceClosePoc_0_2'] = df['close'] - df['pocPrice'].shift(2)
features_df['diffPriceClosePoc_0_3'] = df['close'] - df['pocPrice'].shift(3)
features_df['diffHighPrice_0_1'] = df['high'] - df['high'].shift(1)
features_df['diffHighPrice_0_2'] = df['high'] - df['high'].shift(2)
features_df['diffHighPrice_0_3'] = df['high'] - df['high'].shift(3)
features_df['diffHighPrice_0_4'] = df['high'] - df['high'].shift(4)
features_df['diffLowPrice_0_1'] = df['low'] - df['low'].shift(1)
features_df['diffLowPrice_0_2'] = df['low'] - df['low'].shift(2)
features_df['diffLowPrice_0_3'] = df['low'] - df['low'].shift(3)
features_df['diffLowPrice_0_4'] = df['low'] - df['low'].shift(4)
features_df['diffPriceCloseVWAP'] = df['close'] - df['VWAP']

features_df['atr'] = df['atr']
features_df['bandWidthBB'] = df['bandWidthBB']
features_df['perctBB'] = df['perctBB']


import numpy as np
import pandas as pd

def detect_market_regime(data, period=14, adx_threshold=25):
    # Calcul de l'ADX
    data['plus_dm'] = np.where((data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']),
                               np.maximum(data['high'] - data['high'].shift(1), 0), 0)
    data['minus_dm'] = np.where((data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1)),
                                np.maximum(data['low'].shift(1) - data['low'], 0), 0)
    data['tr'] = np.maximum(data['high'] - data['low'],
                            np.maximum(abs(data['high'] - data['close'].shift(1)),
                                       abs(data['low'] - data['close'].shift(1))))
    data['plus_di'] = 100 * data['plus_dm'].rolling(period).sum() / data['tr'].rolling(period).sum()
    data['minus_di'] = 100 * data['minus_dm'].rolling(period).sum() / data['tr'].rolling(period).sum()
    data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
    data['adx'] = data['dx'].rolling(period).mean()



   # data['market_regimeADX'] = np.where(data['adx'] > adx_threshold, data['adx'], data['adx'])
    data['market_regimeADX'] = np.where(data['adx'] > adx_threshold, data['adx'], 0)
    data['market_regimeADX'] = data['market_regimeADX'].fillna(addDivBy0)
    return data


def range_strength(data, window=30, atr_multiple=3, min_strength=0.1):
    data = data.copy()
    required_columns = ['high', 'low', 'close']
    missing_cols = [col for col in required_columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Les colonnes manquantes dans le DataFrame : {missing_cols}")

    for col in required_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift()),
            np.abs(data['low'] - data['close'].shift())
        )
    )

    data['atr'] = data['tr'].rolling(window=window).mean()
    data['threshold'] = (data['atr'] / data['close'].replace(0, np.nan)) * atr_multiple
    data['rolling_high'] = data['high'].rolling(window=window).max()
    data['rolling_low'] = data['low'].rolling(window=window).min()
    data['range_width'] = (data['rolling_high'] - data['rolling_low']) / data['rolling_low']

    condition = data['range_width'] <= data['threshold']
    data['range_duration'] = condition.astype(int).groupby((~condition).cumsum()).cumsum()
    data['range_strength'] = data['range_duration'] / (1 + data['range_width'])

    # Appliquer un seuil minimum et une transformation logarithmique
    data['range_strength'] = np.where(data['range_strength'] < min_strength, np.nan, data['range_strength'])

    data['log_range_strength'] = np.log1p(data['range_strength'])

    data.drop(['tr'], axis=1, inplace=True)
    data['range_strength'] = data['range_strength'].fillna(addDivBy0)

    return data



# Appliquer range_strength sur une copie de df pour ne pas modifier df

df_with_range_strength = range_strength(df, window=25, atr_multiple=2.5, min_strength=0.01)
# Ajouter la colonne 'range_strength' à features_df
features_df['range_strength'] = df_with_range_strength['range_strength']
# Appliquer detect_market_regime sur une copie de df pour ne pas modifier df
df_copy = df.copy()
df_with_regime = detect_market_regime(df_copy)
# Ajouter la colonne 'market_regime' à features_df
features_df['market_regimeADX'] = df_with_regime['market_regimeADX']


# Nouvelles features - Force du renversement
features_df['bearish_reversal_force'] = np.where(df['volume'] != 0, df['VolAbv'] / df['volume'], addDivBy0)
features_df['bullish_reversal_force'] = np.where(df['volume'] != 0, df['VolBlw'] / df['volume'], addDivBy0)

df['VolAbvAsk'] = df['upTickVolAbvAskDesc'] + df['upTickVolAbvAskAsc'] + \
                  df['downTickVolAbvAskDesc'] + df['downTickVolAbvAskAsc'] + \
                  df['repeatUpTickVolAbvAskDesc'] + df['repeatUpTickVolAbvAskAsc'] + \
                  df['repeatDownTickVolAbvAskDesc'] + df['repeatDownTickVolAbvAskAsc']

df['VolAbvBid'] = df['upTickVolAbvBidDesc'] + df['upTickVolAbvBidAsc'] + \
                  df['downTickVolAbvBidDesc'] + df['downTickVolAbvBidAsc'] + \
                  df['repeatUpTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidAsc'] + \
                  df['repeatDownTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidAsc']

df['VolBlwAsk'] = df['upTickVolBlwAskDesc'] + df['upTickVolBlwAskAsc'] + \
                  df['downTickVolBlwAskDesc'] + df['downTickVolBlwAskAsc'] + \
                  df['repeatUpTickVolBlwAskDesc'] + df['repeatUpTickVolBlwAskAsc'] + \
                  df['repeatDownTickVolBlwAskDesc'] + df['repeatDownTickVolBlwAskAsc']

df['VolBlwBid'] = df['upTickVolBlwBidDesc'] + df['upTickVolBlwBidAsc'] + \
                  df['downTickVolBlwBidDesc'] + df['downTickVolBlwBidAsc'] + \
                  df['repeatUpTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidAsc'] + \
                  df['repeatDownTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidAsc']

# Nouvelles features - Ratio Ask/Bid dans la zone de renversement
features_df['bearish_ask_bid_ratio'] = np.where(df['VolAbvBid'] != 0, df['VolAbvAsk'] / df['VolAbvBid'], addDivBy0)
features_df['bullish_ask_bid_ratio'] = np.where(df['VolBlwAsk'] != 0, df['VolBlwBid'] / df['VolBlwAsk'], addDivBy0)

# Nouvelles features - Features de Momentum:
# Moyenne des volumes
features_df['meanVolx'] = df['volume'].shift().rolling(window=5, min_periods=1).mean()

# Relative delta Momentum
features_df['ratioDeltaBlw'] = np.where(df['VolBlw'] != 0, df['DeltaBlw'] / df['VolBlw'], diffDivBy0)
features_df['ratioDeltaAbv'] = np.where(df['VolAbv'] != 0, df['DeltaAbv'] / df['VolAbv'], diffDivBy0)

# Relatif volume evol
features_df['diffVolCandle_0_1Ratio'] = np.where(features_df['meanVolx'] != 0,
                                            (df['volume'] - df['volume'].shift(1)) / features_df['meanVolx'], diffDivBy0)

# Relatif delta evol
features_df['diffVolDelta_0_1Ratio'] = np.where(features_df['meanVolx'] != 0,
                                           (df['delta'] - df['delta'].shift(1)) / features_df['meanVolx'], diffDivBy0)

# cumDiffVolDelta
features_df['cumDiffVolDeltaRatio'] =  np.where(features_df['meanVolx'] != 0,(df['delta'].shift(1) + df['delta'].shift(2) + \
                                  df['delta'].shift(3) + df['delta'].shift(4) + df['delta'].shift(5))/ features_df['meanVolx'], diffDivBy0)

# Nouvelles features - Features de Volume Profile:
# Importance du POC
features_df['VolPocVolCandleRatio'] = np.where(df['volume'] != 0, df['volPOC'] / df['volume'], addDivBy0)
features_df['pocDeltaPocVolRatio'] = np.where(df['volPOC'] != 0, df['deltaPOC'] / df['volPOC'], diffDivBy0)

# Asymétrie du volume
features_df['asymetrie_volume'] = np.where(df['volume'] != 0, (df['VolAbv'] - df['VolBlw']) / df['volume'], diffDivBy0)

# Nouvelles features - Features Cumulatives sur les 5 dernières bougies:
# Volume spike
features_df['VolCandleMeanxRatio'] = np.where(features_df['meanVolx'] != 0, df['volume'] / features_df['meanVolx'], addDivBy0)

# Nouvelles features - Caractéristiques de la zone de renversement :
features_df['bearish_ask_ratio'] = np.where(df['VolAbv'] != 0, df['VolAbvAsk'] / df['VolAbv'], addDivBy0)
features_df['bearish_bid_ratio'] = np.where(df['VolAbv'] != 0, df['VolAbvBid'] / df['VolAbv'], addDivBy0)
features_df['bullish_ask_ratio'] = np.where(df['VolBlw'] != 0, df['VolBlwAsk'] / df['VolBlw'], addDivBy0)
features_df['bullish_bid_ratio'] = np.where(df['VolBlw'] != 0, df['VolBlwBid'] / df['VolBlw'], addDivBy0)

# Nouvelles features - Dynamique de prix dans la zone de renversement :
features_df['bearish_ask_score'] = np.where(df['VolAbv'] != 0,
                                            (df['downTickVolAbvAskDesc'] + df['repeatDownTickVolAbvAskDesc']-
                                             df['upTickVolAbvAskDesc'] - df['repeatUpTickVolAbvAskDesc'] )/ df['VolAbv'], diffDivBy0)

features_df['bearish_bid_score'] = np.where(df['VolAbv'] != 0,
                                            (df['downTickVolAbvBidDesc']+ df['repeatDownTickVolAbvBidDesc']-
                                             df['upTickVolAbvBidDesc'] - df['repeatUpTickVolAbvBidDesc']
                                              ) / df['VolAbv'], diffDivBy0)
features_df['bearish_imnbScore_score']=features_df['bearish_bid_score'] -features_df['bearish_ask_score']

features_df['bullish_ask_score'] = np.where(df['VolBlw'] != 0,
                                            (df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] -
                                             df['downTickVolBlwAskAsc'] - df['repeatDownTickVolBlwAskAsc']) / df['VolBlw'], diffDivBy0)

features_df['bullish_bid_score'] = np.where(df['VolBlw'] != 0,
                                            (df['upTickVolBlwBidAsc'] + df['repeatUpTickVolBlwBidAsc'] -
                                             df['downTickVolBlwBidAsc'] - df['repeatDownTickVolBlwBidAsc'] )/ df['VolBlw'], diffDivBy0)
features_df['bullish_imnbScore_score']=features_df['bullish_ask_score']-features_df['bullish_bid_score']

# Nouvelles features - Order Flow:
# Imbalances haussières
features_df['bull_imbalance_low_1'] = np.where(df['bidVolLow'] != 0, df['askVolLow_1'] / df['bidVolLow'], addDivBy0)
features_df['bull_imbalance_low_2'] = np.where(df['bidVolLow_1'] != 0, df['askVolLow_2'] / df['bidVolLow_1'], addDivBy0)
features_df['bull_imbalance_low_3'] = np.where(df['bidVolLow_2'] != 0, df['askVolLow_3'] / df['bidVolLow_2'], addDivBy0)
features_df['bull_imbalance_high_0'] = np.where(df['bidVolHigh_1'] != 0, df['askVolHigh'] / df['bidVolHigh_1'], addDivBy0)
features_df['bull_imbalance_high_1'] = np.where(df['bidVolHigh_2'] != 0, df['askVolHigh_1'] / df['bidVolHigh_2'], addDivBy0)
features_df['bull_imbalance_high_2'] = np.where(df['bidVolHigh_3'] != 0, df['askVolHigh_2'] / df['bidVolHigh_3'], addDivBy0)

# Imbalances baissières
features_df['bear_imbalance_low_0'] = np.where(df['askVolLow_1'] != 0, df['bidVolLow'] / df['askVolLow_1'], addDivBy0)
features_df['bear_imbalance_low_1'] = np.where(df['askVolLow_2'] != 0, df['bidVolLow_1'] / df['askVolLow_2'], addDivBy0)
features_df['bear_imbalance_low_2'] = np.where(df['askVolLow_3'] != 0, df['bidVolLow_2'] / df['askVolLow_3'], addDivBy0)
features_df['bear_imbalance_high_1'] = np.where(df['askVolHigh'] != 0, df['bidVolHigh_1'] / df['askVolHigh'], addDivBy0)
features_df['bear_imbalance_high_2'] = np.where(df['askVolHigh_1'] != 0, df['bidVolHigh_2'] / df['askVolHigh_1'], addDivBy0)
features_df['bear_imbalance_high_3'] = np.where(df['askVolHigh_2'] != 0, df['bidVolHigh_3'] / df['askVolHigh_2'], addDivBy0)

# Score d'Imbalance Asymétrique
sell_pressureLow = df['bidVolLow'] + df['bidVolLow_1']
buy_pressureLow = df['askVolLow_1'] + df['askVolLow_2']
total_volumeLow = buy_pressureLow + sell_pressureLow
features_df['imbalance_score_low'] = np.where(total_volumeLow != 0,
                                              (buy_pressureLow - sell_pressureLow) / total_volumeLow, diffDivBy0)

sell_pressureHigh = df['bidVolHigh_1'] + df['bidVolHigh_2']
buy_pressureHigh = df['askVolHigh'] + df['askVolHigh_1']
total_volumeHigh = sell_pressureHigh + buy_pressureHigh
features_df['imbalance_score_high'] = np.where(total_volumeHigh != 0,
                                               (sell_pressureHigh - buy_pressureHigh) / total_volumeHigh, diffDivBy0)

# Finished Auction
features_df['finished_auction_high'] = (df['bidVolHigh'] == 0).astype(int)
features_df['finished_auction_low'] = (df['askVolLow'] == 0).astype(int)
features_df['staked00_high'] = ((df['bidVolHigh'] == 0) & (df['bidVolHigh_1'] == 0)).astype(int)
features_df['staked00_low'] = ((df['askVolLow'] == 0) & (df['askVolLow_1'] == 0)).astype(int)
# Calcul des variables upTick
# Calcul des variables upTick
upTickVolAbvBid = df['upTickVolAbvBidDesc'] + df['upTickVolAbvBidAsc']
upTickVolAbvAsk = df['upTickVolAbvAskDesc'] + df['upTickVolAbvAskAsc']
upTickVolBlwBid = df['upTickVolBlwBidDesc'] + df['upTickVolBlwBidAsc']
upTickVolBlwAsk = df['upTickVolBlwAskDesc'] + df['upTickVolBlwAskAsc']

# Calcul des variables repeat
repeatUpTickVolAbvBid = df['repeatUpTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidAsc']
repeatUpTickVolAbvAsk = df['repeatUpTickVolAbvAskDesc'] + df['repeatUpTickVolAbvAskAsc']
repeatUpTickVolBlwBid = df['repeatUpTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidAsc']
repeatUpTickVolBlwAsk = df['repeatUpTickVolBlwAskDesc'] + df['repeatUpTickVolBlwAskAsc']

repeatDownTickVolAbvBid = df['repeatDownTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidAsc']
repeatDownTickVolAbvAsk = df['repeatDownTickVolAbvAskDesc'] + df['repeatDownTickVolAbvAskAsc']
repeatDownTickVolBlwBid = df['repeatDownTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidAsc']
repeatDownTickVolBlwAsk = df['repeatDownTickVolBlwAskDesc'] + df['repeatDownTickVolBlwAskAsc']

# Feature générale - Order Flow
features_df['bearish_ask_abs_ratio_abv'] = np.where(df['VolAbv'] != 0,
                                                    (upTickVolAbvAsk + repeatUpTickVolAbvAsk + repeatDownTickVolAbvAsk) / df['VolAbv'], addDivBy0)

features_df['bearish_bid_abs_ratio_abv'] = np.where(df['VolAbv'] != 0,
                                                    (upTickVolAbvBid + repeatUpTickVolAbvBid + repeatDownTickVolAbvBid) / df['VolAbv'], addDivBy0)

features_df['bearish_abs_diff_abv'] = np.where(df['VolAbv'] != 0,
                                               ((upTickVolAbvAsk + repeatUpTickVolAbvAsk + repeatDownTickVolAbvAsk) -
                                                (upTickVolAbvBid + repeatUpTickVolAbvBid + repeatDownTickVolAbvBid)) / df['VolAbv'], diffDivBy0)

features_df['bullish_ask_abs_ratio_blw'] = np.where(df['VolBlw'] != 0,
                                                    (upTickVolBlwAsk + repeatUpTickVolBlwAsk + repeatDownTickVolBlwAsk) / df['VolBlw'], addDivBy0)

features_df['bullish_bid_abs_ratio_blw'] = np.where(df['VolBlw'] != 0,
                                                    (upTickVolBlwBid + repeatUpTickVolBlwBid + repeatDownTickVolBlwBid) / df['VolBlw'], addDivBy0)

features_df['bullish_abs_diff_blw'] = np.where(df['VolBlw'] != 0,
                                               ((upTickVolBlwAsk + repeatUpTickVolBlwAsk + repeatDownTickVolBlwAsk) -
                                                (upTickVolBlwBid + repeatUpTickVolBlwBid + repeatDownTickVolBlwBid)) / df['VolBlw'], diffDivBy0)


# Calcul des variables intermédiaires pour BigStand et BigHigh
# Pour la zone Abv (bearish)
upTickVolAbvAsk_bigStand = df['upTickVolAbvAskDesc_bigStand'] + df['upTickVolAbvAskAsc_bigStand']
upTickVolAbvBid_bigStand = df['upTickVolAbvBidDesc_bigStand'] + df['upTickVolAbvBidAsc_bigStand']
repeatUpTickVolAbvAsk_bigStand = df['repeatUpTickVolAbvAskDesc_bigStand'] + df['repeatUpTickVolAbvAskAsc_bigStand']
repeatUpTickVolAbvBid_bigStand = df['repeatUpTickVolAbvBidDesc_bigStand'] + df['repeatUpTickVolAbvBidAsc_bigStand']
repeatDownTickVolAbvAsk_bigStand = df['repeatDownTickVolAbvAskDesc_bigStand'] + df['repeatDownTickVolAbvAskAsc_bigStand']
repeatDownTickVolAbvBid_bigStand = df['repeatDownTickVolAbvBidDesc_bigStand'] + df['repeatDownTickVolAbvBidAsc_bigStand']

upTickVolAbvAsk_bigHigh = df['upTickVolAbvAskDesc_bigHigh'] + df['upTickVolAbvAskAsc_bigHigh']
upTickVolAbvBid_bigHigh = df['upTickVolAbvBidDesc_bigHigh'] + df['upTickVolAbvBidAsc_bigHigh']
repeatUpTickVolAbvAsk_bigHigh = df['repeatUpTickVolAbvAskDesc_bigHigh'] + df['repeatUpTickVolAbvAskAsc_bigHigh']
repeatUpTickVolAbvBid_bigHigh = df['repeatUpTickVolAbvBidDesc_bigHigh'] + df['repeatUpTickVolAbvBidAsc_bigHigh']
repeatDownTickVolAbvAsk_bigHigh = df['repeatDownTickVolAbvAskDesc_bigHigh'] + df['repeatDownTickVolAbvAskAsc_bigHigh']
repeatDownTickVolAbvBid_bigHigh = df['repeatDownTickVolAbvBidDesc_bigHigh'] + df['repeatDownTickVolAbvBidAsc_bigHigh']

# Pour la zone Blw (bullish)
upTickVolBlwAsk_bigStand = df['upTickVolBlwAskDesc_bigStand'] + df['upTickVolBlwAskAsc_bigStand']
upTickVolBlwBid_bigStand = df['upTickVolBlwBidDesc_bigStand'] + df['upTickVolBlwBidAsc_bigStand']
repeatUpTickVolBlwAsk_bigStand = df['repeatUpTickVolBlwAskDesc_bigStand'] + df['repeatUpTickVolBlwAskAsc_bigStand']
repeatUpTickVolBlwBid_bigStand = df['repeatUpTickVolBlwBidDesc_bigStand'] + df['repeatUpTickVolBlwBidAsc_bigStand']
repeatDownTickVolBlwAsk_bigStand = df['repeatDownTickVolBlwAskDesc_bigStand'] + df['repeatDownTickVolBlwAskAsc_bigStand']
repeatDownTickVolBlwBid_bigStand = df['repeatDownTickVolBlwBidDesc_bigStand'] + df['repeatDownTickVolBlwBidAsc_bigStand']

upTickVolBlwAsk_bigHigh = df['upTickVolBlwAskDesc_bigHigh'] + df['upTickVolBlwAskAsc_bigHigh']
upTickVolBlwBid_bigHigh = df['upTickVolBlwBidDesc_bigHigh'] + df['upTickVolBlwBidAsc_bigHigh']
repeatUpTickVolBlwAsk_bigHigh = df['repeatUpTickVolBlwAskDesc_bigHigh'] + df['repeatUpTickVolBlwAskAsc_bigHigh']
repeatUpTickVolBlwBid_bigHigh = df['repeatUpTickVolBlwBidDesc_bigHigh'] + df['repeatUpTickVolBlwBidAsc_bigHigh']
repeatDownTickVolBlwAsk_bigHigh = df['repeatDownTickVolBlwAskDesc_bigHigh'] + df['repeatDownTickVolBlwAskAsc_bigHigh']
repeatDownTickVolBlwBid_bigHigh = df['repeatDownTickVolBlwBidDesc_bigHigh'] + df['repeatDownTickVolBlwBidAsc_bigHigh']

# Calcul des variables intermédiaires pour la zone extrem
# Pour la zone Abv (bearish)
upTickVolAbvAsk_extrem = df['upTickVolAbvAskDesc_extrem'] + df['upTickVolAbvAskAsc_extrem']
upTickVolAbvBid_extrem = df['upTickVolAbvBidDesc_extrem'] + df['upTickVolAbvBidAsc_extrem']
downTickVolAbvAsk_extrem = df['downTickVolAbvAskDesc_extrem'] + df['downTickVolAbvAskAsc_extrem']
downTickVolAbvBid_extrem = df['downTickVolAbvBidDesc_extrem'] + df['downTickVolAbvBidAsc_extrem']
repeatUpTickVolAbvAsk_extrem = df['repeatUpTickVolAbvAskDesc_extrem'] + df['repeatUpTickVolAbvAskAsc_extrem']
repeatUpTickVolAbvBid_extrem = df['repeatUpTickVolAbvBidDesc_extrem'] + df['repeatUpTickVolAbvBidAsc_extrem']
repeatDownTickVolAbvAsk_extrem = df['repeatDownTickVolAbvAskDesc_extrem'] + df['repeatDownTickVolAbvAskAsc_extrem']
repeatDownTickVolAbvBid_extrem = df['repeatDownTickVolAbvBidDesc_extrem'] + df['repeatDownTickVolAbvBidAsc_extrem']

# Pour la zone Blw (bullish)
upTickVolBlwAsk_extrem = df['upTickVolBlwAskDesc_extrem'] + df['upTickVolBlwAskAsc_extrem']
upTickVolBlwBid_extrem = df['upTickVolBlwBidDesc_extrem'] + df['upTickVolBlwBidAsc_extrem']
downTickVolBlwAsk_extrem = df['downTickVolBlwAskDesc_extrem'] + df['downTickVolBlwAskAsc_extrem']
downTickVolBlwBid_extrem = df['downTickVolBlwBidDesc_extrem'] + df['downTickVolBlwBidAsc_extrem']
repeatUpTickVolBlwAsk_extrem = df['repeatUpTickVolBlwAskDesc_extrem'] + df['repeatUpTickVolBlwAskAsc_extrem']
repeatUpTickVolBlwBid_extrem = df['repeatUpTickVolBlwBidDesc_extrem'] + df['repeatUpTickVolBlwBidAsc_extrem']
repeatDownTickVolBlwAsk_extrem = df['repeatDownTickVolBlwAskDesc_extrem'] + df['repeatDownTickVolBlwAskAsc_extrem']
repeatDownTickVolBlwBid_extrem = df['repeatDownTickVolBlwBidDesc_extrem'] + df['repeatDownTickVolBlwBidAsc_extrem']

# Pour les big trades dans la zone extrem
upTickVolAbvAsk_bigStand_extrem = df['upTickVolAbvAskDesc_bigStand_extrem'] + df['upTickVolAbvAskAsc_bigStand_extrem']
upTickVolAbvBid_bigStand_extrem = df['upTickVolAbvBidDesc_bigStand_extrem'] + df['upTickVolAbvBidAsc_bigStand_extrem']
downTickVolAbvAsk_bigStand_extrem = df['downTickVolAbvAskDesc_bigStand_extrem'] + df['downTickVolAbvAskAsc_bigStand_extrem']
downTickVolAbvBid_bigStand_extrem = df['downTickVolAbvBidDesc_bigStand_extrem'] + df['downTickVolAbvBidAsc_bigStand_extrem']
repeatUpTickVolAbvAsk_bigStand_extrem = df['repeatUpTickVolAbvAskDesc_bigStand_extrem'] + df['repeatUpTickVolAbvAskAsc_bigStand_extrem']
repeatUpTickVolAbvBid_bigStand_extrem = df['repeatUpTickVolAbvBidDesc_bigStand_extrem'] + df['repeatUpTickVolAbvBidAsc_bigStand_extrem']
repeatDownTickVolAbvAsk_bigStand_extrem = df['repeatDownTickVolAbvAskDesc_bigStand_extrem'] + df['repeatDownTickVolAbvAskAsc_bigStand_extrem']
repeatDownTickVolAbvBid_bigStand_extrem = df['repeatDownTickVolAbvBidDesc_bigStand_extrem'] + df['repeatDownTickVolAbvBidAsc_bigStand_extrem']

upTickVolBlwAsk_bigStand_extrem = df['upTickVolBlwAskDesc_bigStand_extrem'] + df['upTickVolBlwAskAsc_bigStand_extrem']
upTickVolBlwBid_bigStand_extrem = df['upTickVolBlwBidDesc_bigStand_extrem'] + df['upTickVolBlwBidAsc_bigStand_extrem']
downTickVolBlwAsk_bigStand_extrem = df['downTickVolBlwAskDesc_bigStand_extrem'] + df['downTickVolBlwAskAsc_bigStand_extrem']
downTickVolBlwBid_bigStand_extrem = df['downTickVolBlwBidDesc_bigStand_extrem'] + df['downTickVolBlwBidAsc_bigStand_extrem']
repeatUpTickVolBlwAsk_bigStand_extrem = df['repeatUpTickVolBlwAskDesc_bigStand_extrem'] + df['repeatUpTickVolBlwAskAsc_bigStand_extrem']
repeatUpTickVolBlwBid_bigStand_extrem = df['repeatUpTickVolBlwBidDesc_bigStand_extrem'] + df['repeatUpTickVolBlwBidAsc_bigStand_extrem']
repeatDownTickVolBlwAsk_bigStand_extrem = df['repeatDownTickVolBlwAskDesc_bigStand_extrem'] + df['repeatDownTickVolBlwAskAsc_bigStand_extrem']
repeatDownTickVolBlwBid_bigStand_extrem = df['repeatDownTickVolBlwBidDesc_bigStand_extrem'] + df['repeatDownTickVolBlwBidAsc_bigStand_extrem']

# Calcul des nouvelles features
# BigStand - Bearish (zone Abv)
features_df['bearish_askBigStand_abs_ratio_abv'] = np.where(
    (upTickVolAbvAsk + repeatUpTickVolAbvAsk + repeatDownTickVolAbvAsk) != 0,
    (upTickVolAbvAsk_bigStand + repeatUpTickVolAbvAsk_bigStand + repeatDownTickVolAbvAsk_bigStand) /
    (upTickVolAbvAsk + repeatUpTickVolAbvAsk + repeatDownTickVolAbvAsk),
    addDivBy0)

features_df['bearish_bidBigStand_abs_ratio_abv'] = np.where(
    (upTickVolAbvBid + repeatUpTickVolAbvBid + repeatDownTickVolAbvBid) != 0,
    (upTickVolAbvBid_bigStand + repeatUpTickVolAbvBid_bigStand + repeatDownTickVolAbvBid_bigStand) /
    (upTickVolAbvBid + repeatUpTickVolAbvBid + repeatDownTickVolAbvBid),
    addDivBy0)

features_df['bearish_bigStand_abs_diff_abv'] = np.where(
    df['VolAbv'] != 0,
    ((upTickVolAbvAsk_bigStand + repeatUpTickVolAbvAsk_bigStand + repeatDownTickVolAbvAsk_bigStand) -
     (upTickVolAbvBid_bigStand + repeatUpTickVolAbvBid_bigStand + repeatDownTickVolAbvBid_bigStand)) / df['VolAbv'],
    diffDivBy0)

# BigStand - Bullish (zone Blw)
features_df['bullish_askBigStand_abs_ratio_blw'] = np.where(
    (upTickVolBlwAsk + repeatUpTickVolBlwAsk + repeatDownTickVolBlwAsk) != 0,
    (upTickVolBlwAsk_bigStand + repeatUpTickVolBlwAsk_bigStand + repeatDownTickVolBlwAsk_bigStand) /
    (upTickVolBlwAsk + repeatUpTickVolBlwAsk + repeatDownTickVolBlwAsk),
    addDivBy0)

features_df['bullish_bidBigStand_abs_ratio_blw'] = np.where(
    (upTickVolBlwBid + repeatUpTickVolBlwBid + repeatDownTickVolBlwBid) != 0,
    (upTickVolBlwBid_bigStand + repeatUpTickVolBlwBid_bigStand + repeatDownTickVolBlwBid_bigStand) /
    (upTickVolBlwBid + repeatUpTickVolBlwBid + repeatDownTickVolBlwBid),
    addDivBy0)

features_df['bullish_bigStand_abs_diff_blw'] = np.where(
    df['VolBlw'] != 0,
    ((upTickVolBlwAsk_bigStand + repeatUpTickVolBlwAsk_bigStand + repeatDownTickVolBlwAsk_bigStand) -
     (upTickVolBlwBid_bigStand + repeatUpTickVolBlwBid_bigStand + repeatDownTickVolBlwBid_bigStand)) / df['VolBlw'],
    diffDivBy0)

# BigHigh - Bearish (zone Abv)
features_df['bearish_askBigHigh_abs_ratio_abv'] = np.where(
    (upTickVolAbvAsk + repeatUpTickVolAbvAsk + repeatDownTickVolAbvAsk) != 0,
    (upTickVolAbvAsk_bigHigh + repeatUpTickVolAbvAsk_bigHigh + repeatDownTickVolAbvAsk_bigHigh) /
    (upTickVolAbvAsk + repeatUpTickVolAbvAsk + repeatDownTickVolAbvAsk),
    addDivBy0)

features_df['bearish_bidBigHigh_abs_ratio_abv'] = np.where(
    (upTickVolAbvBid + repeatUpTickVolAbvBid + repeatDownTickVolAbvBid) != 0,
    (upTickVolAbvBid_bigHigh + repeatUpTickVolAbvBid_bigHigh + repeatDownTickVolAbvBid_bigHigh) /
    (upTickVolAbvBid + repeatUpTickVolAbvBid + repeatDownTickVolAbvBid),
    addDivBy0)

features_df['bearish_bigHigh_abs_diff_abv'] = np.where(
    df['VolAbv'] != 0,
    ((upTickVolAbvAsk_bigHigh + repeatUpTickVolAbvAsk_bigHigh + repeatDownTickVolAbvAsk_bigHigh) -
     (upTickVolAbvBid_bigHigh + repeatUpTickVolAbvBid_bigHigh + repeatDownTickVolAbvBid_bigHigh)) / df['VolAbv'],
    diffDivBy0)

# BigHigh - Bullish (zone Blw)
features_df['bullish_askBigHigh_abs_ratio_blw'] = np.where(
    (upTickVolBlwAsk + repeatUpTickVolBlwAsk + repeatDownTickVolBlwAsk) != 0,
    (upTickVolBlwAsk_bigHigh + repeatUpTickVolBlwAsk_bigHigh + repeatDownTickVolBlwAsk_bigHigh) /
    (upTickVolBlwAsk + repeatUpTickVolBlwAsk + repeatDownTickVolBlwAsk),
    addDivBy0)

features_df['bullish_bidBigHigh_abs_ratio_blw'] = np.where(
    (upTickVolBlwBid + repeatUpTickVolBlwBid + repeatDownTickVolBlwBid) != 0,
    (upTickVolBlwBid_bigHigh + repeatUpTickVolBlwBid_bigHigh + repeatDownTickVolBlwBid_bigHigh) /
    (upTickVolBlwBid + repeatUpTickVolBlwBid + repeatDownTickVolBlwBid),
    addDivBy0)

features_df['bullish_bigHigh_abs_diff_blw'] = np.where(
    df['VolBlw'] != 0,
    ((upTickVolBlwAsk_bigHigh + repeatUpTickVolBlwAsk_bigHigh + repeatDownTickVolBlwAsk_bigHigh) -
     (upTickVolBlwBid_bigHigh + repeatUpTickVolBlwBid_bigHigh + repeatDownTickVolBlwBid_bigHigh)) / df['VolBlw'],
    diffDivBy0)

# a) Intensité du renversement dans la zone extrême
features_df['bearish_extrem_revIntensity_ratio'] = np.where(
    df['VolAbv'] != 0,
    ((upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem) -
     (downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem)) / df['VolAbv'],
    diffDivBy0)
features_df['bullish_extrem_revIntensity_ratio'] = np.where(
    df['VolBlw'] != 0,
    ((upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem) -
     (downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem)) / df['VolBlw'],
    diffDivBy0)

# f) Ratio de volume dans la zone extrême par rapport à la zone de renversement
features_df['bearish_extrem_zone_volume_ratio'] = np.where(
    df['VolAbv'] != 0,
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatUpTickVolAbvBid_extrem +
     downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatDownTickVolAbvAsk_extrem) / df['VolAbv'],
    diffDivBy0)
features_df['bullish_extrem_zone_volume_ratio'] = np.where(
    df['VolBlw'] != 0,
    (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatUpTickVolBlwBid_extrem +
     downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatDownTickVolBlwAsk_extrem) / df['VolBlw'],
    diffDivBy0)

# b) Pression acheteur/vendeur dans la zone extrême
features_df['bearish_extrem_pressure_ratio'] = np.where(
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem) != 0,
    (downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem) /
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem),
    addDivBy0)
features_df['bullish_extrem_pressure_ratio'] = np.where(
    (downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem) != 0,
    (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem) /
    (downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem),
    addDivBy0)

# c) Absorption dans la zone extrême
features_df['bearish_extrem_abs_ratio'] = np.where(
    (downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem) != 0,
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem) /
    (downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem),
    addDivBy0)
features_df['bullish_extrem_abs_ratio'] = np.where(
    (downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem) != 0,
    (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem) /
    (downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem),
    addDivBy0)

# i) Comparaison de l'activité extrême vs. reste de la zone de renversement
features_df['bearish_extrem_vs_rest_activity'] = np.where(
    (df['VolAbv'] - (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
                     downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem)) != 0,
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
     downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem) /
    (df['VolAbv'] - (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
                     downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem)),
    addDivBy0)
features_df['bullish_extrem_vs_rest_activity'] = np.where(
    (df['VolBlw'] - (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
                     downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem)) != 0,
    (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
     downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem) /
    (df['VolBlw'] - (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
                     downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem)),
    addDivBy0)

# j) Indicateur de continuation vs. renversement dans la zone extrême
features_df['bearish_continuation_vs_reversal'] = np.where(
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
     downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem) != 0,
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem) /
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
     downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem),
    addDivBy0)
features_df['bullish_continuation_vs_reversal'] = np.where(
    (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
     downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem) != 0,
    (downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem) /
    (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
     downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem),
    addDivBy0)

# k) Ratio de repeat ticks dans la zone extrême
features_df['bearish_repeat_ticks_ratio'] = np.where(
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
     downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem) != 0,
    (repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
     repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem) /
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
     downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem),
    addDivBy0)

features_df['bullish_repeat_ticks_ratio'] = np.where(
    (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
     downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem) != 0,
    (repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
     repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem) /
    (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
     downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem),
    addDivBy0)

# l) Big trades dans la zone extrême
# Pour les bougies bearish (zone Abv)


features_df['bearish_big_trade_ratio_extrem'] = np.where(
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
     downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem) != 0,
    (upTickVolAbvAsk_bigStand_extrem + repeatUpTickVolAbvAsk_bigStand_extrem + repeatDownTickVolAbvAsk_bigStand_extrem +
     downTickVolAbvBid_bigStand_extrem + repeatDownTickVolAbvBid_bigStand_extrem + repeatUpTickVolAbvBid_bigStand_extrem) /
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
     downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem),
    addDivBy0)

features_df['bearish_big_trade_imbalance'] = np.where(
    (upTickVolAbvAsk_bigStand_extrem + repeatUpTickVolAbvAsk_bigStand_extrem + repeatDownTickVolAbvAsk_bigStand_extrem +
     downTickVolAbvBid_bigStand_extrem + repeatDownTickVolAbvBid_bigStand_extrem + repeatUpTickVolAbvBid_bigStand_extrem) != 0,
    ((upTickVolAbvAsk_bigStand_extrem + repeatUpTickVolAbvAsk_bigStand_extrem + repeatDownTickVolAbvAsk_bigStand_extrem) -
     (downTickVolAbvBid_bigStand_extrem + repeatDownTickVolAbvBid_bigStand_extrem + repeatUpTickVolAbvBid_bigStand_extrem)) /
    (upTickVolAbvAsk_bigStand_extrem + repeatUpTickVolAbvAsk_bigStand_extrem + repeatDownTickVolAbvAsk_bigStand_extrem +
     downTickVolAbvBid_bigStand_extrem + repeatDownTickVolAbvBid_bigStand_extrem + repeatUpTickVolAbvBid_bigStand_extrem),
    diffDivBy0)

# Pour les bougies bullish (zone Blw)



features_df['bullish_big_trade_ratio_extrem'] = np.where(
    (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
     downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem) != 0,
    (upTickVolBlwAsk_bigStand_extrem + repeatUpTickVolBlwAsk_bigStand_extrem + repeatDownTickVolBlwAsk_bigStand_extrem +
     downTickVolBlwBid_bigStand_extrem + repeatDownTickVolBlwBid_bigStand_extrem + repeatUpTickVolBlwBid_bigStand_extrem) /
    (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
     downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem),
    addDivBy0)

features_df['bullish_big_trade_imbalance'] = np.where(
    (upTickVolBlwAsk_bigStand_extrem + repeatUpTickVolBlwAsk_bigStand_extrem + repeatDownTickVolBlwAsk_bigStand_extrem +
     downTickVolBlwBid_bigStand_extrem + repeatDownTickVolBlwBid_bigStand_extrem + repeatUpTickVolBlwBid_bigStand_extrem) != 0,
    ((upTickVolBlwAsk_bigStand_extrem + repeatUpTickVolBlwAsk_bigStand_extrem + repeatDownTickVolBlwAsk_bigStand_extrem) -
     (downTickVolBlwBid_bigStand_extrem + repeatDownTickVolBlwBid_bigStand_extrem + repeatUpTickVolBlwBid_bigStand_extrem)) /
    (upTickVolBlwAsk_bigStand_extrem + repeatUpTickVolBlwAsk_bigStand_extrem + repeatDownTickVolBlwAsk_bigStand_extrem +
     downTickVolBlwBid_bigStand_extrem + repeatDownTickVolBlwBid_bigStand_extrem + repeatUpTickVolBlwBid_bigStand_extrem),
    diffDivBy0)

# Nouvelles features exploitant ASC et DSC
print_notification("Calcul des nouvelles features ASC/DSC")

# 1. Dynamique ASC/DSC dans la zone de renversement
features_df['bearish_asc_dsc_ratio'] = np.where(
    (df['downTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidDesc']) != 0,
    (df['upTickVolAbvAskAsc'] + df['repeatUpTickVolAbvAskAsc'] + df['repeatDownTickVolAbvAskAsc']) /
    (df['downTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidDesc']),
    addDivBy0)

features_df['bearish_asc_dynamics'] = np.where(
    df['VolAbv'] != 0,
    (df['upTickVolAbvAskAsc'] + df['repeatUpTickVolAbvAskAsc'] + df['repeatDownTickVolAbvAskAsc']) / df['VolAbv'],
    addDivBy0)

features_df['bearish_dsc_dynamics'] = np.where(
    df['VolAbv'] != 0,
    (df['downTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidDesc']) / df['VolAbv'],
    addDivBy0)

features_df['bullish_asc_dsc_ratio'] = np.where(
    (df['downTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidDesc']) != 0,
    (df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] + df['repeatDownTickVolBlwAskAsc']) /
    (df['downTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidDesc']),
    addDivBy0)

features_df['bullish_asc_dynamics'] = np.where(
    df['VolBlw'] != 0,
    (df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] + df['repeatDownTickVolBlwAskAsc']) / df['VolBlw'],
    addDivBy0)

features_df['bullish_dsc_dynamics'] = np.where(
    df['VolBlw'] != 0,
    (df['downTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidDesc']) / df['VolBlw'],
    addDivBy0)

# 2. Déséquilibre Ask-Bid dans les phases ASC et DSC
print_notification("Calcul des features de déséquilibre Ask-Bid dans les phases ASC et DSC")

features_df['bearish_asc_ask_bid_imbalance'] = np.where(
    df['VolAbv'] != 0,
    (df['upTickVolAbvAskAsc'] + df['repeatUpTickVolAbvAskAsc'] + df['repeatDownTickVolAbvAskAsc']) / df['VolAbv'],
    addDivBy0)

features_df['bearish_dsc_ask_bid_imbalance'] = np.where(
    df['VolAbv'] != 0,
    (df['downTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidDesc']) / df['VolAbv'],
    addDivBy0)

features_df['bearish_imbalance_evolution'] = np.where(
    (df['VolAbv'] != 0),
    features_df['bearish_asc_ask_bid_imbalance'] - features_df['bearish_dsc_ask_bid_imbalance'],
    diffDivBy0)

features_df['bearish_asc_ask_bid_delta_imbalance'] = np.where(
    df['VolAbv'] != 0,
    (df['upTickVolAbvAskAsc'] + df['repeatUpTickVolAbvAskAsc'] + df['repeatDownTickVolAbvAskAsc'] -
     (df['upTickVolAbvBidAsc'] + df['repeatUpTickVolAbvBidAsc'] + df['repeatDownTickVolAbvBidAsc'])) / df['VolAbv'],
    diffDivBy0)

features_df['bearish_dsc_ask_bid_delta_imbalance'] = np.where(
    df['VolAbv'] != 0,
    (df['upTickVolAbvAskDesc'] + df['repeatUpTickVolAbvAskDesc'] + df['repeatDownTickVolAbvAskDesc'] -
     (df['downTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidDesc'])) / df['VolAbv'],
    diffDivBy0)

features_df['bullish_asc_ask_bid_imbalance'] = np.where(
    df['VolBlw'] != 0,
    (df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] + df['repeatDownTickVolBlwAskAsc']) / df['VolBlw'],
    addDivBy0)

features_df['bullish_dsc_ask_bid_imbalance'] = np.where(
    df['VolBlw'] != 0,
    (df['downTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidDesc']) / df['VolBlw'],
    addDivBy0)

features_df['bullish_imbalance_evolution'] = np.where(
    (df['VolBlw'] != 0) ,
    features_df['bullish_asc_ask_bid_imbalance'] - features_df['bullish_dsc_ask_bid_imbalance'],
    diffDivBy0)

features_df['bullish_asc_ask_bid_delta_imbalance'] = np.where(
    df['VolBlw'] != 0,
    (df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] + df['repeatDownTickVolBlwAskAsc'] -
     (df['upTickVolBlwBidAsc'] + df['repeatUpTickVolBlwBidAsc'] + df['repeatDownTickVolBlwBidAsc'])) / df['VolBlw'],
    diffDivBy0)

features_df['bullish_dsc_ask_bid_delta_imbalance'] = np.where(
    df['VolBlw'] != 0,
    (df['upTickVolBlwAskDesc'] + df['repeatUpTickVolBlwAskDesc'] + df['repeatDownTickVolBlwAskDesc'] -
     (df['downTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidDesc'])) / df['VolBlw'],
    diffDivBy0)

# 3. Importance et dynamique de la zone extrême
print_notification("Calcul des features de la zone extrême")

# Features bearish
features_df['extrem_asc_ratio_bearish'] = np.where(
    df['VolAbv'] != 0,
    (df['upTickVolAbvAskAsc_extrem'] + df['repeatUpTickVolAbvAskAsc_extrem'] + df['repeatDownTickVolAbvAskAsc_extrem']) / df['VolAbv'],
    addDivBy0)

features_df['extrem_dsc_ratio_bearish'] = np.where(
    df['VolAbv'] != 0,
    (df['downTickVolAbvBidDesc_extrem'] + df['repeatUpTickVolAbvBidDesc_extrem'] + df['repeatDownTickVolAbvBidDesc_extrem']) / df['VolAbv'],
    addDivBy0)

features_df['extrem_zone_significance_bearish'] = np.where(
    ( df['VolAbv'] != 0),
    (features_df['extrem_asc_ratio_bearish'] + features_df['extrem_dsc_ratio_bearish']) *
    abs(features_df['extrem_asc_ratio_bearish'] - features_df['extrem_dsc_ratio_bearish']),
    diffDivBy0)

features_df['extrem_ask_bid_imbalance_bearish'] = np.where(
    (df['upTickVolAbvAskAsc_extrem'] + df['repeatUpTickVolAbvAskAsc_extrem'] + df['repeatDownTickVolAbvAskAsc_extrem'] +
     df['downTickVolAbvBidDesc_extrem'] + df['repeatUpTickVolAbvBidDesc_extrem'] + df['repeatDownTickVolAbvBidDesc_extrem']) != 0,
    (df['upTickVolAbvAskAsc_extrem'] + df['repeatUpTickVolAbvAskAsc_extrem'] + df['repeatDownTickVolAbvAskAsc_extrem'] -
     (df['downTickVolAbvBidDesc_extrem'] + df['repeatUpTickVolAbvBidDesc_extrem'] + df['repeatDownTickVolAbvBidDesc_extrem'])) /
    (df['upTickVolAbvAskAsc_extrem'] + df['repeatUpTickVolAbvAskAsc_extrem'] + df['repeatDownTickVolAbvAskAsc_extrem'] +
     df['downTickVolAbvBidDesc_extrem'] + df['repeatUpTickVolAbvBidDesc_extrem'] + df['repeatDownTickVolAbvBidDesc_extrem']),
    diffDivBy0)

features_df['extrem_asc_dsc_comparison_bearish'] = np.where(
    ( df['VolAbv'] != 0),
    features_df['extrem_asc_ratio_bearish'] / features_df['extrem_dsc_ratio_bearish'],
    addDivBy0)

features_df['bearish_repeat_ticks_ratio'] = np.where(
    (upTickVolAbvAsk_extrem + downTickVolAbvBid_extrem + repeatUpTickVolAbvAsk_extrem +
     repeatDownTickVolAbvAsk_extrem + repeatUpTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem) != 0,
    (repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
     repeatUpTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem) /
    (upTickVolAbvAsk_extrem + downTickVolAbvBid_extrem + repeatUpTickVolAbvAsk_extrem +
     repeatDownTickVolAbvAsk_extrem + repeatUpTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem),
    addDivBy0)

# Features bullish
features_df['extrem_asc_ratio_bullish'] = np.where(
    df['VolBlw'] != 0,
    (df['upTickVolBlwAskAsc_extrem'] + df['repeatUpTickVolBlwAskAsc_extrem'] + df['repeatDownTickVolBlwAskAsc_extrem']) / df['VolBlw'],
    addDivBy0)

features_df['extrem_dsc_ratio_bullish'] = np.where(
    df['VolBlw'] != 0,
    (df['downTickVolBlwBidDesc_extrem'] + df['repeatUpTickVolBlwBidDesc_extrem'] + df['repeatDownTickVolBlwBidDesc_extrem']) / df['VolBlw'],
    addDivBy0)

features_df['extrem_zone_significance_bullish'] = np.where(
    ( df['VolBlw'] != 0),
    (features_df['extrem_asc_ratio_bullish'] + features_df['extrem_dsc_ratio_bullish']) *
    abs(features_df['extrem_asc_ratio_bullish'] - features_df['extrem_dsc_ratio_bullish']),
    diffDivBy0)

features_df['extrem_ask_bid_imbalance_bullish'] = np.where(
    (df['upTickVolBlwAskAsc_extrem'] + df['repeatUpTickVolBlwAskAsc_extrem'] + df['repeatDownTickVolBlwAskAsc_extrem'] +
     df['downTickVolBlwBidDesc_extrem'] + df['repeatUpTickVolBlwBidDesc_extrem'] + df['repeatDownTickVolBlwBidDesc_extrem']) != 0,
    (df['upTickVolBlwAskAsc_extrem'] + df['repeatUpTickVolBlwAskAsc_extrem'] + df['repeatDownTickVolBlwAskAsc_extrem'] -
     (df['downTickVolBlwBidDesc_extrem'] + df['repeatUpTickVolBlwBidDesc_extrem'] + df['repeatDownTickVolBlwBidDesc_extrem'])) /
    (df['upTickVolBlwAskAsc_extrem'] + df['repeatUpTickVolBlwAskAsc_extrem'] + df['repeatDownTickVolBlwAskAsc_extrem'] +
     df['downTickVolBlwBidDesc_extrem'] + df['repeatUpTickVolBlwBidDesc_extrem'] + df['repeatDownTickVolBlwBidDesc_extrem']),
    diffDivBy0)

features_df['extrem_asc_dsc_comparison_bullish'] = np.where(
    ( df['VolBlw'] != 0),
    features_df['extrem_asc_ratio_bullish'] / features_df['extrem_dsc_ratio_bullish'],
    addDivBy0)

features_df['bullish_repeat_ticks_ratio'] = np.where(
    (upTickVolBlwAsk_extrem + downTickVolBlwBid_extrem + repeatUpTickVolBlwAsk_extrem +
     repeatDownTickVolBlwAsk_extrem + repeatUpTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem) != 0,
    (repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
     repeatUpTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem) /
    (upTickVolBlwAsk_extrem + downTickVolBlwBid_extrem + repeatUpTickVolBlwAsk_extrem +
     repeatDownTickVolBlwAsk_extrem + repeatUpTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem),
    addDivBy0)

features_df['bearish_absorption_ratio'] = np.where(
    (df['upTickVolAbvAskAsc'] + df['repeatUpTickVolAbvAskAsc'] + df['repeatDownTickVolAbvAskAsc']) != 0,
    (df['downTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidDesc']) /
    (df['upTickVolAbvAskAsc'] + df['repeatUpTickVolAbvAskAsc'] + df['repeatDownTickVolAbvAskAsc']),
    addDivBy0)

features_df['bullish_absorption_ratio'] = np.where(
    (df['downTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidDesc']) != 0,
    (df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] + df['repeatDownTickVolBlwAskAsc']) /
    (df['downTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidDesc']),
    addDivBy0)

# 5. Dynamique des gros trades dans la zone extrême
print_notification("Calcul des features de dynamique des gros trades dans la zone extrême")

# Pour les bougies bearish (zone Abv)


features_df['bearish_big_trade_ratio2_extrem'] = np.where(
    (upTickVolAbvAsk_extrem + downTickVolAbvBid_extrem) != 0,
    (upTickVolAbvAsk_bigStand_extrem + repeatUpTickVolAbvAsk_bigStand_extrem + repeatDownTickVolAbvAsk_bigStand_extrem +
     downTickVolAbvBid_bigStand_extrem + repeatUpTickVolAbvBid_bigStand_extrem + repeatDownTickVolAbvBid_bigStand_extrem) /
    (upTickVolAbvAsk_extrem + downTickVolAbvBid_extrem),
    addDivBy0)
# Pour les bougies bullish (zone Blw)
features_df['bullish_big_trade_ratio2_extrem'] = np.where(
    (upTickVolBlwAsk_extrem + downTickVolBlwBid_extrem) != 0,
    (upTickVolBlwAsk_bigStand_extrem + repeatUpTickVolBlwAsk_bigStand_extrem + repeatDownTickVolBlwAsk_bigStand_extrem +
     downTickVolBlwBid_bigStand_extrem + repeatUpTickVolBlwBid_bigStand_extrem + repeatDownTickVolBlwBid_bigStand_extrem) /
    (upTickVolBlwAsk_extrem + downTickVolBlwBid_extrem),
    addDivBy0)

# a. Pour les bougies bearish (zone Abv)
upTickCountAbvAsk=df['upTickCountAbvAskAsc']+df['upTickCountAbvAskDesc']
downTickCountAbvAsk=df['downTickCountAbvAskAsc']+df['downTickCountAbvAskDesc']
repeatUpTickCountAbvAsk=df['repeatUpTickCountAbvAskAsc']+df['repeatUpTickCountAbvAskDesc']
repeatDownTickCountAbvAsk=df['repeatDownTickCountAbvAskAsc']+df['repeatDownTickCountAbvAskDesc']

upTickCountAbvBid=df['upTickCountAbvBidAsc']+df['upTickCountAbvBidDesc']
downTickCountAbvBid=df['downTickCountAbvBidAsc']+df['downTickCountAbvBidDesc']
repeatUpTickCountAbvBid=df['repeatUpTickCountAbvBidAsc']+df['repeatUpTickCountAbvBidDesc']
repeatDownTickCountAbvBid=df['repeatDownTickCountAbvBidAsc']+df['repeatDownTickCountAbvBidDesc']

features_df['total_count_abv'] = (
    upTickCountAbvAsk + downTickCountAbvAsk +
    repeatUpTickCountAbvAsk + repeatDownTickCountAbvAsk +
    upTickCountAbvBid + downTickCountAbvBid +
    repeatUpTickCountAbvBid + repeatDownTickCountAbvBid
)

features_df['absorption_intensity_repeat_bearish_vol'] = np.where(
    df['VolAbv'] != 0,
    (repeatUpTickVolAbvAsk + repeatDownTickVolAbvAsk) / df['VolAbv'],
    addDivBy0)

features_df['absorption_intensity_repeat_bearish_count'] = np.where(
    features_df['total_count_abv'] != 0,
    (repeatUpTickCountAbvAsk + repeatDownTickCountAbvAsk) / features_df['total_count_abv'],
    addDivBy0)

features_df['bearish_repeatAskBid_ratio'] = np.where(
    df['VolAbv'] != 0,
    (repeatUpTickVolAbvAsk + repeatUpTickVolAbvBid + repeatDownTickVolAbvAsk + repeatDownTickVolAbvBid) / df['VolAbv'],
    addDivBy0)



# b. Pour les bougies bullish (zone Blw)
upTickCountBlwAsk=df['upTickCountBlwAskAsc']+df['upTickCountBlwAskDesc']
downTickCountBlwAsk=df['downTickCountBlwAskAsc']+df['downTickCountBlwAskDesc']
repeatUpTickCountBlwAsk=df['repeatUpTickCountBlwAskAsc']+df['repeatUpTickCountBlwAskDesc']
repeatDownTickCountBlwAsk=df['repeatDownTickCountBlwAskAsc']+df['repeatDownTickCountBlwAskDesc']

upTickCountBlwBid=df['upTickCountBlwBidAsc']+df['upTickCountBlwBidDesc']
downTickCountBlwBid=df['downTickCountBlwBidAsc']+df['downTickCountBlwBidDesc']
repeatUpTickCountBlwBid=df['repeatUpTickCountBlwBidAsc']+df['repeatUpTickCountBlwBidDesc']
repeatDownTickCountBlwBid=df['repeatDownTickCountBlwBidAsc']+df['repeatDownTickCountBlwBidDesc']

features_df['total_count_blw'] = (
    upTickCountBlwAsk + downTickCountBlwAsk +
    repeatUpTickCountBlwAsk + repeatDownTickCountBlwAsk +
    upTickCountBlwBid + downTickCountBlwBid +
    repeatUpTickCountBlwBid + repeatDownTickCountBlwBid
)

features_df['absorption_intensity_repeat_bullish_vol'] = np.where(
    df['VolBlw'] != 0,
    (repeatUpTickVolBlwBid + repeatDownTickVolBlwBid) / df['VolBlw'],
    addDivBy0)

features_df['absorption_intensity_repeat_bullish_count'] = np.where(
    features_df['total_count_blw'] != 0,
    (repeatUpTickCountBlwBid + repeatDownTickCountBlwBid) / features_df['total_count_blw'],
    addDivBy0)

features_df['bullish_repeatAskBid_ratio'] = np.where(
    df['VolBlw'] != 0,
    (repeatUpTickVolBlwAsk + repeatUpTickVolBlwBid + repeatDownTickVolBlwAsk + repeatDownTickVolBlwBid) / df['VolBlw'],
    addDivBy0)


features_df['ratio_Count_AbvBlw'] = np.where(
    features_df['total_count_blw'] != 0,
    features_df['total_count_abv']/ features_df['total_count_blw'],
    addDivBy0)
print_notification("Calcul des features de la zone 6Ticks")



# a) Ratio de volume _6Tick
features_df['bearish_volume_ratio_6Tick'] = np.where(df['VolAbv'] != 0, df['VolBlw_6Tick'] / df['VolAbv'], addDivBy0)
features_df['bullish_volume_ratio_6Tick'] = np.where(df['VolBlw'] != 0, df['VolAbv_6Tick'] / df['VolBlw'], addDivBy0)

# b) Delta _6Tick
features_df['bearish_relatif_ratio_6Tick'] = np.where(df['VolBlw_6Tick'] != 0, df['DeltaBlw_6Tick'] / df['VolBlw_6Tick'], diffDivBy0)
features_df['bullish_relatif_ratio_6Tick'] = np.where(df['VolAbv_6Tick'] != 0, df['DeltaAbv_6Tick'] / df['VolAbv_6Tick'], diffDivBy0)

# c) Delta relatif _6Tick
features_df['bearish_relatifDelta_ratio_6Tick'] = np.where(
    (df['VolBlw_6Tick'] != 0) & (df['volume'] != 0) & (df['delta'] != 0),
    (df['DeltaBlw_6Tick'] / df['VolBlw_6Tick']) / (df['delta'] / df['volume']),
    addDivBy0)

features_df['bullish_relatifDelta_ratio_6Tick'] = np.where(
    (df['VolAbv_6Tick'] != 0) & (df['volume'] != 0) & (df['delta'] != 0),
    (df['DeltaAbv_6Tick'] / df['VolAbv_6Tick']) / (df['delta'] / df['volume']),
    addDivBy0)

# d) Pression acheteur dans la zone _6Tick
features_df['bearish_buyer_pressure_6Tick'] = np.where(df['VolBlw_6Tick'] != 0,
    (df['upTickVol6TicksBlwAsk'] + df['repeatUpTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwAsk']) / df['VolBlw_6Tick'], addDivBy0)
features_df['bullish_buyer_pressure_6Tick'] = np.where(df['VolAbv_6Tick'] != 0,
    (df['upTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvAsk'] + df['repeatDownTickVol6TicksAbvAsk']) / df['VolAbv_6Tick'], addDivBy0)

# e) Pression vendeur dans la zone _6Tick
features_df['bearish_seller_pressure_6Tick'] = np.where(df['VolBlw_6Tick'] != 0,
    (df['downTickVol6TicksBlwBid'] + df['repeatDownTickVol6TicksBlwBid'] + df['repeatUpTickVol6TicksBlwBid']) / df['VolBlw_6Tick'], addDivBy0)
features_df['bullish_seller_pressure_6Tick'] = np.where(df['VolAbv_6Tick'] != 0,
    (df['downTickVol6TicksAbvBid'] + df['repeatDownTickVol6TicksAbvBid'] + df['repeatUpTickVol6TicksAbvBid']) / df['VolAbv_6Tick'], addDivBy0)

# f) Absorption dans la zone _6Tick
features_df['bearish_absorption_6Tick'] = np.where(
    (df['downTickVol6TicksBlwBid'] + df['repeatDownTickVol6TicksBlwBid'] + df['repeatUpTickVol6TicksBlwBid']) != 0,
    (df['upTickVol6TicksBlwAsk'] + df['repeatUpTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwAsk']) /
    (df['downTickVol6TicksBlwBid'] + df['repeatDownTickVol6TicksBlwBid'] + df['repeatUpTickVol6TicksBlwBid']), addDivBy0)
features_df['bullish_absorption_6Tick'] = np.where(
    (df['upTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvAsk'] + df['repeatDownTickVol6TicksAbvAsk']) != 0,
    (df['downTickVol6TicksAbvBid'] + df['repeatDownTickVol6TicksAbvBid'] + df['repeatUpTickVol6TicksAbvBid']) /
    (df['upTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvAsk'] + df['repeatDownTickVol6TicksAbvAsk']), addDivBy0)

# g) Ratio de repeat ticks _6Tick
features_df['bearish_repeat_ticks_ratio_6Tick'] = np.where(df['VolBlw_6Tick'] != 0,
    (df['repeatUpTickVol6TicksBlwAsk'] + df['repeatUpTickVol6TicksBlwBid'] +
     df['repeatDownTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwBid']) / df['VolBlw_6Tick'], addDivBy0)
features_df['bullish_repeat_ticks_ratio_6Tick'] = np.where(df['VolAbv_6Tick'] != 0,
    (df['repeatUpTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvBid'] +
     df['repeatDownTickVol6TicksAbvAsk'] + df['repeatDownTickVol6TicksAbvBid']) / df['VolAbv_6Tick'], addDivBy0)

# h) Comparaison de la dynamique de prix
upTickVol6TicksAbv=df['upTickVol6TicksAbvBid']+df['upTickVol6TicksAbvAsk']
downTickVol6TicksAbv=df['downTickVol6TicksAbvBid']+df['downTickVol6TicksAbvAsk']

upTickVol6TicksBlw=df['upTickVol6TicksBlwAsk']+df['upTickVol6TicksBlwBid']
downTickVol6TicksBlw=df['downTickVol6TicksBlwAsk']+df['downTickVol6TicksBlwBid']

features_df['bearish_price_dynamics_comparison_6Tick'] = np.where(
    (upTickVol6TicksBlw + downTickVol6TicksBlw) != 0,
    (df['upTickVol6TicksBlwAsk'] + df['upTickVol6TicksBlwBid'] - df['downTickVol6TicksBlwAsk'] - df['downTickVol6TicksBlwBid']) /
    (upTickVol6TicksBlw + downTickVol6TicksBlw), diffDivBy0)
features_df['bullish_price_dynamics_comparison_6Tick'] = np.where(
    (upTickVol6TicksAbv + downTickVol6TicksAbv) != 0,
    (df['downTickVol6TicksAbvAsk'] + df['downTickVol6TicksAbvBid'] - df['upTickVol6TicksAbvAsk'] - df['upTickVol6TicksAbvBid']) /
    (upTickVol6TicksAbv + downTickVol6TicksAbv), diffDivBy0)

# i) Ratio d'activité Ask vs Bid dans la zone _6Tick
features_df['bearish_activity_bid_ask_ratio_6Tick'] = np.where(
    (df['upTickVol6TicksBlwAsk'] + df['downTickVol6TicksBlwAsk'] + df['repeatUpTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwAsk']) != 0,
    (df['upTickVol6TicksBlwBid'] + df['downTickVol6TicksBlwBid'] + df['repeatUpTickVol6TicksBlwBid'] + df['repeatDownTickVol6TicksBlwBid']) /
    (df['upTickVol6TicksBlwAsk'] + df['downTickVol6TicksBlwAsk'] + df['repeatUpTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwAsk']), addDivBy0)
features_df['bullish_activity_ask_bid_ratio_6Tick'] = np.where(
    (df['upTickVol6TicksAbvBid'] + df['downTickVol6TicksAbvBid'] + df['repeatUpTickVol6TicksAbvBid'] + df['repeatDownTickVol6TicksAbvBid']) != 0,
    (df['upTickVol6TicksAbvAsk'] + df['downTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvAsk'] + df['repeatDownTickVol6TicksAbvAsk']) /
    (df['upTickVol6TicksAbvBid'] + df['downTickVol6TicksAbvBid'] + df['repeatUpTickVol6TicksAbvBid'] + df['repeatDownTickVol6TicksAbvBid']), addDivBy0)

# j) Déséquilibre des repeat ticks
features_df['bearish_repeat_ticks_imbalance_6Tick'] = np.where(df['VolBlw_6Tick'] != 0,
    (df['repeatDownTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwBid'] -
     df['repeatUpTickVol6TicksBlwAsk'] - df['repeatUpTickVol6TicksBlwBid']) / df['VolBlw_6Tick'], diffDivBy0)
features_df['bullish_repeat_ticks_imbalance_6Tick'] = np.where(df['VolAbv_6Tick'] != 0,
    (df['repeatUpTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvBid'] -
     df['repeatDownTickVol6TicksAbvAsk'] - df['repeatDownTickVol6TicksAbvBid']) / df['VolAbv_6Tick'], diffDivBy0)

# k) Déséquilibre global
features_df['bearish_ticks_imbalance_6Tick'] = np.where(df['VolBlw_6Tick'] != 0,
    (df['downTickVol6TicksBlwBid'] + df['repeatDownTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwBid'] -
     df['upTickVol6TicksBlwAsk'] - df['repeatUpTickVol6TicksBlwAsk'] - df['repeatUpTickVol6TicksBlwBid']) / df['VolBlw_6Tick'], diffDivBy0)
features_df['bullish_ticks_imbalance_6Tick'] = np.where(df['VolAbv_6Tick'] != 0,
    (df['upTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvBid'] -
     df['downTickVol6TicksAbvBid'] - df['repeatDownTickVol6TicksAbvAsk'] - df['repeatDownTickVol6TicksAbvBid']) / df['VolAbv_6Tick'], diffDivBy0)

print_notification("Ajout des informations sur les class et les trades")

features_df['class_binaire']=df['class_binaire']
features_df['date']=df['date']
features_df['trade_category']=df['trade_category']

# Enregistrement des fichiers
print_notification("Début de l'enregistrement des fichiers")

# Extraire le nom du fichier et le répertoire
file_dir = os.path.dirname(CONFIG['FILE_PATH'])
file_name = os.path.basename(CONFIG['FILE_PATH'])
def toBeDisplayed_if_s(user_choice, choice):
    # Utilisation de l'opérateur ternaire
    result = True if user_choice == 'd' else (True if user_choice == 's' and choice == True else False)
    return result

## 0) key nom de la feature / 1) Ative Floor / 2) Active Crop / 3) % à Floored / ') % à Croped / 5) Afficher et/ou inclure Features dans fichiers cibles
# choix des features à traiter
column_settings = {
    # Time-based features
    'deltaTimestampOpening':                  (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSection5min':       (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSection5index':     (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSection30min':      (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSection30index':    (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),
    'deltaCustomSectionMin':                  (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),
    'deltaCustomSectionIndex':                (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),

    # Price and volume features
    'candleSizeTicks':                        (True, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffPriceClosePoc_0_0':                  (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffPriceClosePoc_0_1':                  (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffPriceClosePoc_0_2':                  (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffPriceClosePoc_0_3':                  (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffHighPrice_0_1':                      (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffHighPrice_0_2':                      (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffHighPrice_0_3':                      (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffHighPrice_0_4':                        (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffLowPrice_0_1':                       (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffLowPrice_0_2':                       (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffLowPrice_0_3':                       (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffLowPrice_0_4':                         (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffPriceCloseVWAP':                     (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok

    # Technical indicators
    'atr':                                    (True, True, 0.1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'bandWidthBB':                            (True, True, 0.1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'perctBB':                                (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'market_regimeADX':                         (False, False, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),
    'range_strength':                           (False, True, 0.1, 99.5,toBeDisplayed_if_s(user_choice, False)),

    # Reversal and momentum features
    'bearish_reversal_force':                 (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'bullish_reversal_force':                 (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'bearish_ask_bid_ratio':                  (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'bullish_ask_bid_ratio':                  (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'meanVolx':                               (False, True, 1, 99.7,toBeDisplayed_if_s(user_choice, False)),#ok
    'ratioDeltaBlw':                          (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'ratioDeltaAbv':                          (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffVolCandle_0_1Ratio':                 (False, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffVolDelta_0_1Ratio':                  (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'cumDiffVolDeltaRatio':                  (True, True, 0.1, 99.1,toBeDisplayed_if_s(user_choice, False)),#ok

    # Volume profile features
    'VolPocVolCandleRatio':                  (False, False, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'pocDeltaPocVolRatio':                    (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'asymetrie_volume':                       (True, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'VolCandleMeanxRatio':                    (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok

    # Order flow features
    'bearish_ask_ratio':                      (False, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'bearish_bid_ratio':                      (False, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'bullish_ask_ratio':                     (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bullish_bid_ratio':                     (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bearish_ask_score':                      (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bearish_bid_score':                      (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bearish_imnbScore_score':                (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bullish_ask_score':                      (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bullish_bid_score':                      (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bullish_imnbScore_score':                (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1

    # Imbalance features
    'bull_imbalance_low_1':                   (False, True, 1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bull_imbalance_low_2':                  (False, True, 1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bull_imbalance_low_3':                 (False, True, 1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bull_imbalance_high_0':                 (False, True, 1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bull_imbalance_high_1':                  (False, True, 1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bull_imbalance_high_2':                 (False, True, 1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bear_imbalance_low_0':                   (False, True, 1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bear_imbalance_low_1':                  (False, True, 1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bear_imbalance_low_2':                   (False, True, 1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bear_imbalance_high_1':                 (False, True, 1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bear_imbalance_high_2':                 (False, True, 1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bear_imbalance_high_3':                 (False, True, 1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'imbalance_score_low':                    (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok1
    'imbalance_score_high':                   (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok1

    # Auction features
    'finished_auction_high':                  (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok1
    'finished_auction_low':                   (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok1
    'staked00_high':                          (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok1
    'staked00_low':                           (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok1

    # Absorption features
    'bearish_ask_abs_ratio_abv':               (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bearish_bid_abs_ratio_abv':                (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bearish_abs_diff_abv':                    (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bullish_ask_abs_ratio_blw':                (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bullish_bid_abs_ratio_blw':               (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bullish_abs_diff_blw':                     (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1

    # Big trade features
    'bearish_askBigStand_abs_ratio_abv':      (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bearish_bidBigStand_abs_ratio_abv':      (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bearish_bigStand_abs_diff_abv':          (True, False, 0.5, 99,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_askBigStand_abs_ratio_blw':      (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_bidBigStand_abs_ratio_blw':      (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_bigStand_abs_diff_blw':          (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bearish_askBigHigh_abs_ratio_abv':       (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bearish_bidBigHigh_abs_ratio_abv':       (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bearish_bigHigh_abs_diff_abv':           (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_askBigHigh_abs_ratio_blw':       (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_bidBigHigh_abs_ratio_blw':       (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_bigHigh_abs_diff_blw':           (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2

    # Extreme zone features
    'bearish_extrem_revIntensity_ratio':      (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_extrem_revIntensity_ratio':      (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_extrem_zone_volume_ratio':       (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_extrem_zone_volume_ratio':       (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_extrem_pressure_ratio':          (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_extrem_pressure_ratio':          (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_extrem_abs_ratio':               (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_extrem_abs_ratio':               (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_extrem_vs_rest_activity':        (True, True, 1, 98,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_extrem_vs_rest_activity':        (True, True, 1, 98,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_continuation_vs_reversal':       (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_continuation_vs_reversal':       (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_repeat_ticks_ratio':             (True, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_repeat_ticks_ratio':             (True, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_big_trade_ratio_extrem':         (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_big_trade_imbalance':            (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_big_trade_ratio_extrem':          (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_big_trade_imbalance':            (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3

    # Ascending/Descending features
    'bearish_asc_dsc_ratio':                  (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_asc_dynamics':                   (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_dsc_dynamics':                   (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_asc_dsc_ratio':                  (False, True, 1, 97,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_asc_dynamics':                   (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_dsc_dynamics':                   (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3

    # Ask/Bid imbalance features
    'bearish_asc_ask_bid_imbalance':          (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_dsc_ask_bid_imbalance':          (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_imbalance_evolution':            (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_asc_ask_bid_delta_imbalance':    (True, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bearish_dsc_ask_bid_delta_imbalance':    (True, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bullish_asc_ask_bid_imbalance':          (False, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bullish_dsc_ask_bid_imbalance':          (False, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bullish_imbalance_evolution':            (True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bullish_asc_ask_bid_delta_imbalance':    (True, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bullish_dsc_ask_bid_delta_imbalance':    (True, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok4

    # Extreme zone additional features
    'extrem_asc_ratio_bearish':               (False, True, 0.5, 99,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_dsc_ratio_bearish':               (False, True, 0.5, 99,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_zone_significance_bearish':       (False, True, 0.5, 97,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_ask_bid_imbalance_bearish':       (False, False, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_asc_dsc_comparison_bearish':      (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_asc_ratio_bullish':               (False, True, 0.5, 99,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_dsc_ratio_bullish':               (False, True, 0.5, 99,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_zone_significance_bullish':       (False, True, 0.5, 97,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_ask_bid_imbalance_bullish':      (False, False, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_asc_dsc_comparison_bullish':     (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4

    # Absorption and big trade features
    'bearish_absorption_ratio':               (False, True, 0.5, 97,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bullish_absorption_ratio':               (False, True, 0.5, 97,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bearish_big_trade_ratio2_extrem':        (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bullish_big_trade_ratio2_extrem':        (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4

    # Absorption and repeat features
    'total_count_abv':                        (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4
    'absorption_intensity_repeat_bearish_vol':(False, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
    'absorption_intensity_repeat_bearish_count':(False, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bearish_repeatAskBid_ratio':             (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok4
    'total_count_blw':                        (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4
    'absorption_intensity_repeat_bullish_vol':(True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
    'absorption_intensity_repeat_bullish_count':(True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bullish_repeatAskBid_ratio':             (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok4
    'ratio_Count_AbvBlw':                       (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),
    # 6 Ticks zone features
    'bearish_volume_ratio_6Tick':             (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bullish_volume_ratio_6Tick':             (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bearish_relatif_ratio_6Tick':            (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_relatif_ratio_6Tick':            (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_relatifDelta_ratio_6Tick':       (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_relatifDelta_ratio_6Tick':       (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_buyer_pressure_6Tick':           (False, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_buyer_pressure_6Tick':            (False, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_seller_pressure_6Tick': (False, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_seller_pressure_6Tick': (False, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_absorption_6Tick': (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_absorption_6Tick': (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_repeat_ticks_ratio_6Tick': (True, True, 5, 99,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_repeat_ticks_ratio_6Tick': (True, True, 5, 99,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_price_dynamics_comparison_6Tick': (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_price_dynamics_comparison_6Tick': (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_activity_bid_ask_ratio_6Tick': (False, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_activity_ask_bid_ratio_6Tick': (False, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_repeat_ticks_imbalance_6Tick': (True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_repeat_ticks_imbalance_6Tick': (True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_ticks_imbalance_6Tick': (True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_ticks_imbalance_6Tick': (True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok5
}
columns_to_process = list(column_settings.keys())

# Vérification de l'existence des colonnes
missing_columns = [column for column in columns_to_process if column not in features_df.columns]

if missing_columns:
    print("Erreur : Les colonnes suivantes sont manquantes :")
    for column in missing_columns:
        print(f"- {column}")
    print("Le processus va s'arrêter.")
    exit(1)  # Arrête le script avec un code d'erreur

print("Toutes les features nécessaires sont présentes. Poursuite du traitement.")


def calculate_percentiles(df_NANVAlue, columnName, settings, nan_replacement_values=None):
    floor_enabled, crop_enabled, floorInf_percentage, cropSup_percentage, _ = settings[columnName]

    if nan_replacement_values is not None and columnName in nan_replacement_values:
        nan_value = nan_replacement_values[columnName]
        mask = df_NANVAlue[columnName] != nan_value
        nan_count = (~mask).sum()
        print(f"   In calculate_percentiles:")
        print(f"     - Filter out {nan_count} nan replacement value(s) {nan_value} for {columnName}")
    else:
        mask = df_NANVAlue[columnName].notna()
        nan_count = df_NANVAlue[columnName].isna().sum()
        print(f"   In calculate_percentiles:")
        print(f"     - {nan_count} NaN value(s) found in {columnName}")

    filtered_values = df_NANVAlue.loc[mask, columnName]

    floor_value = np.percentile(filtered_values, floorInf_percentage) if floor_enabled else None
    crop_value = np.percentile(filtered_values, cropSup_percentage) if crop_enabled else None

    print(f"     - floor_value: {floor_value}   crop_value: {crop_value}")

    return floor_value, crop_value

import numpy as np
import pandas as pd

def replace_nan_and_inf(df, columns_to_process, start_value, increment, REPLACE_NAN=True):
    current_value = start_value
    nan_replacement_values = {}
    df_replaced = df.copy()

    for column in columns_to_process:
        is_nan = df[column].isna()
        is_inf = np.isinf(df[column])
        nan_count = is_nan.sum()
        inf_count = is_inf.sum()

        if nan_count > 0 or inf_count > 0:
            print(f"Colonne problématique : {column}")
            print(f"Nombre de valeurs NaN : {nan_count}")
            print(f"Nombre de valeurs infinies : {inf_count}")

            if REPLACE_NAN:  # Effectuer le traitement pour NaN et infinies seulement si REPLACE_NAN est True
                if start_value != 0:
                    df_replaced.loc[is_nan | is_inf, column] = current_value
                    nan_replacement_values[column] = current_value
                    total_replacements = nan_count + inf_count
                    print(f"L'option start_value != 0 est activée.")
                    print(f"Les {total_replacements} valeurs NaN et infinies dans la colonne {column} ont été remplacées par {current_value}")
                    print(f"Les valeurs NaN sont remplacées par la valeur choisie par l'utilisateur : {current_value}")
                    if increment != 0:  # Incrémenter seulement si l'incrément est différent de zéro
                        current_value += increment
                else:
                    print(f"Les valeurs NaN et infinies dans la colonne {column} ont été laissées inchangées car start_value est 0")
            else:
                # Si REPLACE_NAN est False, remplacer uniquement les valeurs infinies par NaN
                df_replaced.loc[is_inf, column] = np.nan
                # Compter combien de valeurs infinies ont été remplacées par NaN
                inf_replacements = inf_count
                print(f"REPLACE_NAN est à False.")
                print(f"Les {inf_replacements} valeurs infinies dans la colonne {column} ont été remplacées par NaN")
                print(f"Les {nan_count} valeurs NaN dans la colonne {column} ont été laissées inchangées")
                print(f"Les valeurs NaN ne sont pas remplacées par une valeur choisie par l'utilisateur.")

    return df_replaced, nan_replacement_values



def winsorize(features_NANReplacedVal_df, column, floor_value, crop_value, floor_enabled, crop_enabled,
              nan_replacement_values=None):
    # Créer une copie des données de la colonne spécifiée
    winsorized_data = features_NANReplacedVal_df[column].copy()

    # Assurez-vous que le nom de la série est préservé
    winsorized_data.name = column

    # Créer un masque pour exclure la valeur nan_value si spécifiée
    if nan_replacement_values is not None and column in nan_replacement_values:
        nan_value = nan_replacement_values[column]
        mask = features_NANReplacedVal_df[column] != nan_value
    else:
        # Si pas de valeur à exclure, on crée un masque qui sélectionne toutes les valeurs non-NaN
        mask = features_NANReplacedVal_df[column].notna()

    # Appliquer la winsorisation seulement sur les valeurs non masquées
    if floor_enabled:
        winsorized_data.loc[mask & (winsorized_data < floor_value)] = floor_value

    if crop_enabled:
        winsorized_data.loc[mask & (winsorized_data > crop_value)] = crop_value

    # S'assurer qu'il n'y a pas de NaN dans les données winsorisées
    #winsorized_data = winsorized_data.fillna(nan_replacement_values.get(column, winsorized_data.median()))

    return winsorized_data


def cropFloor_dataSource(features_NANReplacedVal_df, columnName, settings, nan_replacement_values=None):
    floorInf_values, cropSup_values, floorInf_percent, cropSup_percent, _ = settings[columnName]

    floor_valueNANfiltered, crop_valueNANfiltered = calculate_percentiles(
        features_NANReplacedVal_df, columnName, settings, nan_replacement_values)

    return floor_valueNANfiltered, crop_valueNANfiltered, floorInf_values, cropSup_values, floorInf_percent, cropSup_percent


def plot_single_histogram(values_before, winsorized_values_after, column, floor_value, crop_value,
                          floorInf_values, cropSup_values, floorInf_percent, cropSup_percent, ax,
                          nan_replacement_values=None):
    values_before_clean = values_before.dropna()


    sns.histplot(data=pd.DataFrame({column: values_before_clean}), x=column, color="blue", kde=True, ax=ax,
                 label="Initial", alpha=0.7)
    sns.histplot(data=pd.DataFrame({column: winsorized_values_after}), x=column, color="red", kde=True, ax=ax,
                 label="Winsorized", alpha=0.7)

    if floorInf_values:
        ax.axvline(floor_value, color='g', linestyle='--', label=f'Floor ({floorInf_percent}%)')
    if cropSup_values:
        ax.axvline(crop_value, color='r', linestyle='--', label=f'Crop ({cropSup_percent}%)')

    ax.set_title(column, fontsize=8, pad=0)
    ax.legend(fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlabel('')
    ax.xaxis.set_tick_params(labelsize=7)

    if nan_replacement_values and column in nan_replacement_values:
        ax.annotate(f"NaN replaced by: {nan_replacement_values[column]}",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=8, ha='left', va='top')

    # Fonction pour formater les valeurs
    def format_value(value):
        return f"{value:.2f}" if pd.notna(value) else "nan"

    # Afficher les 2 premières et 2 dernières valeurs pour chaque ensemble de données
    initial_values = values_before_clean.sort_values()
    winsorized_values = winsorized_values_after.sort_values()

    initial_text = f"[{format_value(initial_values.iloc[0])} {format_value(initial_values.iloc[1])}]\n[{format_value(initial_values.iloc[-2])} {format_value(initial_values.iloc[-1])}]"
    winsorized_text = f"[{format_value(winsorized_values.iloc[0])} {format_value(winsorized_values.iloc[1])}]\n[{format_value(winsorized_values.iloc[-2])} {format_value(winsorized_values.iloc[-1])}]"

    ax.annotate(initial_text, xy=(0.75, 0.75), xycoords='axes fraction', fontsize=6, ha='left', va='top', color='blue')
    ax.annotate(winsorized_text, xy=(0.75, 0.65), xycoords='axes fraction', fontsize=6, ha='left', va='top', color='red')
def plot_histograms(columns, figsize=(32, 24)):

    n_columns = len(columns)
    ncols = 8
    nrows = (n_columns + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    return fig, axes


# Utilisation

# Paramètres
start_value = REPLACED_NANVALUE_BY
increment = REPLACED_NANVALUE_BY_INDEX

# Appliquer la fonction à features_df
features_NANReplacedVal_df, nan_replacement_values = replace_nan_and_inf(features_df.copy(), columns_to_process,
                                                                         start_value, increment,REPLACE_NAN)
number_of_elementsnan_replacement_values = len(nan_replacement_values)
print(f"Le dictionnaire nan_replacement_values contient {number_of_elementsnan_replacement_values} éléments.")

print(features_NANReplacedVal_df['bearish_bid_score'].describe())
print("Traitement des valeurs NaN et infinies terminé.")

print("Suppression des valeurs NAN ajoutées terminée.")

print("\n")

#recupere la liste du nom des features pour paramettrage
columns = [col for col, settings in column_settings.items() if
               settings[4] or all(not s[4] for s in column_settings.values())]

# Initialisation des variables pour l'affichage si nécessaire
if user_choice.lower() == 'd' or user_choice.lower() == 's':
    fig, axes = plot_histograms(columns, figsize=(32, 24))

# Créez un DataFrame vide avant la boucle
winsorized_df = pd.DataFrame()
winsorized_scaledWithNanValue_df = pd.DataFrame()
# Initialisation des listes temporaires pour stocker les colonnes
winsorized_columns = []
scaled_columns = []

# Parcours la liste des features
for i, columnName in enumerate(columns):
    print(f"\n")
    print(f"Debut de traitement de {columnName}:")
    floor_valueNANfiltered, crop_valueNANfiltered, floorInf_values, cropSup_values, floorInf_percent, cropSup_percent = (
        cropFloor_dataSource( features_NANReplacedVal_df, columnName, column_settings, nan_replacement_values))

    # Winsorisation avec les valeurs NAN
    winsorized_valuesWithNanValue = winsorize(features_NANReplacedVal_df, columnName, floor_valueNANfiltered, crop_valueNANfiltered,
                                        floorInf_values, cropSup_values, nan_replacement_values)

    # Stocker la série winsorisée dans la liste
    winsorized_columns.append(pd.DataFrame({columnName: winsorized_valuesWithNanValue}))

    # Initialiser la colonne dans winsorized_scaledWithNanValue_df
    scaled_column = winsorized_valuesWithNanValue.copy()

    if nan_replacement_values is not None and columnName in nan_replacement_values:
        nan_value = nan_replacement_values[columnName]
        mask = scaled_column != nan_value
        # Sauvegardons les positions des nan_value
        nan_positions = ~mask
    else:
        mask = slice(None)  # Sélectionne toutes les lignes si pas de nan_value
        nan_positions = pd.Series(False, index=winsorized_df.index)  # Série de False

    # Normalisation des valeurs
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(scaled_column.loc[mask].values.reshape(-1, 1)).flatten()

    # Convertir la colonne en float64 si elle ne l'est pas déjà
    scaled_column = scaled_column.astype('float64')

    scaled_column.loc[mask] = normalized_values

    # Remettre les nan_value à leur place seulement s'il y en avait
    if nan_replacement_values is not None and columnName in nan_replacement_values:
        scaled_column.loc[nan_positions] = nan_value

    # Stocker la série normalisée dans la liste
    scaled_columns.append(pd.DataFrame({columnName: scaled_column}))

    # Affichage des graphiques si demandé
    if user_choice.lower() == 'd' or user_choice.lower() == 's':
        winsorized_values_4Plotting = winsorized_valuesWithNanValue[
            winsorized_valuesWithNanValue != nan_replacement_values.get(columnName, np.nan)]
        print(f"   Graphiques de {columnName} avant et après les modifications (colonnes sélectionnées) :")
        print(f"   Taille de winsorized_values_after (sans NaN) pour plotting: {len(winsorized_values_4Plotting)}")
        value_before_df = features_df.copy()
        plot_single_histogram(value_before_df[columnName], winsorized_values_4Plotting, columnName,
                              floor_valueNANfiltered, crop_valueNANfiltered,
                              floorInf_values, cropSup_values, floorInf_percent, cropSup_percent,
                              axes[i], nan_replacement_values)

# Concaténer toutes les colonnes en une seule étape pour chaque DataFrame
winsorized_df = pd.concat(winsorized_columns, axis=1)
winsorized_scaledWithNanValue_df = pd.concat(scaled_columns, axis=1)

print(f"\n")
print("Vérification finale :")
print(f"   - Nombre de colonnes dans winsorized_df : {len(winsorized_df.columns)}")
print(f"   - Nombre de colonnes dans winsorized_scaledWithNanValue_df : {len(winsorized_scaledWithNanValue_df.columns)}")
assert len(winsorized_df.columns) == len(winsorized_scaledWithNanValue_df.columns), "Le nombre de colonnes ne correspond pas entre les DataFrames"


print_notification("Ajout de  'class_binaire', 'date', 'trade_category', 'SessionStartEnd' pour permettre la suite des traitements")
# Colonnes à ajouter
columns_to_add = ['class_binaire', 'candleDir', 'date', 'trade_category', 'SessionStartEnd']

# Vérifiez que toutes les colonnes existent dans df
missing_columns = [col for col in columns_to_add if col not in df.columns]
if missing_columns:
    error_message = f"Erreur: Les colonnes suivantes n'existent pas dans le DataFrame d'entrée: {', '.join(missing_columns)}"
    print(error_message)
    raise ValueError(error_message)

# Si nous arrivons ici, toutes les colonnes existent

# Créez un DataFrame avec les colonnes à ajouter

columns_df = df[columns_to_add]
# Ajoutez ces colonnes à features_df, winsorized_df, et winsorized_scaledWithNanValue_df en une seule opération
features_df = pd.concat([features_df, columns_df], axis=1)
winsorized_df = pd.concat([winsorized_df, columns_df], axis=1)
winsorized_scaledWithNanValue_df = pd.concat([winsorized_scaledWithNanValue_df, columns_df], axis=1)

print_notification("Colonnes 'class_binaire', 'candleDir', 'date', 'trade_category', 'SessionStartEnd' ajoutées")



# Créer le nouveau nom de fichier pour les features originales
new_file_name = "Step5_" + file_name.rsplit('.', 1)[0] + '_feat.csv'

# Construire le chemin complet du nouveau fichier
feat_file = os.path.join(file_dir, new_file_name)

# Sauvegarder le fichier des features originales
print_notification(f"Enregistrement du fichier de features non modifiées : {feat_file}")
features_df.to_csv(feat_file, sep=';', index=False, encoding='iso-8859-1')


# Créer le nouveau nom de fichier pour winsorized_df
winsorized_file_name = "Step5_" + file_name.rsplit('.', 1)[0] + '_feat_winsorized.csv'

# Construire le chemin complet du nouveau fichier winsorized
winsorized_file = os.path.join(file_dir, winsorized_file_name)

# Sauvegarder le fichier winsorized
print_notification(f"Enregistrement du fichier de features winsorisées : {winsorized_file}")
winsorized_df.to_csv(winsorized_file, sep=';', index=False, encoding='iso-8859-1')

# Créer le nouveau nom de fichier pour winsorized_scaledWithNanValue_df
scaled_file_name = "Step5_" + file_name.rsplit('.', 1)[0] + '_feat_winsorizedScaledWithNanVal.csv'

# Construire le chemin complet du nouveau fichier scaled
scaled_file = os.path.join(file_dir, scaled_file_name)

# Sauvegarder le fichier scaled
winsorized_scaledWithNanValue_df.to_csv(scaled_file, sep=';', index=False, encoding='iso-8859-1')
print_notification(f"Enregistrement du fichier de features winsorisées et normalisées : {scaled_file}")

# Affichage final des graphiques si demandé
if user_choice.lower() == 'd' or user_choice.lower() == 's':
    # Supprimer les sous-graphiques vides
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
else:
    print_notification("Calcul des features terminé. Graphiques non affichés.")