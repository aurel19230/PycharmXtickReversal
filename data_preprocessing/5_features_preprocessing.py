import pandas as pd
import numpy as np
from standardFunc import print_notification
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import os

diffDivBy0=np.nan
addDivBy0=np.nan

# Configuration
CONFIG = {
    'NUM_GROUPS': 9,
    'MIN_RANGE': 30,  # en minutes
    'FILE_PATH': r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL\merge\Step4_Step3_Step2_MergedAllFile_Step1_0_merged_extractOnlyFullSession.csv",
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

def load_data(file_path: str) -> pd.DataFrame:
    print_notification("Début du chargement des données")
    df = pd.read_csv(file_path, sep=';', encoding='iso-8859-1')
    print_notification("Données chargées avec succès")
    return df

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
features_df['diffLowPrice_0_1'] = df['low'] - df['low'].shift(1)
features_df['diffLowPrice_0_2'] = df['low'] - df['low'].shift(2)
features_df['diffLowPrice_0_3'] = df['low'] - df['low'].shift(3)
features_df['diffPriceCloseVWAP'] = df['close'] - df['VWAP']

features_df['atr'] = df['atr']
features_df['bandWidthBB'] = df['bandWidthBB']
features_df['perctBB'] = df['perctBB']

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
    0)

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

features_df['class']=df['class']
features_df['date']=df['date']
features_df['trade_category']=df['trade_category']

# Enregistrement des fichiers
print_notification("Début de l'enregistrement des fichiers")

# Extraire le nom du fichier et le répertoire
file_dir = os.path.dirname(CONFIG['FILE_PATH'])
file_name = os.path.basename(CONFIG['FILE_PATH'])

# Créer le nouveau nom de fichier
new_file_name = "Step5_" + file_name.rsplit('.', 1)[0] + '_feat.csv'

# Construire le chemin complet du nouveau fichier
feat_file = os.path.join(file_dir, new_file_name)

# Sauvegarder le fichier
features_df.to_csv(feat_file, sep=';', index=False, encoding='iso-8859-1')
print_notification(f"Fichier de features non modifiées enregistré : {feat_file}")

# Fichier standardisé (_featStand.csv)
standardized_df = features_df.copy()
columns_to_standardize = [
    'deltaTimestampOpening','deltaTimestampOpeningSection5min', 'deltaTimestampOpeningSection5index',
    'deltaTimestampOpeningSection30min', 'deltaTimestampOpeningSection30index',
    'deltaCustomSectionMin', 'deltaCustomSectionIndex',
    'candleSizeTicks', 'diffPriceClosePoc_0_1', 'diffPriceClosePoc_0_2', 'diffPriceClosePoc_0_3',
    'diffHighPrice_0_1', 'diffHighPrice_0_2', 'diffHighPrice_0_3',
    'diffLowPrice_0_1', 'diffLowPrice_0_2', 'diffLowPrice_0_3',
    'diffPriceCloseVWAP', 'diffPriceClosePoc_0_0', 'atr', 'bandWidthBB', 'perctBB',
    'bearish_reversal_force', 'bullish_reversal_force',
    'bearish_ask_bid_ratio', 'bullish_ask_bid_ratio',
    'meanVolx', 'ratioDeltaBlw', 'ratioDeltaAbv',
    'diffVolCandle_0_1Ratio', 'diffVolDelta_0_1Ratio', 'cumDiffVolDeltaRatio',
    'VolPocVolCandleRatio', 'pocDeltaPocVolRatio', 'asymetrie_volume',
    'VolCandleMeanxRatio',
    'bearish_ask_ratio', 'bearish_bid_ratio', 'bullish_ask_ratio', 'bullish_bid_ratio',
    'bearish_ask_score', 'bearish_bid_score', 'bearish_imnbScore_score','bullish_ask_score', 'bullish_bid_score','bullish_imnbScore_score',
    'bull_imbalance_low_1', 'bull_imbalance_low_2', 'bull_imbalance_low_3',
    'bull_imbalance_high_0', 'bull_imbalance_high_1', 'bull_imbalance_high_2',
    'bear_imbalance_low_0', 'bear_imbalance_low_1', 'bear_imbalance_low_2',
    'bear_imbalance_high_1', 'bear_imbalance_high_2', 'bear_imbalance_high_3',
    'imbalance_score_low', 'imbalance_score_high', 'finished_auction_high', 'finished_auction_low',
    'staked00_high', 'staked00_low',
    'bearish_ask_abs_ratio_abv', 'bearish_bid_abs_ratio_abv', 'bearish_abs_diff_abv',
    'bullish_ask_abs_ratio_blw', 'bullish_bid_abs_ratio_blw', 'bullish_abs_diff_blw'
]
# Ajout des nouvelles colonnes à standardiser
columns_to_standardize.extend([
    'bearish_askBigStand_abs_ratio_abv', 'bearish_bidBigStand_abs_ratio_abv', 'bearish_bigStand_abs_diff_abv',
    'bullish_askBigStand_abs_ratio_blw', 'bullish_bidBigStand_abs_ratio_blw', 'bullish_bigStand_abs_diff_blw',
    'bearish_askBigHigh_abs_ratio_abv', 'bearish_bidBigHigh_abs_ratio_abv', 'bearish_bigHigh_abs_diff_abv',
    'bullish_askBigHigh_abs_ratio_blw', 'bullish_bidBigHigh_abs_ratio_blw', 'bullish_bigHigh_abs_diff_blw'
])
# Ajout des nouvelles colonnes à standardiser
new_columns_to_standardize = [
    'bearish_extrem_revIntensity_ratio', 'bullish_extrem_revIntensity_ratio',
    'bearish_extrem_zone_volume_ratio', 'bullish_extrem_zone_volume_ratio',
    'bearish_extrem_pressure_ratio', 'bullish_extrem_pressure_ratio',
    'bearish_extrem_abs_ratio', 'bullish_extrem_abs_ratio',
    'bearish_extrem_vs_rest_activity', 'bullish_extrem_vs_rest_activity',
    'bearish_continuation_vs_reversal', 'bullish_continuation_vs_reversal',
    'bearish_repeat_ticks_ratio', 'bullish_repeat_ticks_ratio',
    'bearish_big_trade_ratio_extrem', 'bearish_big_trade_imbalance',
    'bullish_big_trade_ratio_extrem', 'bullish_big_trade_imbalance'
]

columns_to_standardize.extend(new_columns_to_standardize)


new_asc_dsc_columns = [
    'bearish_asc_dsc_ratio', 'bearish_asc_dynamics', 'bearish_dsc_dynamics',
    'bullish_asc_dsc_ratio', 'bullish_asc_dynamics', 'bullish_dsc_dynamics'
]
columns_to_standardize.extend(new_asc_dsc_columns)

new_imbalance_columns = [
    'bearish_asc_ask_bid_imbalance', 'bearish_dsc_ask_bid_imbalance', 'bearish_imbalance_evolution',
    'bearish_asc_ask_bid_delta_imbalance', 'bearish_dsc_ask_bid_delta_imbalance',
    'bullish_asc_ask_bid_imbalance', 'bullish_dsc_ask_bid_imbalance', 'bullish_imbalance_evolution',
    'bullish_asc_ask_bid_delta_imbalance', 'bullish_dsc_ask_bid_delta_imbalance'
]
columns_to_standardize.extend(new_imbalance_columns)

# Ajout des nouvelles colonnes à la liste des colonnes à standardiser
new_extrem_columns = [
    'extrem_asc_ratio_bearish', 'extrem_dsc_ratio_bearish', 'extrem_zone_significance_bearish',
    'extrem_ask_bid_imbalance_bearish', 'extrem_asc_dsc_comparison_bearish', 'bearish_repeat_ticks_ratio',
    'extrem_asc_ratio_bullish', 'extrem_dsc_ratio_bullish', 'extrem_zone_significance_bullish',
    'extrem_ask_bid_imbalance_bullish', 'extrem_asc_dsc_comparison_bullish', 'bullish_repeat_ticks_ratio'
]
columns_to_standardize.extend(new_extrem_columns)

# Ajout des nouvelles colonnes à la liste des colonnes à standardiser
new_absorption_big_trade_columns = [
    'bearish_absorption_ratio', 'bullish_absorption_ratio',
    'bearish_big_trade_ratio2_extrem', 'bullish_big_trade_ratio2_extrem'
]
columns_to_standardize.extend(new_absorption_big_trade_columns)

new_absorption_repeat_columns = [
    'total_count_abv', 'absorption_intensity_repeat_bearish_vol', 'absorption_intensity_repeat_bearish_count', 'bearish_repeatAskBid_ratio',
    'total_count_blw', 'absorption_intensity_repeat_bullish_vol', 'absorption_intensity_repeat_bullish_count', 'bullish_repeatAskBid_ratio'
]
columns_to_standardize.extend(new_absorption_repeat_columns)
new_6ticks_columns = [
    'bearish_volume_ratio_6Tick', 'bullish_volume_ratio_6Tick',
    'bearish_relatif_ratio_6Tick', 'bullish_relatif_ratio_6Tick',
    'bearish_relatifDelta_ratio_6Tick', 'bullish_relatifDelta_ratio_6Tick',
    'bearish_buyer_pressure_6Tick', 'bullish_buyer_pressure_6Tick',
    'bearish_seller_pressure_6Tick', 'bullish_seller_pressure_6Tick',
    'bearish_absorption_6Tick', 'bullish_absorption_6Tick',
    'bearish_repeat_ticks_ratio_6Tick', 'bullish_repeat_ticks_ratio_6Tick',
    'bearish_price_dynamics_comparison_6Tick', 'bullish_price_dynamics_comparison_6Tick',
    'bearish_activity_bid_ask_ratio_6Tick', 'bullish_activity_ask_bid_ratio_6Tick',
    'bearish_repeat_ticks_imbalance_6Tick', 'bullish_repeat_ticks_imbalance_6Tick',
    'bearish_ticks_imbalance_6Tick', 'bullish_ticks_imbalance_6Tick'
]
columns_to_standardize.extend(new_6ticks_columns)

# Vérification de l'existence des colonnes
missing_columns = [column for column in columns_to_standardize if column not in features_df.columns]

if missing_columns:
    print("Erreur : Les colonnes suivantes sont manquantes :")
    for column in missing_columns:
        print(f"- {column}")
    print("Le processus va s'arrêter.")
    exit(1)  # Arrête le script avec un code d'erreur

print("Toutes les features nécessaires sont présentes. Poursuite du traitement.")

for column in columns_to_standardize:
    if np.isinf(standardized_df[column]).any() or np.isnan(standardized_df[column]).any():
        print(f"Colonne problématique : {column}")
        print(f"Nombre de valeurs infinies : {np.isinf(standardized_df[column]).sum()}")
        print(f"Nombre de valeurs NaN : {np.isnan(standardized_df[column]).sum()}")
