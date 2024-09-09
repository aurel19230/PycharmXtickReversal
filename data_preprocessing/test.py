import pandas as pd
import numpy as np
from standardFunc import print_notification

# Configuration
CONFIG = {
    'NUM_GROUPS': 9,
    'MIN_RANGE': 30,  # en minutes
    'FILE_PATH': r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL\merge\MergedAllFile_030619_300824_merged_extractOnly20LastFullSession.csv",
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
    df = pd.read_csv(file_path, sep=';')
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

# Création d'un DataFrame temporaire pour accumuler les nouvelles colonnes
features_temp = pd.DataFrame(index=df.index)

# Calcul des features
features_temp['deltaTimestampOpening'] = df['deltaTimestampOpening']

features_temp['deltaTimestampOpeningSection5min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x/5))*5, 1350))

unique_sections = sorted(features_temp['deltaTimestampOpeningSection5min'].unique())
section_to_index = {section: index for index, section in enumerate(unique_sections)}
features_temp['deltaTimestampOpeningSection5index'] = features_temp['deltaTimestampOpeningSection5min'].map(section_to_index)

features_temp['deltaTimestampOpeningSection30min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x/30))*30, 1350))

unique_sections = sorted(features_temp['deltaTimestampOpeningSection30min'].unique())
section_to_index = {section: index for index, section in enumerate(unique_sections)}
features_temp['deltaTimestampOpeningSection30index'] = features_temp['deltaTimestampOpeningSection30min'].map(section_to_index)

features_temp['deltaCustomSectionMin'] = df['deltaTimestampOpening'].apply(
    lambda x: get_custom_section(x)['start'])

unique_custom_sections = sorted(features_temp['deltaCustomSectionMin'].unique())
custom_section_to_index = {section: index for index, section in enumerate(unique_custom_sections)}
features_temp['deltaCustomSectionIndex'] = features_temp['deltaCustomSectionMin'].map(custom_section_to_index)

# Features précédentes
features_temp['candleSizeTicks'] = df['candleSizeTicks']
features_temp['diffPriceClosePoc_0_1'] = df['close'] - df['pocPrice'].shift(1)
features_temp['diffPriceClosePoc_0_2'] = df['close'] - df['pocPrice'].shift(2)
features_temp['diffPriceClosePoc_0_3'] = df['close'] - df['pocPrice'].shift(3)
features_temp['diffHighPrice_0_1'] = df['high'] - df['high'].shift(1)
features_temp['diffHighPrice_0_2'] = df['high'] - df['high'].shift(2)
features_temp['diffHighPrice_0_3'] = df['high'] - df['high'].shift(3)
features_temp['diffLowPrice_0_1'] = df['low'] - df['low'].shift(1)
features_temp['diffLowPrice_0_2'] = df['low'] - df['low'].shift(2)
features_temp['diffLowPrice_0_3'] = df['low'] - df['low'].shift(3)
features_temp['diffPriceCloseVWAP'] = df['close'] - df['VWAP']
features_temp['diffPriceClosePoc_0_0'] = df['close'] - df['pocPrice']
features_temp['atr'] = df['atr']
features_temp['bandWidthBB'] = df['bandWidthBB']
features_temp['perctBB'] = df['perctBB']

# Nouvelles features - Force du renversement
features_temp['bearish_reversal_force'] = np.where(df['volume'] != 0, df['VolAbv'] / df['volume'], 0)
features_temp['bullish_reversal_force'] = np.where(df['volume'] != 0, df['VolBlw'] / df['volume'], 0)

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
features_temp['bearish_ask_bid_ratio'] = np.where(df['VolAbvBid'] != 0, df['VolAbvAsk'] / df['VolAbvBid'], -1)
features_temp['bullish_ask_bid_ratio'] = np.where(df['VolBlwAsk'] != 0, df['VolBlwBid'] / df['VolBlwAsk'], -1)

# Nouvelles features - Features de Momentum:
features_temp['meanVolx'] = df['volume'].shift().rolling(window=5, min_periods=1).mean()
features_temp['ratioDeltaBlw'] = np.where(df['VolBlw'] != 0, df['DeltaBlw'] / df['VolBlw'], 0)
features_temp['ratioDeltaAbv'] = np.where(df['VolAbv'] != 0, df['DeltaAbv'] / df['VolAbv'], 0)
features_temp['diffVolCandle_0_1'] = np.where(features_temp['meanVolx'] != 0,
                                              (df['volume'] - df['volume'].shift(1)) / features_temp['meanVolx'], 0)
features_temp['diffVolDelta_0_1'] = np.where(features_temp['meanVolx'] != 0,
                                             (df['delta'] - df['delta'].shift(1)) / features_temp['meanVolx'], 0)
features_temp['cumDiffVolDelta'] = df['delta'].shift(1) + df['delta'].shift(2) + df['delta'].shift(3) + df['delta'].shift(4) + df['delta'].shift(5)

# Nouvelles features - Features de Volume Profile:
features_temp['ratioVolPocVolCandle'] = np.where(df['volume'] != 0, df['volPOC'] / df['volume'], -1)
features_temp['ratioPocDeltaPocVol'] = np.where(df['volPOC'] != 0, df['deltaPOC'] / df['volPOC'], 0)
features_temp['asymetrie_volume'] = np.where(df['volume'] != 0, (df['VolAbv'] - df['VolBlw']) / df['volume'], 0)

# Nouvelles features - Features Cumulatives sur les 5 dernières bougies:
features_temp['ratioVolCandleMeanx'] = np.where(features_temp['meanVolx'] != 0, df['volume'] / features_temp['meanVolx'], -1)

# Nouvelles features - Caractéristiques de la zone de renversement :
features_temp['bearish_ask_ratio'] = np.where(df['VolAbv'] != 0, df['VolAbvAsk'] / df['VolAbv'], -1)
features_temp['bearish_bid_ratio'] = np.where(df['VolAbv'] != 0, df['VolAbvBid'] / df['VolAbv'], -1)
features_temp['bullish_ask_ratio'] = np.where(df['VolBlw'] != 0, df['VolBlwAsk'] / df['VolBlw'], -1)
features_temp['bullish_bid_ratio'] = np.where(df['VolBlw'] != 0, df['VolBlwBid'] / df['VolBlw'], -1)

# Nouvelles features - Dynamique de prix dans la zone de renversement :
features_temp['bearish_ask_score'] = np.where(df['VolAbv'] != 0,
                                              (df['upTickVolAbvAskDesc'] + df['repeatUpTickVolAbvAskDesc'] -
                                               df['downTickVolAbvAskDesc'] - df['repeatDownTickVolAbvAskDesc']) / df['VolAbv'], 0)

features_temp['bearish_bid_score'] = np.where(df['VolAbv'] != 0,
                                              (df['upTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidDesc'] -
                                               df['downTickVolAbvBidDesc'] - df['repeatDownTickVolAbvBidDesc']) / df['VolAbv'], 0)

features_temp['bullish_ask_score'] = np.where(df['VolBlw'] != 0,
                                              (df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] -
                                               df['downTickVolBlwAskAsc'] - df['repeatDownTickVolBlwAskAsc']) / df['VolBlw'], 0)

features_temp['bullish_bid_score'] = np.where(df['VolBlw'] != 0,
                                              (df['upTickVolBlwBidAsc'] + df['repeatUpTickVolBlwBidAsc'] -
                                               df['downTickVolBlwBidAsc'] - df['repeatDownTickVolBlwBidAsc']) / df['VolBlw'], 0)

# Nouvelles features - Order Flow:
features_temp['bull_imbalance_low_1'] = np.where(df['bidVolLow'] != 0, df['askVolLow_1'] / df['bidVolLow'], -1)
features_temp['bull_imbalance_low_2'] = np.where(df['bidVolLow_1'] != 0, df['askVolLow_2'] / df['bidVolLow_1'], -1)
features_temp['bull_imbalance_low_3'] = np.where(df['bidVolLow_2'] != 0, df['askVolLow_3'] / df['bidVolLow_2'], -1)
features_temp['bull_imbalance_high_0'] = np.where(df['bidVolHigh_1'] != 0, df['askVolHigh'] / df['bidVolHigh_1'], -1)
features_temp['bull_imbalance_high_1'] = np.where(df['bidVolHigh_2'] != 0, df['askVolHigh_1'] / df['bidVolHigh_2'], -1)
features_temp['bull_imbalance_high_2'] = np.where(df['bidVolHigh_3'] != 0, df['askVolHigh_2'] / df['bidVolHigh_3'], -1)

# Imbalances baissières
features_temp['bear_imbalance_low_0'] = np.where(df['askVolLow_1'] != 0, df['bidVolLow'] / df['askVolLow_1'], -1)
features_temp['bear_imbalance_low_1'] = np.where(df['askVolLow_2'] != 0, df['bidVolLow_1'] / df['askVolLow_2'], -1)
features_temp['bear_imbalance_low_2'] = np.where(df['askVolLow_3'] != 0, df['bidVolLow_2'] / df['askVolLow_3'], -1)
features_temp['bear_imbalance_high_1'] = np.where(df['askVolHigh'] != 0, df['bidVolHigh_1'] / df['askVolHigh'], -1)
features_temp['bear_imbalance_high_2'] = np.where(df['askVolHigh_1'] != 0, df['bidVolHigh_2'] / df['askVolHigh_1'], -1)
features_temp['bear_imbalance_high_3'] = np.where(df['askVolHigh_2'] != 0, df['bidVolHigh_3'] / df['askVolHigh_2'], -1)

# Score d'Imbalance Asymétrique
sell_pressureLow = df['bidVolLow'] + df['bidVolLow_1']
buy_pressureLow = df['askVolLow_1'] + df['askVolLow_2']
total_volumeLow = buy_pressureLow + sell_pressureLow
features_temp['imbalance_score_low'] = np.where(total_volumeLow != 0,
                                                (buy_pressureLow - sell_pressureLow) / total_volumeLow, 0)

sell_pressureHigh = df['bidVolHigh_1'] + df['bidVolHigh_2']
buy_pressureHigh = df['askVolHigh'] + df['askVolHigh_1']
total_volumeHigh = sell_pressureHigh + buy_pressureHigh
features_temp['imbalance_score_high'] = np.where(total_volumeHigh != 0,
                                                 (sell_pressureHigh - buy_pressureHigh) / total_volumeHigh, 0)

# Finished Auction
features_temp['finished_auction_high'] = (df['bidVolHigh'] == 0).astype(int)
features_temp['finished_auction_low'] = (df['askVolLow'] == 0).astype(int)
features_temp['staked00_high'] = ((df['bidVolHigh'] == 0) & (df['bidVolHigh_1'] == 0)).astype(int)
features_temp['staked00_low'] = ((df['askVolLow'] == 0) & (df['askVolLow_1'] == 0)).astype(int)

# Ajout de toutes les nouvelles features calculées précédemment
features_temp['bearish_askBigStand_abs_ratio_abv'] = np.where(
    (df['upTickVolAbvAskDesc'] + df['repeatUpTickVolAbvAskDesc'] + df['repeatDownTickVolAbvAskDesc']) != 0,
    (df['upTickVolAbvAskDesc_bigStand'] + df['repeatUpTickVolAbvAskDesc_bigStand'] + df['repeatDownTickVolAbvAskDesc_bigStand']) /
    (df['upTickVolAbvAskDesc'] + df['repeatUpTickVolAbvAskDesc'] + df['repeatDownTickVolAbvAskDesc']),
    -1
)

# (Ajouter ici les autres features dans le même style...)

# Concaténer toutes les nouvelles colonnes dans le DataFrame features_df en une seule opération
features_df = pd.concat([features_temp], axis=1)

# Si vous avez besoin de copier le DataFrame pour la défragmentation
features_df = features_df.copy()

print_notification("Calcul des features terminé")

# Enregistrement des fichiers
print_notification("Début de l'enregistrement des fichiers")

# Fichier non standardisé (_feat.csv)
feat_file = CONFIG['FILE_PATH'].rsplit('.', 1)[0] + '_feat.csv'
features_df.to_csv(feat_file, sep=';', index=False)
print_notification(f"Fichier de features non modifiées enregistré : {feat_file}")

print_notification("Script terminé avec succès")
