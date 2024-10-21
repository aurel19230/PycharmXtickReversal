import pandas as pd
import numpy as np
from standardFunc import print_notification
from standardFunc import load_data
import math
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
valueX=np.nan
valueY=np.nan
from sklearn.preprocessing import MinMaxScaler
# Définition de la fonction calculate_max_ratio
import numpy as np

def calculate_max_ratio(values, condition, calc_max=False, std_multiplier=1):
    valid_ratios = values[condition]
    # Exclure les NaN des calculs
    valid_ratios = valid_ratios[~np.isnan(valid_ratios)]

    if len(valid_ratios) > 0:
        if calc_max:
            return valid_ratios.max()
        else:
            mean = np.mean(valid_ratios)
            std = np.std(valid_ratios)
            if mean < 0:
                return mean - std_multiplier * std
            else:
                return mean + std_multiplier * std
    else:
        return 0

ENABLE_PANDAS_METHOD_SCALING=True

DEFAULT_DIV_BY0=False #max_ratio or valuex
user_choice = input("Appuyez sur Entrée pour calculer les features sans la afficher. \n"
                    "Appuyez sur 'd' puis Entrée pour les calculer et les afficher : \n"
                    "Appuyez sur 's' puis Entrée pour les calculer et les afficher :")
if user_choice.lower() == 'd':
    fig_range_input = input("Entrez la plage des figures à afficher au format x_y (par exemple 2_5) : \n")

# Demander à l'utilisateur s'il souhaite ajuster l'axe des abscisses
adjust_xaxis_input=''
if user_choice.lower() == 'd' or user_choice.lower() == 's':
    adjust_xaxis_input = input("Voulez-vous afficher les graphiques entre les valeurs de floor et crop ? (o/n) : ").lower()


adjust_xaxis = adjust_xaxis_input == 'o'

# Nom du fichier
file_name = "Step4_4_0_8TP_1SL_080919_161024_extractOnlyFullSession.csv"
#file_name = "Step4_4_0_8TP_1SL_080919_161024_extractOnly220LastFullSession_OnlyShort.csv"


# Chemin du répertoire
directory_path = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\4_0_8TP_1SL\\merge"

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



df = load_data(CONFIG['FILE_PATH'])



# Afficher la liste complète des colonnes
all_columns = df.columns.tolist()

# Imprimer la liste
print("Liste complète des colonnes:")
for col in all_columns:
    print(col)


print_notification("Début du calcul des features")

# Calcul des features
features_df = pd.DataFrame()
features_df['deltaTimestampOpening'] = df['deltaTimestampOpening']


features_df['deltaTimestampOpeningSection1min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x/15))*15, 1365))

unique_sections = sorted(features_df['deltaTimestampOpeningSection1min'].unique())
section_to_index = {section: index for index, section in enumerate(unique_sections)}
features_df['deltaTimestampOpeningSection1index'] = features_df['deltaTimestampOpeningSection1min'].map(section_to_index)

features_df['deltaTimestampOpeningSection5min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x/5))*5, 1375))

unique_sections = sorted(features_df['deltaTimestampOpeningSection5min'].unique())
section_to_index = {section: index for index, section in enumerate(unique_sections)}
features_df['deltaTimestampOpeningSection5index'] = features_df['deltaTimestampOpeningSection5min'].map(section_to_index)

features_df['deltaTimestampOpeningSection15min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x/15))*15, 1365))

unique_sections = sorted(features_df['deltaTimestampOpeningSection15min'].unique())
section_to_index = {section: index for index, section in enumerate(unique_sections)}
features_df['deltaTimestampOpeningSection15index'] = features_df['deltaTimestampOpeningSection15min'].map(section_to_index)


features_df['deltaTimestampOpeningSection30min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x/30))*30, 1380))

unique_sections = sorted(features_df['deltaTimestampOpeningSection30min'].unique())
section_to_index = {section: index for index, section in enumerate(unique_sections)}
features_df['deltaTimestampOpeningSection30index'] = features_df['deltaTimestampOpeningSection30min'].map(section_to_index)

features_df['deltaCustomSectionMin'] = df['deltaTimestampOpening'].apply(
    lambda x: get_custom_section(x)['start'])

unique_custom_sections = sorted(features_df['deltaCustomSectionMin'].unique())
custom_section_to_index = {section: index for index, section in enumerate(unique_custom_sections)}
features_df['deltaCustomSectionIndex'] = features_df['deltaCustomSectionMin'].map(custom_section_to_index)

# Features précédentes
features_df['VolAbvState'] = np.where(df['VolAbv'] == 0, 0, 1)
features_df['VolBlwState'] = np.where(df['VolBlw'] == 0, 0, 1)
features_df['candleSizeTicks'] = np.where(df['candleSizeTicks'] < 4, np.nan, df['candleSizeTicks'])
features_df['diffPriceClosePoc_0_0'] = df['close'] - df['pocPrice']
features_df['diffPriceClosePoc_0_1'] = df['close'] - df['pocPrice'].shift(1)
features_df['diffPriceClosePoc_0_2'] = df['close'] - df['pocPrice'].shift(2)
features_df['diffPriceClosePoc_0_3'] = df['close'] - df['pocPrice'].shift(3)
features_df['diffPriceClosePoc_0_4'] = df['close'] - df['pocPrice'].shift(4)
features_df['diffPriceClosePoc_0_5'] = df['close'] - df['pocPrice'].shift(5)
#features_df['diffPriceClosePoc_0_6'] = df['close'] - df['pocPrice'].shift(6)


features_df['diffHighPrice_0_1'] = df['high'] - df['high'].shift(1)
features_df['diffHighPrice_0_2'] = df['high'] - df['high'].shift(2)
features_df['diffHighPrice_0_3'] = df['high'] - df['high'].shift(3)
features_df['diffHighPrice_0_4'] = df['high'] - df['high'].shift(4)
features_df['diffHighPrice_0_5'] = df['high'] - df['high'].shift(5)
#features_df['diffHighPrice_0_6'] = df['high'] - df['high'].shift(6)


features_df['diffLowPrice_0_1'] = df['low'] - df['low'].shift(1)
features_df['diffLowPrice_0_2'] = df['low'] - df['low'].shift(2)
features_df['diffLowPrice_0_3'] = df['low'] - df['low'].shift(3)
features_df['diffLowPrice_0_4'] = df['low'] - df['low'].shift(4)
features_df['diffLowPrice_0_5'] = df['low'] - df['low'].shift(5)
#features_df['diffLowPrice_0_6'] = df['low'] - df['low'].shift(6)


features_df['diffPriceCloseVWAP'] = df['close'] - df['VWAP']

# Créer les conditions pour chaque plage
conditions = [
    (df['close'] >= df['VWAP']) & (df['close'] <= df['VWAPsd1Top']),
    (df['close'] > df['VWAPsd1Top']) & (df['close'] <= df['VWAPsd2Top']),
    (df['close'] > df['VWAPsd2Top']) & (df['close'] <= df['VWAPsd3Top']),
    (df['close'] > df['VWAPsd3Top']) & (df['close'] <= df['VWAPsd4Top']),
    (df['close'] > df['VWAPsd4Top']),
    (df['close'] < df['VWAP']) & (df['close'] >= df['VWAPsd1Bot']),
    (df['close'] < df['VWAPsd1Bot']) & (df['close'] >= df['VWAPsd2Bot']),
    (df['close'] < df['VWAPsd2Bot']) & (df['close'] >= df['VWAPsd3Bot']),
    (df['close'] < df['VWAPsd3Bot']) & (df['close'] >= df['VWAPsd4Bot']),
    (df['close'] < df['VWAPsd4Bot'])
]

# Créer les valeurs correspondantes pour chaque condition
values = [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]

# Utiliser np.select pour créer la nouvelle feature
features_df['diffPriceCloseVWAPbyIndex'] = np.select(conditions, values, default=0)

features_df['atr'] = df['atr']
features_df['bandWidthBB'] = df['bandWidthBB']
features_df['perctBB'] = df['perctBB']


import numpy as np

def detect_market_regimeADX(data, period=14, adx_threshold=25):
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


    data['market_regimeADX'] = np.where(data['adx'] > adx_threshold, data['adx'], data['adx'])
    #data['market_regimeADX'] = data['market_regimeADX'].fillna(addDivBy0 if DEFAULT_DIV_BY0 else valueX)
    # Calcul du pourcentage de valeurs inférieures à adx_threshold
    total_count = len(data['adx'])
    below_threshold_count = (data['adx'] < adx_threshold).sum()
    regimeAdx_pct_infThreshold = (below_threshold_count / total_count) * 100

    print(f"Pourcentage de valeurs ADX inférieures à {adx_threshold}: {regimeAdx_pct_infThreshold:.2f}%")

    return data, regimeAdx_pct_infThreshold


def range_strength(data, range_strength_, window=14, atr_multiple=2, min_strength=0.05):
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
    data[range_strength_] = data['range_duration'] / (1 + data['range_width'])

    # Appliquer un seuil minimum et une transformation logarithmique
    data[range_strength_] = np.where(data[range_strength_] < min_strength, np.nan, data[range_strength_])

    # data['log_range_strength'] = np.log1p(data[range_strength_])

    # Calculer le pourcentage de temps en range et hors range
    total_periods = len(data)
    in_range_periods = (data[range_strength_].notna()).sum()
    out_of_range_periods = total_periods - in_range_periods

    range_strength_percent_in_range = (in_range_periods / total_periods) * 100
    range_strength_percent_out_of_range = (out_of_range_periods / total_periods) * 100

    print(f"Pourcentage de temps en range: {range_strength_percent_in_range:.2f}%")
    print(f"Pourcentage de temps hors range: {range_strength_percent_out_of_range:.2f}%")

    data.drop(['tr'], axis=1, inplace=True)

    return data, range_strength_percent_in_range


def valueArea_pct(data, nbPeriods):
    # Calculate the difference between the high and low value area bands
    bands_difference = data[f'vaH_{nbPeriods}periods'] - data[f'vaL_{nbPeriods}periods']

    # Avoid division by zero by assigning 0 where bands_difference is zero
    result = np.where(bands_difference != 0,
                      (data['close'] - data[f'vaPoc_{nbPeriods}periods']) / bands_difference,
                      0)

    # Convert the result into a pandas Series
    return pd.Series(result, index=data.index)


# Apply the function for different periods
import numpy as np
from itertools import combinations

# Liste des périodes à analyser
periods = [6, 11, 16,21]

for nbPeriods in periods:
    # Calcul du pourcentage de la zone de valeur
    features_df[f'perct_VA{nbPeriods}P'] = np.where(
        valueArea_pct(df, nbPeriods) != 0,
        valueArea_pct(df, nbPeriods),
        np.nan
    )

    # Calcul du ratio delta volume
    features_df[f'ratio_delta_vol_VA{nbPeriods}P'] = np.where(
        df[f'vaVol_{nbPeriods}periods'] != 0,
        df[f'vaDelta_{nbPeriods}periods'] / df[f'vaVol_{nbPeriods}periods'],
        np.nan
    )

    # Différence entre le prix de clôture et le POC
    features_df[f'diffPriceClose_VA{nbPeriods}PPoc'] = np.where(
        df[f'vaPoc_{nbPeriods}periods']!= 0,
        df['close'] - df[f'vaPoc_{nbPeriods}periods'],
        np.nan
    )

    # Différence entre le prix de clôture et VAH
    features_df[f'diffPriceClose_VA{nbPeriods}PvaH'] = np.where(
        df[f'vaH_{nbPeriods}periods']!= 0,
        df['close'] - df[f'vaH_{nbPeriods}periods'],
        np.nan
    )

    # Différence entre le prix de clôture et VAL
    features_df[f'diffPriceClose_VA{nbPeriods}PvaL'] = np.where(
        df[f'vaL_{nbPeriods}periods']!= 0,
        df['close'] - df[f'vaL_{nbPeriods}periods'],
        np.nan
    )

# Génération des combinaisons de périodes
period_combinations = [(6, 11), (6, 16), (6, 21),(11, 21)]

for nbPeriods1, nbPeriods2 in period_combinations:
    # --- Proposition 1 : Chevauchement des zones de valeur ---

    # Récupération des VAH et VAL pour les deux périodes
    vaH_p1 = df[f'vaH_{nbPeriods1}periods']
    vaL_p1 = df[f'vaL_{nbPeriods1}periods']
    vaH_p2 = df[f'vaH_{nbPeriods2}periods']
    vaL_p2 = df[f'vaL_{nbPeriods2}periods']

    # Calcul du chevauchement
    min_VAH = np.minimum(vaH_p1, vaH_p2)
    max_VAL = np.maximum(vaL_p1, vaL_p2)
    overlap = np.maximum(0, min_VAH - max_VAL)

    # Calcul de l'étendue totale des zones de valeur combinées
    max_VAH_total = np.maximum(vaH_p1, vaH_p2)
    min_VAL_total = np.minimum(vaL_p1, vaL_p2)
    total_range = max_VAH_total - min_VAL_total

    # Calcul du ratio de chevauchement normalisé
    condition = (total_range != 0) & (vaH_p1 != 0) & (vaH_p2 != 0)&(vaL_p1 != 0) & (vaL_p2 != 0)
    overlap_ratio = np.where(condition, overlap / total_range, np.nan)

    # Ajout de la nouvelle feature au dataframe features_df
    features_df[f'overlap_ratio_VA_{nbPeriods1}P_{nbPeriods2}P'] = overlap_ratio

    # --- Proposition 2 : Analyse des POC ---

    # Récupération des POC pour les deux périodes
    poc_p1 = df[f'vaPoc_{nbPeriods1}periods']
    poc_p2 = df[f'vaPoc_{nbPeriods2}periods']

    # Calcul de la différence absolue entre les POC
    condition = (poc_p1 != 0) & (poc_p2 != 0)
    poc_diff = np.where(condition, poc_p1 - poc_p2, np.nan)

    # Calcul de la valeur moyenne des POC pour normalisation
    average_POC = (poc_p1 + poc_p2) / 2

    # Calcul du ratio de différence normalisé
    condition = (average_POC != 0) & (poc_p1 != 0) & (poc_p2 != 0)
    poc_diff_ratio = np.where(condition, np.abs(poc_diff) / average_POC, np.nan)

    # Ajout des nouvelles features au dataframe features_df
    features_df[f'poc_diff_{nbPeriods1}P_{nbPeriods2}P'] = poc_diff
    features_df[f'poc_diff_ratio_{nbPeriods1}P_{nbPeriods2}P'] = poc_diff_ratio

# Appliquer range_strength sur une copie de df pour ne pas modifier df
df_copy1 = df.copy()
df_with_range_strength_10_32 , range_strength_percent_in_range_10_32= range_strength(df_copy1,'range_strength_10_32', window=10, atr_multiple=3.2, min_strength=0.1)
df_with_range_strength_5_23 , range_strength_percent_in_range_5_23= range_strength(df_copy1, 'range_strength_5_23',window=5, atr_multiple=2.3, min_strength=0.1)

# Ajouter la colonne 'range_strength' à features_df
features_df['range_strength_10_32'] = df_with_range_strength_10_32['range_strength_10_32']
features_df['range_strength_5_23'] = df_with_range_strength_5_23['range_strength_5_23']

# Appliquer detect_market_regime sur une copie de df pour ne pas modifier df
df_copy = df.copy()
df_with_regime,regimeAdx_pct_infThreshold = detect_market_regimeADX(df_copy, period=14, adx_threshold=25)
# Ajouter la colonne 'market_regime' à features_df
features_df['market_regimeADX'] = df_with_regime['market_regimeADX']
features_df['is_in_range_10_32'] = df_with_range_strength_10_32['range_strength_10_32'].notna().astype(int)
features_df['is_in_range_5_23'] = df_with_range_strength_5_23['range_strength_5_23'].notna().astype(int)

conditions = [
    (features_df['market_regimeADX'] < 25),
    (features_df['market_regimeADX'] >= 25) & (features_df['market_regimeADX'] < 50),
    (features_df['market_regimeADX'] >= 50) & (features_df['market_regimeADX'] < 75),
    (features_df['market_regimeADX'] >= 75)
]

choices = [0, 1, 2, 3]

features_df['market_regimeADX_state'] = np.select(conditions, choices, default=np.nan)

# Nouvelles features - Force du renversement
features_df['bearish_reversal_force'] = np.where(df['volume'] != 0, df['VolAbv'] / df['volume'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_reversal_force'] = np.where(df['volume'] != 0, df['VolBlw'] / df['volume'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)

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
features_df['bearish_ask_bid_ratio'] = np.where(df['VolAbvBid'] != 0, df['VolAbvAsk'] / df['VolAbvBid'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_ask_bid_ratio'] = np.where(df['VolBlwAsk'] != 0, df['VolBlwBid'] / df['VolBlwAsk'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Nouvelles features - Features de Momentum:
# Moyenne des volumes
features_df['meanVolx'] = df['volume'].shift().rolling(window=5, min_periods=1).mean()

# Relative delta Momentum
features_df['ratioDeltaBlw'] = np.where(df['VolBlw'] != 0, df['DeltaBlw'] / df['VolBlw'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['ratioDeltaAbv'] = np.where(df['VolAbv'] != 0, df['DeltaAbv'] / df['VolAbv'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Relatif volume evol
features_df['diffVolCandle_0_1Ratio'] = np.where(features_df['meanVolx'] != 0,
                                            (df['volume'] - df['volume'].shift(1)) / features_df['meanVolx'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Relatif delta evol
features_df['diffVolDelta_0_1Ratio'] = np.where(features_df['meanVolx'] != 0,
                                           (df['delta'] - df['delta'].shift(1)) / features_df['meanVolx'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# cumDiffVolDelta
features_df['cumDiffVolDeltaRatio'] =  np.where(features_df['meanVolx'] != 0,(df['delta'].shift(1) + df['delta'].shift(2) + \
                                  df['delta'].shift(3) + df['delta'].shift(4) + df['delta'].shift(5))/ features_df['meanVolx'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Nouvelles features - Features de Volume Profile:
# Importance du POC
features_df['VolPocVolCandleRatio'] = np.where(df['volume'] != 0, df['volPOC'] / df['volume'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['pocDeltaPocVolRatio'] = np.where(df['volPOC'] != 0, df['deltaPOC'] / df['volPOC'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Asymétrie du volume
features_df['VolAbv_vol_ratio'] = np.where(df['volume'] != 0, (df['VolAbv']) / df['volume'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['VolBlw_vol_ratio'] = np.where(df['volume'] != 0, (df['VolBlw']) / df['volume'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)


features_df['asymetrie_volume'] = np.where(df['volume'] != 0, (df['VolAbv'] - df['VolBlw']) / df['volume'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Nouvelles features - Features Cumulatives sur les 5 dernières bougies:
# Volume spike
features_df['VolCandleMeanxRatio'] = np.where(features_df['meanVolx'] != 0, df['volume'] / features_df['meanVolx'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Nouvelles features - Caractéristiques de la zone de renversement :
features_df['bearish_ask_ratio'] = np.where(df['VolAbv'] != 0, df['VolAbvAsk'] / df['VolAbv'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bearish_bid_ratio'] = np.where(df['VolAbv'] != 0, df['VolAbvBid'] / df['VolAbv'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_ask_ratio'] = np.where(df['VolBlw'] != 0, df['VolBlwAsk'] / df['VolBlw'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_bid_ratio'] = np.where(df['VolBlw'] != 0, df['VolBlwBid'] / df['VolBlw'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Nouvelles features - Dynamique de prix dans la zone de renversement :
features_df['bearish_ask_score'] = np.where(df['VolAbv'] != 0,
                                            (df['downTickVolAbvAskDesc'] + df['repeatDownTickVolAbvAskDesc']-
                                             df['upTickVolAbvAskDesc'] - df['repeatUpTickVolAbvAskDesc'] )/ df['VolAbv'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bearish_bid_score'] = np.where(df['VolAbv'] != 0,
                                            (df['downTickVolAbvBidDesc']+ df['repeatDownTickVolAbvBidDesc']-
                                             df['upTickVolAbvBidDesc'] - df['repeatUpTickVolAbvBidDesc']
                                              ) / df['VolAbv'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bearish_imnbScore_score']=features_df['bearish_bid_score'] -features_df['bearish_ask_score']

features_df['bullish_ask_score'] = np.where(df['VolBlw'] != 0,
                                            (df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] -
                                             df['downTickVolBlwAskAsc'] - df['repeatDownTickVolBlwAskAsc']) / df['VolBlw'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bullish_bid_score'] = np.where(df['VolBlw'] != 0,
                                            (df['upTickVolBlwBidAsc'] + df['repeatUpTickVolBlwBidAsc'] -
                                             df['downTickVolBlwBidAsc'] - df['repeatDownTickVolBlwBidAsc'] )/ df['VolBlw'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_imnbScore_score']=features_df['bullish_ask_score']-features_df['bullish_bid_score']

# Nouvelles features - Order Flow:
# Imbalances haussières
features_df['bull_imbalance_low_1'] = np.where(
    df['bidVolLow'] != 0,
    df['askVolLow_1'] / df['bidVolLow'],
    addDivBy0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['askVolLow_1'] / df['bidVolLow'],
            df['bidVolLow'] != 0
        )
    )
)
# Imbalances haussières
features_df['bull_imbalance_low_2'] = np.where(
    df['bidVolLow_1'] != 0,
    df['askVolLow_2'] / df['bidVolLow_1'],
    addDivBy0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['askVolLow_2'] / df['bidVolLow_1'],
            df['bidVolLow_1'] != 0
        )
    )
)

features_df['bull_imbalance_low_3'] = np.where(
    df['bidVolLow_2'] != 0,
    df['askVolLow_3'] / df['bidVolLow_2'],
    addDivBy0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['askVolLow_3'] / df['bidVolLow_2'],
            df['bidVolLow_2'] != 0
        )
    )
)

features_df['bull_imbalance_high_0'] = np.where(
    df['bidVolHigh_1'] != 0,
    df['askVolHigh'] / df['bidVolHigh_1'],
    addDivBy0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['askVolHigh'] / df['bidVolHigh_1'],
            df['bidVolHigh_1'] != 0
        )
    )
)

features_df['bull_imbalance_high_1'] = np.where(
    df['bidVolHigh_2'] != 0,
    df['askVolHigh_1'] / df['bidVolHigh_2'],
    addDivBy0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['askVolHigh_1'] / df['bidVolHigh_2'],
            df['bidVolHigh_2'] != 0
        )
    )
)

features_df['bull_imbalance_high_2'] = np.where(
    df['bidVolHigh_3'] != 0,
    df['askVolHigh_2'] / df['bidVolHigh_3'],
    addDivBy0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['askVolHigh_2'] / df['bidVolHigh_3'],
            df['bidVolHigh_3'] != 0
        )
    )
)

# Imbalances baissières
features_df['bear_imbalance_low_0'] = np.where(
    df['askVolLow_1'] != 0,
    df['bidVolLow'] / df['askVolLow_1'],
    addDivBy0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['bidVolLow'] / df['askVolLow_1'],
            df['askVolLow_1'] != 0
        )
    )
)

features_df['bear_imbalance_low_1'] = np.where(
    df['askVolLow_2'] != 0,
    df['bidVolLow_1'] / df['askVolLow_2'],
    addDivBy0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['bidVolLow_1'] / df['askVolLow_2'],
            df['askVolLow_2'] != 0
        )
    )
)

features_df['bear_imbalance_low_2'] = np.where(
    df['askVolLow_3'] != 0,
    df['bidVolLow_2'] / df['askVolLow_3'],
    addDivBy0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['bidVolLow_2'] / df['askVolLow_3'],
            df['askVolLow_3'] != 0
        )
    )
)

features_df['bear_imbalance_high_1'] = np.where(
    df['askVolHigh'] != 0,
    df['bidVolHigh_1'] / df['askVolHigh'],
    addDivBy0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['bidVolHigh_1'] / df['askVolHigh'],
            df['askVolHigh'] != 0
        )
    )
)

features_df['bear_imbalance_high_2'] = np.where(
    df['askVolHigh_1'] != 0,
    df['bidVolHigh_2'] / df['askVolHigh_1'],
    addDivBy0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['bidVolHigh_2'] / df['askVolHigh_1'],
            df['askVolHigh_1'] != 0
        )
    )
)

features_df['bear_imbalance_high_3'] = np.where(
    df['askVolHigh_2'] != 0,
    df['bidVolHigh_3'] / df['askVolHigh_2'],
    addDivBy0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['bidVolHigh_3'] / df['askVolHigh_2'],
            df['askVolHigh_2'] != 0
        )
    )
)
# Score d'Imbalance Asymétrique
sell_pressureLow = df['bidVolLow'] + df['bidVolLow_1']
buy_pressureLow = df['askVolLow_1'] + df['askVolLow_2']
total_volumeLow = buy_pressureLow + sell_pressureLow
features_df['imbalance_score_low'] = np.where(total_volumeLow != 0,
                                              (buy_pressureLow - sell_pressureLow) / total_volumeLow, diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

sell_pressureHigh = df['bidVolHigh_1'] + df['bidVolHigh_2']
buy_pressureHigh = df['askVolHigh'] + df['askVolHigh_1']
total_volumeHigh = sell_pressureHigh + buy_pressureHigh
features_df['imbalance_score_high'] = np.where(total_volumeHigh != 0,
                                               (sell_pressureHigh - buy_pressureHigh) / total_volumeHigh, diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

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
                                                    (upTickVolAbvAsk + repeatUpTickVolAbvAsk + repeatDownTickVolAbvAsk) / df['VolAbv'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bearish_bid_abs_ratio_abv'] = np.where(df['VolAbv'] != 0,
                                                    (upTickVolAbvBid + repeatUpTickVolAbvBid + repeatDownTickVolAbvBid) / df['VolAbv'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bearish_abs_diff_abv'] = np.where(df['VolAbv'] != 0,
                                               ((upTickVolAbvAsk + repeatUpTickVolAbvAsk + repeatDownTickVolAbvAsk) -
                                                (upTickVolAbvBid + repeatUpTickVolAbvBid + repeatDownTickVolAbvBid)) / df['VolAbv'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bullish_ask_abs_ratio_blw'] = np.where(df['VolBlw'] != 0,
                                                    (upTickVolBlwAsk + repeatUpTickVolBlwAsk + repeatDownTickVolBlwAsk) / df['VolBlw'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bullish_bid_abs_ratio_blw'] = np.where(df['VolBlw'] != 0,
                                                    (upTickVolBlwBid + repeatUpTickVolBlwBid + repeatDownTickVolBlwBid) / df['VolBlw'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bullish_abs_diff_blw'] = np.where(df['VolBlw'] != 0,
                                               ((upTickVolBlwAsk + repeatUpTickVolBlwAsk + repeatDownTickVolBlwAsk) -
                                                (upTickVolBlwBid + repeatUpTickVolBlwBid + repeatDownTickVolBlwBid)) / df['VolBlw'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)


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

#----- Calcul de bearish_askBigStand_abs_ratio_abv et bearish_askBigStand_abs_ratio_abv_special
vol_ask_abv = upTickVolAbvAsk + repeatUpTickVolAbvAsk + repeatDownTickVolAbvAsk
vol_ask_abv_bigStand = upTickVolAbvAsk_bigStand + repeatUpTickVolAbvAsk_bigStand + repeatDownTickVolAbvAsk_bigStand
ratio_bearish = np.where(vol_ask_abv != 0, vol_ask_abv_bigStand / vol_ask_abv, 0)
max_ratio_bearish = calculate_max_ratio(ratio_bearish, vol_ask_abv != 0)
features_df['bearish_askBigStand_abs_ratio_abv'] = np.where(
    df['VolAbv'] == 0, valueY,  # Si VolAbv est nul, le ratio d'absorption est 0
    np.where(
        vol_ask_abv != 0,
        ratio_bearish,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish  # Utilise le ratio maximal ou une valeur spécifique
    )
)
features_df['bearish_askBigStand_abs_ratio_abv_special']=np.where(vol_ask_abv == 0, 2,0)
#-----
#----- Calcul de bearish_bidBigStand_abs_ratio_abv et bearish_bidBigStand_abs_ratio_abv_special
vol_bid_abv = upTickVolAbvBid + repeatUpTickVolAbvBid + repeatDownTickVolAbvBid
vol_bid_abv_bigStand = upTickVolAbvBid_bigStand + repeatUpTickVolAbvBid_bigStand + repeatDownTickVolAbvBid_bigStand
ratio_bearish_bid = np.where(vol_bid_abv != 0, vol_bid_abv_bigStand / vol_bid_abv, 0)
max_ratio_bearish_bid = calculate_max_ratio(ratio_bearish_bid, vol_bid_abv != 0)
features_df['bearish_bidBigStand_abs_ratio_abv'] = np.where(
    df['VolAbv'] == 0, valueY,  # Si VolAbv est nul, le ratio d'absorption est 0
    np.where(
        vol_bid_abv != 0,
        ratio_bearish_bid,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_bid  # Utilise le ratio maximal ou une valeur spécifique
    )
)
features_df['bearish_bidBigStand_abs_ratio_abv_special'] =np.where(vol_bid_abv == 0, 3, 0)
#-----

features_df['bearish_bigStand_abs_diff_abv'] = np.where(
    df['VolAbv'] != 0,
    ((upTickVolAbvAsk_bigStand + repeatUpTickVolAbvAsk_bigStand + repeatDownTickVolAbvAsk_bigStand) -
     (upTickVolAbvBid_bigStand + repeatUpTickVolAbvBid_bigStand + repeatDownTickVolAbvBid_bigStand)) / df['VolAbv'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# BigStand - Bullish (zone Blw)
#----- Calcul de bullish_askBigStand_abs_ratio_blw et bullish_askBigStand_abs_ratio_blw_special
vol_ask_blw = upTickVolBlwAsk + repeatUpTickVolBlwAsk + repeatDownTickVolBlwAsk
vol_ask_blw_bigStand = upTickVolBlwAsk_bigStand + repeatUpTickVolBlwAsk_bigStand + repeatDownTickVolBlwAsk_bigStand
ratio_bullish_ask = np.where(vol_ask_blw != 0, vol_ask_blw_bigStand / vol_ask_blw, 0)
max_ratio_bullish_ask = calculate_max_ratio(ratio_bullish_ask, vol_ask_blw != 0)
features_df['bullish_askBigStand_abs_ratio_blw'] = np.where(
    df['VolBlw'] == 0, valueY,  # Si VolBlw est nul, le ratio d'absorption est 0
    np.where(
        vol_ask_blw != 0,
        ratio_bullish_ask,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_ask  # Utilise le ratio maximal ou une valeur spécifique
    )
)
features_df['bullish_askBigStand_abs_ratio_blw_special'] =np.where(vol_ask_blw == 0, 4, 0)  # Marque les cas où le dénominateur est nul mais VolBlw n'est pas nul


#----- Calcul de bullish_bidBigStand_abs_ratio_blw et bullish_bidBigStand_abs_ratio_blw_special
vol_bid_blw = upTickVolBlwBid + repeatUpTickVolBlwBid + repeatDownTickVolBlwBid
vol_bid_blw_bigStand = upTickVolBlwBid_bigStand + repeatUpTickVolBlwBid_bigStand + repeatDownTickVolBlwBid_bigStand
ratio_bullish_bid = np.where(vol_bid_blw != 0, vol_bid_blw_bigStand / vol_bid_blw, 0)
max_ratio_bullish_bid = calculate_max_ratio(ratio_bullish_bid, vol_bid_blw != 0)
features_df['bullish_bidBigStand_abs_ratio_blw'] = np.where(
    df['VolBlw'] == 0, valueY,  # Si VolBlw est nul, le ratio d'absorption est 0
    np.where(
        vol_bid_blw != 0,
        ratio_bullish_bid,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_bid  # Utilise le ratio maximal ou une valeur spécifique
    )
)
features_df['bullish_bidBigStand_abs_ratio_blw_special'] =np.where(vol_bid_blw == 0, 5, 0)  # Marque les cas où le dénominateur est nul mais VolBlw n'est pas nul

#-----


features_df['bullish_bigStand_abs_diff_blw'] = np.where(
    df['VolBlw'] != 0,
    ((upTickVolBlwAsk_bigStand + repeatUpTickVolBlwAsk_bigStand + repeatDownTickVolBlwAsk_bigStand) -
     (upTickVolBlwBid_bigStand + repeatUpTickVolBlwBid_bigStand + repeatDownTickVolBlwBid_bigStand)) / df['VolBlw'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# BigHigh - Bearish (zone Abv)
#----- Calcul de bearish_askBigHigh_abs_ratio_abv et bearish_askBigHigh_abs_ratio_abv_special
vol_ask_abv_bighigh = upTickVolAbvAsk + repeatUpTickVolAbvAsk + repeatDownTickVolAbvAsk
vol_ask_abv_bighigh_high = upTickVolAbvAsk_bigHigh + repeatUpTickVolAbvAsk_bigHigh + repeatDownTickVolAbvAsk_bigHigh
ratio_bearish_askBigHigh = np.where(vol_ask_abv_bighigh != 0, vol_ask_abv_bighigh_high / vol_ask_abv_bighigh, 0)
max_ratio_bearish_askBigHigh = calculate_max_ratio(ratio_bearish_askBigHigh, vol_ask_abv_bighigh != 0)

features_df['bearish_askBigHigh_abs_ratio_abv'] = np.where(
    df['VolAbv'] == 0, valueY,  # Si vol_ask_abv_bighigh est nul, le ratio est 0
    np.where(
        vol_ask_abv_bighigh != 0,
        ratio_bearish_askBigHigh,  # Calcul normal
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_askBigHigh  # Cas où le dénominateur est nul
    )
)

features_df['bearish_askBigHigh_abs_ratio_abv_special'] =np.where(vol_ask_abv_bighigh_high == 0, 6, 0)  # Marque les cas où vol_ask_abv_bighigh_high est nul mais vol_ask_abv_bighigh n'est pas nul

#----- Calcul de bearish_bidBigHigh_abs_ratio_abv et bearish_bidBigHigh_abs_ratio_abv_special
vol_bid_abv_bighigh = upTickVolAbvBid + repeatUpTickVolAbvBid + repeatDownTickVolAbvBid
vol_bid_abv_bighigh_high = upTickVolAbvBid_bigHigh + repeatUpTickVolAbvBid_bigHigh + repeatDownTickVolAbvBid_bigHigh
ratio_bearish_bidBigHigh = np.where(vol_bid_abv_bighigh != 0, vol_bid_abv_bighigh_high / vol_bid_abv_bighigh, 0)
max_ratio_bearish_bidBigHigh = calculate_max_ratio(ratio_bearish_bidBigHigh, vol_bid_abv_bighigh != 0)

features_df['bearish_bidBigHigh_abs_ratio_abv'] = np.where(
    df['VolAbv'] == 0, valueY,  # Si vol_bid_abv_bighigh est nul, le ratio est 0
    np.where(
        vol_bid_abv_bighigh != 0,
        ratio_bearish_bidBigHigh,  # Calcul normal
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_bidBigHigh  # Cas où le dénominateur est nul
    )
)

features_df['bearish_bidBigHigh_abs_ratio_abv_special'] =np.where(vol_bid_abv_bighigh_high == 0, 7, 0)


features_df['bearish_bigHigh_abs_diff_abv'] = np.where(
    df['VolAbv'] != 0,
    ((upTickVolAbvAsk_bigHigh + repeatUpTickVolAbvAsk_bigHigh + repeatDownTickVolAbvAsk_bigHigh) -
     (upTickVolAbvBid_bigHigh + repeatUpTickVolAbvBid_bigHigh + repeatDownTickVolAbvBid_bigHigh)) / df['VolAbv'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# BigHigh - Bullish (zone Blw)
#----- Calcul de bullish_askBigHigh_abs_ratio_blw et bullish_askBigHigh_abs_ratio_blw_special
vol_ask_blw = upTickVolBlwAsk + repeatUpTickVolBlwAsk + repeatDownTickVolBlwAsk
vol_ask_blw_bigHigh = upTickVolBlwAsk_bigHigh + repeatUpTickVolBlwAsk_bigHigh + repeatDownTickVolBlwAsk_bigHigh
ratio_bullish_ask = np.where(vol_ask_blw != 0, vol_ask_blw_bigHigh / vol_ask_blw, 0)
max_ratio_bullish_ask = calculate_max_ratio(ratio_bullish_ask, vol_ask_blw != 0)
features_df['bullish_askBigHigh_abs_ratio_blw'] = np.where(
    df['VolBlw'] == 0, valueY,  # Si VolBlw est nul, le ratio d'absorption est 0
    np.where(
        vol_ask_blw != 0,
        ratio_bullish_ask,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_ask  # Utilise le ratio maximal ou une valeur spécifique
    )
)
features_df['bullish_askBigHigh_abs_ratio_blw_special'] =np.where(vol_ask_blw == 0, 8, 0)

#----- Calcul de bullish_bidBigHigh_abs_ratio_blw et bullish_bidBigHigh_abs_ratio_blw_special
vol_bid_blw = upTickVolBlwBid + repeatUpTickVolBlwBid + repeatDownTickVolBlwBid
vol_bid_blw_bigHigh = upTickVolBlwBid_bigHigh + repeatUpTickVolBlwBid_bigHigh + repeatDownTickVolBlwBid_bigHigh
ratio_bullish_bid = np.where(vol_bid_blw != 0, vol_bid_blw_bigHigh / vol_bid_blw, 0)
max_ratio_bullish_bid = calculate_max_ratio(ratio_bullish_bid, vol_bid_blw != 0)
features_df['bullish_bidBigHigh_abs_ratio_blw'] = np.where(
    df['VolBlw'] == 0, valueY,  # Si VolBlw est nul, le ratio d'absorption est 0
    np.where(
        vol_bid_blw != 0,
        ratio_bullish_bid,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_bid  # Utilise le ratio maximal ou une valeur spécifique
    )
)
features_df['bullish_bidBigHigh_abs_ratio_blw_special'] = np.where(vol_bid_blw == 0, 9, 0)
#-----

features_df['bullish_bigHigh_abs_diff_blw'] = np.where(
    df['VolBlw'] != 0,
    ((upTickVolBlwAsk_bigHigh + repeatUpTickVolBlwAsk_bigHigh + repeatDownTickVolBlwAsk_bigHigh) -
     (upTickVolBlwBid_bigHigh + repeatUpTickVolBlwBid_bigHigh + repeatDownTickVolBlwBid_bigHigh)) / df['VolBlw'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# a) Intensité du renversement dans la zone extrême
features_df['bearish_extrem_revIntensity_ratio'] = np.where(
    df['VolAbv'] != 0,
    ((upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem) -
     (downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem)) / df['VolAbv'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_extrem_revIntensity_ratio'] = np.where(
    df['VolBlw'] != 0,
    ((upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem) -
     (downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem)) / df['VolBlw'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# f) Ratio de volume dans la zone extrême par rapport à la zone de renversement
features_df['bearish_extrem_zone_volume_ratio'] = np.where(
    df['VolAbv'] != 0,
    (upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatUpTickVolAbvBid_extrem +
     downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatDownTickVolAbvAsk_extrem) / df['VolAbv'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_extrem_zone_volume_ratio'] = np.where(
    df['VolBlw'] != 0,
    (upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatUpTickVolBlwBid_extrem +
     downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatDownTickVolBlwAsk_extrem) / df['VolBlw'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# b) Pression acheteur/vendeur dans la zone extrême
#----- Calcul de bearish_extrem_pressure_ratio et bearish_extrem_pressure_ratio_special
vol_ask_extrem_abv = upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem
vol_bid_extrem_abv = downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem
ratio_bearish_pressure = np.where(vol_ask_extrem_abv != 0, vol_bid_extrem_abv / vol_ask_extrem_abv, 0)
max_ratio_bearish_pressure = calculate_max_ratio(ratio_bearish_pressure, vol_ask_extrem_abv != 0)
features_df['bearish_extrem_pressure_ratio'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        vol_ask_extrem_abv != 0,
        ratio_bearish_pressure,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_pressure
    )
)
features_df['bearish_extrem_pressure_ratio_special'] =np.where(vol_bid_extrem_abv == 0, 10, 0)

#----- Calcul de bullish_extrem_pressure_ratio et bullish_extrem_pressure_ratio_special
vol_bid_extrem_blw = downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem
vol_ask_extrem_blw = upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem
ratio_bullish_pressure = np.where(vol_bid_extrem_blw != 0, vol_ask_extrem_blw / vol_bid_extrem_blw, 0)
max_ratio_bullish_pressure = calculate_max_ratio(ratio_bullish_pressure, vol_bid_extrem_blw != 0)
features_df['bullish_extrem_pressure_ratio'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        vol_bid_extrem_blw != 0,
        ratio_bullish_pressure,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_pressure
    )
)
features_df['bullish_extrem_pressure_ratio_special'] =np.where(vol_ask_extrem_blw == 0, 11, 0)
#-----


# c) Absorption dans la zone extrême
#----- Calcul de bearish_extrem_abs_ratio et bearish_extrem_abs_ratio_special
vol_bid_abs_extrem_abv = downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem
vol_ask_abs_extrem_abv = upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem
ratio_bearish_abs = np.where(vol_bid_abs_extrem_abv != 0, vol_ask_abs_extrem_abv / vol_bid_abs_extrem_abv, 0)
max_ratio_bearish_abs = calculate_max_ratio(ratio_bearish_abs, vol_bid_abs_extrem_abv != 0)
features_df['bearish_extrem_abs_ratio'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        vol_bid_abs_extrem_abv != 0,
        ratio_bearish_abs,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_abs
    )
)

features_df['bearish_extrem_abs_ratio_special'] =np.where(vol_ask_abs_extrem_abv == 0, 12, 0)

#----- Calcul de bullish_extrem_abs_ratio et bullish_extrem_abs_ratio_special
vol_bid_abs_extrem_blw = downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem
vol_ask_abs_extrem_blw = upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem
ratio_bullish_abs = np.where(vol_bid_abs_extrem_blw != 0, vol_ask_abs_extrem_blw / vol_bid_abs_extrem_blw, 0)
max_ratio_bullish_abs = calculate_max_ratio(ratio_bullish_abs, vol_bid_abs_extrem_blw != 0)
features_df['bullish_extrem_abs_ratio'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        vol_bid_abs_extrem_blw != 0,
        ratio_bullish_abs,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_abs
    )
)
features_df['bullish_extrem_abs_ratio_special'] = np.where(vol_ask_abs_extrem_blw == 0, 13, 0)
#-----


# d) Comparaison de l'activité extrême vs. reste de la zone de renversement
#----- Calcul de bearish_extrem_vs_rest_activity et bearish_extrem_vs_rest_activity_special
vol_extrem_bearish = (
    upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
    downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem
)
rest_vol_bearish = df['VolAbv'] - vol_extrem_bearish
ratio_bearish_activity = np.where(rest_vol_bearish != 0, vol_extrem_bearish / rest_vol_bearish, 0)
max_ratio_bearish_activity = calculate_max_ratio(ratio_bearish_activity, rest_vol_bearish != 0)
features_df['bearish_extrem_vs_rest_activity'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        rest_vol_bearish != 0,
        ratio_bearish_activity,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_activity
    )
)
features_df['bearish_extrem_vs_rest_activity_special'] = np.where(rest_vol_bearish == 0, 14, 0)

#----- Calcul de bullish_extrem_vs_rest_activity et bullish_extrem_vs_rest_activity_special
vol_extrem_bullish = (
    upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
    downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem
)
rest_vol_bullish = df['VolBlw'] - vol_extrem_bullish
ratio_bullish_activity = np.where(rest_vol_bullish != 0, vol_extrem_bullish / rest_vol_bullish, 0)
max_ratio_bullish_activity = calculate_max_ratio(ratio_bullish_activity, rest_vol_bullish != 0)
features_df['bullish_extrem_vs_rest_activity'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        rest_vol_bullish != 0,
        ratio_bullish_activity,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_activity
    )
)
features_df['bullish_extrem_vs_rest_activity_special'] = np.where(rest_vol_bullish == 0, 15, 0)
#-----

# e) Indicateur de continuation vs. renversement dans la zone extrême
#----- Calcul de bearish_continuation_vs_reversal et bearish_continuation_vs_reversal_special
total_vol_bearish_extrem = (
    upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
    downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem
)
continuation_vol_bearish_extrem = (
    upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem
)
ratio_bearish_continuation = np.where(total_vol_bearish_extrem != 0, continuation_vol_bearish_extrem / total_vol_bearish_extrem, 0)
max_ratio_bearish_continuation = calculate_max_ratio(ratio_bearish_continuation, total_vol_bearish_extrem != 0)
features_df['bearish_continuation_vs_reversal'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        total_vol_bearish_extrem != 0,
        ratio_bearish_continuation,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_continuation
    )
)
features_df['bearish_continuation_vs_reversal_special'] = np.where(continuation_vol_bearish_extrem == 0, 16, 0)


#----- Calcul de bullish_continuation_vs_reversal et bullish_continuation_vs_reversal_special
total_vol_bullish_extrem = (
    upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
    downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem
)
continuation_vol_bullish_extrem = (
    downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem
)
ratio_bullish_continuation = np.where(total_vol_bullish_extrem != 0, continuation_vol_bullish_extrem / total_vol_bullish_extrem, 0)
max_ratio_bullish_continuation = calculate_max_ratio(ratio_bullish_continuation, total_vol_bullish_extrem != 0)
features_df['bullish_continuation_vs_reversal'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        total_vol_bullish_extrem != 0,
        ratio_bullish_continuation,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_continuation
    )
)
features_df['bullish_continuation_vs_reversal_special'] =np.where(continuation_vol_bullish_extrem == 0, 17, 0)
#-----


# f) Ratio de repeat ticks dans la zone extrême
#----- Calcul de bearish_repeat_ticks_ratio et bearish_repeat_ticks_ratio_special
total_vol_bearish_extrem = (
    upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
    downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem
)
repeat_vol_bearish_extrem = (
    repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
    repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem
)
ratio_bearish_repeat_ticks = np.where(total_vol_bearish_extrem != 0, repeat_vol_bearish_extrem / total_vol_bearish_extrem, 0)
max_ratio_bearish_repeat_ticks = calculate_max_ratio(ratio_bearish_repeat_ticks, total_vol_bearish_extrem != 0)
features_df['bearish_repeat_ticks_ratio'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        total_vol_bearish_extrem != 0,
        ratio_bearish_repeat_ticks,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_repeat_ticks
    )
)
features_df['bearish_repeat_ticks_ratio_special'] =np.where(repeat_vol_bearish_extrem == 0, 18, 0)

#----- Calcul de bullish_repeat_ticks_ratio et bullish_repeat_ticks_ratio_special
total_vol_bullish_extrem = (
    upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
    downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem
)
repeat_vol_bullish_extrem = (
    repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
    repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem
)
ratio_bullish_repeat_ticks = np.where(total_vol_bullish_extrem != 0, repeat_vol_bullish_extrem / total_vol_bullish_extrem, 0)
max_ratio_bullish_repeat_ticks = calculate_max_ratio(ratio_bullish_repeat_ticks, total_vol_bullish_extrem != 0)
features_df['bullish_repeat_ticks_ratio'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        total_vol_bullish_extrem != 0,
        ratio_bullish_repeat_ticks,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_repeat_ticks
    )
)
features_df['bullish_repeat_ticks_ratio_special'] = np.where(repeat_vol_bullish_extrem == 0, 19, 0)


# g) Big trades dans la zone extrême
# Pour les bougies bearish (zone Abv)


#----- Calcul de bearish_big_trade_ratio_extrem et bearish_big_trade_ratio_extrem_special
total_vol_bearish_extrem = (
    upTickVolAbvAsk_extrem + repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
    downTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem + repeatUpTickVolAbvBid_extrem
)
big_trade_vol_bearish_extrem = (
    upTickVolAbvAsk_bigStand_extrem + repeatUpTickVolAbvAsk_bigStand_extrem + repeatDownTickVolAbvAsk_bigStand_extrem +
    downTickVolAbvBid_bigStand_extrem + repeatDownTickVolAbvBid_bigStand_extrem + repeatUpTickVolAbvBid_bigStand_extrem
)
ratio_bearish_big_trade = np.where(total_vol_bearish_extrem != 0, big_trade_vol_bearish_extrem / total_vol_bearish_extrem, 0)
max_ratio_bearish_big_trade = calculate_max_ratio(ratio_bearish_big_trade, total_vol_bearish_extrem != 0)
features_df['bearish_big_trade_ratio_extrem'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        total_vol_bearish_extrem != 0,
        ratio_bearish_big_trade,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_big_trade
    )
)
features_df['bearish_big_trade_ratio_extrem_special'] =np.where(big_trade_vol_bearish_extrem == 0, 21, 0)
#----- Calcul de bearish_big_trade_imbalance et bearish_big_trade_imbalance_special
imbalance_vol_bearish_big_trade = (
    (upTickVolAbvAsk_bigStand_extrem + repeatUpTickVolAbvAsk_bigStand_extrem + repeatDownTickVolAbvAsk_bigStand_extrem) -
    (downTickVolAbvBid_bigStand_extrem + repeatDownTickVolAbvBid_bigStand_extrem + repeatUpTickVolAbvBid_bigStand_extrem)
)
total_vol_bearish_big_trade = (
    upTickVolAbvAsk_bigStand_extrem + repeatUpTickVolAbvAsk_bigStand_extrem + repeatDownTickVolAbvAsk_bigStand_extrem +
    downTickVolAbvBid_bigStand_extrem + repeatDownTickVolAbvBid_bigStand_extrem + repeatUpTickVolAbvBid_bigStand_extrem
)
ratio_bearish_big_trade_imbalance = np.where(total_vol_bearish_big_trade != 0, imbalance_vol_bearish_big_trade / total_vol_bearish_big_trade, 0)
max_ratio_bearish_big_trade_imbalance = calculate_max_ratio(ratio_bearish_big_trade_imbalance, total_vol_bearish_big_trade != 0)
features_df['bearish_big_trade_imbalance'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        total_vol_bearish_big_trade != 0,
        ratio_bearish_big_trade_imbalance,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_big_trade_imbalance
    )
)
features_df['bearish_big_trade_imbalance_special'] = np.where(imbalance_vol_bearish_big_trade == 0, 22, 0)
#-----


# Pour les bougies bullish (zone Blw)
#----- Calcul de bullish_big_trade_ratio_extrem et bullish_big_trade_ratio_extrem_special
total_vol_bullish_extrem = (
    upTickVolBlwAsk_extrem + repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
    downTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem + repeatUpTickVolBlwBid_extrem
)
big_trade_vol_bullish_extrem = (
    upTickVolBlwAsk_bigStand_extrem + repeatUpTickVolBlwAsk_bigStand_extrem + repeatDownTickVolBlwAsk_bigStand_extrem +
    downTickVolBlwBid_bigStand_extrem + repeatDownTickVolBlwBid_bigStand_extrem + repeatUpTickVolBlwBid_bigStand_extrem
)
ratio_bullish_big_trade = np.where(total_vol_bullish_extrem != 0, big_trade_vol_bullish_extrem / total_vol_bullish_extrem, 0)
max_ratio_bullish_big_trade = calculate_max_ratio(ratio_bullish_big_trade, total_vol_bullish_extrem != 0)
features_df['bullish_big_trade_ratio_extrem'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        total_vol_bullish_extrem != 0,
        ratio_bullish_big_trade,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_big_trade
    )
)
features_df['bullish_big_trade_ratio_extrem_special'] = np.where(big_trade_vol_bullish_extrem == 0, 23, 0)
#----- Calcul de bullish_big_trade_imbalance et bullish_big_trade_imbalance_special
imbalance_vol_bullish_big_trade = (
    (upTickVolBlwAsk_bigStand_extrem + repeatUpTickVolBlwAsk_bigStand_extrem + repeatDownTickVolBlwAsk_bigStand_extrem) -
    (downTickVolBlwBid_bigStand_extrem + repeatDownTickVolBlwBid_bigStand_extrem + repeatUpTickVolBlwBid_bigStand_extrem)
)
total_vol_bullish_big_trade = (
    upTickVolBlwAsk_bigStand_extrem + repeatUpTickVolBlwAsk_bigStand_extrem + repeatDownTickVolBlwAsk_bigStand_extrem +
    downTickVolBlwBid_bigStand_extrem + repeatDownTickVolBlwBid_bigStand_extrem + repeatUpTickVolBlwBid_bigStand_extrem
)
ratio_bullish_big_trade_imbalance = np.where(total_vol_bullish_big_trade != 0, imbalance_vol_bullish_big_trade / total_vol_bullish_big_trade, 0)
max_ratio_bullish_big_trade_imbalance = calculate_max_ratio(ratio_bullish_big_trade_imbalance, total_vol_bullish_big_trade != 0)
features_df['bullish_big_trade_imbalance'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        total_vol_bullish_big_trade != 0,
        ratio_bullish_big_trade_imbalance,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_big_trade_imbalance
    )
)
features_df['bullish_big_trade_imbalance_special'] =np.where(imbalance_vol_bullish_big_trade == 0, 24, 0)
#-----


# Nouvelles features exploitant ASC et DSC
print_notification("Calcul des nouvelles features ASC/DSC")

# 1. Dynamique ASC/DSC dans la zone de renversement
#----- Calcul de bearish_asc_dsc_ratio et bearish_asc_dsc_ratio_special
vol_bid_desc_bearish = df['downTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidDesc']
vol_ask_asc_bearish = df['upTickVolAbvAskAsc'] + df['repeatUpTickVolAbvAskAsc'] + df['repeatDownTickVolAbvAskAsc']
ratio_bearish_asc_dsc = np.where(vol_bid_desc_bearish != 0, vol_ask_asc_bearish / vol_bid_desc_bearish, 0)
max_ratio_bearish_asc_dsc = calculate_max_ratio(ratio_bearish_asc_dsc, vol_bid_desc_bearish != 0)
features_df['bearish_asc_dsc_ratio'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        vol_bid_desc_bearish != 0,
        ratio_bearish_asc_dsc,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_asc_dsc
    )
)
features_df['bearish_asc_dsc_ratio_special'] = np.where(vol_ask_asc_bearish == 0, 25, 0)
#-----


features_df['bearish_asc_dynamics'] = np.where(
    df['VolAbv'] != 0,
    (df['upTickVolAbvAskAsc'] + df['repeatUpTickVolAbvAskAsc'] + df['repeatDownTickVolAbvAskAsc']) / df['VolAbv'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bearish_dsc_dynamics'] = np.where(
    df['VolAbv'] != 0,
    (df['downTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidDesc']) / df['VolAbv'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

#----- Calcul de bullish_asc_dsc_ratio et bullish_asc_dsc_ratio_special
vol_bid_desc_bullish = df['downTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidDesc']
vol_ask_asc_bullish = df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] + df['repeatDownTickVolBlwAskAsc']
ratio_bullish_asc_dsc = np.where(vol_bid_desc_bullish != 0, vol_ask_asc_bullish / vol_bid_desc_bullish, 0)
max_ratio_bullish_asc_dsc = calculate_max_ratio(ratio_bullish_asc_dsc, vol_bid_desc_bullish != 0)
features_df['bullish_asc_dsc_ratio'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        vol_bid_desc_bullish != 0,
        ratio_bullish_asc_dsc,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_asc_dsc
    )
)
features_df['bullish_asc_dsc_ratio_special'] = np.where(vol_ask_asc_bullish == 0, 26, 0)
#-----


features_df['bullish_asc_dynamics'] = np.where(
    df['VolBlw'] != 0,
    (df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] + df['repeatDownTickVolBlwAskAsc']) / df['VolBlw'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bullish_dsc_dynamics'] = np.where(
    df['VolBlw'] != 0,
    (df['downTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidDesc']) / df['VolBlw'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

# 2. Déséquilibre Ask-Bid dans les phases ASC et DSC
print_notification("Calcul des features de déséquilibre Ask-Bid dans les phases ASC et DSC")

features_df['bearish_asc_ask_bid_imbalance'] = np.where(
    df['VolAbv'] != 0,
    (df['upTickVolAbvAskAsc'] + df['repeatUpTickVolAbvAskAsc'] + df['repeatDownTickVolAbvAskAsc']) / df['VolAbv'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bearish_dsc_ask_bid_imbalance'] = np.where(
    df['VolAbv'] != 0,
    (df['downTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidDesc']) / df['VolAbv'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bearish_imbalance_evolution'] = np.where(
    (df['VolAbv'] != 0),
    features_df['bearish_asc_ask_bid_imbalance'] - features_df['bearish_dsc_ask_bid_imbalance'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bearish_asc_ask_bid_delta_imbalance'] = np.where(
    df['VolAbv'] != 0,
    (df['upTickVolAbvAskAsc'] + df['repeatUpTickVolAbvAskAsc'] + df['repeatDownTickVolAbvAskAsc'] -
     (df['upTickVolAbvBidAsc'] + df['repeatUpTickVolAbvBidAsc'] + df['repeatDownTickVolAbvBidAsc'])) / df['VolAbv'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bearish_dsc_ask_bid_delta_imbalance'] = np.where(
    df['VolAbv'] != 0,
    (df['upTickVolAbvAskDesc'] + df['repeatUpTickVolAbvAskDesc'] + df['repeatDownTickVolAbvAskDesc'] -
     (df['downTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidDesc'])) / df['VolAbv'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bullish_asc_ask_bid_imbalance'] = np.where(
    df['VolBlw'] != 0,
    (df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] + df['repeatDownTickVolBlwAskAsc']) / df['VolBlw'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bullish_dsc_ask_bid_imbalance'] = np.where(
    df['VolBlw'] != 0,
    (df['downTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidDesc']) / df['VolBlw'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bullish_imbalance_evolution'] = np.where(
    (df['VolBlw'] != 0) ,
    features_df['bullish_asc_ask_bid_imbalance'] - features_df['bullish_dsc_ask_bid_imbalance'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bullish_asc_ask_bid_delta_imbalance'] = np.where(
    df['VolBlw'] != 0,
    (df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] + df['repeatDownTickVolBlwAskAsc'] -
     (df['upTickVolBlwBidAsc'] + df['repeatUpTickVolBlwBidAsc'] + df['repeatDownTickVolBlwBidAsc'])) / df['VolBlw'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bullish_dsc_ask_bid_delta_imbalance'] = np.where(
    df['VolBlw'] != 0,
    (df['upTickVolBlwAskDesc'] + df['repeatUpTickVolBlwAskDesc'] + df['repeatDownTickVolBlwAskDesc'] -
     (df['downTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidDesc'])) / df['VolBlw'],
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# 3. Importance et dynamique de la zone extrême
print_notification("Calcul des features de la zone extrême")

# Features bearish
features_df['extrem_asc_ratio_bearish'] = np.where(
    df['VolAbv'] != 0,
    (df['upTickVolAbvAskAsc_extrem'] + df['repeatUpTickVolAbvAskAsc_extrem'] + df['repeatDownTickVolAbvAskAsc_extrem']) / df['VolAbv'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['extrem_dsc_ratio_bearish'] = np.where(
    df['VolAbv'] != 0,
    (df['downTickVolAbvBidDesc_extrem'] + df['repeatUpTickVolAbvBidDesc_extrem'] + df['repeatDownTickVolAbvBidDesc_extrem']) / df['VolAbv'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['extrem_zone_significance_bearish'] = np.where(
    ( df['VolAbv'] != 0),
    (features_df['extrem_asc_ratio_bearish'] + features_df['extrem_dsc_ratio_bearish']) *
    abs(features_df['extrem_asc_ratio_bearish'] - features_df['extrem_dsc_ratio_bearish']),
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

#----- Calcul de extrem_ask_bid_imbalance_bearish et extrem_ask_bid_imbalance_bearish_special
total_vol_extrem_bearish = (
    df['upTickVolAbvAskAsc_extrem'] + df['repeatUpTickVolAbvAskAsc_extrem'] + df['repeatDownTickVolAbvAskAsc_extrem'] +
    df['downTickVolAbvBidDesc_extrem'] + df['repeatUpTickVolAbvBidDesc_extrem'] + df['repeatDownTickVolAbvBidDesc_extrem']
)
imbalance_vol_extrem_bearish = (
    df['upTickVolAbvAskAsc_extrem'] + df['repeatUpTickVolAbvAskAsc_extrem'] + df['repeatDownTickVolAbvAskAsc_extrem'] -
    (df['downTickVolAbvBidDesc_extrem'] + df['repeatUpTickVolAbvBidDesc_extrem'] + df['repeatDownTickVolAbvBidDesc_extrem'])
)
ratio_extrem_ask_bid_imbalance_bearish = np.where(total_vol_extrem_bearish != 0, imbalance_vol_extrem_bearish / total_vol_extrem_bearish, 0)
max_ratio_extrem_ask_bid_imbalance_bearish = calculate_max_ratio(ratio_extrem_ask_bid_imbalance_bearish, total_vol_extrem_bearish != 0)
features_df['extrem_ask_bid_imbalance_bearish'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        total_vol_extrem_bearish != 0,
        ratio_extrem_ask_bid_imbalance_bearish,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_extrem_ask_bid_imbalance_bearish
    )
)
features_df['extrem_ask_bid_imbalance_bearish_special'] = np.where(imbalance_vol_extrem_bearish == 0, 27, 0)
#-----


features_df['extrem_asc_dsc_comparison_bearish'] = np.where(
    ( df['VolAbv'] != 0),
    features_df['extrem_asc_ratio_bearish'] / features_df['extrem_dsc_ratio_bearish'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

#----- Calcul de bearish_repeat_ticks_ratio et bearish_repeat_ticks_ratio_special
total_vol_bearish_extrem = (
    upTickVolAbvAsk_extrem + downTickVolAbvBid_extrem + repeatUpTickVolAbvAsk_extrem +
    repeatDownTickVolAbvAsk_extrem + repeatUpTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem
)
repeat_vol_bearish_extrem = (
    repeatUpTickVolAbvAsk_extrem + repeatDownTickVolAbvAsk_extrem +
    repeatUpTickVolAbvBid_extrem + repeatDownTickVolAbvBid_extrem
)
ratio_bearish_repeat_ticks = np.where(total_vol_bearish_extrem != 0, repeat_vol_bearish_extrem / total_vol_bearish_extrem, 0)
max_ratio_bearish_repeat_ticks = calculate_max_ratio(ratio_bearish_repeat_ticks, total_vol_bearish_extrem != 0)
features_df['bearish_repeat_ticks_ratio'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        total_vol_bearish_extrem != 0,
        ratio_bearish_repeat_ticks,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_repeat_ticks
    )
)
features_df['bearish_repeat_ticks_ratio_special'] = np.where(repeat_vol_bearish_extrem == 0, 28, 0)
#-----


# Features bullish
features_df['extrem_asc_ratio_bullish'] = np.where(
    df['VolBlw'] != 0,
    (df['upTickVolBlwAskAsc_extrem'] + df['repeatUpTickVolBlwAskAsc_extrem'] + df['repeatDownTickVolBlwAskAsc_extrem']) / df['VolBlw'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['extrem_dsc_ratio_bullish'] = np.where(
    df['VolBlw'] != 0,
    (df['downTickVolBlwBidDesc_extrem'] + df['repeatUpTickVolBlwBidDesc_extrem'] + df['repeatDownTickVolBlwBidDesc_extrem']) / df['VolBlw'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['extrem_zone_significance_bullish'] = np.where(
    ( df['VolBlw'] != 0),
    (features_df['extrem_asc_ratio_bullish'] + features_df['extrem_dsc_ratio_bullish']) *
    abs(features_df['extrem_asc_ratio_bullish'] - features_df['extrem_dsc_ratio_bullish']),
    diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

#----- Calcul de extrem_ask_bid_imbalance_bullish et extrem_ask_bid_imbalance_bullish_special
total_vol_extrem_bullish = (
    df['upTickVolBlwAskAsc_extrem'] + df['repeatUpTickVolBlwAskAsc_extrem'] + df['repeatDownTickVolBlwAskAsc_extrem'] +
    df['downTickVolBlwBidDesc_extrem'] + df['repeatUpTickVolBlwBidDesc_extrem'] + df['repeatDownTickVolBlwBidDesc_extrem']
)
imbalance_vol_extrem_bullish = (
    df['upTickVolBlwAskAsc_extrem'] + df['repeatUpTickVolBlwAskAsc_extrem'] + df['repeatDownTickVolBlwAskAsc_extrem'] -
    (df['downTickVolBlwBidDesc_extrem'] + df['repeatUpTickVolBlwBidDesc_extrem'] + df['repeatDownTickVolBlwBidDesc_extrem'])
)
ratio_extrem_ask_bid_imbalance_bullish = np.where(total_vol_extrem_bullish != 0, imbalance_vol_extrem_bullish / total_vol_extrem_bullish, 0)
max_ratio_extrem_ask_bid_imbalance_bullish = calculate_max_ratio(ratio_extrem_ask_bid_imbalance_bullish, total_vol_extrem_bullish != 0)
features_df['extrem_ask_bid_imbalance_bullish'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        total_vol_extrem_bullish != 0,
        ratio_extrem_ask_bid_imbalance_bullish,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_extrem_ask_bid_imbalance_bullish
    )
)
features_df['extrem_ask_bid_imbalance_bullish_special'] = np.where(imbalance_vol_extrem_bullish == 0, 29, 0)
#-----


features_df['extrem_asc_dsc_comparison_bullish'] = np.where(
    ( df['VolBlw'] != 0),
    features_df['extrem_asc_ratio_bullish'] / features_df['extrem_dsc_ratio_bullish'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

#----- Calcul de bullish_repeat_ticks_ratio et bullish_repeat_ticks_ratio_special
total_vol_bullish_extrem = (
    upTickVolBlwAsk_extrem + downTickVolBlwBid_extrem + repeatUpTickVolBlwAsk_extrem +
    repeatDownTickVolBlwAsk_extrem + repeatUpTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem
)
repeat_vol_bullish_extrem = (
    repeatUpTickVolBlwAsk_extrem + repeatDownTickVolBlwAsk_extrem +
    repeatUpTickVolBlwBid_extrem + repeatDownTickVolBlwBid_extrem
)
ratio_bullish_repeat_ticks = np.where(total_vol_bullish_extrem != 0, repeat_vol_bullish_extrem / total_vol_bullish_extrem, 0)
max_ratio_bullish_repeat_ticks = calculate_max_ratio(ratio_bullish_repeat_ticks, total_vol_bullish_extrem != 0)
features_df['bullish_repeat_ticks_ratio'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        total_vol_bullish_extrem != 0,
        ratio_bullish_repeat_ticks,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_repeat_ticks
    )
)
features_df['bullish_repeat_ticks_ratio_special'] = np.where(repeat_vol_bullish_extrem == 0, 30, 0)
#-----

#----- Calcul de bearish_absorption_ratio et bearish_absorption_ratio_special
vol_ask_asc_bearish = df['upTickVolAbvAskAsc'] + df['repeatUpTickVolAbvAskAsc'] + df['repeatDownTickVolAbvAskAsc']
vol_bid_desc_bearish = df['downTickVolAbvBidDesc'] + df['repeatUpTickVolAbvBidDesc'] + df['repeatDownTickVolAbvBidDesc']
ratio_bearish_absorption = np.where(vol_ask_asc_bearish != 0, vol_bid_desc_bearish / vol_ask_asc_bearish, 0)
max_ratio_bearish_absorption = calculate_max_ratio(ratio_bearish_absorption, vol_ask_asc_bearish != 0)
features_df['bearish_absorption_ratio'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        vol_ask_asc_bearish != 0,
        ratio_bearish_absorption,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_absorption
    )
)
features_df['bearish_absorption_ratio_special'] =np.where(vol_bid_desc_bearish == 0, 31, 0)

#----- Calcul de bullish_absorption_ratio et bullish_absorption_ratio_special
vol_bid_desc_bullish = df['downTickVolBlwBidDesc'] + df['repeatUpTickVolBlwBidDesc'] + df['repeatDownTickVolBlwBidDesc']
vol_ask_asc_bullish = df['upTickVolBlwAskAsc'] + df['repeatUpTickVolBlwAskAsc'] + df['repeatDownTickVolBlwAskAsc']
ratio_bullish_absorption = np.where(vol_bid_desc_bullish != 0, vol_ask_asc_bullish / vol_bid_desc_bullish, 0)
max_ratio_bullish_absorption = calculate_max_ratio(ratio_bullish_absorption, vol_bid_desc_bullish != 0)
features_df['bullish_absorption_ratio'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        vol_bid_desc_bullish != 0,
        ratio_bullish_absorption,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_absorption
    )
)
features_df['bullish_absorption_ratio_special'] = np.where(vol_ask_asc_bullish == 0, 32, 0)
#-----


# 5. Dynamique des gros trades dans la zone extrême
print_notification("Calcul des features de dynamique des gros trades dans la zone extrême")

# Pour les bougies bearish (zone Abv)
#----- Calcul de bearish_big_trade_ratio2_extrem et bearish_big_trade_ratio2_extrem_special
total_vol_bearish_extrem = upTickVolAbvAsk_extrem + downTickVolAbvBid_extrem
big_trade_vol_bearish_extrem = (
    upTickVolAbvAsk_bigStand_extrem + repeatUpTickVolAbvAsk_bigStand_extrem + repeatDownTickVolAbvAsk_bigStand_extrem +
    downTickVolAbvBid_bigStand_extrem + repeatUpTickVolAbvBid_bigStand_extrem + repeatDownTickVolAbvBid_bigStand_extrem
)
ratio_bearish_big_trade2 = np.where(total_vol_bearish_extrem != 0, big_trade_vol_bearish_extrem / total_vol_bearish_extrem, 0)
max_ratio_bearish_big_trade2 = calculate_max_ratio(ratio_bearish_big_trade2, total_vol_bearish_extrem != 0)
features_df['bearish_big_trade_ratio2_extrem'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        total_vol_bearish_extrem != 0,
        ratio_bearish_big_trade2,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_big_trade2
    )
)
features_df['bearish_big_trade_ratio2_extrem_special'] = np.where(big_trade_vol_bearish_extrem == 0, 33, 0)

#----- Calcul de bullish_big_trade_ratio2_extrem et bullish_big_trade_ratio2_extrem_special
total_vol_bullish_extrem = upTickVolBlwAsk_extrem + downTickVolBlwBid_extrem
big_trade_vol_bullish_extrem = (
    upTickVolBlwAsk_bigStand_extrem + repeatUpTickVolBlwAsk_bigStand_extrem + repeatDownTickVolBlwAsk_bigStand_extrem +
    downTickVolBlwBid_bigStand_extrem + repeatUpTickVolBlwBid_bigStand_extrem + repeatDownTickVolBlwBid_bigStand_extrem
)
ratio_bullish_big_trade2 = np.where(total_vol_bullish_extrem != 0, big_trade_vol_bullish_extrem / total_vol_bullish_extrem, 0)
max_ratio_bullish_big_trade2 = calculate_max_ratio(ratio_bullish_big_trade2, total_vol_bullish_extrem != 0)
features_df['bullish_big_trade_ratio2_extrem'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        total_vol_bullish_extrem != 0,
        ratio_bullish_big_trade2,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_big_trade2
    )
)
features_df['bullish_big_trade_ratio2_extrem_special'] = np.where(big_trade_vol_bullish_extrem == 0, 34, 0)
#-----


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
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

#----- Calcul de absorption_intensity_repeat_bearish_count et absorption_intensity_repeat_bearish_count_special
repeat_count_bearish = repeatUpTickCountAbvAsk + repeatDownTickCountAbvAsk
total_count_bearish = features_df['total_count_abv']
ratio_absorption_intensity_repeat_bearish = np.where(total_count_bearish != 0, repeat_count_bearish / total_count_bearish, 0)
max_ratio_absorption_intensity_repeat_bearish = calculate_max_ratio(ratio_absorption_intensity_repeat_bearish, total_count_bearish != 0)
features_df['absorption_intensity_repeat_bearish_count'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        total_count_bearish != 0,
        ratio_absorption_intensity_repeat_bearish,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_absorption_intensity_repeat_bearish
    )
)
features_df['absorption_intensity_repeat_bearish_count_special'] = np.where(repeat_count_bearish == 0, 35, 0)
#-----


features_df['bearish_repeatAskBid_ratio'] = np.where(
    df['VolAbv'] != 0,
    (repeatUpTickVolAbvAsk + repeatUpTickVolAbvBid + repeatDownTickVolAbvAsk + repeatDownTickVolAbvBid) / df['VolAbv'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)



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
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

#----- Calcul de absorption_intensity_repeat_bullish_count et absorption_intensity_repeat_bullish_count_special
repeat_count_bullish = repeatUpTickCountBlwBid + repeatDownTickCountBlwBid
total_count_bullish = features_df['total_count_blw']
ratio_absorption_intensity_repeat_bullish = np.where(total_count_bullish != 0, repeat_count_bullish / total_count_bullish, 0)
max_ratio_absorption_intensity_repeat_bullish = calculate_max_ratio(ratio_absorption_intensity_repeat_bullish, total_count_bullish != 0)
features_df['absorption_intensity_repeat_bullish_count'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        total_count_bullish != 0,
        ratio_absorption_intensity_repeat_bullish,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_absorption_intensity_repeat_bullish
    )
)
features_df['absorption_intensity_repeat_bullish_count_special'] = np.where(repeat_count_bullish == 0, 36, 0)
#-----


features_df['bullish_repeatAskBid_ratio'] = np.where(
    df['VolBlw'] != 0,
    (repeatUpTickVolBlwAsk + repeatUpTickVolBlwBid + repeatDownTickVolBlwAsk + repeatDownTickVolBlwBid) / df['VolBlw'],
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)


features_df['count_AbvBlw_asym_ratio'] = np.where(
    features_df['total_count_abv'] + features_df['total_count_blw'] != 0,
    (features_df['total_count_abv'] - features_df['total_count_blw']) /
    (features_df['total_count_abv'] + features_df['total_count_blw']),
    addDivBy0 if DEFAULT_DIV_BY0 else 0
)
features_df['count_abv_tot_ratio'] = np.where(
    features_df['total_count_abv'] + features_df['total_count_blw'] != 0,
    (features_df['total_count_abv']) /
    (features_df['total_count_abv'] + features_df['total_count_blw']),
    addDivBy0 if DEFAULT_DIV_BY0 else 0
)

features_df['count_blw_tot_ratio'] = np.where(
    features_df['total_count_abv'] + features_df['total_count_blw'] != 0,
    (features_df['total_count_blw']) /
    (features_df['total_count_abv'] + features_df['total_count_blw']),
    addDivBy0 if DEFAULT_DIV_BY0 else 0
)


print_notification("Calcul des features de la zone 6Ticks")

# a) Ratio de volume _6Tick
features_df['bearish_volume_ratio_6Tick'] = np.where(df['VolBlw'] != 0, df['VolBlw_6Tick'] / df['VolBlw'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_volume_ratio_6Tick'] = np.where(df['VolAbv'] != 0, df['VolAbv_6Tick'] / df['VolAbv'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)

# b) Delta _6Tick
features_df['bearish_relatif_ratio_6Tick'] = np.where(df['VolBlw_6Tick'] != 0, df['DeltaBlw_6Tick'] / df['VolBlw_6Tick'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_relatif_ratio_6Tick'] = np.where(df['VolAbv_6Tick'] != 0, df['DeltaAbv_6Tick'] / df['VolAbv_6Tick'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# c) Delta relatif _6Tick
features_df['bearish_relatifDelta_ratio_6Tick'] = np.where(
    (df['VolBlw_6Tick'] != 0) & (df['volume'] != 0) & (df['delta'] != 0),
    (df['DeltaBlw_6Tick'] / df['VolBlw_6Tick']) / (df['delta'] / df['volume']),
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['bullish_relatifDelta_ratio_6Tick'] = np.where(
    (df['VolAbv_6Tick'] != 0) & (df['volume'] != 0) & (df['delta'] != 0),
    (df['DeltaAbv_6Tick'] / df['VolAbv_6Tick']) / (df['delta'] / df['volume']),
    addDivBy0 if DEFAULT_DIV_BY0 else valueX)

# d) Pression acheteur dans la zone _6Tick
features_df['bearish_buyer_pressure_6Tick'] = np.where(df['VolBlw_6Tick'] != 0,
    (df['upTickVol6TicksBlwAsk'] + df['repeatUpTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwAsk']) / df['VolBlw_6Tick'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_buyer_pressure_6Tick'] = np.where(df['VolAbv_6Tick'] != 0,
    (df['upTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvAsk'] + df['repeatDownTickVol6TicksAbvAsk']) / df['VolAbv_6Tick'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)

# e) Pression vendeur dans la zone _6Tick
features_df['bearish_seller_pressure_6Tick'] = np.where(df['VolBlw_6Tick'] != 0,
    (df['downTickVol6TicksBlwBid'] + df['repeatDownTickVol6TicksBlwBid'] + df['repeatUpTickVol6TicksBlwBid']) / df['VolBlw_6Tick'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_seller_pressure_6Tick'] = np.where(df['VolAbv_6Tick'] != 0,
    (df['downTickVol6TicksAbvBid'] + df['repeatDownTickVol6TicksAbvBid'] + df['repeatUpTickVol6TicksAbvBid']) / df['VolAbv_6Tick'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)

# f) Absorption dans la zone _6Tick
#----- Calcul de bearish_absorption_6Tick et bearish_absorption_6Tick_special
vol_bid_6ticks_bearish = df['downTickVol6TicksBlwBid'] + df['repeatDownTickVol6TicksBlwBid'] + df['repeatUpTickVol6TicksBlwBid']
vol_ask_6ticks_bearish = df['upTickVol6TicksBlwAsk'] + df['repeatUpTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwAsk']
ratio_bearish_absorption_6Tick = np.where(vol_bid_6ticks_bearish != 0, vol_ask_6ticks_bearish / vol_bid_6ticks_bearish, 0)
max_ratio_bearish_absorption_6Tick = calculate_max_ratio(ratio_bearish_absorption_6Tick, vol_bid_6ticks_bearish != 0)
features_df['bearish_absorption_6Tick'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        vol_bid_6ticks_bearish != 0,
        ratio_bearish_absorption_6Tick,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_absorption_6Tick
    )
)
features_df['bearish_absorption_6Tick_special'] = np.where(vol_ask_6ticks_bearish == 0, 37, 0)

#----- Calcul de bullish_absorption_6Tick et bullish_absorption_6Tick_special
vol_ask_6ticks_bullish = df['upTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvAsk'] + df['repeatDownTickVol6TicksAbvAsk']
vol_bid_6ticks_bullish = df['downTickVol6TicksAbvBid'] + df['repeatDownTickVol6TicksAbvBid'] + df['repeatUpTickVol6TicksAbvBid']
ratio_bullish_absorption_6Tick = np.where(vol_ask_6ticks_bullish != 0, vol_bid_6ticks_bullish / vol_ask_6ticks_bullish, 0)
max_ratio_bullish_absorption_6Tick = calculate_max_ratio(ratio_bullish_absorption_6Tick, vol_ask_6ticks_bullish != 0)
features_df['bullish_absorption_6Tick'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        vol_ask_6ticks_bullish != 0,
        ratio_bullish_absorption_6Tick,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_absorption_6Tick
    )
)
features_df['bullish_absorption_6Tick_special'] = np.where(vol_bid_6ticks_bullish == 0, 38, 0)
#-----


# g) Ratio de repeat ticks _6Tick
features_df['bearish_repeat_ticks_ratio_6Tick'] = np.where(df['VolBlw_6Tick'] != 0,
    (df['repeatUpTickVol6TicksBlwAsk'] + df['repeatUpTickVol6TicksBlwBid'] +
     df['repeatDownTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwBid']) / df['VolBlw_6Tick'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_repeat_ticks_ratio_6Tick'] = np.where(df['VolAbv_6Tick'] != 0,
    (df['repeatUpTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvBid'] +
     df['repeatDownTickVol6TicksAbvAsk'] + df['repeatDownTickVol6TicksAbvBid']) / df['VolAbv_6Tick'], addDivBy0 if DEFAULT_DIV_BY0 else valueX)

# h) Comparaison de la dynamique de prix
upTickVol6TicksAbv=df['upTickVol6TicksAbvBid']+df['upTickVol6TicksAbvAsk']
downTickVol6TicksAbv=df['downTickVol6TicksAbvBid']+df['downTickVol6TicksAbvAsk']

upTickVol6TicksBlw=df['upTickVol6TicksBlwAsk']+df['upTickVol6TicksBlwBid']
downTickVol6TicksBlw=df['downTickVol6TicksBlwAsk']+df['downTickVol6TicksBlwBid']

#----- Calcul de bearish_price_dynamics_comparison_6Tick et bearish_price_dynamics_comparison_6Tick_special
total_vol_6ticks_bearish = upTickVol6TicksBlw + downTickVol6TicksBlw
price_dynamics_bearish = (
    df['upTickVol6TicksBlwAsk'] + df['upTickVol6TicksBlwBid'] - df['downTickVol6TicksBlwAsk'] - df['downTickVol6TicksBlwBid']
)
ratio_bearish_price_dynamics_6Tick = np.where(total_vol_6ticks_bearish != 0, price_dynamics_bearish / total_vol_6ticks_bearish, 0)
max_ratio_bearish_price_dynamics_6Tick = calculate_max_ratio(ratio_bearish_price_dynamics_6Tick, total_vol_6ticks_bearish != 0)
features_df['bearish_price_dynamics_comparison_6Tick'] = np.where(
    df['VolBlw'] == 0, valueY,
    np.where(
        total_vol_6ticks_bearish != 0,
        ratio_bearish_price_dynamics_6Tick,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bearish_price_dynamics_6Tick
    )
)
features_df['bearish_price_dynamics_comparison_6Tick_special'] = np.where(price_dynamics_bearish == 0, 39, 0)

#----- Calcul de bullish_price_dynamics_comparison_6Tick et bullish_price_dynamics_comparison_6Tick_special
total_vol_6ticks_bullish = upTickVol6TicksAbv + downTickVol6TicksAbv
price_dynamics_bullish = (
    df['downTickVol6TicksAbvAsk'] + df['downTickVol6TicksAbvBid'] - df['upTickVol6TicksAbvAsk'] - df['upTickVol6TicksAbvBid']
)
ratio_bullish_price_dynamics_6Tick = np.where(total_vol_6ticks_bullish != 0, price_dynamics_bullish / total_vol_6ticks_bullish, 0)
max_ratio_bullish_price_dynamics_6Tick = calculate_max_ratio(ratio_bullish_price_dynamics_6Tick, total_vol_6ticks_bullish != 0)
features_df['bullish_price_dynamics_comparison_6Tick'] = np.where(
    df['VolAbv'] == 0, valueY,
    np.where(
        total_vol_6ticks_bullish != 0,
        ratio_bullish_price_dynamics_6Tick,
        diffDivBy0 if DEFAULT_DIV_BY0 else max_ratio_bullish_price_dynamics_6Tick
    )
)
features_df['bullish_price_dynamics_comparison_6Tick_special'] = np.where(price_dynamics_bullish == 0, 40, 0)
#-----

# i) Ratio d'activité Ask vs Bid dans la zone _6Tick
features_df['bearish_activity_bid_ask_ratio_6Tick'] = np.where(
    (df['upTickVol6TicksBlwAsk'] + df['downTickVol6TicksBlwAsk'] + df['repeatUpTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwAsk']) != 0,
    (df['upTickVol6TicksBlwBid'] + df['downTickVol6TicksBlwBid'] + df['repeatUpTickVol6TicksBlwBid'] + df['repeatDownTickVol6TicksBlwBid']) /
    (df['upTickVol6TicksBlwAsk'] + df['downTickVol6TicksBlwAsk'] + df['repeatUpTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwAsk']), addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_activity_ask_bid_ratio_6Tick'] = np.where(
    (df['upTickVol6TicksAbvBid'] + df['downTickVol6TicksAbvBid'] + df['repeatUpTickVol6TicksAbvBid'] + df['repeatDownTickVol6TicksAbvBid']) != 0,
    (df['upTickVol6TicksAbvAsk'] + df['downTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvAsk'] + df['repeatDownTickVol6TicksAbvAsk']) /
    (df['upTickVol6TicksAbvBid'] + df['downTickVol6TicksAbvBid'] + df['repeatUpTickVol6TicksAbvBid'] + df['repeatDownTickVol6TicksAbvBid']), addDivBy0 if DEFAULT_DIV_BY0 else valueX)

# j) Déséquilibre des repeat ticks
features_df['bearish_repeat_ticks_imbalance_6Tick'] = np.where(df['VolBlw_6Tick'] != 0,
    (df['repeatDownTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwBid'] -
     df['repeatUpTickVol6TicksBlwAsk'] - df['repeatUpTickVol6TicksBlwBid']) / df['VolBlw_6Tick'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_repeat_ticks_imbalance_6Tick'] = np.where(df['VolAbv_6Tick'] != 0,
    (df['repeatUpTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvBid'] -
     df['repeatDownTickVol6TicksAbvAsk'] - df['repeatDownTickVol6TicksAbvBid']) / df['VolAbv_6Tick'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# k) Déséquilibre global
features_df['bearish_ticks_imbalance_6Tick'] = np.where(df['VolBlw_6Tick'] != 0,
    (df['downTickVol6TicksBlwBid'] + df['repeatDownTickVol6TicksBlwAsk'] + df['repeatDownTickVol6TicksBlwBid'] -
     df['upTickVol6TicksBlwAsk'] - df['repeatUpTickVol6TicksBlwAsk'] - df['repeatUpTickVol6TicksBlwBid']) / df['VolBlw_6Tick'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_ticks_imbalance_6Tick'] = np.where(df['VolAbv_6Tick'] != 0,
    (df['upTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvAsk'] + df['repeatUpTickVol6TicksAbvBid'] -
     df['downTickVol6TicksAbvBid'] - df['repeatDownTickVol6TicksAbvAsk'] - df['repeatDownTickVol6TicksAbvBid']) / df['VolAbv_6Tick'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

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
    'deltaTimestampOpeningSection1min': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSection1index': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSection5min':       (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSection5index':     (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSection15min': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSection15index': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSection30min':      (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSection30index':    (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),
    'deltaCustomSectionMin':                  (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),
    'deltaCustomSectionIndex':                (False, False, 10, 90,toBeDisplayed_if_s(user_choice, False)),

    # Price and volume features
    'VolAbvState': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'VolBlwState': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'candleSizeTicks':                        (True, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffPriceClosePoc_0_0':                  (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffPriceClosePoc_0_1':                  (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffPriceClosePoc_0_2':                  (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffPriceClosePoc_0_3':                  (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffPriceClosePoc_0_4': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClosePoc_0_5': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    #'diffPriceClosePoc_0_6': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok

    'diffHighPrice_0_1':                      (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffHighPrice_0_2':                      (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffHighPrice_0_3':                      (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffHighPrice_0_4':                        (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffHighPrice_0_5':                        (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    #'diffHighPrice_0_6': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok

    'diffLowPrice_0_1':                       (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffLowPrice_0_2':                       (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffLowPrice_0_3':                       (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffLowPrice_0_4':                         (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffLowPrice_0_5':                     (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    #'diffLowPrice_0_6': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok

    'diffPriceCloseVWAP':                     (True, True, 1, 99,toBeDisplayed_if_s(user_choice, True)),#ok
    'diffPriceCloseVWAPbyIndex':                     (False, False, 1, 99,toBeDisplayed_if_s(user_choice, True)),#ok


    # Technical indicators
    'atr':                                    (True, True, 0.1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'bandWidthBB':                            (True, True, 0.1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'perctBB':                                (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok

    'perct_VA6P':                               (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'ratio_delta_vol_VA6P':                     (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClose_VA6PPoc':                     (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA6PvaH':                     (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA6PvaL':                     (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'perct_VA11P':                               (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'ratio_delta_vol_VA11P':                    (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClose_VA11PPoc':                  (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA11PvaH':                  (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA11PvaL':                  (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'perct_VA16P':                                (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'ratio_delta_vol_VA16P':                    (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClose_VA16PPoc':                  (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA16PvaH':                  (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA16PvaL':                  (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'perct_VA21P':                                (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'ratio_delta_vol_VA21P':                    (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClose_VA21PPoc':                  (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA21PvaH':                  (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA21PvaL':                  (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':

    # Chevauchement des Zones de Valeur
    'overlap_ratio_VA_6P_11P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'overlap_ratio_VA_6P_16P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'overlap_ratio_VA_6P_21P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'overlap_ratio_VA_11P_21P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # Analyse des POC
    'poc_diff_6P_11P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_ratio_6P_11P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_6P_16P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_ratio_6P_16P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_6P_21P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_ratio_6P_21P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_11P_21P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_ratio_11P_21P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),


    'market_regimeADX':                       (False, True, 0.5, 99.8,toBeDisplayed_if_s(user_choice, True)),
    'market_regimeADX_state':                 (False, False, 0.5, 99.8,toBeDisplayed_if_s(user_choice, True)),
    'range_strength_10_32':                   (False, True, 0.1, 99.5,toBeDisplayed_if_s(user_choice, True)),
    'range_strength_5_23':                    (False, True, 0.1, 99.5, toBeDisplayed_if_s(user_choice, True)),
    'is_in_range_10_32':                      (False, False, 0.5, 99.8, toBeDisplayed_if_s(user_choice, True)),
    'is_in_range_5_23':                       (False, False, 0.5, 99.8, toBeDisplayed_if_s(user_choice, True)),

    # Reversal and momentum features
    'bearish_reversal_force':                 (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'bullish_reversal_force':                 (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'bearish_ask_bid_ratio':                  (False, True, 1, 98,toBeDisplayed_if_s(user_choice, False)),#ok
    'bullish_ask_bid_ratio':                  (False, True, 1, 98,toBeDisplayed_if_s(user_choice, False)),#ok
    'meanVolx':                               (False, True, 1, 99.7,toBeDisplayed_if_s(user_choice, False)),#ok
    'ratioDeltaBlw':                          (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'ratioDeltaAbv':                          (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffVolCandle_0_1Ratio':                 (False, True, 1, 98.5,toBeDisplayed_if_s(user_choice, False)),#ok
    'diffVolDelta_0_1Ratio':                  (True, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'cumDiffVolDeltaRatio':                  (True, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok

    # Volume profile features
    'VolPocVolCandleRatio':                  (False, False, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'pocDeltaPocVolRatio':                    (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'VolAbv_vol_ratio': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok
    'VolBlw_vol_ratio': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok

    'asymetrie_volume':                       (True, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'VolCandleMeanxRatio':                    (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok

    # Order flow features
    'bearish_ask_ratio':                      (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok
    'bearish_bid_ratio':                      (False, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok
    'bullish_ask_ratio':                     (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bullish_bid_ratio':                     (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bearish_ask_score':                      (True, True, 3, 99,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bearish_bid_score':                      (True, True, 3, 99,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bearish_imnbScore_score':                (True, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bullish_ask_score':                      (True, True, 3, 98,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bullish_bid_score':                      (True, True, 3, 98,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bullish_imnbScore_score':                (True, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok1

    # Imbalance features
    'bull_imbalance_low_1':                   (False, True, 1, 96.5,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bull_imbalance_low_2':                  (False, True, 1, 96.5,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bull_imbalance_low_3':                 (False, True, 1, 96.5,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bull_imbalance_high_0':                 (False, True, 1, 96.5,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bull_imbalance_high_1':                  (False, True, 1, 96.5,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bull_imbalance_high_2':                 (False, True, 1, 96.5,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bear_imbalance_low_0':                   (False, True, 1, 96.5,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bear_imbalance_low_1':                  (False, True, 1, 96.5,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bear_imbalance_low_2':                   (False, True, 1, 96.5,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bear_imbalance_high_1':                 (False, True, 1, 96.5,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bear_imbalance_high_2':                 (False, True, 1, 96.5,toBeDisplayed_if_s(user_choice, False)),#ok1
    'bear_imbalance_high_3':                 (False, True, 1, 96.5,toBeDisplayed_if_s(user_choice, False)),#ok1
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
    'bearish_askBigStand_abs_ratio_abv_special': (False, False, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok2
    'bearish_bidBigStand_abs_ratio_abv':      (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bearish_bidBigStand_abs_ratio_abv_special': (False, False, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok2
    'bearish_bigStand_abs_diff_abv':          (True, False, 0.5, 99,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_askBigStand_abs_ratio_blw':      (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_askBigStand_abs_ratio_blw_special':      (False, False, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_bidBigStand_abs_ratio_blw':      (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_bidBigStand_abs_ratio_blw_special': (False, False, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok2
    'bullish_bigStand_abs_diff_blw':          (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bearish_askBigHigh_abs_ratio_abv':       (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bearish_askBigHigh_abs_ratio_abv_special': (False, False, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok2
    'bearish_bidBigHigh_abs_ratio_abv':       (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bearish_bidBigHigh_abs_ratio_abv_special': (False, False, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok2
    'bearish_bigHigh_abs_diff_abv':           (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_askBigHigh_abs_ratio_blw':       (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_askBigHigh_abs_ratio_blw_special': (False, False, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok2
    'bullish_bidBigHigh_abs_ratio_blw':       (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2
    'bullish_bidBigHigh_abs_ratio_blw_special': (False, False, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok2
    'bullish_bigHigh_abs_diff_blw':           (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok2

    # Extreme zone features
    'bearish_extrem_revIntensity_ratio':      (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_extrem_revIntensity_ratio':      (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_extrem_zone_volume_ratio':       (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_extrem_zone_volume_ratio':       (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_extrem_pressure_ratio':          (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_extrem_pressure_ratio_special': (False, False, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok3
    'bullish_extrem_pressure_ratio':          (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_extrem_pressure_ratio_special': (False, False, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok3
    'bearish_extrem_abs_ratio':               (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_extrem_abs_ratio_special':               (False, False, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_extrem_abs_ratio':               (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_extrem_abs_ratio_special': (False, False, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok3
    'bearish_extrem_vs_rest_activity':        (True, True, 1, 98,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_extrem_vs_rest_activity_special': (False, False, 1, 98, toBeDisplayed_if_s(user_choice, False)),  # ok3
    'bullish_extrem_vs_rest_activity':        (True, True, 1, 98,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_extrem_vs_rest_activity_special':        (False, False, 1, 98,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_continuation_vs_reversal':       (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_continuation_vs_reversal_special': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok3
    'bullish_continuation_vs_reversal':       (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_continuation_vs_reversal_special': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok3
    'bearish_repeat_ticks_ratio':             (True, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_repeat_ticks_ratio_special': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok3
    'bullish_repeat_ticks_ratio':             (True, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_repeat_ticks_ratio_special': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok3
    'bearish_big_trade_ratio_extrem':         (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
'bearish_big_trade_ratio_extrem_special':         (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_big_trade_imbalance':            (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_big_trade_imbalance_special':            (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_big_trade_ratio_extrem':          (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_big_trade_ratio_extrem_special': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok3
    'bullish_big_trade_imbalance':            (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_big_trade_imbalance_special': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok3

    # Ascending/Descending features
    'bearish_asc_dsc_ratio':                  (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
'bearish_asc_dsc_ratio_special':                  (False, False, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_asc_dynamics':                   (False, True, 1, 99,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bearish_dsc_dynamics':                   (False, True, 1, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_asc_dsc_ratio':                  (False, True, 1, 97,toBeDisplayed_if_s(user_choice, False)),#ok3
    'bullish_asc_dsc_ratio_special': (False, False, 1, 97, toBeDisplayed_if_s(user_choice, False)),  # ok3
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
'extrem_ask_bid_imbalance_bearish_special':       (False, False, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_asc_dsc_comparison_bearish':      (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_asc_ratio_bullish':               (False, True, 0.5, 99,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_dsc_ratio_bullish':               (False, True, 0.5, 99,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_zone_significance_bullish':       (False, True, 0.5, 97,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_ask_bid_imbalance_bullish':      (False, False, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok4
    'extrem_ask_bid_imbalance_bullish_special': (False, False, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok4
    'extrem_asc_dsc_comparison_bullish':     (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4

    # Absorption and big trade features
    'bearish_absorption_ratio':               (False, True, 0.5, 97,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bearish_absorption_ratio_special': (False, False, 0.5, 97, toBeDisplayed_if_s(user_choice, False)),  # ok4
    'bullish_absorption_ratio':               (False, True, 0.5, 97,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bullish_absorption_ratio_special': (False, False, 0.5, 97, toBeDisplayed_if_s(user_choice, False)),  # ok4
    'bearish_big_trade_ratio2_extrem':        (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bearish_big_trade_ratio2_extrem_special': (False, False, 0.5, 98, toBeDisplayed_if_s(user_choice, False)),  # ok4
    'bullish_big_trade_ratio2_extrem':        (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bullish_big_trade_ratio2_extrem_special': (False, False, 0.5, 98, toBeDisplayed_if_s(user_choice, False)),  # ok4

    # Absorption and repeat features
    'total_count_abv':                        (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4
    'absorption_intensity_repeat_bearish_vol':(False, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
    'absorption_intensity_repeat_bearish_count':(False, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
'absorption_intensity_repeat_bearish_count_special':(False, False, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bearish_repeatAskBid_ratio':             (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok4
    'total_count_blw':                        (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok4
    'absorption_intensity_repeat_bullish_vol':(True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
    'absorption_intensity_repeat_bullish_count':(True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
'absorption_intensity_repeat_bullish_count_special':(False, False, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok4
    'bullish_repeatAskBid_ratio':             (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok4
    'count_AbvBlw_asym_ratio':                       (True, True, 0.1, 99.9,toBeDisplayed_if_s(user_choice, False)),
    'count_blw_tot_ratio': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'count_abv_tot_ratio': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),

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
    'bearish_absorption_6Tick_special': (False, False, 0.5, 98, toBeDisplayed_if_s(user_choice, False)),  # ok5
    'bullish_absorption_6Tick': (False, True, 0.5, 98,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_absorption_6Tick_special': (False, False, 0.5, 98, toBeDisplayed_if_s(user_choice, False)),  # ok5
    'bearish_repeat_ticks_ratio_6Tick': (True, True, 5, 99,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_repeat_ticks_ratio_6Tick': (True, True, 5, 99,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_price_dynamics_comparison_6Tick': (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_price_dynamics_comparison_6Tick_special': (False, False, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok5
    'bullish_price_dynamics_comparison_6Tick': (True, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
'bullish_price_dynamics_comparison_6Tick_special': (False, False, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_activity_bid_ask_ratio_6Tick': (False, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_activity_ask_bid_ratio_6Tick': (False, True, 0.5, 99.5,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_repeat_ticks_imbalance_6Tick': (True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_repeat_ticks_imbalance_6Tick': (True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bearish_ticks_imbalance_6Tick': (True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok5
    'bullish_ticks_imbalance_6Tick': (True, True, 0.5, 99.9,toBeDisplayed_if_s(user_choice, False)),#ok5
}
columns_to_process = list(column_settings.keys())

# Vérification de l'existence des colonnes
# Vérification des colonnes manquantes dans features_df
missing_columns = [column for column in columns_to_process if column not in features_df.columns]

# Vérification des colonnes supplémentaires dans features_df
columns_to_exclude = ['class_binaire', 'date', 'trade_category']

extra_columns = [column for column in features_df.columns
                 if column not in columns_to_process
                 and column not in columns_to_exclude]
if missing_columns or extra_columns:
    if missing_columns:
        print("Erreur : Les colonnes suivantes sont manquantes dans features_df :")
        for column in missing_columns:
            print(f"- {column}")

    if extra_columns:
        print("Erreur : Les colonnes suivantes sont présentes dans features_df mais pas dans columns_to_process :")
        for column in extra_columns:
            print(f"- {column}")

    print("Le processus va s'arrêter en raison de différences dans les colonnes.")
    exit(1)  # Arrête le script avec un code d'erreur

print("Toutes les features nécessaires sont présentes et aucune colonne supplémentaire n'a été détectée. Poursuite du traitement.")


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
        # Combiner les masques pour NaN et Inf en une seule opération
        is_nan_or_inf = df[column].isna() | np.isinf(df[column])
        total_replacements = is_nan_or_inf.sum()

        if total_replacements > 0:
            nan_count = df[column].isna().sum()
            inf_count = np.isinf(df[column]).sum()

            print(f"Colonne problématique : {column}")
            print(f"Nombre de valeurs NaN : {nan_count}")
            print(f"Nombre de valeurs infinies : {inf_count}")

            if REPLACE_NAN:
                if start_value != 0:
                    df_replaced.loc[is_nan_or_inf, column] = current_value
                    nan_replacement_values[column] = current_value
                    print(f"L'option start_value != 0 est activée.")
                    print(f"Les {total_replacements} valeurs NaN et infinies dans la colonne '{column}' ont été remplacées par {current_value}")
                    if increment != 0:
                        current_value += increment
                else:
                    print(f"Les valeurs NaN et infinies dans la colonne '{column}' ont été laissées inchangées car start_value est 0")
            else:
                # Remplacer uniquement les valeurs infinies par NaN
                df_replaced.loc[np.isinf(df[column]), column] = np.nan
                inf_replacements = inf_count
                print(f"REPLACE_NAN est à False.")
                print(f"Les {inf_replacements} valeurs infinies dans la colonne '{column}' ont été remplacées par NaN")
                print(f"Les {nan_count} valeurs NaN dans la colonne '{column}' ont été laissées inchangées")
                print("Les valeurs NaN ne sont pas remplacées par une valeur choisie par l'utilisateur.")

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


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_single_histogram(values_before, winsorized_values_after, column, floor_value, crop_value,
                          floorInf_values, cropSup_values, floorInf_percent, cropSup_percent, ax,
                          nan_replacement_values=None, range_strength_percent_in_range_10_32=None,
                          range_strength_percent_in_range_5_23=None, regimeAdx_pct_infThreshold=None,
                          adjust_xaxis=True):
    values_before_clean = values_before.dropna()

    sns.histplot(data=pd.DataFrame({column: values_before_clean}), x=column, color="blue", kde=False, ax=ax, alpha=0.7)
    sns.histplot(data=pd.DataFrame({column: winsorized_values_after}), x=column, color="red", kde=False, ax=ax,
                 alpha=0.7)

    if floorInf_values:
        ax.axvline(floor_value, color='g', linestyle='--', label=f'Floor ({floorInf_percent}%)')
    if cropSup_values:
        ax.axvline(crop_value, color='y', linestyle='--', label=f'Crop ({cropSup_percent}%)')

    def format_value(value):
        return f"{value:.2f}" if pd.notna(value) else "nan"

    initial_values = values_before_clean.sort_values()
    winsorized_values = winsorized_values_after.dropna().sort_values()

    ax.axvline(initial_values.iloc[0], color='blue',
               label=f'Init ({format_value(initial_values.iloc[0])}, {format_value(initial_values.iloc[-1])})')
    ax.axvline(winsorized_values.iloc[0], color='red',
               label=f'Winso ({format_value(winsorized_values.iloc[0])}, {format_value(winsorized_values.iloc[-1])})')

    if adjust_xaxis:
        # Assurez-vous que x_min prend en compte les valeurs négatives
        x_min = min(winsorized_values_after.min(), floor_value) if floorInf_values else winsorized_values_after.min()
        x_max = max(winsorized_values_after.max(), crop_value) if cropSup_values else winsorized_values_after.max()
        ax.set_xlim(left=x_min, right=x_max)

    # Keep the title
    ax.set_title(column, fontsize=6, pad=0.1)  # Title is kept

    # Clear the x-axis label to avoid duplication
    ax.set_xlabel('')  # This will clear the default x-axis label
    ax.set_ylabel('')
    # Reduce the font size of the legend
    ax.legend(fontsize=5)

    ax.tick_params(axis='both', which='major', labelsize=4.5)
    ax.xaxis.set_tick_params(labelsize=4.5, pad=0.1)
    ax.yaxis.set_tick_params(labelsize=4.5, pad=0.1)

    if nan_replacement_values and column in nan_replacement_values:
        ax.annotate(f"NaN replaced by: {nan_replacement_values[column]}",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=5, ha='left', va='top')

    nan_count = winsorized_values_after.isna().sum()
    inf_count = np.isinf(winsorized_values_after).sum()
    nan_proportion = nan_count / len(winsorized_values_after)
    color_proportion = 'green' if nan_proportion < 0.3 else 'red'

    annotation_text = (
        f"Winsorized column:\n"
        f"Remaining NaN: {nan_count}\n"
        f"Remaining Inf: {inf_count}\n"
        f"nb period: {len(winsorized_values_after)}\n"
        f"% de np.nan : {nan_proportion:.2%}"
    )

    if column == 'range_strength_10_32' and range_strength_percent_in_range_10_32 is not None:
        annotation_text += f"\n% time in range: {range_strength_percent_in_range_10_32:.2f}%"
    elif column == 'range_strength_5_23' and range_strength_percent_in_range_5_23 is not None:
        annotation_text += f"\n% time in range: {range_strength_percent_in_range_5_23:.2f}%"
    elif column == 'market_regimeADX' and regimeAdx_pct_infThreshold is not None:
        annotation_text += f"\n% ADX < threshold: {regimeAdx_pct_infThreshold:.2f}%"

    ax.annotate(
        annotation_text,
        xy=(0.05, 0.85),
        xycoords='axes fraction',
        fontsize=5,
        ha='left',
        va='top',
        color=color_proportion,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
    )


def plot_histograms_multi_figure(columns, figsize=(28, 20), graphs_per_figure=40):
    n_columns = len(columns)
    ncols = 7
    nrows = 4
    graphs_per_figure = ncols * nrows
    n_figures = math.ceil(n_columns / graphs_per_figure)

    figures = []
    all_axes = []

    for fig_num in range(n_figures):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        figures.append(fig)
        axes = axes.flatten()
        all_axes.extend(axes)

        # Hide unused subplots
        for ax in axes[n_columns - fig_num * graphs_per_figure:]:
            ax.set_visible(False)

    return figures, all_axes
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

# Récupérer la liste du nom des features pour paramétrage
all_columns = [col for col, settings in column_settings.items() if
               settings[4] or all(not s[4] for s in column_settings.values())]

# Déterminer les colonnes à traiter en fonction du choix de l'utilisateur
if user_choice.lower() == 'd':
    try:
        x, y = map(int, fig_range_input.split('_'))
        start_index = (x - 1) * 28  # 28 graphiques par figure
        end_index = y * 28
        columns_to_process = all_columns[start_index:end_index]
    except ValueError:
        print("Erreur : Format invalide. Traitement de toutes les colonnes.")
        columns_to_process = all_columns
elif user_choice.lower() == 's':
    columns_to_process = all_columns
else:
    columns_to_process = all_columns  # Traiter toutes les colonnes si aucun affichage n'est demandé

# Initialisation des variables pour l'affichage si nécessaire
if user_choice.lower() in ['d', 's']:
    figures, all_axes = plot_histograms_multi_figure(columns_to_process, figsize=(28, 20))

# Initialisation des DataFrames avec le même index que le DataFrame d'entrée
winsorized_df = pd.DataFrame(index=features_NANReplacedVal_df.index)
winsorized_scaledWithNanValue_df = pd.DataFrame(index=features_NANReplacedVal_df.index)

total_features = len(columns_to_process)

# Parcours de la liste des features
for i, columnName in enumerate(columns_to_process):
    current_feature = i+1
    print(f"\nFeature ({current_feature}/{total_features}) -> Début du traitement de {columnName}:")

    # Récupérer les valeurs pour la winsorisation
    floor_valueNANfiltered, crop_valueNANfiltered, floorInf_values, cropSup_values, floorInf_percent, cropSup_percent = (
        cropFloor_dataSource(features_NANReplacedVal_df, columnName, column_settings, nan_replacement_values)
    )

    # Winsorisation avec les valeurs NaN
    winsorized_valuesWithNanValue = winsorize(
        features_NANReplacedVal_df,
        columnName,
        floor_valueNANfiltered,
        crop_valueNANfiltered,
        floorInf_values,
        cropSup_values,
        nan_replacement_values
    )

    # Assignation directe de la série winsorisée au DataFrame
    winsorized_df[columnName] = winsorized_valuesWithNanValue

    # Préparation de la colonne pour la normalisation
    scaled_column = winsorized_valuesWithNanValue.copy()
    if ENABLE_PANDAS_METHOD_SCALING:
        # Sélectionner la colonne 'columnName'
        column_to_scale = winsorized_df[columnName]
        scaled_column = (column_to_scale - column_to_scale.min(skipna=True)) / (
                    column_to_scale.max(skipna=True) - column_to_scale.min(skipna=True))
        #tester nan_policy='omit' pour minmaxscaler ?

    else:
        if nan_replacement_values is not None and columnName in nan_replacement_values:
            nan_value = nan_replacement_values[columnName]
            mask = scaled_column != nan_value
            # Sauvegarde des positions des nan_value
            nan_positions = ~mask
        else:
            mask = scaled_column.notna()  # Sélectionne les lignes non-NaN
            nan_positions = scaled_column.isna()  # Positions des valeurs NaN (si nécessaire)

        # Normalisation des valeurs
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(scaled_column.loc[mask].values.reshape(-1, 1)).flatten()

        # Convertir la colonne en float64 si ce n'est pas déjà le cas
        scaled_column = scaled_column.astype('float64')

        # Assignation des valeurs normalisées aux positions correspondantes
        scaled_column.loc[mask] = normalized_values

        # Remettre les nan_value à leur place seulement s'il y en avait
        if nan_replacement_values is not None and columnName in nan_replacement_values:
            scaled_column.loc[nan_positions] = nan_value

    # Assignation directe de la colonne normalisée au DataFrame
    winsorized_scaledWithNanValue_df[columnName] = scaled_column

    # Affichage des graphiques si demandé
    if user_choice.lower() in ['d', 's']:
        winsorized_values_4Plotting = winsorized_valuesWithNanValue[
            winsorized_valuesWithNanValue != nan_replacement_values.get(columnName, np.nan)
            ]
        print(f"   Graphiques de {columnName} avant et après les modifications (colonnes sélectionnées) :")
        print(f"   Taille de winsorized_values_after (sans NaN) pour plotting: {len(winsorized_values_4Plotting)}")

        value_before_df = features_df.copy()
        plot_single_histogram(
            value_before_df[columnName],
            winsorized_values_4Plotting,
            columnName,
            floor_valueNANfiltered,
            crop_valueNANfiltered,
            floorInf_values,
            cropSup_values,
            floorInf_percent,
            cropSup_percent,
            all_axes[i],
            nan_replacement_values,
            range_strength_percent_in_range_10_32=(
                range_strength_percent_in_range_10_32 if columnName == 'range_strength_10_32' else None
            ),
            range_strength_percent_in_range_5_23=(
                range_strength_percent_in_range_5_23 if columnName == 'range_strength_5_23' else None
            ),
            regimeAdx_pct_infThreshold=(
                regimeAdx_pct_infThreshold if columnName == 'market_regimeADX' else None
            ),
            adjust_xaxis=adjust_xaxis
        )

print("\n")
print("Vérification finale :")
print(f"   - Nombre de colonnes dans winsorized_df : {len(winsorized_df.columns)}")
print(
    f"   - Nombre de colonnes dans winsorized_scaledWithNanValue_df : {len(winsorized_scaledWithNanValue_df.columns)}")
assert len(winsorized_df.columns) == len(
    winsorized_scaledWithNanValue_df.columns), "Le nombre de colonnes ne correspond pas entre les DataFrames"

print(f"\n")
print("Vérification finale :")
print(f"   - Nombre de colonnes dans winsorized_df : {len(winsorized_df.columns)}")
print(f"   - Nombre de colonnes dans winsorized_scaledWithNanValue_df : {len(winsorized_scaledWithNanValue_df.columns)}")
assert len(winsorized_df.columns) == len(winsorized_scaledWithNanValue_df.columns), "Le nombre de colonnes ne correspond pas entre les DataFrames"


print_notification("Ajout de  'timeStampOpening', class_binaire', 'date', 'trade_category', 'SessionStartEnd' pour permettre la suite des traitements")
# Colonnes à ajouter
columns_to_add = ['timeStampOpening', 'class_binaire', 'candleDir', 'date', 'trade_category', 'SessionStartEnd']

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

print_notification("Colonnes 'timeStampOpening','class_binaire', 'candleDir', 'date', 'trade_category', 'SessionStartEnd' ajoutées")



file_without_extension = os.path.splitext(file_name)[0]
file_without_extension = file_without_extension.replace("Step4", "Step5")


# Créer le nouveau nom de fichier pour les features originales
new_file_name = file_without_extension + '_feat.csv'

# Construire le chemin complet du nouveau fichier
feat_file = os.path.join(file_dir, new_file_name)

# Sauvegarder le fichier des features originales
print_notification(f"Enregistrement du fichier de features non modifiées : {feat_file}")
features_df.to_csv(feat_file, sep=';', index=False, encoding='iso-8859-1')


# Créer le nouveau nom de fichier pour winsorized_df
winsorized_file_name = file_without_extension+ '_feat_winsorized.csv'

# Construire le chemin complet du nouveau fichier winsorized
winsorized_file = os.path.join(file_dir, winsorized_file_name)

# Sauvegarder le fichier winsorized
print_notification(f"Enregistrement du fichier de features winsorisées : {winsorized_file}")
winsorized_df.to_csv(winsorized_file, sep=';', index=False, encoding='iso-8859-1')

# Créer le nouveau nom de fichier pour winsorized_scaledWithNanValue_df
scaled_file_name = file_without_extension+ '_feat_winsorizedScaledWithNanVal.csv'

# Construire le chemin complet du nouveau fichier scaled
scaled_file = os.path.join(file_dir, scaled_file_name)

# Sauvegarder le fichier scaled
winsorized_scaledWithNanValue_df.to_csv(scaled_file, sep=';', index=False, encoding='iso-8859-1')
print_notification(f"Enregistrement du fichier de features winsorisées et normalisées : {scaled_file}")


# Affichage final des graphiques si demandé
if user_choice.lower() in ['d', 's']:
    if user_choice.lower() == 'd':
        try:
            x, y = map(int, fig_range_input.split('_'))
            figures_to_show = [fig for fig in figures if x <= fig.number <= y]
        except ValueError:
            print("Erreur : Format invalide. Affichage de toutes les figures.")
            figures_to_show = figures
    else:
        figures_to_show = figures  # Si 's', toutes les figures seront affichées

    for fig in figures_to_show:
        print(f"Affichage de la figure: {fig.number}")
        plt.figure(fig.number)
        plt.tight_layout()
    plt.show()