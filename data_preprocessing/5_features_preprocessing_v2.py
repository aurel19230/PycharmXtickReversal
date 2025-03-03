import pandas as pd
import numpy as np
from func_standard import print_notification, load_data, calculate_naked_poc_distances, CUSTOM_SESSIONS, \
    save_features_with_sessions,remplace_0_nan_reg_slope_p_2d,process_reg_slope_replacement
from definition import *
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
valueX = np.nan
valueY = np.nan
from sklearn.preprocessing import MinMaxScaler
# Définition de la fonction calculate_max_ratio
import numpy as np
import time


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


ENABLE_PANDAS_METHOD_SCALING = True

DEFAULT_DIV_BY0 = True  # max_ratio or valuex
user_choice = input("Appuyez sur Entrée pour calculer les features sans la afficher. \n"
                    "Appuyez sur 'd' puis Entrée pour les calculer et les afficher : \n"
                    "Appuyez sur 's' puis Entrée pour les calculer et les afficher :")
if user_choice.lower() == 'd':
    fig_range_input = input("Entrez la plage des figures à afficher au format x_y (par exemple 2_5) : \n")

# Demander à l'utilisateur s'il souhaite ajuster l'axe des abscisses
adjust_xaxis_input = ''
if user_choice.lower() == 'd' or user_choice.lower() == 's':
    adjust_xaxis_input = input(
        "Voulez-vous afficher les graphiques entre les valeurs de floor et crop ? (o/n) : ").lower()

adjust_xaxis = adjust_xaxis_input == 'o'

# Nom du fichier

file_name = "Step4_5_0_5TP_0SL_030124_270125_extractOnlyFullSession_OnlyShort.csv"

# Chemin du répertoire
directory_path = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\5_0_5TP_0SL\merge"

# Construction du chemin complet du fichier
file_path = os.path.join(directory_path, file_name)

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


def get_custom_section(minutes: int, custom_sections: dict) -> dict:
    """
    Retourne la section correspondant au nombre de minutes dans custom_sections.
    """
    for section_name, section in custom_sections.items():
        if section['start'] <= minutes < section['end']:
            return section
    # Retourne la dernière section si aucune correspondance
    return list(custom_sections.values())[-1]


df = load_data(CONFIG['FILE_PATH'])

# Calcul de la moyenne de trade_pnl pour chaque classe
mean_pnl = df.groupby('class_binaire')['trade_pnl'].mean()

print("\nMoyenne de trade_pnl par classe:")
print(f"Classe 0 (Perdants): {mean_pnl[0]:.2f}")
print(f"Classe 1 (Gagnants): {mean_pnl[1]:.2f}")
stats_pnl = df.groupby('class_binaire')['trade_pnl'].agg(['count', 'mean', 'std'])
print("\nStatistiques de trade_pnl par classe:")
print(stats_pnl)

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

# Session 1 minute
features_df['deltaTimestampOpeningSession1min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x / 1)) * 1, 1379))  # 23h = 1380 minutes - 1

unique_sections = sorted(features_df['deltaTimestampOpeningSession1min'].unique())
section_to_index = {section: index for index, section in enumerate(unique_sections)}
features_df['deltaTimestampOpeningSession1index'] = features_df['deltaTimestampOpeningSession1min'].map(
    section_to_index)

# Session 5 minutes
features_df['deltaTimestampOpeningSession5min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x / 5)) * 5, 1375))  # Dernier multiple de 5 < 1380

unique_sections = sorted(features_df['deltaTimestampOpeningSession5min'].unique())
section_to_index = {section: index for index, section in enumerate(unique_sections)}
features_df['deltaTimestampOpeningSession5index'] = features_df['deltaTimestampOpeningSession5min'].map(
    section_to_index)

# Session 15 minutes
features_df['deltaTimestampOpeningSession15min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x / 15)) * 15, 1365))  # Dernier multiple de 15 < 1380

unique_sections = sorted(features_df['deltaTimestampOpeningSession15min'].unique())
section_to_index = {section: index for index, section in enumerate(unique_sections)}
features_df['deltaTimestampOpeningSession15index'] = features_df['deltaTimestampOpeningSession15min'].map(
    section_to_index)

# Session 30 minutes
features_df['deltaTimestampOpeningSession30min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x / 30)) * 30, 1350))  # Dernier multiple de 30 < 1380

unique_sections = sorted(features_df['deltaTimestampOpeningSession30min'].unique())
section_to_index = {section: index for index, section in enumerate(unique_sections)}
features_df['deltaTimestampOpeningSession30index'] = features_df['deltaTimestampOpeningSession30min'].map(
    section_to_index)

# Custom session
features_df['deltaCustomSessionMin'] = df['deltaTimestampOpening'].apply(
    lambda x: get_custom_section(x, CUSTOM_SESSIONS)['start']
)


def get_custom_section_index(minutes: int, custom_sections: dict) -> int:
    """
    Retourne le session_type_index correspondant au nombre de minutes dans custom_sections.

    Args:
        minutes (int): Nombre de minutes depuis 22h00
        custom_sections (dict): Dictionnaire des sections personnalisées

    Returns:
        int: session_type_index de la section correspondante
    """
    for section in custom_sections.values():
        if section['start'] <= minutes < section['end']:
            return section['session_type_index']
    # Retourne le session_type_index de la dernière section si aucune correspondance
    return list(custom_sections.values())[-1]['session_type_index']


# Application sur features_df
features_df['deltaCustomSessionIndex'] = features_df['deltaTimestampOpening'].apply(
    lambda x: get_custom_section_index(x, CUSTOM_SESSIONS)
)

import numpy as np
from numba import jit


@jit(nopython=True)
def fast_linear_regression_slope(x, y):
    """Calcule la pente de régression linéaire de manière optimisée"""
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_xx = np.sum(x * x)

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    return slope


@jit(nopython=True)
def calculate_slopes(close_values: np.ndarray, session_starts: np.ndarray, window: int) -> np.ndarray:
    n = len(close_values)
    results = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)

    for i in range(n):
        # On cherche le début de la session actuelle
        session_start_idx = -1

        # Remonter pour trouver le début de session
        for j in range(i, -1, -1):  # On remonte jusqu'au début si nécessaire
            if session_starts[j]:
                session_start_idx = j
                break

        # S'il y a assez de barres depuis le début de session
        bars_since_start = i - session_start_idx + 1

        if bars_since_start >= window:
            end_idx = i + 1
            start_idx = end_idx - window
            # Vérifier que start_idx est après le début de session
            if start_idx >= session_start_idx:
                y = close_values[start_idx:end_idx]
                results[i] = fast_linear_regression_slope(x, y)

    return results


def apply_optimized_slope_calculation(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Applique le calcul optimisé des pentes

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame contenant les données
    window : int
        Taille de la fenêtre pour le calcul

    Returns:
    --------
    pd.Series : Série des pentes calculées
    """
    # Préparation des données numpy
    close_values = data['close'].values
    session_starts = (data['SessionStartEnd'] == 10).values

    # Calcul des pentes
    slopes = calculate_slopes(close_values, session_starts, window)

    # Conversion en pandas Series
    return pd.Series(slopes, index=data.index)


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
from numba import jit


@jit(nopython=True)
def calculate_slopes_and_r2_numba(close_values, session_starts, window):
    """
    Calcule les pentes et les coefficients R² pour une série temporelle de manière optimisée avec Numba.

    Parameters:
    -----------
    close_values : np.ndarray
        Valeurs de clôture des prix.
    session_starts : np.ndarray
        Masque indiquant les débuts de session.
    window : int
        Taille de la fenêtre pour le calcul.

    Returns:
    --------
    np.ndarray, np.ndarray :
        Deux tableaux numpy contenant respectivement les pentes et les coefficients R².
    """
    n = len(close_values)
    slopes = np.full(n, np.nan)
    r2s = np.full(n, np.nan)

    # Pré-calculer les x pour toutes les fenêtres
    x = np.arange(window)
    x_mean = np.mean(x)
    x_diff = x - x_mean
    x_var = np.sum(x_diff ** 2)

    for i in range(window - 1, n):
        # Vérifier que la fenêtre est valide (pas de début de session à l'intérieur)
        if np.any(session_starts[i - window + 1:i + 1]):
            continue

        # Extraire les données de la fenêtre
        y = close_values[i - window + 1:i + 1]

        # Calculer la pente (slope)
        y_mean = np.mean(y)
        y_diff = y - y_mean
        slope = np.sum(x_diff * y_diff) / x_var
        slopes[i] = slope

        # Calculer le R²
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - (slope * x + y_mean)) ** 2)
        r2s[i] = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

    return slopes, r2s


def apply_optimized_slope_r2_calculation(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Applique le calcul optimisé des pentes et des coefficients R² avec Numba.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame contenant les données.
    window : int
        Taille de la fenêtre pour le calcul.

    Returns:
    --------
    pd.DataFrame : DataFrame contenant deux colonnes : slope et r2.
    """

    print(f"  apply_optimized_slope_r2_calculation(df, window) {window} ")
    # Préparation des données numpy
    close_values = data['close'].values
    session_starts = (data['SessionStartEnd'] == 10).values

    # Calcul des pentes et des coefficients R²
    slopes, r2s = calculate_slopes_and_r2_numba(close_values, session_starts, window)

    # Conversion en pandas DataFrame
    results_df = pd.DataFrame({
        f'linear_slope_{window}': slopes,
        f'linear_slope_r2_{window}': r2s
    }, index=data.index)

    return results_df


# Utilisation
windows = [
    #6, 14, 21,30, 40,
     50]
for window in windows:
    slope_r2_df = apply_optimized_slope_r2_calculation(df, window)
    features_df = pd.concat([features_df, slope_r2_df], axis=1)

#
# def enhanced_close_to_sma_ratio(
#         data: pd.DataFrame,
#         window: int,
# ) -> pd.DataFrame:
#     """
#     Calcule pour chaque point :
#       - le ratio (close - sma) / sma
#       - le z-score de ce ratio par rapport à son écart-type (rolling)
#
#     Gère les cas où std = 0 en utilisant :
#         diffDivBy0 if DEFAULT_DIV_BY0 else valueX
#
#     :param data: DataFrame avec au moins la colonne 'close'
#     :param window: nombre de périodes pour le calcul rolling (moyenne + écart-type)
#     :param diffDivBy0: valeur si on divise par 0 et que DEFAULT_DIV_BY0 = True
#     :param DEFAULT_DIV_BY0: booléen, si True, alors on utilise diffDivBy0 comme valeur de fallback
#     :param valueX: valeur si on divise par 0 et que DEFAULT_DIV_BY0 = False
#     :return: DataFrame avec close_sma_ratio_{window} et close_sma_zscore_{window}
#     """
#
#     # Calcul de la SMA
#     sma = data['close'].rolling(window=window, min_periods=1).mean()
#
#     # Ratio (close - sma) / sma
#     ratio = (data['close'] - sma)# / sma
#
#     # Écart-type (rolling) du ratio
#     std = ratio.rolling(window=window).std()
#
#     # Calcul du z-score en évitant la division par zéro.
#     # Si std != 0, on fait ratio / std
#     # Sinon, on applique la logique diffDivBy0 if DEFAULT_DIV_BY0 else valueX
#     z_score_array = np.where(
#         std != 0,
#         ratio / std,
#         diffDivBy0 if DEFAULT_DIV_BY0 else valueX
#     )
#
#     # Convertit le tableau en Series pour garder le même index
#     z_score = pd.Series(z_score_array, index=ratio.index)
#
#     # On renvoie un DataFrame avec deux colonnes
#     return pd.DataFrame({
#         f'close_sma_ratio_{window}': ratio,
#         f'close_sma_zscore_{window}': z_score
#     })


# windows_sma = [6, 14, 21, 30, 40, 50]
# for window in windows_sma:
#     results = enhanced_close_to_sma_ratio(df, window)
#     features_df[f'close_sma_ratio_{window}'] = results[f'close_sma_ratio_{window}']
#     features_df[f'close_sma_zscore_{window}'] = results[f'close_sma_zscore_{window}']

import numpy as np
from numba import jit
import pandas as pd


@jit(nopython=True)
def fast_calculate_previous_session_slope(close_values: np.ndarray, session_type_index: np.ndarray) -> np.ndarray:
    """
    Calcule rapidement la pente de la session précédente
    """
    n = len(close_values)
    slopes = np.full(n, np.nan)

    # Variables pour tracker la session précédente
    prev_session_start = 0
    prev_session_type = session_type_index[0]

    for i in range(1, n):
        curr_type = session_type_index[i]

        # Détection changement de session
        if curr_type != session_type_index[i - 1]:
            # Calculer la pente de la session précédente
            if prev_session_start < i - 1:  # S'assurer qu'il y a des points pour la régression
                x = np.arange(float(i - prev_session_start))
                y = close_values[prev_session_start:i]
                n_points = len(x)
                sum_x = np.sum(x)
                sum_y = np.sum(y)
                sum_xy = np.sum(x * y)
                sum_xx = np.sum(x * x)
                slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_xx - sum_x * sum_x)

                # Assigner la pente à la nouvelle session
                j = i
                while j < n and session_type_index[j] == curr_type:
                    slopes[j] = slope
                    j += 1

            # Mettre à jour les indices pour la prochaine session
            prev_session_start = i
            prev_session_type = curr_type

    return slopes


def calculate_previous_session_slope(df, data: pd.DataFrame) -> pd.Series:
    """
    Wrapper pandas pour le calcul des pentes
    """

    if len(features_df) != len(data):
        raise ValueError(f"Dimensions mismatch: features_df has {len(features_df)} rows but data has {len(data)} rows")

    close_values = df['close'].values
    session_type_index = data['deltaCustomSessionIndex'].values

    slopes = fast_calculate_previous_session_slope(close_values, session_type_index)
    return pd.Series(slopes, index=data.index)


# Ajout de la colonne à features_df
features_df['linear_slope_prevSession'] = calculate_previous_session_slope(df, features_df)


# Version originale pour comparaison
def linear_regression_slope_market_trend(series):
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return model.coef_[0][0]


def calculate_absorpsion_features(df, candle_rev_tick):
    # Création d'un nouveau DataFrame pour stocker uniquement les colonnes d'absorption
    absorption_features = pd.DataFrame(index=df.index)

    # Initialisation des colonnes d'absorption
    for tick in range(3, candle_rev_tick + 1):
        absorption_features[f'is_absorpsion_{tick}ticks_low'] = 0
        absorption_features[f'is_absorpsion_{tick}ticks_high'] = 0

    # Logique pour "low"
    for tick in range(3, candle_rev_tick + 1):
        condition_low = (df['askVolLow'] - df['bidVolLow']) < 0
        for i in range(1, tick):
            condition_low &= (df[f'askVolLow_{i}'] - df[f'bidVolLow_{i}']) < 0

        absorption_features[f'is_absorpsion_{tick}ticks_low'] = condition_low.astype(int)

        if tick >= 4:
            for t in range(3, tick):
                absorption_features[f'is_absorpsion_{t}ticks_low'] = absorption_features[
                                                                         f'is_absorpsion_{t}ticks_low'] | condition_low.astype(
                    int)

    # Logique pour "high"
    for tick in range(3, candle_rev_tick + 1):
        condition_high = (df['askVolHigh'] - df['bidVolHigh']) > 0
        for i in range(1, tick):
            condition_high &= (df[f'askVolHigh_{i}'] - df[f'bidVolHigh_{i}']) > 0

        absorption_features[f'is_absorpsion_{tick}ticks_high'] = condition_high.astype(int)

        if tick >= 4:
            for t in range(3, tick):
                absorption_features[f'is_absorpsion_{t}ticks_high'] = absorption_features[
                                                                          f'is_absorpsion_{t}ticks_high'] | condition_high.astype(
                    int)

    return absorption_features


def calculate_candle_rev_tick(df):
    """
    Calcule la valeur de CANDLE_REV_TICK en fonction des conditions spécifiées, en déterminant
    dynamiquement le minimum incrément non nul entre les valeurs de la colonne 'close'.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'candleDir', 'high', 'close'.

    Returns:
        int: La valeur de CANDLE_REV_TICK si toutes les valeurs sont identiques pour les 4 premières occurrences.

    Raises:
        ValueError: Si les valeurs calculées diffèrent pour les 4 premières occurrences où candleDir == 1.
    """
    # Calculer la différence absolue entre les valeurs de 'close'
    df['close_diff'] = df['close'].diff().abs()

    # Identifier le minimum incrément non nul
    minimum_increment = df['close_diff'][df['close_diff'] > 0].min()

    # Vérifier si le minimum incrément est bien défini
    if pd.isna(minimum_increment):
        raise ValueError("Impossible de calculer le minimum incrément non nul.")

    print(minimum_increment)
    # Filtrer les lignes où candleDir == -1
    filtered_df = df[df['candleDir'] == -1]

    # Calculer (high - close) * minimum_increment pour les 4 premières occurrences
    values = ((filtered_df['high'] - filtered_df['close']) * (1 / minimum_increment)).iloc[1:5] + 1

    # Vérifier si toutes les valeurs sont identiques
    if not all(values == values.iloc[0]):
        raise ValueError(
            "Les valeurs de (high - close) * minimum_increment diffèrent pour les 4 premières occurrences.")

    # Retourner la valeur commune
    return int(values.iloc[0])



# Appliquer la fonction

candle_rev_tick = calculate_candle_rev_tick(df)
print(candle_rev_tick)
# Calculer les features d'absorption
absorption_features = calculate_absorpsion_features(df, candle_rev_tick)

# Ajouter les colonnes à features_df
features_df = pd.concat([features_df, absorption_features], axis=1)
"""
# Test de performance
def benchmark(data, window):
    start = time.time()
    # Première exécution pour compiler le code numba
    _ = apply_optimized_slope_calculation(data, window)

    # Test réel
    start = time.time()
    optimized = apply_optimized_slope_calculation(data, window)
    optimized_time = time.time() - start

    start = time.time()
    original = apply_slope_with_session_check(data, window)
    original_time = time.time() - start

    print(f"Temps version optimisée: {optimized_time:.4f}s")
    print(f"Temps version originale: {original_time:.4f}s")
    print(f"Accélération: {original_time / optimized_time:.2f}x")

    # Vérification que les résultats sont identiques
    np.testing.assert_almost_equal(optimized.values, original.values, decimal=5)
    print("Les résultats sont identiques ✓")
"""

# benchmark(df, 14)


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
# features_df['diffPriceClosePoc_0_6'] = df['close'] - df['pocPrice'].shift(6)


features_df['diffHighPrice_0_1'] = df['high'] - df['high'].shift(1)
features_df['diffHighPrice_0_2'] = df['high'] - df['high'].shift(2)
features_df['diffHighPrice_0_3'] = df['high'] - df['high'].shift(3)
features_df['diffHighPrice_0_4'] = df['high'] - df['high'].shift(4)
features_df['diffHighPrice_0_5'] = df['high'] - df['high'].shift(5)
# features_df['diffHighPrice_0_6'] = df['high'] - df['high'].shift(6)


features_df['diffLowPrice_0_1'] = df['low'] - df['low'].shift(1)
features_df['diffLowPrice_0_2'] = df['low'] - df['low'].shift(2)
features_df['diffLowPrice_0_3'] = df['low'] - df['low'].shift(3)
features_df['diffLowPrice_0_4'] = df['low'] - df['low'].shift(4)
features_df['diffLowPrice_0_5'] = df['low'] - df['low'].shift(5)
# features_df['diffLowPrice_0_6'] = df['low'] - df['low'].shift(6)


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
    # data['market_regimeADX'] = data['market_regimeADX'].fillna(addDivBy0 if DEFAULT_DIV_BY0 else valueX)
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

    # Calculate percentage relative to POC, handling division by zero with np.nan
    result = np.where(bands_difference != 0,
                      (data['close'] - data[f'vaPoc_{nbPeriods}periods']) / bands_difference,
                      np.nan)

    # Convert the result into a pandas Series
    return pd.Series(result, index=data.index)


# Apply the function for different periods

# Liste des périodes à analyser
periods = [6, 11, 16, 21]
for nbPeriods in periods:
    # Calculate the percentage of the value area using pd.notnull()
    value_area = valueArea_pct(df, nbPeriods)
    features_df[f'perct_VA{nbPeriods}P'] = np.where(
        pd.notnull(value_area),
        value_area,
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
        df[f'vaPoc_{nbPeriods}periods'] != 0,
        df['close'] - df[f'vaPoc_{nbPeriods}periods'],
        np.nan
    )

    # Différence entre le prix de clôture et VAH
    features_df[f'diffPriceClose_VA{nbPeriods}PvaH'] = np.where(
        df[f'vaH_{nbPeriods}periods'] != 0,
        df['close'] - df[f'vaH_{nbPeriods}periods'],
        np.nan
    )

    # Différence entre le prix de clôture et VAL
    features_df[f'diffPriceClose_VA{nbPeriods}PvaL'] = np.where(
        df[f'vaL_{nbPeriods}periods'] != 0,
        df['close'] - df[f'vaL_{nbPeriods}periods'],
        np.nan
    )

# Génération des combinaisons de périodes
period_combinations = [(6, 11), (6, 16), (6, 21), (11, 21)]

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
    condition = (total_range != 0) & (vaH_p1 != 0) & (vaH_p2 != 0) & (vaL_p1 != 0) & (vaL_p2 != 0)
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
df_with_range_strength_10_32, range_strength_percent_in_range_10_32 = range_strength(df_copy1, 'range_strength_10_32',
                                                                                     window=10, atr_multiple=3.2,
                                                                                     min_strength=0.1)
df_with_range_strength_5_23, range_strength_percent_in_range_5_23 = range_strength(df_copy1, 'range_strength_5_23',
                                                                                   window=5, atr_multiple=2.3,
                                                                                   min_strength=0.1)



# Appliquer detect_market_regime sur une copie de df pour ne pas modifier df
df_copy = df.copy()
df_with_regime, regimeAdx_pct_infThreshold = detect_market_regimeADX(df_copy, period=14, adx_threshold=25)
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
features_df['bearish_reversal_force'] = np.where(df['volume'] != 0, df['VolAbv'] / df['volume'],
                                                 addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['bullish_reversal_force'] = np.where(df['volume'] != 0, df['VolBlw'] / df['volume'],
                                                 addDivBy0 if DEFAULT_DIV_BY0 else valueX)


# Nouvelles features - Features de Momentum:
# Moyenne des volumes
features_df['meanVolx'] = df['volume'].shift().rolling(window=5, min_periods=1).mean()


# Relatif volume evol
features_df['diffVolCandle_0_1Ratio'] = np.where(df['volume'].shift(1) != 0,
                                                 (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1),
                                                 diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Relatif delta evol
features_df['diffVolDelta_0_0Ratio'] = np.where(df['volume'] != 0,
                                                df['delta'] / df['volume'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['diffVolDelta_1_1Ratio'] = np.where(df['volume'].shift(1) != 0,
                                                df['delta'].shift(1) / df['volume'].shift(1),
                                                diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['diffVolDelta_2_2Ratio'] = np.where(df['volume'].shift(2) != 0,
                                                df['delta'].shift(2) / df['volume'].shift(2),
                                                diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['diffVolDelta_3_3Ratio'] = np.where(df['volume'].shift(3) != 0,
                                                df['delta'].shift(3) / df['volume'].shift(3),
                                                diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['diffVolDelta_0_1Ratio'] = np.where(df['delta'].shift(1) != 0,
                                                (df['delta'] - df['delta'].shift(1)) / df['delta'].shift(1),
                                                diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# cumDiffVolDelta
features_df['cumDiffVolDeltaRatio'] = np.where(features_df['meanVolx'] != 0,
                                               (df['delta'].shift(1) + df['delta'].shift(2) + \
                                                df['delta'].shift(3) + df['delta'].shift(4) + df['delta'].shift(5)) /
                                               features_df['meanVolx'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Nouvelles features - Features de Volume Profile:
# Importance du POC
features_df['VolPocVolCandleRatio'] = np.where(df['volume'] != 0, df['volPOC'] / df['volume'],
                                               addDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['pocDeltaPocVolRatio'] = np.where(df['volPOC'] != 0, df['deltaPOC'] / df['volPOC'],
                                              diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Asymétrie du volume
features_df['VolAbv_vol_ratio'] = np.where(df['volume'] != 0, (df['VolAbv']) / df['volume'],
                                           diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
features_df['VolBlw_vol_ratio'] = np.where(df['volume'] != 0, (df['VolBlw']) / df['volume'],
                                           diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['asymetrie_volume'] = np.where(df['volume'] != 0, (df['VolAbv'] - df['VolBlw']) / df['volume'],
                                           diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Nouvelles features - Features Cumulatives sur les 5 dernières bougies:
# Volume spike
features_df['VolCandleMeanxRatio'] = np.where(features_df['meanVolx'] != 0, df['volume'] / features_df['meanVolx'],
                                              addDivBy0 if DEFAULT_DIV_BY0 else valueX)


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
                                              (buy_pressureLow - sell_pressureLow) / total_volumeLow,
                                              diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

sell_pressureHigh = df['bidVolHigh_1'] + df['bidVolHigh_2']
buy_pressureHigh = df['askVolHigh'] + df['askVolHigh_1']
total_volumeHigh = sell_pressureHigh + buy_pressureHigh
features_df['imbalance_score_high'] = np.where(total_volumeHigh != 0,
                                               (sell_pressureHigh - buy_pressureHigh) / total_volumeHigh,
                                               diffDivBy0 if DEFAULT_DIV_BY0 else valueX)

# Finished Auction
features_df['finished_auction_high'] = (df['bidVolHigh'] == 0).astype(int)
features_df['finished_auction_low'] = (df['askVolLow'] == 0).astype(int)
features_df['staked00_high'] = ((df['bidVolHigh'] == 0) & (df['bidVolHigh_1'] == 0)).astype(int)
features_df['staked00_low'] = ((df['askVolLow'] == 0) & (df['askVolLow_1'] == 0)).astype(int)

dist_above, dist_below = calculate_naked_poc_distances(df)

features_df["naked_poc_dist_above"] = dist_above
features_df["naked_poc_dist_below"] = dist_below
print_notification("Ajout des informations sur les class et les trades")


features_df['diffPriceCloseVAH_0'] = df ['close']- df ['va_high_0']
features_df['diffPriceCloseVAL_0'] = df ['close']- df ['va_low_0']
features_df['ratio_delta_vol_VA_0'] = np.where(
    df['va_vol_0'] != 0,  # Condition
    df['va_delta_0'] / df['va_vol_0'],  # Valeur si la condition est vraie
    np.nan  # Valeur si la condition est fausse
)
# Identifier les débuts de session




#add processing metrics
features_df['class_binaire'] = df['class_binaire']
features_df['date'] = df['date']
features_df['trade_category'] = df['trade_category']

# Enregistrement des fichiers
print_notification("Début de l'enregistrement des fichiers")

# Extraire le nom du fichier et le répertoire
file_dir = os.path.dirname(CONFIG['FILE_PATH'])
file_name = os.path.basename(CONFIG['FILE_PATH'])


def toBeDisplayed_if_s(user_choice, choice):
    # Utilisation de l'opérateur ternaire
    result = True if user_choice == 'd' else (True if user_choice == 's' and choice == True else False)
    return result

# Ajouter les colonnes d'absorption au dictionnaire
absorption_settings = {f'is_absorpsion_{tick}ticks_{direction}': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False))
                      for tick in range(3, candle_rev_tick + 1)
                      for direction in ['low', 'high']}

# Définition de toutes les colonnes requises avec leurs identifiants
colonnes_a_transferer = [
    # Ratios de mouvement de base
    'ratio_volRevMove_volImpulsMove',  # //1
    'ratio_deltaImpulsMove_volImpulsMove',  # //2
    'ratio_deltaRevMove_volRevMove',  # //3

    # Ratios de zones
    'ratio_volZone1_volExtrem',  # //3.1
    'ratio_deltaZone1_volZone1',  # //3.2
    'ratio_deltaExtrem_volExtrem',  # //3.3

    # Ratios de zones de continuation
    'ratio_VolRevZone_XticksContZone',  # //4
    'ratioDeltaXticksContZone_VolXticksContZone',  # //5

    # Ratios de force
    'ratio_impulsMoveStrengthVol_XRevZone',  # //6
    'ratio_revMoveStrengthVol_XRevZone',  # //7

    # Type d'imbalance
    'imbType_contZone',  # //8

    # Ratios détaillés des zones
    'ratio_volRevMoveZone1_volImpulsMoveExtrem_XRevZone',  # //9.09
    'ratio_volRevMoveZone1_volRevMoveExtrem_XRevZone',  # //9.10
    'ratio_deltaRevMoveZone1_volRevMoveZone1',  # //9.11
    'ratio_deltaRevMoveExtrem_volRevMoveExtrem',  # //9.12
    'ratio_volImpulsMoveExtrem_volImpulsMoveZone1_XRevZone',  # //9.13
    'ratio_deltaImpulsMoveZone1_volImpulsMoveZone1',  # //9.14
    'ratio_deltaImpulsMoveExtrem_volImpulsMoveExtrem_XRevZone',  # //9.15

    # Métriques DOM et autres
    'cumDOM_AskBid_avgRatio',  # //10
    'cumDOM_AskBid_pullStack_avgDiff_ratio',  # //11
    'delta_impulsMove_XRevZone_bigStand_extrem',  # //12
    'delta_revMove_XRevZone_bigStand_extrem',  # //13

    # Ratios divers
    'ratio_delta_VaVolVa',  # //14
    'borderVa_vs_close',  # //15
    'ratio_volRevZone_VolCandle',  # //16
    'ratio_deltaRevZone_VolCandle',  # //17

    # process_reg_slope_replacement 18 19 20 21

    # Temps
    'timeElapsed2LastBar'  # //22
]

# Vérification des colonnes manquantes
colonnes_manquantes = [col for col in colonnes_a_transferer if col not in df.columns]

# Si des colonnes sont manquantes, lever une erreur avec la liste détaillée
if colonnes_manquantes:
    raise ValueError(f"Les colonnes suivantes sont manquantes dans le DataFrame source :\n" +
                     "\n".join(f"- {col}" for col in colonnes_manquantes))

# Si toutes les colonnes sont présentes, effectuer le transfert
for colonne in colonnes_a_transferer:
    features_df[colonne] = df[colonne]

#recopie les données sierra chart et met des 0 si pas asser de données en début de session
# Liste des fenêtres
# Usage example:
windows_list = [5, 10, 15, 30]
session_starts = (df['SessionStartEnd'] == 10).values
df_results = process_reg_slope_replacement(df, session_starts, windows_list, reg_feature_prefix="sc_reg_slope_")
# Fusionner avec features_df (assurez-vous que l'index est aligné)
features_df = pd.concat([features_df, df_results], axis=1)

df_results = process_reg_slope_replacement(df, session_starts, windows_list, reg_feature_prefix="sc_reg_std_")
# Fusionner avec features_df (assurez-vous que l'index est aligné)
features_df = pd.concat([features_df, df_results], axis=1)

print("Transfert réussi ! Toutes les colonnes ont été copiées avec succès.")

## 0) key nom de la feature / 1) Ative Floor / 2) Active Crop / 3) % à Floored / ') % à Croped / 5) Afficher et/ou inclure Features dans fichiers cibles
# choix des features à traiter
column_settings = {
    # Time-based features
    'deltaTimestampOpening': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession1min': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession1index': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession5min': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession5index': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession15min': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession15index': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession30min': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession30index': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaCustomSessionMin': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaCustomSessionIndex': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

    # Price and volume features
    'VolAbvState': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'VolBlwState': (False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'candleSizeTicks': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClosePoc_0_0': (True, True, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClosePoc_0_1': (True, True, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClosePoc_0_2': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClosePoc_0_3': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClosePoc_0_4': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClosePoc_0_5': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    # 'diffPriceClosePoc_0_6': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok

    'diffHighPrice_0_1': (True, True, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffHighPrice_0_2': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffHighPrice_0_3': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffHighPrice_0_4': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffHighPrice_0_5': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    # 'diffHighPrice_0_6': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok

    'diffLowPrice_0_1': (True, True, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffLowPrice_0_2': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffLowPrice_0_3': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffLowPrice_0_4': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffLowPrice_0_5': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    # 'diffLowPrice_0_6': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok

    'diffPriceCloseVWAP': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, True)),  # ok
    'diffPriceCloseVWAPbyIndex': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, True)),  # ok

    # Technical indicators
    'atr': (True, True, 0.1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok
    'bandWidthBB': (True, True, 0.1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok
    'perctBB': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok

    'perct_VA6P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'ratio_delta_vol_VA6P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClose_VA6PPoc': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA6PvaH': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA6PvaL': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'perct_VA11P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'ratio_delta_vol_VA11P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClose_VA11PPoc': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA11PvaH': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA11PvaL': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'perct_VA16P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'ratio_delta_vol_VA16P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClose_VA16PPoc': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA16PvaH': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA16PvaL': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'perct_VA21P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'ratio_delta_vol_VA21P': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffPriceClose_VA21PPoc': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA21PvaH': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':
    'diffPriceClose_VA21PvaL': (True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok':

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

    'market_regimeADX': (False, True, 0.5, 99.8, toBeDisplayed_if_s(user_choice, True)),
    'market_regimeADX_state': (False, False, 0.5, 99.8, toBeDisplayed_if_s(user_choice, True)),

    'is_in_range_10_32': (False, False, 0.5, 99.8, toBeDisplayed_if_s(user_choice, True)),
    'is_in_range_5_23': (False, False, 0.5, 99.8, toBeDisplayed_if_s(user_choice, True)),

    # Reversal and momentum features
    'bearish_reversal_force': (False, True, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok
    'bullish_reversal_force': (False, True, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),  # ok
      'meanVolx': (False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),  # ok

    'diffVolCandle_0_1Ratio': (False, True, 1, 98.5, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffVolDelta_0_1Ratio': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffVolDelta_0_0Ratio': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok

    'diffVolDelta_1_1Ratio': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffVolDelta_2_2Ratio': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok
    'diffVolDelta_3_3Ratio': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok

    'cumDiffVolDeltaRatio': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok

    # Volume profile features
    'VolPocVolCandleRatio': (False, False, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),  # ok
    'pocDeltaPocVolRatio': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok
    'VolAbv_vol_ratio': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok
    'VolBlw_vol_ratio': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok

    'asymetrie_volume': (True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok
    'VolCandleMeanxRatio': (False, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok


    # Imbalance features
    'bull_imbalance_low_1': (False, True, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'bull_imbalance_low_2': (False, True, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'bull_imbalance_low_3': (False, True, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'bull_imbalance_high_0': (False, True, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'bull_imbalance_high_1': (False, True, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'bull_imbalance_high_2': (False, True, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'bear_imbalance_low_0': (False, True, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'bear_imbalance_low_1': (False, True, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'bear_imbalance_low_2': (False, True, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'bear_imbalance_high_1': (False, True, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'bear_imbalance_high_2': (False, True, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'bear_imbalance_high_3': (False, True, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'imbalance_score_low': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'imbalance_score_high': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok1

    # Auction features
    'finished_auction_high': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'finished_auction_low': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'staked00_high': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok1
    'staked00_low': (False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),  # ok1

    'naked_poc_dist_above': (True, True, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'naked_poc_dist_below': (True, True, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'linear_slope_6': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'linear_slope_14': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'linear_slope_21': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    #'linear_slope_30': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    #'linear_slope_40': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'linear_slope_50': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'linear_slope_r2_6': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'linear_slope_r2_14': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'linear_slope_r2_21': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    #'linear_slope_r2_30': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    #'linear_slope_r2_40': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'linear_slope_r2_50': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'linear_slope_prevSession': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'close_sma_ratio_6': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'close_sma_ratio_14': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'close_sma_ratio_21': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'close_sma_ratio_30': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'close_sma_ratio_40': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'close_sma_ratio_50': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'close_sma_zscore_6': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'close_sma_zscore_14': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'close_sma_zscore_21': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'close_sma_zscore_30': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'close_sma_zscore_40': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'close_sma_zscore_50': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceCloseVAH_0': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceCloseVAL_0': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_delta_vol_VA_0': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # Ratios de volume et de mouvement
    'ratio_volRevMove_volImpulsMove': (False, True, 0.0, 98, toBeDisplayed_if_s(user_choice, False)),      # 1 - Ratio volume reversion/impulsion
    'ratio_deltaImpulsMove_volImpulsMove': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)), # 2 - Delta/Volume ratio pour mouvement impulsif
    'ratio_deltaRevMove_volRevMove': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),       # 3 - Delta/Volume ratio pour mouvement de reversion

    # Ratios de zones
    'ratio_volZone1_volExtrem': (False, True, 0.0, 98, toBeDisplayed_if_s(user_choice, False)),           # 3.1 - Ratio volume Zone1/Extreme
    'ratio_deltaZone1_volZone1': (False, False, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),          # 3.2 - Delta/Volume ratio pour Zone1
    'ratio_deltaExtrem_volExtrem': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),        # 3.3 - Delta/Volume ratio pour zone Extreme

    # Ratios de zones de continuation
    'ratio_VolRevZone_XticksContZone': (False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),    # 4 - Ratio volume reversion/continuation
    'ratioDeltaXticksContZone_VolXticksContZone': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)), # 5 - Delta/Volume ratio zone continuation

    # Ratios de force de mouvement
    'ratio_impulsMoveStrengthVol_XRevZone': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)), # 6 - Force volumique mouvement impulsif
    'ratio_revMoveStrengthVol_XRevZone': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),    # 7 - Force volumique mouvement reversion

    # Type d'imbalance
    'imbType_contZone': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),                     # 8 - Type d'imbalance zone continuation

    # Ratios détaillés de zones
    'ratio_volRevMoveZone1_volImpulsMoveExtrem_XRevZone': (False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)), # 9.09
    'ratio_volRevMoveZone1_volRevMoveExtrem_XRevZone': (False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),    # 9.10
    'ratio_deltaRevMoveZone1_volRevMoveZone1': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),            # 9.11
    'ratio_deltaRevMoveExtrem_volRevMoveExtrem': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),          # 9.12
    'ratio_volImpulsMoveExtrem_volImpulsMoveZone1_XRevZone': (False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)), # 9.13
    'ratio_deltaImpulsMoveZone1_volImpulsMoveZone1': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),         # 9.14
    'ratio_deltaImpulsMoveExtrem_volImpulsMoveExtrem_XRevZone': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)), # 9.15

    # Métriques DOM et VA
    'cumDOM_AskBid_avgRatio': (False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),                # 10 - Ratio moyen Ask/Bid cumulé
    'cumDOM_AskBid_pullStack_avgDiff_ratio': (False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)), # 11 - Ratio différence moyenne pull stack
    'delta_impulsMove_XRevZone_bigStand_extrem': (False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)), # 12
    'delta_revMove_XRevZone_bigStand_extrem': (False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),    # 13

    # Ratios divers
    'ratio_delta_VaVolVa': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),                   # 14 - Ratio delta/volume VA
    'borderVa_vs_close': (False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),                     # 15 - Position relative à la bordure VA
    'ratio_volRevZone_VolCandle': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),           # 16 - Ratio volume reversion/bougie
    'ratio_deltaRevZone_VolCandle': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),         # 17 - Ratio delta reversion/volume bougie

     'sc_reg_slope_5P': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)), # generated bu sierra chart      18
     'sc_reg_std_5P': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),# generated bu sierra chart         19
     'sc_reg_slope_10P': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),# generated bu sierra chart
     'sc_reg_std_10P': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),# generated bu sierra chart
     'sc_reg_slope_15P': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),# generated bu sierra chart      20
     'sc_reg_std_15P': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),# generated bu sierra chart        21
     'sc_reg_slope_30P': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),# generated bu sierra chart
     'sc_reg_std_30P': (False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),# generated bu sierra chart
# Temps
    'timeElapsed2LastBar': (False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),                   # 22 - Temps écoulé depuis dernière barre
    **absorption_settings  # Fusionner les dictionnaires
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

print(
    "Toutes les features nécessaires sont présentes et aucune colonne supplémentaire n'a été détectée. Poursuite du traitement.")

def calculate_percentiles(df_NANValue, columnName, settings, nan_replacement_values=None):
    """
    Calcule les percentiles tout en gérant les valeurs NaN et les valeurs de remplacement.
    Évite les erreurs en cas de colonne entièrement NaN ou filtrée.
    """

    # Récupération des paramètres de winsorisation
    floor_enabled, crop_enabled, floorInf_percentage, cropSup_percentage, _ = settings[columnName]

    # Gestion des valeurs de remplacement NaN
    if nan_replacement_values is not None and columnName in nan_replacement_values:
        nan_value = nan_replacement_values[columnName]
        mask = df_NANValue[columnName] != nan_value
        nan_count = (~mask).sum()
        print(f"   In calculate_percentiles:")
        print(f"     - Filter out {nan_count} nan replacement value(s) {nan_value} for {columnName}")
    else:
        mask = df_NANValue[columnName].notna()
        nan_count = df_NANValue[columnName].isna().sum()
        print(f"   In calculate_percentiles:")
        print(f"     - {nan_count} NaN value(s) found in {columnName}")

    # Filtrage des valeurs valides
    filtered_values = df_NANValue.loc[mask, columnName].values

    # 🚨 Vérification si filtered_values est vide
    if filtered_values.size == 0:
        print(f"⚠️ Warning: No valid values found in '{columnName}', skipping percentile calculation.")
        return None, None  # Ou des valeurs par défaut, ex: return 0, 1

    # Calcul des percentiles en fonction des options activées
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
                    print(
                        f"Les {total_replacements} valeurs NaN et infinies dans la colonne '{column}' ont été remplacées par {current_value}")
                    if increment != 0:
                        current_value += increment
                else:
                    print(
                        f"Les valeurs NaN et infinies dans la colonne '{column}' ont été laissées inchangées car start_value est 0")
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
    # winsorized_data = winsorized_data.fillna(nan_replacement_values.get(column, winsorized_data.median()))

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
                                                                         start_value, increment, REPLACE_NAN)
number_of_elementsnan_replacement_values = len(nan_replacement_values)
print(f"Le dictionnaire nan_replacement_values contient {number_of_elementsnan_replacement_values} éléments.")

# print(features_NANReplacedVal_df['bearish_bid_score'].describe())
# print("Traitement des valeurs NaN et infinies terminé.")
#
# print("Suppression des valeurs NAN ajoutées terminée.")

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
    current_feature = i + 1
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
    """
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
    """
    """
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
    """
print("\n")
print("Vérification finale :")
print(f"   - Nombre de colonnes dans winsorized_df : {len(winsorized_df.columns)}")

print(f"\n")

# print(f"   - Nombre de colonnes dans winsorized_scaledWithNanValue_df : {len(winsorized_scaledWithNanValue_df.columns)}")
# assert len(winsorized_df.columns) == len(winsorized_scaledWithNanValue_df.columns), "Le nombre de colonnes ne correspond pas entre les DataFrames"


print_notification(
    "Ajout de  'timeStampOpening', class_binaire', 'date', 'trade_category', 'SessionStartEnd' pour permettre la suite des traitements")
# Colonnes à ajouter
columns_to_add = ['timeStampOpening', 'class_binaire', 'candleDir', 'date', 'trade_category', 'SessionStartEnd',
                  'close', 'high', 'low','trade_pnl', 'tp1_pnl','tp2_pnl','tp3_pnl','sl_pnl']

# Vérifiez que toutes les colonnes existent dans df
missing_columns = [col for col in columns_to_add if col not in df.columns]
if missing_columns:
    error_message = f"Erreur: Les colonnes suivantes n'existent pas dans le DataFrame d'entrée: {', '.join(missing_columns)}"
    print(error_message)
    raise ValueError(error_message)

# Si nous arrivons ici, toutes les colonnes existent

# Créez un DataFrame avec les colonnes à ajouter

columns_df = df[columns_to_add]
# Ajoutez ces colonnes à features_df, winsorized_df en une seule opération
features_NANReplacedVal_df = pd.concat([features_NANReplacedVal_df, columns_df], axis=1)
winsorized_df = pd.concat([winsorized_df, columns_df], axis=1)

# winsorized_scaledWithNanValue_df = pd.concat([winsorized_scaledWithNanValue_df, columns_df], axis=1)

print_notification(
    "Colonnes 'timeStampOpening','class_binaire', 'candleDir', 'date', 'trade_category', 'SessionStartEnd' , 'close', "
    "'trade_pnl', 'tp1_pnl','tp2_pnl','tp3_pnl','sl_pnl' ajoutées")

file_without_extension = os.path.splitext(file_name)[0]
file_without_extension = file_without_extension.replace("Step4", "Step5")

# Créer le nouveau nom de fichier pour les features originales
new_file_name = file_without_extension + '_feat.csv'

# Construire le chemin complet du nouveau fichier
feat_file = os.path.join(file_dir, new_file_name)

# Créer le nouveau nom de fichier pour winsorized_df
winsorized_file_name = file_without_extension + '_feat_winsorized.csv'

# Construire le chemin complet du nouveau fichier winsorized
winsorized_file = os.path.join(file_dir, winsorized_file_name)

# Sauvegarder le fichier des features originales
print_notification(f"Enregistrement du fichier de features non modifiées : {feat_file}")
save_features_with_sessions(features_NANReplacedVal_df, CUSTOM_SESSIONS, feat_file)

# Sauvegarder le fichier winsorized
print_notification(f"Enregistrement du fichier de features winsorisées : {winsorized_file}")
save_features_with_sessions(winsorized_df, CUSTOM_SESSIONS, winsorized_file)
"""
# Créer le nouveau nom de fichier pour winsorized_scaledWithNanValue_df
scaled_file_name = file_without_extension+ '_feat_winsorizedScaledWithNanVal.csv'

# Construire le chemin complet du nouveau fichier scaled
scaled_file = os.path.join(file_dir, scaled_file_name)

# Sauvegarder le fichier scaled
winsorized_scaledWithNanValue_df.to_csv(scaled_file, sep=';', index=False, encoding='iso-8859-1')
print_notification(f"Enregistrement du fichier de features winsorisées et normalisées : {scaled_file}")
"""

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
