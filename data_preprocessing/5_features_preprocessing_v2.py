import pandas as pd
import numpy as np
from func_standard import print_notification, load_data, calculate_naked_poc_distances, CUSTOM_SESSIONS, \
    save_features_with_sessions,remplace_0_nan_reg_slope_p_2d,process_reg_slope_replacement, calculate_slopes_and_r2_numba,calculate_atr,calculate_percent_bb,enhanced_close_to_sma_ratio
from definition import *
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import os
import numpy as np


diffDivBy0 = np.nan
addDivBy0 = np.nan
valueX = np.nan
valueY = np.nan
from sklearn.preprocessing import MinMaxScaler
# Définition de la fonction calculate_max_ratio
import numpy as np
import time

import warnings
from pandas.errors import PerformanceWarning
# Nom du fichier

file_name = "Step4_version2_100325_260325_bugFixTradeResult1_extractOnlyFullSession_OnlyShort.csv"
#file_name = "Step4_5_0_5TP_1SL_150924_280225_bugFixTradeResult_extractOnlyFullSession_OnlyShort.csv"
#file_name = "Step4_version2_170924_110325_bugFixTradeResult1_extractOnlyFullSession_OnlyShort.csv"
# Chemin du répertoire
directory_path =  r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version2\merge"
directory_path =  r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\version2\merge\extend"

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

df = load_data(CONFIG['FILE_PATH'])
# Ignorer tous les avertissements de performance pandas
warnings.filterwarnings("ignore", category=PerformanceWarning)
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
        # print(f"   In calculate_percentiles:")
        # print(f"     - Filter out {nan_count} nan replacement value(s) {nan_value} for {columnName}")
    else:
        mask = df_NANValue[columnName].notna()
        nan_count = df_NANValue[columnName].isna().sum()
        # print(f"   In calculate_percentiles:")
        # print(f"     - {nan_count} NaN value(s) found in {columnName}")

    # Filtrage des valeurs valides
    filtered_values = df_NANValue.loc[mask, columnName].values

    # 🚨 Vérification si filtered_values est vide
    if filtered_values.size == 0:
        print(f"⚠️ Warning: No valid values found in '{columnName}', skipping percentile calculation.")
        return None, None  # Ou des valeurs par défaut, ex: return 0, 1

    # Calcul des percentiles en fonction des options activées
    floor_value = np.percentile(filtered_values, floorInf_percentage) if floor_enabled else None
    crop_value = np.percentile(filtered_values, cropSup_percentage) if crop_enabled else None

    # print(f"     - floor_value: {floor_value}   crop_value: {crop_value}")

    return floor_value, crop_value

import numpy as np
import pandas as pd


def replace_nan_and_inf(df, columns_to_process, REPLACE_NAN=True):
    # Paramètres
    start_value = REPLACED_NANVALUE_BY
    increment = REPLACED_NANVALUE_BY_INDEX
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

    number_of_elementsnan_replacement_values = len(nan_replacement_values)
    print(f"Le dictionnaire nan_replacement_values contient {number_of_elementsnan_replacement_values} éléments.")
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


def cropFloor_dataSource(features_NANReplacedVal_df, columnName, floorInf_booleen, cropSup_booleen, floorInf_percent,
                         cropSup_percent, nan_replacement_values=None):
    """
    Calcule les percentiles (floor et crop) tout en gérant les valeurs NaN et les valeurs de remplacement.
    """
    # Gestion des valeurs de remplacement NaN
    if nan_replacement_values is not None and columnName in nan_replacement_values:
        nan_value = nan_replacement_values[columnName]
        mask = features_NANReplacedVal_df[columnName] != nan_value
    else:
        mask = features_NANReplacedVal_df[columnName].notna()

    # Filtrage des valeurs valides
    filtered_values = features_NANReplacedVal_df.loc[mask, columnName].values

    # Vérification si filtered_values est vide
    if filtered_values.size == 0:
        print(f"⚠️ Warning: No valid values found in '{columnName}', skipping percentile calculation.")
        return None, None, floorInf_booleen, cropSup_booleen, floorInf_percent, cropSup_percent

    # Calcul des percentiles en fonction des options activées
    floor_valueNANfiltered = np.percentile(filtered_values, floorInf_percent) if floorInf_booleen else None
    crop_valueNANfiltered = np.percentile(filtered_values, cropSup_percent) if cropSup_booleen else None

    return floor_valueNANfiltered, crop_valueNANfiltered, floorInf_booleen, cropSup_booleen, floorInf_percent, cropSup_percent


import numpy as np

import numpy as np


def apply_winsorization(features_NANReplacedVal_df, columnName, floorInf_booleen, cropSup_booleen, floorInf_percent,
                        cropSup_percent, nan_replacement_values=None):
    """
    Calcule les percentiles et applique la winsorisation sur les données.
    """
    # Récupérer les valeurs pour la winsorisation
    floor_valueNANfiltered, crop_valueNANfiltered, _, _, _, _ = cropFloor_dataSource(
        features_NANReplacedVal_df,
        columnName,
        floorInf_booleen,
        cropSup_booleen,
        floorInf_percent,
        cropSup_percent,
        nan_replacement_values
    )

    # Winsorisation avec les valeurs NaN
    winsorized_valuesWithNanValue = winsorize(
        features_NANReplacedVal_df,
        columnName,
        floor_valueNANfiltered,
        crop_valueNANfiltered,
        floorInf_booleen,
        cropSup_booleen,
        nan_replacement_values
    )

    return winsorized_valuesWithNanValue


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
fig_range_input=''
DEFAULT_DIV_BY0 = True  # max_ratio or valuex
# user_choice = input("Appuyez sur Entrée pour calculer les features sans la afficher. \n"
#                     "Appuyez sur 'd' puis Entrée pour les calculer et les afficher : \n"
#                     "Appuyez sur 's' puis Entrée pour les calculer et les afficher :")
# if user_choice.lower() == 'd':
#     fig_range_input = input("Entrez la plage des figures à afficher au format x_y (par exemple 2_5) : \n")

# Demander à l'utilisateur s'il souhaite ajuster l'axe des abscisses
adjust_xaxis_input = ''
user_choice=''
if user_choice.lower() == 'd' or user_choice.lower() == 's':
    adjust_xaxis_input = input(
        "Voulez-vous afficher les graphiques entre les valeurs de floor et crop ? (o/n) : ").lower()

adjust_xaxis = adjust_xaxis_input == 'o'



def get_custom_section(minutes: int, custom_sections: dict) -> dict:
    """
    Retourne la section correspondant au nombre de minutes dans custom_sections.
    """
    for section_name, section in custom_sections.items():
        if section['start'] <= minutes < section['end']:
            return section
    # Retourne la dernière section si aucune correspondance
    return list(custom_sections.values())[-1]



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
    slopes, r2s,stds = calculate_slopes_and_r2_numba(close_values, session_starts, window)

    # Conversion en pandas DataFrame
    results_df = pd.DataFrame({
        f'linear_slope_{window}': slopes,
        f'linear_slope_r2_{window}': r2s,
        f'linear_slope_stds_{window}': stds
    }, index=data.index)

    return results_df


# Utilisation
windows = [
    #6, 14, 21,30, 40,
     10,50]
for window in windows:
    slope_r2_df = apply_optimized_slope_r2_calculation(df, window)
    features_df = pd.concat([features_df, slope_r2_df], axis=1)



windows_sma = [6, 14, 21, 30]
for window in windows_sma:
    ratio, zscore = enhanced_close_to_sma_ratio(df, window)
    features_df[f'close_sma_ratio_{window}'] = ratio
    features_df[f'close_sma_zscore_{window}'] = zscore

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

    Sélectionne 4 occurrences à partir de la 100e ligne du DataFrame, plutôt que les 4 premières.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes 'candleDir', 'high', 'close'.

    Returns:
        int: La valeur de CANDLE_REV_TICK si toutes les valeurs sont identiques pour les 4 occurrences.

    Raises:
        ValueError: Si les valeurs calculées diffèrent pour les 4 occurrences sélectionnées où candleDir == -1.
    """
    # Calculer la différence absolue entre les valeurs de 'close'
    df['close_diff'] = df['close'].diff().abs()

    # Identifier le minimum incrément non nul
    minimum_increment = df['close_diff'][df['close_diff'] > 0].min()

    # Vérifier si le minimum incrément est bien défini
    if pd.isna(minimum_increment):
        raise ValueError("Impossible de calculer le minimum incrément non nul.")

    print(f"Minimum increment: {minimum_increment}")

    # Filtrer les lignes où candleDir == -1
    filtered_df = df[df['candleDir'] == -1]

    # S'assurer qu'il y a au moins 100 lignes + 4 occurrences où candleDir == -1
    if len(filtered_df) < 104:
        raise ValueError(
            f"Pas assez d'occurrences où candleDir == -1 (trouvé {len(filtered_df)}, besoin d'au moins 104)")

    # Sélectionner 4 occurrences à partir de la 100e ligne
    selected_rows = filtered_df.iloc[100:].head(4)

    # Vérifier qu'on a bien 4 lignes
    if len(selected_rows) < 4:
        raise ValueError(
            f"Pas assez d'occurrences à partir de la 100e ligne (trouvé {len(selected_rows)}, besoin de 4)")

    # Calculer (high - close) * minimum_increment pour les 4 occurrences sélectionnées
    values = ((selected_rows['high'] - selected_rows['close']) * (1 / minimum_increment)) + 1

    # Vérifier si toutes les valeurs sont identiques
    if not all(values == values.iloc[0]):
        raise ValueError(
            "Les valeurs de (high - close) * minimum_increment diffèrent pour les 4 occurrences sélectionnées.")

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

features_df['diffPocPrice_0_1'] = df['pocPrice'] - df['pocPrice'].shift(1)
features_df['diffPocPrice_1_2'] = df['pocPrice'].shift(1) - df['pocPrice'].shift(2)
features_df['diffPocPrice_2_3'] = df['pocPrice'].shift(2) - df['pocPrice'].shift(3)
features_df['diffPocPrice_0_2'] = df['pocPrice'] - df['pocPrice'].shift(2)



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
# Définir la période comme variable
nb_periods = 5

# Moyenne des volumes sur les nb_periods dernières périodes (de t-1 à t-nb_periods)
features_df['meanVolx'] = df['volume'].shift(1).rolling(window=nb_periods, min_periods=1).mean()

# Somme des deltas sur les mêmes nb_periods périodes
features_df['cumDiffVolDeltaRatio'] = np.where(features_df['meanVolx'] != 0,
                                              sum(df['delta'].shift(i) for i in range(1, nb_periods + 1)) /
                                              features_df['meanVolx'], diffDivBy0 if DEFAULT_DIV_BY0 else valueX)
# Nouvelles features - Features de Volume Profile:
# Importance du POC
volconZone_zoneReversal = np.where(df['candleDir'] == -1, df['VolAbv'], df['VolBlw']) + df['vol_XticksContZone']


features_df['VolPocVolCandleRatio'] = np.where(df['volume'] != 0, df['volPOC'] / df['volume'],
                                               addDivBy0 if DEFAULT_DIV_BY0 else valueX)

features_df['VolPocVolRevesalXContRatio'] = np.where(volconZone_zoneReversal != 0, df['volPOC'] / volconZone_zoneReversal,
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

Imb_Div0=-6
Imb_zone=-2
# Nouvelles features - Order Flow:
# Imbalances haussières
features_df['bull_imbalance_low_1'] = np.where(
    df['bidVolLow'] == 0,
    Imb_Div0,
    np.where(
        (df['bidVolLow'] >= 1) & (df['bidVolLow'] <= 1),
        Imb_zone,
        df['askVolLow_1'] / df['bidVolLow']
    )
)
# Imbalances haussières
# Version simplifiée avec intervalle
features_df['bull_imbalance_low_2'] = np.where(
    df['bidVolLow_1'] == 0,
    Imb_Div0,
    np.where(
        (df['bidVolLow_1'] >= 1) & (df['bidVolLow_1'] <= 2),
        Imb_zone,
        df['askVolLow_2'] / df['bidVolLow_1']
    )
)


# # Définir des limites adaptées à votre distribution
# bins = [-np.inf, -3, 0, 1.4, 4, 6, np.inf]
# features_df['bull_imbalance_low_2'] = pd.cut(features_df['bull_imbalance_low_2'], bins=bins, labels=False)

# Imbalances haussières
# Version simplifiée avec intervalle
features_df['bull_imbalance_low_3'] = np.where(
    df['bidVolLow_2'] == 0,
    Imb_Div0,
    np.where(
        (df['bidVolLow_2'] >= 1) & (df['bidVolLow_2'] <= 1),
        Imb_zone,
        df['askVolLow_3'] / df['bidVolLow_2']
    )
)


features_df['bull_imbalance_high_0'] = np.where(
    df['bidVolHigh_1'] == 0,
    Imb_Div0,
    np.where(
        (df['bidVolHigh_1'] >= 1) & (df['bidVolHigh_1'] <= 1),
        Imb_zone,
        df['askVolHigh'] / df['bidVolHigh_1']
    )
)

features_df['bull_imbalance_high_1'] = np.where(
    df['bidVolHigh_2'] == 0,
    Imb_Div0,
    np.where(
        (df['bidVolHigh_2'] >= 1) & (df['bidVolHigh_2'] <= 1),
        Imb_zone,
        df['askVolHigh_1'] / df['bidVolHigh_2']
    )
)

features_df['bull_imbalance_high_2'] = np.where(
    df['bidVolHigh_3'] == 0,
    Imb_Div0,
    np.where(
        (df['bidVolHigh_3'] >= 1) & (df['bidVolHigh_3'] <= 1),
        Imb_zone,
        df['askVolHigh_2'] / df['bidVolHigh_3']
    )
)

from stats_sc.standard_stat_sc import *



# Imbalances baissières
features_df['bear_imbalance_low_0'] = np.where(
    df['askVolLow_1'] != 0,
    df['bidVolLow'] / df['askVolLow_1'],
    Imb_Div0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['bidVolLow'] / df['askVolLow_1'],
            df['askVolLow_1'] != 0
        )
    )
)

features_df['bear_imbalance_low_1'] = np.where(
    df['askVolLow_2'] != 0,
    df['bidVolLow_1'] / df['askVolLow_2'],
    Imb_Div0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['bidVolLow_1'] / df['askVolLow_2'],
            df['askVolLow_2'] != 0
        )
    )
)

features_df['bear_imbalance_low_2'] = np.where(
    df['askVolLow_3'] != 0,
    df['bidVolLow_2'] / df['askVolLow_3'],
    Imb_Div0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['bidVolLow_2'] / df['askVolLow_3'],
            df['askVolLow_3'] != 0
        )
    )
)

# Imbalances baissières
features_df['bear_imbalance_low_0'] = np.where(
    df['askVolLow_1'] == 0,
    Imb_Div0,
    np.where(
        (df['askVolLow_1'] >= 1) & (df['askVolLow_1'] <= 1),
        Imb_zone,
        df['bidVolLow'] / df['askVolLow_1']
    )
)

features_df['bear_imbalance_low_1'] = np.where(
    df['askVolLow_2'] == 0,
    Imb_Div0,
    np.where(
        (df['askVolLow_2'] >= 1) & (df['askVolLow_2'] <= 1),
        Imb_zone,
        df['bidVolLow_1'] / df['askVolLow_2']
    )
)

features_df['bear_imbalance_low_2'] = np.where(
    df['askVolLow_3'] == 0,
    Imb_Div0,
    np.where(
        (df['askVolLow_3'] >= 1) & (df['askVolLow_3'] <= 1),
        Imb_zone,
        df['bidVolLow_2'] / df['askVolLow_3']
    )
)

features_df['bear_imbalance_high_1'] = np.where(
    df['askVolHigh'] != 0,
    df['bidVolHigh_1'] / df['askVolHigh'],
    Imb_Div0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['bidVolHigh_1'] / df['askVolHigh'],
            df['askVolHigh'] != 0
        )
    )
)

features_df['bear_imbalance_high_2'] = np.where(
    df['askVolHigh_1'] != 0,
    df['bidVolHigh_2'] / df['askVolHigh_1'],
    Imb_Div0 if DEFAULT_DIV_BY0 else (
        calculate_max_ratio(
            df['bidVolHigh_2'] / df['askVolHigh_1'],
            df['askVolHigh_1'] != 0
        )
    )
)

features_df['bear_imbalance_high_3'] = np.where(
    df['askVolHigh_2'] != 0,
    df['bidVolHigh_3'] / df['askVolHigh_2'],
    Imb_Div0 if DEFAULT_DIV_BY0 else (
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
absorption_settings = {f'is_absorpsion_{tick}ticks_{direction}': ("winsor",None,False, False, 10, 90, toBeDisplayed_if_s(user_choice, False))
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

#features_df['diffLowPriceMean_2_3']=(features_df['diffLowPrice_0_2']+features_df['diffLowPrice_0_3']
                                     #+features_df['diffLowPrice_0_3']
 #                                    )/2

## 0) key nom de la feature / 1) Ative Floor / 2) Active Crop / 3) % à Floored / ') % à Croped / 5) Afficher et/ou inclure Features dans fichiers cibles
# choix des features à traiter



# Liste de toutes les colonnes à inclure si la condition est remplie
colonnes_a_inclure = [
    "ratio_vol_VolCont_ZoneA_xTicksContZone",
    "ratio_delta_VolCont_ZoneA_xTicksContZone",
    "ratio_vol_VolCont_ZoneB_xTicksContZone",
    "ratio_delta_VolCont_ZoneB_xTicksContZone",
    "ratio_vol_VolCont_ZoneC_xTicksContZone",
    "ratio_delta_VolCont_ZoneC_xTicksContZone"
]

# Vérifier si la colonne spécifique est présente dans df
if "ratio_vol_VolCont_ZoneA_xTicksContZone" in df.columns:
    # Vérifier que toutes les colonnes existent dans df
    colonnes_existantes = [col for col in colonnes_a_inclure if col in df.columns]

    # Si features_df n'existe pas encore, le créer avec ces colonnes
    if 'features_df' not in locals():
        features_df = df[colonnes_existantes].copy()
    # Sinon, ajouter ces colonnes à features_df existant
    else:
        for col in colonnes_existantes:
            features_df[col] = df[col]

    print(f"Colonnes ajoutées à features_df: {colonnes_existantes}")
else:
    print("La colonne 'ratio_vol_VolCont_ZoneA_xTicksContZone' n'est pas présente dans df")


def add_stochastic_force_indicators(df, features_df,
                                    k_period_overbought, d_period_overbought,
                                    k_period_oversold, d_period_oversold,
                                    overbought_threshold=80, oversold_threshold=20,
                                    fi_short=1, fi_long=6):
    """
    Ajoute le Stochastique Rapide et le Force Index aux features,
    avec des périodes distinctes pour les zones de surachat et survente.

    Paramètres:
    - df: DataFrame source contenant les données brutes
    - features_df: DataFrame de destination pour les features
    - k_period_overbought: Période %K pour la détection de surachat
    - d_period_overbought: Période %D pour la détection de surachat
    - k_period_oversold: Période %K pour la détection de survente
    - d_period_oversold: Période %D pour la détection de survente
    - overbought_threshold: Seuil de surachat (défaut: 80)
    - oversold_threshold: Seuil de survente (défaut: 20)
    - fi_short: Période court terme pour le Force Index
    - fi_long: Période long terme pour le Force Index

    Retourne:
    - features_df avec les nouvelles colonnes d'indicateurs techniques
    """
    if (k_period_overbought is None or d_period_overbought is None or
            k_period_oversold is None or d_period_oversold is None):
        raise ValueError("Toutes les périodes pour surachat et survente doivent être spécifiées")

    try:
        # Assurer que les données d'entrée sont numériques
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        volume = pd.to_numeric(df['volume'], errors='coerce')
        candle_dir = pd.to_numeric(df['candleDir'], errors='coerce')
        session_starts = (df['SessionStartEnd'] == 10).values

        # Calculer les stochastiques avec périodes spécifiques
        k_overbought, d_overbought = compute_stoch(high, low, close,session_starts,
                                                   k_period_overbought,
                                                   d_period_overbought,
                                                   fill_value=50)

        k_oversold, d_oversold = compute_stoch(high, low, close,session_starts,
                                               k_period_oversold,
                                               d_period_oversold,
                                               fill_value=50)

        # Ajouter les stochastiques spécifiques au DataFrame
        features_df['stoch_k_overbought'] = k_overbought
        features_df['stoch_d_overbought'] = d_overbought
        features_df['stoch_k_oversold'] = k_oversold
        features_df['stoch_d_oversold'] = d_oversold

        # Force Index
        price_change = close.diff().fillna(0)
        force_index_raw = price_change * volume

        # Force Index court terme
        features_df[f'force_index_{fi_short}'] = pd.Series(force_index_raw).ewm(span=fi_short,
                                                                                adjust=False).mean().values

        # Force Index long terme
        features_df[f'force_index_{fi_long}'] = pd.Series(force_index_raw).ewm(span=fi_long, adjust=False).mean().values

        # Assurer que les séries sont numériques pour les opérations suivantes
        stoch_k_overbought = pd.Series(features_df['stoch_k_overbought']).astype(float)
        stoch_d_overbought = pd.Series(features_df['stoch_d_overbought']).astype(float)
        stoch_k_oversold = pd.Series(features_df['stoch_k_oversold']).astype(float)
        stoch_d_oversold = pd.Series(features_df['stoch_d_oversold']).astype(float)

        force_short = pd.Series(features_df[f'force_index_{fi_short}']).astype(float)
        force_long = pd.Series(features_df[f'force_index_{fi_long}']).astype(float)

        # Features dérivées du Stochastique - pour chaque variante
        # Crossover pour la variante surachat
        stoch_cross_overbought = np.zeros(len(stoch_k_overbought))
        for i in range(1, len(stoch_k_overbought)):
            if (pd.notna(stoch_k_overbought[i]) and pd.notna(stoch_d_overbought[i]) and
                    pd.notna(stoch_k_overbought[i - 1]) and pd.notna(stoch_d_overbought[i - 1])):
                if (stoch_k_overbought[i - 1] < stoch_d_overbought[i - 1] and
                        stoch_k_overbought[i] > stoch_d_overbought[i]):
                    stoch_cross_overbought[i] = 1
                elif (stoch_k_overbought[i - 1] > stoch_d_overbought[i - 1] and
                      stoch_k_overbought[i] < stoch_d_overbought[i]):
                    stoch_cross_overbought[i] = -1

        # Crossover pour la variante survente
        stoch_cross_oversold = np.zeros(len(stoch_k_oversold))
        for i in range(1, len(stoch_k_oversold)):
            if (pd.notna(stoch_k_oversold[i]) and pd.notna(stoch_d_oversold[i]) and
                    pd.notna(stoch_k_oversold[i - 1]) and pd.notna(stoch_d_oversold[i - 1])):
                if (stoch_k_oversold[i - 1] < stoch_d_oversold[i - 1] and
                        stoch_k_oversold[i] > stoch_d_oversold[i]):
                    stoch_cross_oversold[i] = 1
                elif (stoch_k_oversold[i - 1] > stoch_d_oversold[i - 1] and
                      stoch_k_oversold[i] < stoch_d_oversold[i]):
                    stoch_cross_oversold[i] = -1



        # Zones de surachat/survente avec leurs stochastiques respectifs
        features_df['is_stoch_overbought'] = np.where(stoch_k_overbought > overbought_threshold, 1, 0)
        features_df['is_stoch_oversold'] = np.where(stoch_k_oversold < oversold_threshold, 1, 0)

        # Features dérivées du Force Index
        avg_volume_20 = volume.rolling(window=4).mean().fillna(volume)

        # Normalisation avec gestion des divisions par zéro
        fi_short_norm = np.where(avg_volume_20 > 0, force_short / avg_volume_20, 0)
        fi_long_norm = np.where(avg_volume_20 > 0, force_long / avg_volume_20, 0)

        features_df[f'force_index_{fi_short}_norm'] = fi_short_norm
        features_df[f'force_index_{fi_long}_norm'] = fi_long_norm

        # Divergence entre Force Index court et long terme
        features_df['force_index_divergence'] = fi_short_norm - fi_long_norm

        # Momentum basé sur le Force Index
        features_df['fi_momentum'] = np.sign(force_short) * np.abs(fi_short_norm)



        return features_df

    except Exception as e:
        print(f"Erreur dans add_stochastic_force_indicators: {str(e)}")
        # En cas d'erreur, retourner le DataFrame original sans modifications
        return features_df


def add_atr(df, features_df, atr_period_range=14, atr_period_extrem=14,
            atr_low_threshold_range=2, atr_high_threshold_range=5,
            atr_low_threshold_extrem=1):
    """
    Ajoute l'indicateur ATR (Average True Range) et des signaux dérivés au DataFrame de features.
    Utilise potentiellement des périodes différentes pour les indicateurs de range et extremLow.

    Paramètres:
    - df: DataFrame contenant les colonnes 'high', 'low', 'close'
    - features_df: DataFrame où ajouter les colonnes liées à l'ATR
    - atr_period_range: Période de calcul de l'ATR pour l'indicateur range (défaut: 14)
    - atr_period_extrem: Période de calcul de l'ATR pour l'indicateur extremLow (défaut: 14)
    - atr_low_threshold_range: Seuil bas pour la plage modérée d'ATR (défaut: 2)
    - atr_high_threshold_range: Seuil haut pour la plage modérée d'ATR (défaut: 5)
    - atr_low_threshold_extrem: Seuil bas pour les valeurs extrêmes d'ATR (défaut: 1)

    Retourne:
    - features_df enrichi des colonnes ATR et dérivées
    """
    try:
        # Calcul de l'ATR avec la période optimisée pour l'indicateur range
        atr_values_range = calculate_atr(df, atr_period_range)

        # Calcul de l'ATR avec la période optimisée pour l'indicateur extremLow
        # Si les périodes sont identiques, éviter de calculer deux fois
        if atr_period_range == atr_period_extrem:
            atr_values_extrem = atr_values_range
        else:
            atr_values_extrem = calculate_atr(df, atr_period_extrem)

        # Ajouter les valeurs brutes d'ATR au DataFrame de features
        features_df['atr_range'] = atr_values_range
        features_df['atr_extrem'] = atr_values_extrem

        # Créer l'indicateur pour la plage "modérée" d'ATR (optimisée pour le win rate)
        features_df['is_atr_range'] = np.where(
            (atr_values_range > atr_low_threshold_range) & (atr_values_range < atr_high_threshold_range),
            1, 0
        )

        # Créer l'indicateur pour les valeurs extrêmement basses d'ATR
        features_df['is_atr_extremLow'] = np.where(
            (atr_values_extrem < atr_low_threshold_extrem),
            1, 0
        )

        # S'assurer que toutes les colonnes sont numériques
        for col in ['atr_range', 'atr_extrem', 'is_atr_range', 'is_atr_extremLow']:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)


    except Exception as e:
        print(f"Erreur dans add_atr: {str(e)}")
        # En cas d'erreur, tenter de renvoyer au moins les colonnes existantes
        if 'atr_range' not in features_df.columns:
            features_df['atr_range'] = 0
        if 'atr_extrem' not in features_df.columns:
            features_df['atr_extrem'] = 0
        if 'is_atr_range' not in features_df.columns:
            features_df['is_atr_range'] = 0
        if 'is_atr_extremLow' not in features_df.columns:
            features_df['is_atr_extremLow'] = 0

    return features_df


def add_regression_slope(df, features_df,
                         period_range=14, period_extrem=14,
                         slope_range_threshold_low=0.1, slope_range_threshold_high=0.5,
                         slope_extrem_threshold_low=0.1, slope_extrem_threshold_high=0.5):
    """
    Ajoute les indicateurs de régression de pente au DataFrame de features.
    Utilise des seuils spécifiques à chaque période.

    Paramètres:
    - df: DataFrame contenant les données de prix
    - features_df: DataFrame où ajouter les indicateurs
    - period_low: Période pour le calcul de la pente de l'indicateur is_rangeSlope
    - period_high: Période pour le calcul de la pente de l'indicateur is_extremSlope
    - slope_range_threshold_low: Seuil bas pour la détection des pentes modérées (period_low)
    - slope_extrem_threshold_low: Seuil haut pour la détection des pentes modérées (period_low)
    - slope_range_threshold_high: Seuil bas pour la détection des pentes fortes (period_high)
    - slope_extrem_threshold_high: Seuil haut pour la détection des pentes fortes (period_high)

    Retourne:
    - features_df enrichi des indicateurs de pente
    """
    try:
        close = pd.to_numeric(df['close'], errors='coerce').values
        session_starts = (df['SessionStartEnd'] == 10).values

        # Calcul des pentes pour l'indicateur is_rangeSlope
        slopes_low, r2_low, std_low = calculate_slopes_and_r2_numba(close, session_starts, period_range)

        # Calcul des pentes pour l'indicateur is_extremSlope (uniquement si période différente)
        if period_range == period_extrem:
            slopes_high = slopes_low
        else:
            slopes_high, r2_high, std_high = calculate_slopes_and_r2_numba(close, session_starts, period_extrem)

        # Ajouter les valeurs brutes au DataFrame de features
        features_df['slope_range'] = slopes_low
        if period_range != period_extrem:
            features_df['slope_extrem'] = slopes_high
        else:
            features_df['slope_extrem'] = slopes_low

        # Créer l'indicateur is_rangeSlope (pentes modérées optimisées pour maximiser le win rate)
        # is_rangeSlope = 1 quand la pente est entre slope_range_threshold_low et slope_extrem_threshold_low
        features_df['is_rangeSlope'] = np.where(
            (slopes_low > slope_range_threshold_low) & (slopes_low < slope_range_threshold_high),
            1, 0
        )

        # Créer l'indicateur is_extremSlope (pentes fortes optimisées pour minimiser le win rate)
        # is_extremSlope = 1 quand la pente est soit inférieure à slope_range_threshold_high
        # soit supérieure à slope_extrem_threshold_high
        features_df['is_extremSlope'] = np.where(
            (slopes_high < slope_extrem_threshold_low) | (slopes_high > slope_extrem_threshold_high),
            1, 0
        )

        # S'assurer que toutes les colonnes sont numériques
        for col in ['slope_range', 'slope_extrem', 'is_rangeSlope', 'is_extremSlope']:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)


    except Exception as e:
        print(f"Erreur dans add_regression_slope: {str(e)}")
        # En cas d'erreur, tenter de renvoyer au moins les colonnes existantes
        if 'slope_range' not in features_df.columns:
            features_df['slope_range'] = 0
        if 'slope_extrem' not in features_df.columns:
            features_df['slope_extrem'] = 0
        if 'is_rangeSlope' not in features_df.columns:
            features_df['is_rangeSlope'] = 0
        if 'is_extremSlope' not in features_df.columns:
            features_df['is_extremSlope'] = 0

    return features_df


def add_zscore(df, features_df,
               period_range=14, period_extrem=14,
               zscore_range_threshold_low=-2.0, zscore_range_threshold_high=0.5,
               zscore_extrem_threshold_low=-2.0, zscore_extrem_threshold_high=0.5):
    """
    Ajoute les indicateurs de Z-Score au DataFrame de features.
    Utilise des seuils spécifiques à chaque période.

    Paramètres:
    - df: DataFrame contenant les données de prix
    - features_df: DataFrame où ajouter les indicateurs
    - period_range: Période pour le calcul du Z-Score de l'indicateur is_zscore_range
    - period_extrem: Période pour le calcul du Z-Score de l'indicateur is_zscore_extrem
                    (Si 0, seul l'indicateur is_zscore_range sera calculé)
    - zscore_range_threshold_low: Seuil bas pour la zone modérée du Z-Score
    - zscore_range_threshold_high: Seuil haut pour la zone modérée du Z-Score
    - zscore_extrem_threshold_low: Seuil bas pour la zone extrême du Z-Score
    - zscore_extrem_threshold_high: Seuil haut pour la zone extrême du Z-Score

    Retourne:
    - features_df enrichi des indicateurs de Z-Score
    """
    try:
        # Vérifier que period_range est valide (> 0)
        if period_range <= 0:
            print(f"Erreur: period_range doit être > 0 (valeur actuelle: {period_range})")
            return features_df

        # Calcul du Z-Score pour l'indicateur is_zscore_range
        _, zscores_range = enhanced_close_to_sma_ratio(df, period_range)

        # Ajouter les valeurs brutes au DataFrame de features
        features_df['zscore_range'] = zscores_range

        # Créer l'indicateur is_zscore_range (Z-Scores modérés optimisés pour maximiser le win rate)
        features_df['is_zscore_range'] = np.where(
            (zscores_range > zscore_range_threshold_low) & (zscores_range < zscore_range_threshold_high),
            1, 0
        )

        # Calculer et ajouter is_zscore_extrem seulement si period_extrem > 0
        if period_extrem > 0:
            # Calcul du Z-Score pour l'indicateur is_zscore_extrem
            if period_range == period_extrem:
                zscores_extrem = zscores_range
            else:
                _, zscores_extrem = enhanced_close_to_sma_ratio(df, period_extrem)

            # Ajouter les valeurs brutes
            features_df['zscore_extrem'] = zscores_extrem

            # Créer l'indicateur is_zscore_extrem
            features_df['is_zscore_extrem'] = np.where(
                (zscores_extrem < zscore_extrem_threshold_low) | (zscores_extrem > zscore_extrem_threshold_high),
                1, 0
            )
        else:
            # Si period_extrem est 0, ne pas calculer is_zscore_extrem
            features_df['zscore_extrem'] = 0
            features_df['is_zscore_extrem'] = 0
            print("Avertissement: period_extrem = 0, is_zscore_extrem est fixé à 0")

        # S'assurer que toutes les colonnes sont numériques
        for col in ['zscore_range', 'zscore_extrem', 'is_zscore_range', 'is_zscore_extrem']:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        import traceback
        print(f"Erreur dans add_zscore: {str(e)}")
        traceback.print_exc()
        # En cas d'erreur, tenter de renvoyer au moins les colonnes existantes
        if 'zscore_range' not in features_df.columns:
            features_df['zscore_range'] = 0
        if 'zscore_extrem' not in features_df.columns:
            features_df['zscore_extrem'] = 0
        if 'is_zscore_range' not in features_df.columns:
            features_df['is_zscore_range'] = 0
        if 'is_zscore_extrem' not in features_df.columns:
            features_df['is_zscore_extrem'] = 0

    return features_df

def add_perctBB_simu(df, features_df,
                     period_high=105, period_low=5,
                     std_dev_high=1.9481898795476222, std_dev_low=0.23237747131209152,
                     bb_high_threshold=0.6550726973429961, bb_low_threshold=0.2891135240579008):
    """
    Ajoute l'indicateur Percent B (%B) des bandes de Bollinger sur des périodes potentiellement
    différentes pour les zones hautes et basses, ainsi que des indicateurs dérivés.
    Version optimisée utilisant directement les fonctions Numba.

    Paramètres:
    - df: DataFrame contenant les données de prix
    - features_df: DataFrame où ajouter les colonnes liées au %B
    - period_high: Période de calcul pour la zone haute (ex: 105)
    - period_low: Période de calcul pour la zone basse (ex: 5)
    - std_dev_high: Nombre d'écarts-types pour la zone haute (ex: 1.95)
    - std_dev_low: Nombre d'écarts-types pour la zone basse (ex: 0.23)
    - bb_high_threshold: Seuil haut pour la zone haute (ex: 0.65)
    - bb_low_threshold: Seuil bas pour la zone basse (ex: 0.29)

    Retourne:
    - features_df enrichi des colonnes %B et dérivées
    """
    try:
        # Calcul du %B pour la zone haute (obtenir directement le tableau NumPy)
        percent_b_high_values = calculate_percent_bb(
            df=df, period=period_high, std_dev=std_dev_high, fill_value=0, return_array=True
        )

        # Créer un DataFrame temporaire pour l'affichage si nécessaire
        percent_b_high_df = pd.DataFrame({'percent_b': percent_b_high_values}, index=df.index)
        print(percent_b_high_df.head(200))

        # Calcul du %B pour la zone basse (uniquement si différente)
        if period_high == period_low and std_dev_high == std_dev_low:
            percent_b_low_values = percent_b_high_values
        else:
            percent_b_low_values = calculate_percent_bb(
                df=df, period=period_low, std_dev=std_dev_low, fill_value=0, return_array=True
            )

        # Ajouter les indicateurs %B bruts
        features_df['percent_b_high'] = percent_b_high_values
        if period_high != period_low or std_dev_high != std_dev_low:
            features_df['percent_b_low'] = percent_b_low_values
        else:
            features_df['percent_b_high'] = percent_b_high_values

        # Créer l'indicateur is_bb_high (zone haute optimisée pour maximiser le win rate)
        features_df['is_bb_high'] = np.where(
            (percent_b_high_values >= bb_high_threshold),
            1, 0
        )

        # Créer l'indicateur is_bb_low (zone basse optimisée pour minimiser le win rate)
        features_df['is_bb_low'] = np.where(
            (percent_b_low_values <= bb_low_threshold),
            1, 0
        )

        # S'assurer que toutes les colonnes sont numériques
        for col in features_df.columns:
            if col.startswith('percent_b') or col.startswith('is_bb'):
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"Erreur dans add_perctBB_simu: {str(e)}")
        # En cas d'erreur, tenter de renvoyer au moins les colonnes essentielles
        columns_to_check = [
            'percent_b_high', 'percent_b_low',
            'is_bb_high', 'is_bb_low'
        ]

        for col in columns_to_check:
            if col not in features_df.columns:
                features_df[col] = 0

    return features_df

def add_vwap(df, features_df,
             vwap_range_threshold_low=-2.6705237017186305, vwap_range_threshold_high=1.47028136092062,
             vwap_extrem_threshold_low=-30.3195, vwap_extrem_threshold_high=49.1878):
    """
    Ajoute les indicateurs basés sur la différence entre le prix et le VWAP
    pour identifier les zones favorables et défavorables pour les positions short.
    Utilise la colonne 'diffPriceCloseVWAP' déjà présente dans le DataFrame.

    Paramètres:
    - df: DataFrame contenant la colonne 'diffPriceCloseVWAP'
    - features_df: DataFrame où ajouter les indicateurs dérivés
    - vwap_range_threshold_low: Seuil bas pour la zone favorable (différence avec VWAP)
    - vwap_range_threshold_high: Seuil haut pour la zone favorable (différence avec VWAP)
    - vwap_extrem_threshold_low: Seuil bas pour la zone non favorable (différence avec VWAP)
    - vwap_extrem_threshold_high: Seuil haut pour la zone non favorable (différence avec VWAP)

    Retourne:
    - features_df enrichi des colonnes d'indicateurs VWAP
    """
    try:
        # Récupérer la différence prix-VWAP déjà calculée
        diff_vwap = pd.to_numeric(features_df['diffPriceCloseVWAP'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')


        # Normaliser la différence par rapport au prix de clôture (pourcentage)
        # Si le prix est 0, utiliser 1 pour éviter la division par zéro
        norm_diff_vwap = np.where(close > 0, diff_vwap / close, diff_vwap)
        features_df['norm_diff_vwap'] = norm_diff_vwap

        # Créer l'indicateur is_vwap_shortArea (zone favorable pour les shorts)
        # Typiquement, quand le prix est modérément au-dessus du VWAP
        features_df['is_vwap_shortArea'] = np.where(
            (diff_vwap > vwap_range_threshold_low) & (diff_vwap < vwap_range_threshold_high),
            1, 0
        )

        # Créer l'indicateur is_vwap_notShortArea (zone non favorable pour les shorts)
        # Typiquement, quand le prix est trop au-dessus ou en-dessous du VWAP
        features_df['is_vwap_notShortArea'] = np.where(
            (diff_vwap < vwap_extrem_threshold_low) | (diff_vwap > vwap_extrem_threshold_high),
            1, 0
        )

        # Croisements du VWAP
        vwap_cross = np.zeros_like(diff_vwap)
        for i in range(1, len(diff_vwap)):
            if diff_vwap[i - 1] < 0 and diff_vwap[i] > 0:
                vwap_cross[i] = 1  # Prix croise au-dessus du VWAP
            elif diff_vwap[i - 1] > 0 and diff_vwap[i] < 0:
                vwap_cross[i] = -1  # Prix croise en-dessous du VWAP



        # S'assurer que toutes les colonnes sont numériques
        vwap_columns = ['norm_diff_vwap', 'is_vwap_shortArea',
                        'is_vwap_notShortArea']
        for col in vwap_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"Erreur dans add_vwap: {str(e)}")
        # En cas d'erreur, initialiser les colonnes principales à 0
        required_columns = [
            'norm_diff_vwap', 'is_vwap_shortArea', 'is_vwap_notShortArea',
        ]
        for col in required_columns:
            if col not in features_df.columns:
                features_df[col] = 0

    return features_df




def add_std_regression(df, features_df,
                       period_range=14, period_extrem=14,
                       std_low_threshold_range=0.1, std_high_threshold_range=0.5,
                       std_low_threshold_extrem=0.1, std_high_threshold_extrem=0.5):
    """
    Ajoute les indicateurs de volatilité basés sur l'écart-type de régression au DataFrame de features.
    Utilise des seuils spécifiques à chaque période.

    Paramètres:
    - df: DataFrame contenant les données de prix
    - features_df: DataFrame où ajouter les indicateurs
    - period_range: Période pour le calcul de l'écart-type de l'indicateur range_volatility
    - period_extrem: Période pour le calcul de l'écart-type de l'indicateur extrem_volatility
    - std_low_threshold_range: Seuil bas pour la détection de volatilité modérée
    - std_high_threshold_range: Seuil haut pour la détection de volatilité modérée
    - std_low_threshold_extrem: Seuil bas pour la détection de volatilité extrême
    - std_high_threshold_extrem: Seuil haut pour la détection de volatilité extrême

    Retourne:
    - features_df enrichi des indicateurs de volatilité
    """
    try:
        close = pd.to_numeric(df['close'], errors='coerce').values
        session_starts = (df['SessionStartEnd'] == 10).values

        # Calcul des écarts-types pour l'indicateur range_volatility
        _, _, stds_range = calculate_slopes_and_r2_numba(close, session_starts, period_range)

        # Calcul des écarts-types pour l'indicateur extrem_volatility (uniquement si période différente)
        if period_range == period_extrem:
            stds_extrem = stds_range
        else:
            _, _, stds_extrem = calculate_slopes_and_r2_numba(close, session_starts, period_extrem)

        # Ajouter les valeurs brutes au DataFrame de features
        features_df['std_range'] = stds_range
        if period_range != period_extrem:
            features_df['std_extrem'] = stds_extrem
        else:
            features_df['std_extrem'] = stds_range

        # Créer l'indicateur is_range_volatility (volatilité modérée optimisée pour maximiser le win rate)
        # is_range_volatility = 1 quand l'écart-type est entre std_low_threshold_range et std_high_threshold_range
        features_df['is_range_volatility_std'] = np.where(
            (stds_range > std_low_threshold_range) & (stds_range < std_high_threshold_range),
            1, 0
        )

        # Créer l'indicateur is_extrem_volatility (volatilité extrême optimisée pour minimiser le win rate)
        # is_extrem_volatility = 1 quand l'écart-type est soit inférieur à std_low_threshold_extrem
        # soit supérieur à std_high_threshold_extrem
        features_df['is_extrem_volatility_std'] = np.where(
            (stds_extrem < std_low_threshold_extrem) | (stds_extrem > std_high_threshold_extrem),
            1, 0
        )

        # S'assurer que toutes les colonnes sont numériques
        for col in ['std_range', 'std_extrem', 'is_range_volatility', 'is_extrem_volatility']:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)





    except Exception as e:
        print(f"Erreur dans add_std_regression: {str(e)}")
        # En cas d'erreur, tenter de renvoyer au moins les colonnes existantes
        if 'std_range' not in features_df.columns:
            features_df['std_range'] = 0
        if 'std_extrem' not in features_df.columns:
            features_df['std_extrem'] = 0
        if 'is_range_volatility' not in features_df.columns:
            features_df['is_range_volatility'] = 0
        if 'is_extrem_volatility' not in features_df.columns:
            features_df['is_extrem_volatility'] = 0

    return features_df
def add_r2_regression(df, features_df,
                    period_range=14, period_extrem=14,
                    r2_low_threshold_range=0.3, r2_high_threshold_range=0.7,
                    r2_low_threshold_extrem=0.3, r2_high_threshold_extrem=0.7):
    """
    Ajoute les indicateurs de volatilité basés sur le R² de régression au DataFrame de features.
    Utilise des seuils spécifiques à chaque période.

    Paramètres:
    - df: DataFrame contenant les données de prix
    - features_df: DataFrame où ajouter les indicateurs
    - period_range: Période pour le calcul du R² de l'indicateur range_volatility
    - period_extrem: Période pour le calcul du R² de l'indicateur extrem_volatility
    - r2_low_threshold_range: Seuil bas pour la détection de volatilité modérée
    - r2_high_threshold_range: Seuil haut pour la détection de volatilité modérée
    - r2_low_threshold_extrem: Seuil bas pour la détection de volatilité extrême
    - r2_high_threshold_extrem: Seuil haut pour la détection de volatilité extrême

    Retourne:
    - features_df enrichi des indicateurs de volatilité basés sur R²
    """
    try:
        close = pd.to_numeric(df['close'], errors='coerce').values
        session_starts = (df['SessionStartEnd'] == 10).values

        # Calcul des R² pour l'indicateur range_volatility
        slopes_range, r2s_range, stds_range = calculate_slopes_and_r2_numba(close, session_starts, period_range)

        # Calcul des R² pour l'indicateur extrem_volatility (uniquement si période différente)
        if period_range == period_extrem:
            r2s_extrem = r2s_range
        else:
            _, r2s_extrem, _ = calculate_slopes_and_r2_numba(close, session_starts, period_extrem)

        # Ajouter les valeurs brutes au DataFrame de features
        features_df['r2_range'] = r2s_range
        if period_range != period_extrem:
            features_df['r2_extrem'] = r2s_extrem
        else:
            features_df['r2_extrem'] = r2s_range

        # Créer l'indicateur is_range_volatility (volatilité modérée optimisée pour maximiser le win rate)
        # is_range_volatility = 1 quand le R² est entre r2_low_threshold_range et r2_high_threshold_range
        features_df['is_range_volatility_r2'] = np.where(
            (r2s_range > r2_low_threshold_range) & (r2s_range < r2_high_threshold_range),
            1, 0
        )

        # Créer l'indicateur is_extrem_volatility (volatilité extrême optimisée pour minimiser le win rate)
        # is_extrem_volatility = 1 quand le R² est soit inférieur à r2_low_threshold_extrem
        # soit supérieur à r2_high_threshold_extrem
        features_df['is_extrem_volatility_r2'] = np.where(
            (r2s_extrem < r2_low_threshold_extrem) | (r2s_extrem > r2_high_threshold_extrem),
            1, 0
        )

        # S'assurer que toutes les colonnes sont numériques
        for col in ['r2_range', 'r2_extrem', 'is_range_volatility_r2', 'is_extrem_volatility_r2']:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"Erreur dans add_r2_regression: {str(e)}")
        # En cas d'erreur, tenter de renvoyer au moins les colonnes existantes
        if 'r2_range' not in features_df.columns:
            features_df['r2_range'] = 0
        if 'r2_extrem' not in features_df.columns:
            features_df['r2_extrem'] = 0
        if 'is_range_volatility_r2' not in features_df.columns:
            features_df['is_range_volatility_r2'] = 0
        if 'is_extrem_volatility_r2' not in features_df.columns:
            features_df['is_extrem_volatility_r2'] = 0

    return features_df


def add_williams_r(df, features_df,
                   period_overbought=14, period_oversold=14,
                   overbought_threshold=-20, oversold_threshold=-80):
    """
    Ajoute l'indicateur Williams %R sur des périodes potentiellement différentes pour
    surachat et survente, ainsi que des indicateurs dérivés.

    Paramètres:
    - df: DataFrame contenant les colonnes 'high', 'low', 'close'
    - features_df: DataFrame où ajouter les colonnes liées au Williams %R
    - period_overbought: Période de calcul pour le surachat (ex: 14)
    - period_oversold: Période de calcul pour la survente (ex: 14)
    - overbought_threshold: Seuil de surachat (défaut: -20)
    - oversold_threshold: Seuil de survente (défaut: -80)

    Retourne:
    - features_df enrichi des colonnes Williams %R et dérivées
    """
    try:
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        session_starts = (df['SessionStartEnd'] == 10).values

        # Calcul du Williams %R pour la période de surachat
        will_r_overbought = compute_wr(high, low, close, session_starts=session_starts,
                                       period=period_overbought, fill_value=-50)

        # Calcul du Williams %R pour la période de survente (uniquement si différente)
        if period_overbought == period_oversold:
            will_r_oversold = will_r_overbought
        else:
            will_r_oversold = compute_wr(high, low, close, session_starts=session_starts,
                                         period=period_oversold, fill_value=-50)

        # Ajouter les indicateurs Williams %R bruts
        features_df['williams_r_overbought'] = will_r_overbought
        if period_overbought != period_oversold:
            features_df['williams_r_oversold'] = will_r_oversold
        else:
            features_df['williams_r_standard'] = will_r_overbought

        # Ajouter les indicateurs de surachat/survente
        features_df['is_williams_r_overbought'] = np.where(will_r_overbought >= overbought_threshold, 1, 0)
        features_df['is_williams_r_oversold'] = np.where(will_r_oversold <= oversold_threshold, 1, 0)


        # Calculer les changements de zone (sortie de surachat/survente)
        will_r_overbought_series = pd.Series(will_r_overbought)
        will_r_oversold_series = pd.Series(will_r_oversold)

        # Sortie de la zone de surachat (signal baissier)
        exit_overbought = (will_r_overbought_series.shift(1) >= overbought_threshold) & \
                          (will_r_overbought_series < overbought_threshold)

        # Sortie de la zone de survente (signal haussier)
        exit_oversold = (will_r_oversold_series.shift(1) <= oversold_threshold) & \
                        (will_r_oversold_series > oversold_threshold)



        # S'assurer que toutes les colonnes sont numériques
        for col in features_df.columns:
            if col.startswith('williams_r') or col.startswith('is_williams_r'):
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"Erreur dans add_williams_r: {str(e)}")
        # En cas d'erreur, tenter de renvoyer au moins les colonnes essentielles
        columns_to_check = [
           'williams_r_oversold', 'will_r_overbought',
            'is_williams_r_overbought','is_williams_r_oversold',

        ]

        for col in columns_to_check:
            if col not in features_df.columns:
                features_df[col] = 0

    return features_df

def add_rsi(df, features_df, period=14):
    """
    Ajoute l'indicateur RSI (Relative Strength Index) sur la période spécifiée.

    Paramètres:
    - df: DataFrame contenant la colonne 'close'
    - features_df: DataFrame où ajouter la colonne 'rsi_{period}'
    - period: Période de calcul (ex: 14)

    Retourne:
    - features_df enrichi de la colonne RSI
    """
    try:
        close = pd.to_numeric(df['close'], errors='coerce')

        # Différence du cours de clôture
        delta = close.diff().fillna(0)

        # Gains (>=0) et pertes (<=0)
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        # Moyenne (simple ou EMA) des gains/pertes
        # Ici on utilise l'EMA pour un RSI plus classique
        avg_gains = gains.ewm(alpha=1/period, adjust=False).mean()
        avg_losses = losses.ewm(alpha=1/period, adjust=False).mean()

        # Éviter division par zéro
        rs = np.where(avg_losses == 0, 0, avg_gains / avg_losses)

        # RSI
        rsi = 100 - (100 / (1 + rs))
        features_df[f'rsi_'] = rsi

    except Exception as e:
        print(f"Erreur dans add_rsi: {str(e)}")

    return features_df


def add_mfi(df, features_df,
            overbought_period, oversold_period,
            overbought_threshold=80, oversold_threshold=20):
    """
    Ajoute l'indicateur MFI (Money Flow Index) avec des périodes obligatoires
    et distinctes pour les zones de surachat et survente.

    Paramètres:
    - df: DataFrame contenant 'high', 'low', 'close', 'volume'
    - features_df: DataFrame où ajouter les colonnes MFI
    - overbought_period: Période spécifique pour la détection de surachat (obligatoire)
    - oversold_period: Période spécifique pour la détection de survente (obligatoire)
    - overbought_threshold: Seuil de surachat (défaut: 80)
    - oversold_threshold: Seuil de survente (défaut: 20)

    Retourne:
    - features_df enrichi des colonnes MFI et dérivées

    Lève:
    - ValueError si overbought_period ou oversold_period est None
    """



    if overbought_period is None or oversold_period is None:
        raise ValueError("Les périodes de surachat et de survente doivent être spécifiées")

    try:
        session_starts = (df['SessionStartEnd'] == 10).values
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        volume = pd.to_numeric(df['volume'], errors='coerce')

        # Calcul des MFI avec périodes spécifiques pour surachat/survente
        is_mfi_overbought = compute_mfi(high, low, close,volume,session_starts, period=overbought_period, fill_value=50)
        is_mfi_oversold = compute_mfi(high, low, close,volume,session_starts,period=oversold_period, fill_value=50)

        # Indicateurs principaux avec périodes distinctes
        features_df['mfi_overbought_period'] = is_mfi_overbought
        features_df['mfi_oversold_period'] = is_mfi_oversold

        # Indicateurs de surachat/survente avec périodes spécifiques
        features_df['is_mfi_overbought'] = np.where(is_mfi_overbought > overbought_threshold, 1, 0)
        features_df['is_mfi_oversold'] = np.where(is_mfi_oversold < oversold_threshold, 1, 0)

        # Indicateur de changement de zone (basé sur les MFI spécifiques)
        mfi_overbought_series = pd.Series(is_mfi_overbought)
        mfi_oversold_series = pd.Series(is_mfi_oversold)

        # Sortie de la zone de surachat (signal baissier)
        exit_overbought = (mfi_overbought_series.shift(1) > overbought_threshold) & (
                    mfi_overbought_series <= overbought_threshold)

        # Sortie de la zone de survente (signal haussier)
        exit_oversold = (mfi_oversold_series.shift(1) < oversold_threshold) & (
                    mfi_oversold_series >= oversold_threshold)



        # Normalisation entre 0 et 1


        # S'assurer que toutes les colonnes sont numériques
        columns = ['mfi_overbought_period', 'mfi_oversold_period',
                   'is_mfi_overbought', 'is_mfi_oversold']


        for col in columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"Erreur dans add_mfi: {str(e)}")
        raise  # Relancer l'exception pour la traiter en amont

    return features_df


def add_mfi_divergence(df, features_df,
                       mfi_period_bearish=14, mfi_period_antiBear=14,
                       div_lookback_bearish=10, div_lookback_antiBear=10,
                       min_price_increase=0.005, min_mfi_decrease=0.005,
                       min_price_decrease=0.005, min_mfi_increase=0.005):
    """
    Ajoute les indicateurs de divergence MFI/prix pour les stratégies short.
    Utilise les mêmes conditions que la fonction objective pour détecter
    les signaux de divergence baissière et anti-divergence, avec la possibilité
    d'utiliser des périodes différentes pour chaque type de divergence.

    Paramètres:
    - df: DataFrame contenant 'high', 'low', 'close', 'volume'
    - features_df: DataFrame où ajouter les colonnes de divergence
    - mfi_period_bearish: Période MFI pour la divergence baissière (ex: 14)
    - mfi_period_antiBear: Période MFI pour l'anti-divergence (ex: 14)
    - div_lookback_bearish: Période lookback pour la divergence baissière (ex: 10)
    - div_lookback_antiBear: Période lookback pour l'anti-divergence (ex: 10)
    - min_price_increase: Seuil minimal d'augmentation de prix en % pour divergence baissière
    - min_mfi_decrease: Seuil minimal de diminution de MFI en % pour divergence baissière
    - min_price_decrease: Seuil minimal de diminution de prix en % pour anti-divergence
    - min_mfi_increase: Seuil minimal d'augmentation de MFI en % pour anti-divergence

    Retourne:
    - features_df enrichi des colonnes de divergence MFI
    """
    try:
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        volume = pd.to_numeric(df['volume'], errors='coerce')
        session_starts = (df['SessionStartEnd'] == 10).values

        # Calcul du MFI pour divergence baissière
        mfi_values_bearish = compute_mfi(high, low, close, volume, session_starts, period=mfi_period_bearish,
                                         fill_value=50)
        mfi_series_bearish = pd.Series(mfi_values_bearish)

        # Calcul du MFI pour anti-divergence (uniquement si période différente)
        if mfi_period_bearish == mfi_period_antiBear:
            mfi_values_antiBear = mfi_values_bearish
            mfi_series_antiBear = mfi_series_bearish
        else:
            mfi_values_antiBear = compute_mfi(high, low, close, volume, session_starts, period=mfi_period_antiBear,
                                              fill_value=50)
            mfi_series_antiBear = pd.Series(mfi_values_antiBear)

        # Ajouter les valeurs brutes du MFI au DataFrame
        if mfi_period_bearish == mfi_period_antiBear:
            features_df['mfi'] = mfi_values_bearish
        else:
            features_df['mfi_bearish'] = mfi_values_bearish
            features_df['mfi_antiBear'] = mfi_values_antiBear

        # --------- Divergence baissière (signal d'entrée short) ---------
        # Détection des divergences baissières
        price_pct_change_bearish = close.pct_change(div_lookback_bearish).fillna(0)
        mfi_pct_change_bearish = mfi_series_bearish.pct_change(div_lookback_bearish).fillna(0)

        # Conditions pour une divergence baissière efficace
        price_increase = price_pct_change_bearish > min_price_increase
        mfi_decrease = mfi_pct_change_bearish < -min_mfi_decrease

        # Prix fait un nouveau haut relatif
        price_rolling_max = pd.Series(close).rolling(window=div_lookback_bearish).max().shift(1)
        price_new_high = (close > price_rolling_max).fillna(False)

        # Définir la divergence baissière avec les mêmes critères que dans l'objective
        features_df['is_mfi_shortDiv'] = np.where(
            (price_new_high | price_increase) &  # Prix fait un nouveau haut ou augmente significativement
            (mfi_decrease),  # MFI diminue
            1, 0
        )

        # --------- Anti-divergence (signal d'évitement de short) ---------
        # Calculs spécifiques pour l'anti-divergence avec ses propres périodes
        price_pct_change_antiBear = close.pct_change(div_lookback_antiBear).fillna(0)
        mfi_pct_change_antiBear = mfi_series_antiBear.pct_change(div_lookback_antiBear).fillna(0)

        # Conditions pour une anti-divergence (mauvais win rate)
        price_decrease = price_pct_change_antiBear < -min_price_decrease  # Prix diminue
        mfi_increase = mfi_pct_change_antiBear > min_mfi_increase  # MFI augmente

        # Prix fait un nouveau bas relatif
        price_rolling_min = pd.Series(close).rolling(window=div_lookback_antiBear).min().shift(1)
        price_new_low = (close < price_rolling_min).fillna(False)

        # Définir l'anti-divergence avec les critères exacts de l'objective
        features_df['is_mfi_antiShortDiv'] = np.where(
            (price_new_low | price_decrease) &  # Prix fait un nouveau bas ou diminue significativement
            (mfi_increase),  # MFI augmente
            1, 0
        )

        # --------- Versions traditionnelles des divergences (pour référence) ---------
        # Utiliser les périodes bearish pour les divergences traditionnelles
        price_highs = pd.Series(close).rolling(window=div_lookback_bearish).max()
        price_lows = pd.Series(close).rolling(window=div_lookback_bearish).min()
        mfi_highs = mfi_series_bearish.rolling(window=div_lookback_bearish).max()
        mfi_lows = mfi_series_bearish.rolling(window=div_lookback_bearish).min()

        # Nouveaux sommets/creux (comparaison avec la période précédente)
        price_new_high_simple = close > price_highs.shift(1)
        price_new_low_simple = close < price_lows.shift(1)
        mfi_new_high = mfi_series_bearish > mfi_highs.shift(1)
        mfi_new_low = mfi_series_bearish < mfi_lows.shift(1)

           # S'assurer que toutes les colonnes MFI sont numériques
        mfi_columns = [col for col in features_df.columns if 'mfi' in col]
        for col in mfi_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)

    except Exception as e:
        print(f"Erreur dans add_mfi_divergence: {str(e)}")
        # En cas d'erreur, initialiser les colonnes principales à 0
        required_columns = [
            'mfi_bearish', 'mfi_antiBear', 'is_mfi_shortDiv', 'is_mfi_antiShortDiv',

        ]
        for col in required_columns:
            if col not in features_df.columns:
                features_df[col] = 0

    return features_df

def add_macd(df, features_df, short_period=12, long_period=26, signal_period=9):
    """
    Ajoute les indicateurs MACD (Moving Average Convergence Divergence)
    et sa ligne de signal.

    Paramètres:
    - df: DataFrame contenant la colonne 'close'
    - features_df: DataFrame où ajouter les colonnes:
        * macd
        * macd_signal
        * macd_hist
    - short_period: Période de l'EMA courte (par défaut 12)
    - long_period: Période de l'EMA longue (par défaut 26)
    - signal_period: Période de la ligne de signal (par défaut 9)

    Retourne:
    - features_df enrichi de 'macd', 'macd_signal', et 'macd_hist'
    """
    try:
        close = pd.to_numeric(df['close'], errors='coerce')

        ema_short = close.ewm(span=short_period, adjust=False).mean()
        ema_long = close.ewm(span=long_period, adjust=False).mean()
        macd = ema_short - ema_long
        macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
        macd_hist = macd - macd_signal

        features_df['macd'] = macd
        features_df['macd_signal'] = macd_signal
        features_df['macd_hist'] = macd_hist

    except Exception as e:
        print(f"Erreur dans add_macd: {str(e)}")

    return features_df






def add_adx(df, features_df, period=14):
    """
    Ajoute l'Average Directional Index (ADX).

    Paramètres:
    - df: DataFrame contenant 'high', 'low', 'close'
    - features_df: DataFrame où ajouter la colonne 'adx_{period}'
    - period: Période pour le calcul (ex: 14)

    Retourne:
    - features_df enrichi de la colonne ADX
    """
    try:
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')

        # Calcul du True Range
        shift_close = close.shift(1).fillna(close[0])
        tr = pd.DataFrame({
            'tr1': high - low,
            'tr2': (high - shift_close).abs(),
            'tr3': (low - shift_close).abs()
        }).max(axis=1)

        # +DM et -DM
        shift_high = high.shift(1).fillna(high[0])
        shift_low = low.shift(1).fillna(low[0])

        plus_dm = (high - shift_high).clip(lower=0)
        minus_dm = (shift_low - low).clip(lower=0)

        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm <= plus_dm] = 0

        # Moyenne exponentielle ou simple
        tr_ewm = tr.ewm(span=period, adjust=False).mean()
        plus_dm_ewm = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_ewm = minus_dm.ewm(span=period, adjust=False).mean()

        # +DI et -DI
        plus_di = 100 * (plus_dm_ewm / tr_ewm)
        minus_di = 100 * (minus_dm_ewm / tr_ewm)

        # DX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

        # ADX
        adx = dx.ewm(span=period, adjust=False).mean()

        features_df[f'adx_'] = adx
        features_df[f'plus_di_'] = plus_di
        features_df[f'minus_di_'] = minus_di

    except Exception as e:
        print(f"Erreur dans add_adx: {str(e)}")

    return features_df



features_df = add_rsi(df, features_df, period=5)
features_df = add_macd(df, features_df, short_period=4, long_period=8, signal_period=5)


features_df = add_regression_slope(df, features_df, period_range=28, period_extrem=30,
                     slope_range_threshold_low=0.2715994135835932 , slope_range_threshold_high=0.3842233665393566 ,
                     slope_extrem_threshold_low=-0.299867692475089, slope_extrem_threshold_high=0.5280173704178184 )

features_df = add_atr(df, features_df, atr_period_range=12, atr_period_extrem=25,
            atr_low_threshold_range= 2.2906256448366022, atr_high_threshold_range= 2.6612737528788495,
            atr_low_threshold_extrem=1.5360453527266502)

features_df = add_vwap(df, features_df,
         vwap_range_threshold_low=-2.6705237017186305, vwap_range_threshold_high=1.47028136092062,
         vwap_extrem_threshold_low=-29.9292, vwap_extrem_threshold_high=45.8839)



features_df = add_zscore(df, features_df,
              period_range=48, period_extrem=0,
              zscore_range_threshold_low=-0.3435, zscore_range_threshold_high=0.2173,
              zscore_extrem_threshold_low=-0, zscore_extrem_threshold_high=0)

# features_df = add_perctBB_simu(df, features_df,
#                  period_high=105, period_low=5,
#                  std_dev_high=1.7885122738322288, std_dev_low=1.2465031028992493,
#                  bb_high_threshold=1.0446251463502378, bb_low_threshold=0.5936224871229059)

features_df = add_std_regression(df, features_df,
                  period_range=13, period_extrem=46,
                  std_low_threshold_range=0.8468560606715232, std_high_threshold_range=0.9342162977682715,
                  std_low_threshold_extrem=1.3744010118158592 ,std_high_threshold_extrem=5.0650563837214735)

features_df = add_r2_regression(df, features_df,
                    period_range=21, period_extrem=53,
                    r2_low_threshold_range=0.10281944161014597, r2_high_threshold_range= 0.18548155526479806,
                    r2_low_threshold_extrem= 0.020222174899276364, r2_high_threshold_extrem=0.8665041310372354)

features_df = add_stochastic_force_indicators(df, features_df,
                                    k_period_overbought=42, d_period_overbought=41,
                                    k_period_oversold=105, d_period_oversold=169,
                                    overbought_threshold=93, oversold_threshold=21,
                                    fi_short=4, fi_long=4)

features_df = add_williams_r(df, features_df, period_overbought=42, period_oversold=106, overbought_threshold=-7, oversold_threshold=-79)

features_df = add_mfi(df, features_df, overbought_period=27,oversold_period=50,overbought_threshold=71, oversold_threshold=39)

features_df = add_mfi_divergence(df, features_df,
                       mfi_period_bearish=11, mfi_period_antiBear=14,
                       div_lookback_bearish=7, div_lookback_antiBear=18,
                       min_price_increase=0.0007855280106081092, min_mfi_decrease=0.00018,
                       min_price_decrease=0.000892667864022656, min_mfi_increase=0.00093)

column_settings = {
    # Time-based features
    'deltaTimestampOpening': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession1min': (
    "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession1index': (
    "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession5min': (
    "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession5index': (
    "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession15min': (
    "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession15index': (
    "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession30min': (
    "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaTimestampOpeningSession30index': (
    "winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaCustomSessionMin': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'deltaCustomSessionIndex': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

    # Stochastic indicators
    'stoch_k_overbought': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'stoch_k_oversold': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'stoch_d_overbought': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'stoch_d_oversold': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),


    # Force index indicators
    'force_index_4': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'force_index_4': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'force_index_4_norm': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'force_index_4_norm': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'force_index_divergence': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'fi_momentum': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'is_stoch_overbought': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'is_stoch_oversold': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),


    # Other technical indicators
    'rsi_': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'macd': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'macd_signal': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'macd_hist': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    #'adx_': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    #'plus_di_': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    #'minus_di_': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

    # Williams R indicators
    'is_williams_r_overbought': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'is_williams_r_oversold': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'williams_r_overbought': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'williams_r_oversold': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

    # MFI indicators
    #'mfi': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'is_mfi_overbought': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'is_mfi_oversold': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'mfi_overbought_period': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'mfi_oversold_period': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

    'mfi_bearish': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'mfi_antiBear': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'is_mfi_shortDiv': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'is_mfi_antiShortDiv': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),



    # Price and volume features
    'VolAbvState': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'VolBlwState': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'candleSizeTicks': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClosePoc_0_0': ("winsor", None, True, True, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClosePoc_0_1': ("winsor", None, True, True, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClosePoc_0_2': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClosePoc_0_3': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClosePoc_0_4': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClosePoc_0_5': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # High/Low price differentials
    'diffHighPrice_0_1': ("winsor", None, True, True, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),
    'diffHighPrice_0_2': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffHighPrice_0_3': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
    'diffHighPrice_0_4': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
    'diffHighPrice_0_5': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffLowPrice_0_1': ("winsor", None, False, False, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),
    'diffLowPrice_0_2': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffLowPrice_0_3': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffLowPrice_0_4': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
    'diffLowPrice_0_5': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),

    # POC price differentials
    'diffPocPrice_0_1': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'diffPocPrice_1_2': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'diffPocPrice_2_3': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),
    'diffPocPrice_0_2': ("winsor", None, False, False, 10, 90, toBeDisplayed_if_s(user_choice, False)),

    # VWAP related
    'diffPriceCloseVWAP': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, True)),
    'diffPriceCloseVWAPbyIndex': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, True)),

    # Technical indicators
    'atr': ("winsor", None, True, True, 0.1, 99, toBeDisplayed_if_s(user_choice, False)),
    'atr_range': ("winsor", None, False, False, 0.1, 99, toBeDisplayed_if_s(user_choice, False)),
    'atr_extrem': ("winsor", None, False, False, 0.1, 99, toBeDisplayed_if_s(user_choice, False)),
    'is_atr_range': ("winsor", None, False, False, 0.1, 99, toBeDisplayed_if_s(user_choice, False)),
    'is_atr_extremLow': ("winsor", None, False, False, 0.1, 99, toBeDisplayed_if_s(user_choice, False)),

    'bandWidthBB': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
    'perctBB': ("winsor", None, True, True, 12, 92, toBeDisplayed_if_s(user_choice, False)),

    # VA (Value Area) metrics
    'perct_VA6P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_delta_vol_VA6P': ("winsor", None, True, True, 4, 98, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClose_VA6PPoc': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClose_VA6PvaH': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClose_VA6PvaL': ("winsor", None, True, True, 12, 88, toBeDisplayed_if_s(user_choice, False)),
    'perct_VA11P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_delta_vol_VA11P': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClose_VA11PPoc': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClose_VA11PvaH': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClose_VA11PvaL': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'perct_VA16P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_delta_vol_VA16P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClose_VA16PPoc': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClose_VA16PvaH': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClose_VA16PvaL': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'perct_VA21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_delta_vol_VA21P': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClose_VA21PPoc': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClose_VA21PvaH': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceClose_VA21PvaL': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # VA overlap ratios
    'overlap_ratio_VA_6P_11P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'overlap_ratio_VA_6P_16P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'overlap_ratio_VA_6P_21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'overlap_ratio_VA_11P_21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # POC analysis
    'poc_diff_6P_11P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_ratio_6P_11P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_6P_16P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_ratio_6P_16P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_6P_21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_ratio_6P_21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_11P_21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'poc_diff_ratio_11P_21P': ("winsor", None, True, True, 0.1, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # Market regime metrics
    'market_regimeADX': ("winsor", None, True, False, 2, 99, toBeDisplayed_if_s(user_choice, True)),
    'market_regimeADX_state': ("winsor", None, False, False, 0.5, 99.8, toBeDisplayed_if_s(user_choice, True)),
    'is_in_range_10_32': ("winsor", None, False, False, 0.5, 99.8, toBeDisplayed_if_s(user_choice, True)),
    'is_in_range_5_23': ("winsor", None, False, False, 0.5, 99.8, toBeDisplayed_if_s(user_choice, True)),

    # Reversal and momentum features
    'bearish_reversal_force': ("winsor", None, False, True, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),
    'bullish_reversal_force': ("winsor", None, False, True, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),
    'meanVolx': ("winsor", None, False, True, 1, 99.7, toBeDisplayed_if_s(user_choice, False)),
    'diffVolCandle_0_1Ratio': ("winsor", None, False, True, 1, 98.5, toBeDisplayed_if_s(user_choice, False)),
    'diffVolDelta_0_1Ratio': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
    'diffVolDelta_0_0Ratio': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
    'diffVolDelta_1_1Ratio': ("winsor", None, True, True, 2.5, 97.5, toBeDisplayed_if_s(user_choice, False)),
    'diffVolDelta_2_2Ratio': ("winsor", None, True, True, 5, 95, toBeDisplayed_if_s(user_choice, False)),
    'diffVolDelta_3_3Ratio': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
    'cumDiffVolDeltaRatio': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),

    # Volume profile features
    'VolPocVolCandleRatio': ("winsor", None, True, True, 2, 65, toBeDisplayed_if_s(user_choice, False)),
    'VolPocVolRevesalXContRatio': ("winsor", None, True, True, 2, 95, toBeDisplayed_if_s(user_choice, False)),
    'pocDeltaPocVolRatio': ("winsor", None, True, True, 5, 95, toBeDisplayed_if_s(user_choice, False)),
    'VolAbv_vol_ratio': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
    'VolBlw_vol_ratio': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
    'asymetrie_volume': ("winsor", None, True, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),
    'VolCandleMeanxRatio': ("winsor", None, False, True, 1, 99, toBeDisplayed_if_s(user_choice, False)),

    # Imbalance features
    'bull_imbalance_low_1': ("winsor", None, False, False, 1, 99.5, toBeDisplayed_if_s(user_choice, False)),
    'bull_imbalance_low_2': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)), #Yeo-Johnson
    'bull_imbalance_low_3': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
    'bull_imbalance_high_0': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
    'bull_imbalance_high_1': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
    'bull_imbalance_high_2': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
    'bear_imbalance_low_0': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
    'bear_imbalance_low_1': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
    'bear_imbalance_low_2': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
    'bear_imbalance_high_1': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
    'bear_imbalance_high_2': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
    'bear_imbalance_high_3': ("winsor", None, False, False, 1, 96.5, toBeDisplayed_if_s(user_choice, False)),
    'imbalance_score_low': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),
    'imbalance_score_high': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),

    # Auction features
    'finished_auction_high': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),
    'finished_auction_low': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),
    'staked00_high': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),
    'staked00_low': ("winsor", None, False, False, 1, 99, toBeDisplayed_if_s(user_choice, False)),

    # POC distances
    'naked_poc_dist_above': ("winsor", None, True, True, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'naked_poc_dist_below': ("winsor", None, True, True, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # Linear slope metrics
    'linear_slope_10': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'linear_slope_r2_10': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'linear_slope_stds_10': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    'linear_slope_50': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'linear_slope_r2_50': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'linear_slope_stds_50': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    'linear_slope_prevSession': ("winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),

    # SMA ratio metrics
    'close_sma_ratio_6': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'close_sma_ratio_14': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'close_sma_ratio_21': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'close_sma_ratio_30': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # SMA z-score metrics
    'close_sma_zscore_6': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'close_sma_zscore_14': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'close_sma_zscore_21': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'close_sma_zscore_30': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # VA price differences
    'diffPriceCloseVAH_0': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'diffPriceCloseVAL_0': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_delta_vol_VA_0': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # Volume and movement ratios
    'ratio_volRevMove_volImpulsMove': (
    "winsor", None, False, True, 0.0, 80, toBeDisplayed_if_s(user_choice, False)),
    'ratio_deltaImpulsMove_volImpulsMove': (
    "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_deltaRevMove_volRevMove': ("winsor", None, True, True, 3, 80, toBeDisplayed_if_s(user_choice, False)),

    # Zone ratios
    'ratio_volZone1_volExtrem': ("winsor", None, False, True, 0.0, 98, toBeDisplayed_if_s(user_choice, False)),
    'ratio_deltaZone1_volZone1': ("winsor", None, True, True, 6, 96, toBeDisplayed_if_s(user_choice, False)),
    'ratio_deltaExtrem_volExtrem': ("winsor", None, True, True, 2, 97, toBeDisplayed_if_s(user_choice, False)),

    # Continuation zone ratios
    'ratio_VolRevZone_XticksContZone': (
    "winsor", None, False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),
    'ratioDeltaXticksContZone_VolXticksContZone': (
    "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # Movement strength ratios
    'ratio_impulsMoveStrengthVol_XRevZone': (
    "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_revMoveStrengthVol_XRevZone': (
    "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # Imbalance type
    'imbType_contZone': ("winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # Detailed zone ratios
    'ratio_volRevMoveZone1_volImpulsMoveExtrem_XRevZone': (
    "winsor", None, False, True, 0.0, 90, toBeDisplayed_if_s(user_choice, False)),
    'ratio_volRevMoveZone1_volRevMoveExtrem_XRevZone': (
    "winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),
    'ratio_deltaRevMoveZone1_volRevMoveZone1': (
    "winsor", None, True, True, 2, 95, toBeDisplayed_if_s(user_choice, False)),
    'ratio_deltaRevMoveExtrem_volRevMoveExtrem': (
    "winsor", None, True, True, 2, 98, toBeDisplayed_if_s(user_choice, False)),
    'ratio_volImpulsMoveExtrem_volImpulsMoveZone1_XRevZone': (
    "winsor", None, False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),
    'ratio_deltaImpulsMoveZone1_volImpulsMoveZone1': (
    "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_deltaImpulsMoveExtrem_volImpulsMoveExtrem_XRevZone': (
    "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # DOM and VA metrics
    'cumDOM_AskBid_avgRatio': ("winsor", None, False, True, 0.0, 98, toBeDisplayed_if_s(user_choice, False)),
    'cumDOM_AskBid_pullStack_avgDiff_ratio': (
    "winsor", None, True, True, 2, 99, toBeDisplayed_if_s(user_choice, False)),
    'delta_impulsMove_XRevZone_bigStand_extrem': (
    "winsor", None, False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),
    'delta_revMove_XRevZone_bigStand_extrem': (
    "winsor", None, False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),

    # Misc ratios
    'ratio_delta_VaVolVa': ("winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'borderVa_vs_close': ("winsor", None, False, True, 0.0, 99, toBeDisplayed_if_s(user_choice, False)),
    'ratio_volRevZone_VolCandle': (
    "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_deltaRevZone_VolCandle': (
    "winsor", None, False, False, 0.0, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # Sierra chart regression metrics
    'sc_reg_slope_5P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'sc_reg_std_5P_2': ("winsor", None, False, True, 0.5, 95, toBeDisplayed_if_s(user_choice, False)),
    'sc_reg_slope_10P_2': ("winsor", None, False, False, 0.5, 99.5, toBeDisplayed_if_s(user_choice, False)),
    'sc_reg_std_10P_2': ("winsor", None, False, True, 0.5, 95, toBeDisplayed_if_s(user_choice, False)),
    'sc_reg_slope_15P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'sc_reg_std_15P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'sc_reg_slope_30P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'sc_reg_std_30P_2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    'slope_range': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'slope_extrem': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'is_rangeSlope': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'is_extremSlope': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    'norm_diff_vwap': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'is_vwap_shortArea': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'is_vwap_notShortArea': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    'std_range': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'std_extrem': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'is_range_volatility_std': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'is_extrem_volatility_std': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    'zscore_range': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'zscore_extrem': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'is_zscore_range': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'is_zscore_extrem': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # 'percent_b_high': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'percent_b_low': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'is_bb_high': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    # 'is_bb_low': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    'r2_range': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'r2_extrem': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'is_range_volatility_r2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'is_extrem_volatility_r2': ("winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # Zone volume ratios
    'ratio_vol_VolCont_ZoneA_xTicksContZone': (
    "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_delta_VolCont_ZoneA_xTicksContZone': (
    "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_vol_VolCont_ZoneB_xTicksContZone': (
    "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_delta_VolCont_ZoneB_xTicksContZone': (
    "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_vol_VolCont_ZoneC_xTicksContZone': (
    "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),
    'ratio_delta_VolCont_ZoneC_xTicksContZone': (
    "winsor", None, False, False, 0.5, 99.9, toBeDisplayed_if_s(user_choice, False)),

    # Time-related
    'timeElapsed2LastBar': ("winsor", None, False, True, 0.0, 95, toBeDisplayed_if_s(user_choice, False)),

    # Include any absorption settings
    **absorption_settings
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

# Utilisation



# Appliquer la fonction à features_df
features_NANReplacedVal_df, nan_replacement_values = replace_nan_and_inf(features_df.copy(), columns_to_process,
                                                                        REPLACE_NAN)


# Initialisation des DataFrames avec le même index que le DataFrame d'entrée
outliersTransform_df = pd.DataFrame(index=features_NANReplacedVal_df.index)
winsorized_scaledWithNanValue_df = pd.DataFrame(index=features_NANReplacedVal_df.index)

total_features = len(columns_to_process)

for i, columnName in enumerate(columns_to_process):
    # Récupérer les paramètres de la colonne
    (
        transformation_method,
        transformation_params,
        floorInf_booleen,
        cropSup_booleen,
        floorInf_percent,
        cropSup_percent,
        _
    ) = column_settings[columnName]

    # Selon la méthode, on applique le traitement adéquat
    if transformation_method == "winsor":
        # On applique la winsorisation
        outliersTransform_df[columnName] = apply_winsorization(
            features_NANReplacedVal_df,
            columnName,
            floorInf_booleen,
            cropSup_booleen,
            floorInf_percent,
            cropSup_percent,
            nan_replacement_values
        )
    elif transformation_method == "log":
        #outliersTransform_df[columnName] = np.log(features_NANReplacedVal_df['columnName'])

        #outliersTransform_df[columnName] = transformer.fit_transform(features_NANReplacedVal_df[[columnName]])
        outliersTransform_df[columnName] = np.sqrt(features_NANReplacedVal_df[columnName])

    elif transformation_method == "Yeo-Johnson":
        # Exemple: placeholder d'une fonction Yeo-Johnson
        # ==============================================
        def apply_yeo_johnson(features_NANReplacedVal_df, columnName, transformation_params=None, standardize=False):
            """
            Applique la transformation Yeo-Johnson via scikit-learn sur une seule colonne.
            Si la colonne contient des NaN, la transformation est ignorée.
            """
            # Vérification des valeurs NaN
            from sklearn.preprocessing import PowerTransformer

            if features_NANReplacedVal_df[columnName].isna().any():
                print(f"⚠️ Warning: Column '{columnName}' contains NaN values. Transformation skipped.")
                exit(27)
                return features_NANReplacedVal_df[columnName]

            # Récupération des valeurs valides
            valid_values = features_NANReplacedVal_df[[columnName]].values  # Scikit-learn exige un 2D array

            if valid_values.size == 0:
                raise ValueError(
                    f"🚨 Error: No valid values found in '{columnName}', cannot apply Yeo-Johnson transformation.")

            # Application de la transformation Yeo-Johnson via PowerTransformer
            transformer = PowerTransformer(method='yeo-johnson', standardize=standardize)
            transformed_values = transformer.fit_transform(valid_values)

            # Mise à jour des valeurs transformées dans le DataFrame
            transformed_column = features_NANReplacedVal_df[columnName].copy()
            transformed_column[:] = transformed_values.flatten()

            return transformed_column

        outliersTransform_df[columnName] = apply_yeo_johnson(
            features_NANReplacedVal_df,
            columnName,
            transformation_params
        )
        # ==============================================

    else:
        print("error aucune transformation")

        exit(45)

features_NANReplacedVal_df = features_NANReplacedVal_df[columns_to_process]

print(f"arpès on a features_NANReplacedVal_df:{features_NANReplacedVal_df.shape}")
print(f"arpès on a outliersTransform_df:{outliersTransform_df.shape}")

print("\n")
print("Vérification finale :")
print(f"   - Nombre de colonnes dans outliersTransform_df : {len(outliersTransform_df.columns)}")

print(f"\n")

# print(f"   - Nombre de colonnes dans winsorized_scaledWithNanValue_df : {len(winsorized_scaledWithNanValue_df.columns)}")
# assert len(outliersTransform_df.columns) == len(winsorized_scaledWithNanValue_df.columns), "Le nombre de colonnes ne correspond pas entre les DataFrames"


print_notification(
    "Ajout de  'timeStampOpening', class_binaire', 'date', 'trade_category', 'SessionStartEnd' pour permettre la suite des traitements")
# Colonnes à ajouter
columns_to_add = ['timeStampOpening', 'class_binaire', 'candleDir', 'date', 'trade_category', 'SessionStartEnd',
                  'close', 'high', 'low','trade_pnl', 'tp1_pnl','tp2_pnl','tp3_pnl','sl_pnl','trade_pnl_theoric','tp1_pnl_theoric','sl_pnl_theoric']

# Vérifiez que toutes les colonnes existent dans df
missing_columns = [col for col in columns_to_add if col not in df.columns]
if missing_columns:
    error_message = f"Erreur: Les colonnes suivantes n'existent pas dans le DataFrame d'entrée: {', '.join(missing_columns)}"
    print(error_message)
    raise ValueError(error_message)

# Si nous arrivons ici, toutes les colonnes existent

# Créez un DataFrame avec les colonnes à ajouter

columns_df = df[columns_to_add]
# Ajoutez ces colonnes à features_df, outliersTransform_df en une seule opération
features_NANReplacedVal_df = pd.concat([features_NANReplacedVal_df, columns_df], axis=1)
outliersTransform_df = pd.concat([outliersTransform_df, columns_df], axis=1)


# winsorized_scaledWithNanValue_df = pd.concat([winsorized_scaledWithNanValue_df, columns_df], axis=1)

print_notification(
    "Colonnes 'timeStampOpening','class_binaire', 'candleDir', 'date', 'trade_category', 'SessionStartEnd' , 'close', "
    "'trade_pnl', 'tp1_pnl','tp2_pnl','tp3_pnl','sl_pnl','trade_pnl_theoric','tp1_pnl_theoric','sl_pnl_theoric' ajoutées")

file_without_extension = os.path.splitext(file_name)[0]
file_without_extension = file_without_extension.replace("Step4", "Step5")

# Créer le nouveau nom de fichier pour les features originales
new_file_name = file_without_extension + '_feat.csv'

# Construire le chemin complet du nouveau fichier
feat_file = os.path.join(file_dir, new_file_name)

# Créer le nouveau nom de fichier pour outliersTransform_df
winsorized_file_name = file_without_extension + '_feat_winsorized.csv'

# Construire le chemin complet du nouveau fichier winsorized
winsorized_file = os.path.join(file_dir, winsorized_file_name)

# Sauvegarder le fichier des features originales
#print_notification(f"Enregistrement du fichier de features non modifiées : {feat_file}")
#save_features_with_sessions(features_NANReplacedVal_df, CUSTOM_SESSIONS, feat_file)

# Sauvegarder le fichier winsorized
print_notification(f"Enregistrement du fichier de features winsorisées : {winsorized_file}")
save_features_with_sessions(outliersTransform_df, CUSTOM_SESSIONS, winsorized_file)
