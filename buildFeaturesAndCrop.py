import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from numba import jit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from standardFunc import plot_feature_histograms_by_class

# Définition des paramètres
folder_path = 'C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/data_simuNew/4_0_4TP_1SL/merge'
input_file_name="20190303_20240524_4TicksRev"

input_file = os.path.join(folder_path, f'{input_file_name}.csv')

output_file_before = os.path.join(folder_path, f'{input_file_name}Features.csv')
output_file_after = os.path.join(folder_path, f'{input_file_name}FeaturesCropped_wihSelection.csv')
output_file_afterPCA = os.path.join(folder_path, f'{input_file_name}FeaturesCropped_wihSelectionPCA.csv')

# Specify the output file path
output_file_before_label0_1 = os.path.join(folder_path, f'{input_file_name}FeaturesLabel0_1.csv')
output_file_before_dirShort = os.path.join(folder_path, f'{input_file_name}FeaturesDirShort.csv')


separator=';'
window_sizeMean=8
# Chargement du dataset
data = pd.read_csv(input_file, sep=separator)

# Remplacer les valeurs dans la colonne 'tradeResult'
data['tradeResult'] = data['tradeResult'].replace({2: 1, -1: 0, 0: 5})

# Get the count of each class
class_counts = data['tradeResult'].value_counts()

# Calculate the percentage of each class
class_percentages = data['tradeResult'].value_counts(normalize=True) * 100

# Display the number of classes and the percentage of each class
for cls in class_counts.index:
    print(f"Class {cls}: Count = {class_counts[cls]}, Percentage = {class_percentages[cls]:.2f}%")
class_1_percentage = class_percentages[1]
class_1_and_0_percentage = class_percentages[0] + class_percentages[1]
class_1_relative_percentage = (class_1_percentage / class_1_and_0_percentage) * 100
print(f"Percentage of Class 1 relative to Class 0 and Class 1 combined: {class_1_relative_percentage:.2f}%")


# Création de nouvelles colonnes à partir de la bougie de la ligne actuelle
data['diffPriceClosePoc_0_0'] = data['close'] - data['pocPrice']
data['diffPriceCloseHigh_0_0'] = data['close'] - data['high']
data['diffPriceCloseLow_0_0'] = data['close'] - data['low']

data['meanVolx'] = data['volume'].rolling(window=window_sizeMean, min_periods=1).mean()
data['ratioVolCandleMeanx'] = data['volume'] / data['meanVolx']
data['ratioVolPocVolCandle'] = data['volPOC'] / data['volume']
data['ratioPocDeltaPocVol'] = data['deltaPOC'] / data['volPOC']
data['ratioVolBlw'] = data['VolBlw'] / data['volume']
data['ratioVolAbv'] = data['VolAbv'] / data['volume']
data['ratioDeltaBlw'] = np.where(data['VolBlw'] != 0, data['DeltaBlw'] / data['VolBlw'], 0)
data['ratioDeltaAbv'] = np.where(data['VolAbv'] != 0, data['DeltaAbv'] / data['VolAbv'], 0)
data['diffPriceCloseVWAP'] = data['close'] - data['VWAP']
data['diffPriceCloseVWAPsd3Top'] = data['close'] - data['VWAPsd3Top']
data['diffPriceCloseVWAPsd3Bot'] = data['close'] - data['VWAPsd3Bot']
data['imbFactorAskL'] = data['imbFactorAskL']
data['imbFactorBidH'] = data['imbFactorBidH']
data['bidVolumeAtBarLow'] = data['bidVolumeAtBarLow']
data['askVolumeAtBarLow'] = data['askVolumeAtBarLow']
data['bidVolumeAtBarHigh'] = data['bidVolumeAtBarHigh']
data['askVolumeAtBarHigh'] = data['askVolumeAtBarHigh']

# Création de nouvelles colonnes à partir de la soustraction entre la bougie de la ligne actuelle et la bougie précédente
data['diffPocPrice_0_1'] = data['pocPrice'] - data['pocPrice'].shift(1)
data['diffHighPrice_0_1'] = data['high'] - data['high'].shift(1)
data['diffLowPrice_0_1'] = data['low'] - data['low'].shift(1)
data['diffVolCandle_0_1'] = (data['volume'] - data['volume'].shift(1)) / data['meanVolx']
data['diffVolDelta_0_1'] = (data['delta'] - data['delta'].shift(1)) / data['meanVolx']

# Afficher les lignes contenant des valeurs NaN
print("Lignes contenant des valeurs NaN :")
print(data[data.isnull().any(axis=1)])

# Afficher le nombre de valeurs manquantes (NaN)
print(f"\nNombre de valeurs manquantes (NaN) : {data.isnull().sum().sum()}")

# Arrondir les nouvelles colonnes à 2 chiffres après la virgule
data = data.round(2)

# Vérifier et remplacer les valeurs NaN dans chaque colonne
for column in data.columns:
    data[column] = data[column].bfill()

# Sauvegarder le dataset avant les modifications dans un fichier
#data[['timeStamp', 'deltaTimestamp', 'diffPriceClosePoc_0_0', 'diffPriceCloseHigh_0_0', 'diffPriceCloseLow_0_0',
    #     'ratioVolCandleMeanx', 'ratioVolPocVolCandle', 'ratioPocDeltaPocVol', 'ratioVolBlw', 'ratioVolAbv',
    # 'ratioDeltaBlw', 'ratioDeltaAbv', 'diffPriceCloseVWAP', 'diffPriceCloseVWAPsd3Top', 'diffPriceCloseVWAPsd3Bot',
    #  'imbFactorAskL', 'imbFactorBidH', 'bidVolumeAtBarLow', 'askVolumeAtBarLow', 'bidVolumeAtBarHigh',
    #  'askVolumeAtBarHigh', 'diffPocPrice_0_1', 'diffHighPrice_0_1', 'diffLowPrice_0_1', 'diffVolCandle_0_1',
#  'diffVolDelta_0_1', 'tradeDir', 'tradeResult']].to_csv(output_file_before, sep=';', index=False)


# Filter the DataFrame to keep only rows where 'tradeDir' is -1 for keeping only the short trade
dataDirShort= data[data['tradeDir'].isin([-1])]

# Save the filtered DataFrame to a CSV file
dataDirShort[['timeStamp', 'deltaTimestamp', 'diffPriceClosePoc_0_0', 'diffPriceCloseHigh_0_0', 'diffPriceCloseLow_0_0',
              'ratioVolCandleMeanx', 'ratioVolPocVolCandle', 'ratioPocDeltaPocVol', 'ratioVolBlw', 'ratioVolAbv',
              'ratioDeltaBlw', 'ratioDeltaAbv', 'diffPriceCloseVWAP', 'diffPriceCloseVWAPsd3Top', 'diffPriceCloseVWAPsd3Bot',
              'imbFactorAskL', 'imbFactorBidH', 'bidVolumeAtBarLow', 'askVolumeAtBarLow', 'bidVolumeAtBarHigh',
              'askVolumeAtBarHigh', 'diffPocPrice_0_1', 'diffHighPrice_0_1', 'diffLowPrice_0_1', 'diffVolCandle_0_1',
              'diffVolDelta_0_1', 'tradeDir', 'tradeResult']].to_csv(output_file_before_dirShort, sep=';', index=False)

# Filter the DataFrame to keep only rows where 'tradeResult' is 0 or 1
dataLabel0_1 = data[data['tradeResult'].isin([0, 1])]

# Sélectionner les colonnes pour l'analyse PCA
columns_for_pca = ['diffPriceClosePoc_0_0', 'diffPriceCloseHigh_0_0', 'diffPriceCloseLow_0_0', 'ratioVolCandleMeanx',
                   'ratioVolPocVolCandle', 'ratioPocDeltaPocVol', 'ratioVolBlw', 'ratioVolAbv', 'ratioDeltaBlw',
                   'ratioDeltaAbv', 'diffPriceCloseVWAP', 'diffPriceCloseVWAPsd3Top', 'diffPriceCloseVWAPsd3Bot',
                   'imbFactorAskL', 'imbFactorBidH', 'bidVolumeAtBarLow', 'askVolumeAtBarLow', 'bidVolumeAtBarHigh',
                   'askVolumeAtBarHigh', 'diffPocPrice_0_1', 'diffHighPrice_0_1', 'diffLowPrice_0_1', 'diffVolCandle_0_1',
                   'diffVolDelta_0_1']
X = data[columns_for_pca]


# Dictionnaire pour paramétrer le crop et le floor pour chaque colonne
column_settings = {
    'deltaTimestamp':    (False, False, 10, 90),
    'diffPriceClosePoc_0_0':    (False, False, 10, 90),
    'diffPriceCloseHigh_0_0':   (True, True, 10, 90),
    'diffPriceCloseLow_0_0':    (True, True, 10, 90),
    'ratioVolCandleMeanx':      (True, True, 10, 90),
    'ratioVolPocVolCandle':     (True, True, 10, 90),
    'ratioPocDeltaPocVol':      (True, True, 10, 90),
    'ratioVolBlw':              (False, False, 0, 99),
    'ratioVolAbv':              (False, False, 0, 99),
    'ratioDeltaBlw':            (True, True, 10, 90),
    'ratioDeltaAbv':            (True, True, 10, 90),
    'diffPriceCloseVWAP':       (True, True, 10, 90),
    'diffPriceCloseVWAPsd3Top': (True, True, 10, 90),
    'diffPriceCloseVWAPsd3Bot': (True, True, 10, 90),
    'imbFactorAskL':            (True, True, 10, 90),
    'imbFactorBidH':            (True, True, 10, 90),
    'bidVolumeAtBarLow':        (False, True, 10, 90),
    'askVolumeAtBarLow':        (False, True, 0, 90),
    'bidVolumeAtBarHigh':       (False, True, 0, 90),
    'askVolumeAtBarHigh':       (False, True, 0, 90),
    'diffPocPrice_0_1':         (True, True, 10, 90),
       'diffHighPrice_0_1':        (True, True, 10, 90),
       'diffLowPrice_0_1':         (True, True, 10, 90),
      'diffVolCandle_0_1':        (True, True, 10, 90),
      'diffVolDelta_0_1':         (True, True, 10, 90),
}


@jit(nopython=True)
def calculate_percentiles(column_values, floorInf_percentage, cropSup_percentage):
    sorted_values = np.sort(column_values[~np.isnan(column_values)])
    floor_value = np.percentile(sorted_values, floorInf_percentage)
    crop_value = np.percentile(sorted_values, cropSup_percentage)
    return floor_value, crop_value


def processFloorCrop(data, column_settings):
    data_processed = data.copy()
    for column, (floorInf_values, cropSup_values, floorInf_percentage, cropSup_percentage) in column_settings.items():
        column_values = data[column].values
        floorInf_value, cropSup_value = calculate_percentiles(column_values, floorInf_percentage, cropSup_percentage)

        if floorInf_values:
            data_processed.loc[data_processed[column] < floorInf_value, column] = floorInf_value
        if cropSup_values:
            data_processed.loc[data_processed[column] > cropSup_value, column] = cropSup_value

    return data_processed


def plot_histograms(data_before, data_after, column_settings, figsize=(32, 24)):
    columns = list(column_settings.keys())
    n_columns = len(columns)
    ncols = 12
    nrows = (n_columns + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows * 2, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, column in enumerate(columns):
        column_values_before = data_before[column].values
        floorInf_values, cropSup_values, floorInf_percentage, cropSup_percentage = column_settings[column]
        floor_value, crop_value = calculate_percentiles(column_values_before, floorInf_percentage, cropSup_percentage)

        # Graphique avant modifications
        sns.histplot(data=pd.DataFrame({column: column_values_before}), x=column, ax=axes[i * 2], kde=True)

        axes[i * 2].axvline(crop_value, color='r', linestyle='--',
                            label=f'{cropSup_percentage}% correspond à {crop_value:.2f}')
        axes[i * 2].axvline(floor_value, color='g', linestyle='--',
                            label=f'{floorInf_percentage}% correspond à {floor_value:.2f}')

        axes[i * 2].set_xlabel('', fontsize=7)  # Supprime la légende de l'axe x
        axes[i * 2].set_title(f'{column} - Avant', fontsize=7)
        axes[i * 2].legend(loc='upper right', ncol=1, fontsize=7)
        axes[i * 2].set_ylabel('', fontsize=7)

        # Graphique après modifications
        column_values_after = data_after[column].values

        sns.histplot(data=pd.DataFrame({column: column_values_after}), x=column, ax=axes[i * 2 + 1], kde=True)

        axes[i * 2 + 1].set_xlabel('', fontsize=7)  # Supprime la légende de l'axe x
        title = f'{column} - Après'
        if floorInf_values and cropSup_values:
            title += ' (Floor / Crop)'
        elif floorInf_values:
            title += ' (Floor)'
        elif cropSup_values:
            title += ' (Crop)'
        axes[i * 2 + 1].set_title(title, fontsize=7)
        axes[i * 2 + 1].set_ylabel('', fontsize=7)

    plt.tight_layout()
    plt.show()


# Appel de la fonction pour calculer les données après les modifications
data_processed = processFloorCrop(data, column_settings)
data_processedLabel0_1= processFloorCrop(dataLabel0_1, column_settings)







# Appel de la fonction pour afficher les histogrammes avant et après les modifications
print("Graphiques avant et après les modifications :")
#plot_histograms(data, data_processed, column_settings, figsize=(32, 24))
#plot_histograms(dataLabel0_1, data_processedLabel0_1, column_settings, figsize=(32, 24))



def plot_feature_kde_by_class(data, column_settings, figsize=(32, 24)):
    columns = list(column_settings.keys())
    n_columns = len(columns)
    ncols = 6
    nrows = (n_columns + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, column in enumerate(columns):
        # Tracer le KDE pour la classe 0 en rouge
        sns.kdeplot(data=data[data['tradeResult'] == 0], x=column, color='red', label='Classe 0', ax=axes[i])

        # Tracer le KDE pour la classe 1 en bleu
        sns.kdeplot(data=data[data['tradeResult'] == 1], x=column, color='blue', label='Classe 1', ax=axes[i])

        # Ajouter un titre et des étiquettes d'axe
      #  axes[i].set_title(f'{column}')
       # axes[i].set_xlabel('Valeur de la feature')
       # axes[i].set_ylabel('Densité de probabilité')

        # Ajouter une légende
        axes[i].legend()

    plt.tight_layout()
    plt.show()

# Appel de la fonction pour afficher les KDE par classe
print("KDE des features par classe :")
#plot_feature_kde_by_class(dataLabel0_1, column_settings, figsize=(32, 24))
#plot_feature_kde_by_class(dataDirShort, column_settings, figsize=(32, 20))



import matplotlib.pyplot as plt
import seaborn as sns




################################ PCA ####################################
# Sélectionner les colonnes pour l'analyse PCA
columns_for_pca = ['diffPriceClosePoc_0_0', 'diffPriceCloseHigh_0_0', 'diffPriceCloseLow_0_0', 'ratioVolCandleMeanx',
                   'ratioVolPocVolCandle', 'ratioPocDeltaPocVol', 'ratioVolBlw', 'ratioVolAbv', 'ratioDeltaBlw',
                   'ratioDeltaAbv', 'diffPriceCloseVWAP', 'diffPriceCloseVWAPsd3Top', 'diffPriceCloseVWAPsd3Bot',
                   'imbFactorAskL', 'imbFactorBidH', 'bidVolumeAtBarLow', 'askVolumeAtBarLow', 'bidVolumeAtBarHigh',
                   'askVolumeAtBarHigh', 'diffPocPrice_0_1', 'diffHighPrice_0_1', 'diffLowPrice_0_1', 'diffVolCandle_0_1',
                   'diffVolDelta_0_1']
X = data_processed[columns_for_pca]
#X = data[columns_for_pca]

# Standardisation des données
scaler = StandardScaler()
#scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(dataDirShort)

# Appliquer PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
cumulative_variance_threshold = 0.8  # Seuil de variance cumulée pour déterminer le nombre de composantes principales à conserver

# Déterminer le nombre de composantes principales à conserver
cumulative_variance_ratio = pca.explained_variance_ratio_.cumsum()
n_components = sum(cumulative_variance_ratio < cumulative_variance_threshold) + 1
print(f"Nombre de composantes principales à conserver pour capturer {cumulative_variance_threshold * 100}% de la variance : {n_components}")

# Créer un DataFrame avec les données PCA et les colonnes supplémentaires
pca_columns = [f'PCA_{i+1}' for i in range(n_components)]
pca_data = pd.DataFrame(X_pca[:, :n_components], columns=pca_columns)
pca_data = pd.concat([dataDirShort[['timeStamp', 'deltaTimestamp']], pca_data, dataDirShort[['tradeDir', 'tradeResult']]], axis=1)

# Sauvegarder les données PCA dans un fichier
output_file_afterPCA = os.path.join(folder_path, f'{input_file_name}FeaturesPCA{n_components}.csv')
pca_data.to_csv(output_file_afterPCA, sep=';', index=False)

################################ END PCA ####################################


intervals = [(10, 860)
            ,
            # (880, 920), (935, 1195), (1210, 1300)
             ]

def in_intervals(x):
    for start, end in intervals:
        if start <= x <= end:
            return True
    return False

# Appliquer la fonction in_intervals à la colonne deltaTimestamp
mask = dataDirShort['deltaTimestamp'].apply(in_intervals)

# Filtrer le DataFrame en utilisant le masque
dataDirShort_filtered = dataDirShort[mask]


print(dataDirShort_filtered)

print(dataDirShort['tradeResult'])

# Appel de la fonction pour afficher les histogrammes par classe
#print("Histogrammes des features par classe :")
dataDirShort_FloorCrop= processFloorCrop(dataDirShort, column_settings)

#plot_feature_histograms_by_class(pca_data, pca_column_settings, figsize=(32, 24))


#pca_column_settings = {column: (False, False, 10, 90) for column in pca_columns}
# Get the count of each class
class_counts = dataDirShort['tradeResult'].value_counts()

# Calculate the percentage of each class
class_percentages = dataDirShort['tradeResult'].value_counts(normalize=True) * 100

# Display the number of classes and the percentage of each class
for cls in class_counts.index:
    print(f"dataDirShort -> Class {cls}: Count = {class_counts[cls]}, Percentage = {class_percentages[cls]:.2f}%")
class_1_percentage = class_percentages[1]
class_1_and_0_percentage = class_percentages[0] + class_percentages[1]
class_1_relative_percentage = (class_1_percentage / class_1_and_0_percentage) * 100
print(f"dataDirShort -> Percentage of Class 1 relative to Class 0 and Class 1 combined: {class_1_relative_percentage:.2f}%")

# Get the count of each class
class_counts = dataDirShort_filtered['tradeResult'].value_counts()
# Calculate the percentage of each class
class_percentages = dataDirShort_filtered['tradeResult'].value_counts(normalize=True) * 100

# Display the number of classes and the percentage of each class
for cls in class_counts.index:
    print(f"dataDirShort_filtered -> Class {cls}: Count = {class_counts[cls]}, Percentage = {class_percentages[cls]:.2f}%")
class_1_percentage = class_percentages[1]
class_1_and_0_percentage = class_percentages[0] + class_percentages[1]
class_1_relative_percentage = (class_1_percentage / class_1_and_0_percentage) * 100
print(f"dataDirShort_filtered -> Percentage of Class 1 relative to Class 0 and Class 1 combined: {class_1_relative_percentage:.2f}%")



plot_feature_histograms_by_class(dataDirShort_filtered, 'tradeResult',column_settings, figsize=(32, 24))
#
# loadings = pca.components_
# loadings_df = pd.DataFrame(loadings, columns=columns_for_pca, index=[f'PC{i+1}' for i in range(loadings.shape[0])])
#
# print(loadings_df.loc['PC1'])
# plt.figure(figsize=(10, 6))
# sns.barplot(x=loadings_df.loc['PC1'], y=loadings_df.columns)
# plt.title('PCA 7 Loadings')
# plt.show()

