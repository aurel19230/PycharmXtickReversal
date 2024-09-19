import standardFunc
from standardFunc import convert_to_unix_timestamp, convert_from_unix_timestamp, timestamp_to_date_utc, date_to_timestamp_utc
import os
import csv
from datetime import datetime
import numpy as np
from numba import njit, prange
from standardFunc import print_notification

# Demander à l'utilisateur s'il souhaite ajouter la colonne "date"
add_date_column = input("Voulez-vous ajouter une colonne 'date' ? (Appuyez sur Entrée pour non, ou tapez 'd' pour oui) : ")

# Répertoire contenant les fichiers d'entrée et de sortie
directory = "C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/simu/4_0_4TP_1SL/"
directoryMerge = "C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/simu/4_0_4TP_1SL/merge13092024"

# #on suppose les fichiers d'entrée on une base de nom fixe
#le suffixe permet de les distinguer entre eux.
#le nom du fichier de sortie est choisi également
suffixe="_191120_061222"
if add_date_column.lower() == 'd':
    withDate="_withDate"
else:
    withDate=""
candles_file = "of_raw_candles_dataNew"+suffixe+".csv"
trades_file = "of_trade_result_dataNew"+suffixe+".csv"
output_file_name = "4TicksRev"+suffixe+withDate+".csv"
# Délimiteur CSV
delimiter = ";"

# Lire les données de of_raw_candles_data
print_notification("Lecture des données de 'of_raw_candles_data'")
with open(os.path.join(directory, candles_file), 'r') as file:
    reader = csv.reader(file, delimiter=delimiter)
    headers = next(reader)  # Lire les en-têtes
    candles_data = []
    for row in reader:
        values = [float(val) for val in row]  # Conserver toutes les colonnes
        candles_data.append(values)
    candles_data = np.array(candles_data, dtype=np.float64)

# Lire les données de trade_result_data
print_notification("Lecture des données de 'of_trade_result_data'")
with open(os.path.join(directory, trades_file), 'r') as file:
    reader = csv.reader(file, delimiter=delimiter)
    headers_trade = next(reader)  # Lire les en-têtes
    trades_data = []
    for row in reader:
        index = int(float(row[0]))  # Convertir en float puis en int
        trade_result = int(row[1])
        trade_pnl = float(row[2])  # Ajouter la colonne trade_pnl
        trades_data.append([index, trade_result, trade_pnl])
    trades_data = np.array(trades_data, dtype=np.float64)  # Modifier le type de données en float64

# Extraire les dates du premier et du dernier timestamp de of_raw_candles_data
start_timestamp = int(candles_data[0, 0])
end_timestamp = int(candles_data[-1, 0])
start_date = datetime.utcfromtimestamp(start_timestamp).strftime('%d%m%y')
end_date = datetime.utcfromtimestamp(end_timestamp).strftime('%d%m%y')

# Nom du fichier de sortie avec les dates et le nom de fichier variable
output_file = f"Step1_{start_date}_{end_date}_{output_file_name}"

# Fonction optimisée avec Numba pour traiter les données sans la colonne "date"
@njit(parallel=True)
def process_data_numba(candles_data, trades_data):
    result = np.empty((len(candles_data), len(candles_data[0]) + 3), dtype=np.float64)  # Ajouter une colonne supplémentaire pour trade_pnl

    for i in prange(len(candles_data)):
        index = int(candles_data[i, 4])  # L'index est maintenant à la 5ème colonne
        match = trades_data[trades_data[:, 0] == index]
        if len(match) > 0:
            trade_result = int(match[0, 1])
            trade_pnl = float(match[0, 2])  # Récupérer la valeur de trade_pnl
            if trade_result == 2:
                trade_dir = -1
                trade_res = 1
            elif trade_result == 1:
                trade_dir = 1
                trade_res = 1
            elif trade_result == -1:
                trade_dir = 1
                trade_res = -1
            elif trade_result == -2:
                trade_dir = -1
                trade_res = -1
            else:
                trade_dir = 0
                trade_res = 0
        else:
            trade_dir = 0
            trade_res = 99
            trade_pnl = 0.0  # Définir trade_pnl à 0.0 s'il n'y a pas de correspondance

        row = np.concatenate((candles_data[i], np.array([trade_dir, trade_res, trade_pnl], dtype=np.float64)))  # Ajouter trade_pnl à la ligne
        result[i] = row
    return result

# Fonction pour traiter les données avec la colonne "date" (sans Numba)
def process_data_with_date(candles_data, trades_data):
    result = []

    for i in range(len(candles_data)):
        index = int(candles_data[i, 4])  # L'index est maintenant à la 5ème colonne
        match = trades_data[trades_data[:, 0] == index]
        if len(match) > 0:
            trade_result = int(match[0, 1])
            trade_pnl = float(match[0, 2])  # Récupérer la valeur de trade_pnl
            if trade_result == 2:
                trade_dir = -1
                trade_res = 1
            elif trade_result == 1:
                trade_dir = 1
                trade_res = 1
            elif trade_result == -1:
                trade_dir = 1
                trade_res = -1
            elif trade_result == -2:
                trade_dir = -1
                trade_res = -1
            else:
                trade_dir = 0
                trade_res = 0
        else:
            trade_dir = 0
            trade_res = 99
            trade_pnl = 0.0  # Définir trade_pnl à 0.0 s'il n'y a pas de correspondance

        timestamp = int(candles_data[i, 0])
        formatted_date = timestamp_to_date_utc(timestamp)
        row = [formatted_date] + candles_data[i].tolist() + [trade_dir, trade_res, trade_pnl]  # Ajouter trade_pnl à la ligne
        result.append(row)

    return result

# Traiter les données en fonction de l'option add_date_column
if add_date_column.lower() == 'd':
    print_notification("Traitement des données avec ajout de la colonne 'date'")
    merged_data = process_data_with_date(candles_data, trades_data)
else:
    print_notification("Traitement des données sans ajout de la colonne 'date'")
    merged_data = process_data_numba(candles_data, trades_data)

# Écrire les données fusionnées dans le fichier de sortie
print_notification(f"Écriture des données fusionnées dans le fichier : {output_file}")
with open(os.path.join(directoryMerge, output_file), 'w', newline='') as file:
    writer = csv.writer(file, delimiter=delimiter)

    # Écrire les en-têtes des colonnes
    if add_date_column.lower() == 'd':
        headers = ['dateUTC'] + headers
    headers.extend(['tradeDir', 'tradeResult', 'trade_pnl'])  # Ajouter 'trade_pnl' aux en-têtes
    writer.writerow(headers)

    # Écrire les données traitées
    if add_date_column.lower() == 'd':
        for row in merged_data:
            writer.writerow(row)
    else:
        for row in merged_data:
            writer.writerow(row.tolist())

print_notification(f"Fusion des fichiers terminée. Le fichier de sortie '{output_file}' a été créé.")
