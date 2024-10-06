import standardFunc
from standardFunc import convert_to_unix_timestamp, convert_from_unix_timestamp, timestamp_to_date_utc, date_to_timestamp_utc
import os
import csv
from datetime import datetime
import numpy as np
from numba import njit, prange
from standardFunc import print_notification

# Fonction optimisée avec Numba pour traiter les données sans la colonne "date"
@njit(parallel=True)
def apply_20_10_logic_numba(data):
    for i in prange(3, len(data)):
        if int(data[i, 3]) == 10:
            data[i - 1, 3] = 20
        if (data[i, 1] < 300 and data[i - 1, 1] > 600) and (int(data[i, 3]) != 10 and int(data[i - 1, 3]) != 20):
            data[i - 1, 3] = 20
            data[i, 3] = 10
    return data

def apply_20_10_logic_noNumba(data):
    for i in range(4, len(data)):
        if int(data[i][4]) == 10:
            data[i - 1][4] = 20
        if (data[i][2] < 300 and data[i - 1][2] > 600) and (int(data[i][4]) != 10 and int(data[i - 1][4]) != 20):
            print(f"Valeur à {data[i - 1][0]} : {data[i][0]}")
            user_input = input("Voulez-vous ajouter un 20 à data[i-1][4] et un 10 à data[i][4]? (y/n): ")
            if user_input.lower() == 'y':
                data[i - 1][4] = 20
                data[i][4] = 10
    return data

@njit(parallel=True)
def process_data_numba(candles_data, trades_data):
    result = np.empty((len(candles_data), len(candles_data[0]) + 3), dtype=np.float64)

    for i in prange(len(candles_data)):
        index = int(candles_data[i, 4])
        match = trades_data[trades_data[:, 0] == index]
        if len(match) > 0:
            trade_result = int(match[0, 1])
            trade_pnl = float(match[0, 2])
            if trade_result == 2:
                trade_dir, trade_res = -1, 1
            elif trade_result == 1:
                trade_dir, trade_res = 1, 1
            elif trade_result == -1:
                trade_dir, trade_res = 1, -1
            elif trade_result == -2:
                trade_dir, trade_res = -1, -1
            else:
                trade_dir, trade_res = 0, 0
        else:
            trade_dir, trade_res, trade_pnl = 0, 99, 0.0

        result[i] = np.concatenate((candles_data[i], np.array([trade_dir, trade_res, trade_pnl], dtype=np.float64)))

    result = apply_20_10_logic_numba(result)
    return result

def process_data_with_date(candles_data, trades_data):
    result = []

    for i in range(len(candles_data)):
        index = int(candles_data[i, 4])
        match = trades_data[trades_data[:, 0] == index]
        if len(match) > 0:
            trade_result = int(match[0, 1])
            trade_pnl = float(match[0, 2])
            if trade_result == 2:
                trade_dir, trade_res = -1, 1
            elif trade_result == 1:
                trade_dir, trade_res = 1, 1
            elif trade_result == -1:
                trade_dir, trade_res = 1, -1
            elif trade_result == -2:
                trade_dir, trade_res = -1, -1
            else:
                trade_dir, trade_res = 0, 0
        else:
            trade_dir, trade_res, trade_pnl = 0, 99, 0.0

        timestamp = int(candles_data[i, 0])
        formatted_date = timestamp_to_date_utc(timestamp)
        row = [formatted_date] + candles_data[i].tolist() + [trade_dir, trade_res, trade_pnl]
        result.append(row)

    result = apply_20_10_logic_noNumba(result)
    return result

def main():
    # Demander à l'utilisateur s'il souhaite ajouter la colonne "date"
    add_date_column = input("Voulez-vous ajouter une colonne 'date' ? (Appuyez sur Entrée pour non, ou tapez 'd' pour oui) : ")

    # Répertoire contenant les fichiers d'entrée et de sortie
    directory = "C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/simu/4_0_4TP_1SL_04102024/"
    directoryMerge = "C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/simu/4_0_4TP_1SL_04102024/merge"

    suffixe = "_0"
    withDate = "_withDate" if add_date_column.lower() == 'd' else ""
    candles_file = f"of_raw_candles_dataNew{suffixe}.csv"
    trades_file = f"of_trade_result_dataNew{suffixe}.csv"
    output_file_name = f"4TicksRev{suffixe}{withDate}.csv"
    delimiter = ";"

    # Lire les données
    print_notification(f"Lecture des données de 'of_raw_candles_data{suffixe}'.csv")
    with open(os.path.join(directory, candles_file), 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader)  # Lire les en-têtes
        candles_data = np.array([row for row in reader], dtype=float)

    print_notification("Lecture des données de 'of_trade_result_data'")
    trades_data = np.genfromtxt(os.path.join(directory, trades_file), delimiter=delimiter, skip_header=1)

    # Extraire les dates
    start_timestamp, end_timestamp = int(candles_data[0, 0]), int(candles_data[-1, 0])
    start_date = datetime.utcfromtimestamp(start_timestamp).strftime('%d%m%y')
    end_date = datetime.utcfromtimestamp(end_timestamp).strftime('%d%m%y')

    # Nom du fichier de sortie
    output_file = f"Step1_{start_date}_{end_date}_{output_file_name}"

    # Traiter les données
    if add_date_column.lower() == 'd':
        print_notification("Traitement des données avec ajout de la colonne 'date'")
        merged_data = process_data_with_date(candles_data, trades_data)
    else:
        print_notification("Traitement des données sans ajout de la colonne 'date'")
        merged_data = process_data_numba(candles_data, trades_data)

    # Écrire les données fusionnées
    print_notification(f"Écriture des données fusionnées dans le fichier : {output_file}")
    with open(os.path.join(directoryMerge, output_file), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=delimiter)
        if add_date_column.lower() == 'd':
            headers = ['dateUTC'] + headers
        headers.extend(['tradeDir', 'tradeResult', 'trade_pnl'])
        writer.writerow(headers)
        writer.writerows(merged_data)

    print_notification(f"Fusion des fichiers terminée. Le fichier de sortie '{output_file}' a été créé.")

if __name__ == "__main__":
    main()