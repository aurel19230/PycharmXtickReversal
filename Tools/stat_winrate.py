import pandas as pd
import numpy as np
import os
from standardFunc import load_data

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

FILE_NAME_ = "Step5_Step4_Step3_Step2_MergedAllFile_Step1_4_merged_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
DIRECTORY_PATH_ = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL\merge"
file_path = os.path.join(DIRECTORY_PATH_, FILE_NAME_)
initial_df = load_data(file_path)
print("Taille initiale du DataFrame:", len(initial_df))


def calculate_stats(df, title):
    total = len(df)
    wins = (df['class_binaire'] == 1).sum()
    losses = (df['class_binaire'] == 0).sum()
    excluded = total - wins - losses
    winrate = wins / (wins + losses) if (wins + losses) > 0 else 0

    print(f"\n{title}:")
    print(f"Total observations: {total}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Exclus (class_binaire == 99): {excluded}")
    print(f"Winrate: {winrate:.2%}")


# Statistiques initiales
calculate_stats(initial_df, "Statistiques initiales")

# Statistiques après filtrage de VWAP
df_vwap_filtered = initial_df[initial_df['diffPriceCloseVWAP'] >= 0].copy()
calculate_stats(df_vwap_filtered, "Statistiques après filtrage de VWAP (diffPriceCloseVWAP >= 0)")
print(f"Nombre de lignes exclues par le filtrage VWAP: {len(initial_df) - len(df_vwap_filtered)}")

# Statistiques après filtrage de perctBB
df_perctBB_filtered = initial_df[initial_df['perctBB'] > 0.75].copy()
calculate_stats(df_perctBB_filtered, "Statistiques après filtrage de perctBB (> 0.75)")
print(f"Nombre de lignes exclues par le filtrage perctBB: {len(initial_df) - len(df_perctBB_filtered)}")

# Statistiques après filtrage de finished_auction_high
df_auction_filtered = initial_df[initial_df['finished_auction_high'] == 1].copy()
calculate_stats(df_auction_filtered, "Statistiques après filtrage de finished_auction_high (== 1)")
print(f"Nombre de lignes exclues par le filtrage finished_auction_high: {len(initial_df) - len(df_auction_filtered)}")

# Statistiques sur bear_imbalance_high_1
initial_df['bear_imbalance_high_1_modified'] = np.where(initial_df['bear_imbalance_high_1'] > 3,
                                                        initial_df['bear_imbalance_high_1'], 0)

zero_count = (initial_df['bear_imbalance_high_1_modified'] == 0).sum()
above_three_count = (initial_df['bear_imbalance_high_1_modified'] > 3).sum()

print("\nStatistiques sur bear_imbalance_high_1 après modification:")
print(f"Nombre de valeurs égales à 0 : {zero_count}")
print(f"Nombre de valeurs supérieures à 3 : {above_three_count}")
print(f"Pourcentage de valeurs égales à 0 : {zero_count / len(initial_df):.2%}")
print(f"Pourcentage de valeurs supérieures à 3 : {above_three_count / len(initial_df):.2%}")

df_bear_filtered = initial_df[initial_df['bear_imbalance_high_1_modified'] > 0].copy()
calculate_stats(df_bear_filtered, "Statistiques après filtrage de bear_imbalance_high_1 (> 0)")
print(f"Nombre de lignes exclues par le filtrage bear_imbalance_high_1: {len(initial_df) - len(df_bear_filtered)}")

print("\nStatistiques finales sur bear_imbalance_high_1 dans df_bear_filtered:")
print(f"Valeur minimale : {df_bear_filtered['bear_imbalance_high_1_modified'].min()}")
print(f"Valeur maximale : {df_bear_filtered['bear_imbalance_high_1_modified'].max()}")

print("\nAperçu des premières lignes de bear_imbalance_high_1 dans df_bear_filtered:")
print(df_bear_filtered['bear_imbalance_high_1_modified'].head(10))

# Statistiques après application de tous les filtres
df_all_filtered = initial_df[
    #(initial_df['diffPriceCloseVWAP'] >= 0) &
    #(initial_df['perctBB'] > 0.75) &
    (initial_df['finished_auction_high'] == 1) &
    (initial_df['bear_imbalance_high_1_modified'] > 0)
    ].copy()
calculate_stats(df_all_filtered, "Statistiques après application de tous les filtres")
print(f"Nombre total de lignes exclues par tous les filtrages: {len(initial_df) - len(df_all_filtered)}")