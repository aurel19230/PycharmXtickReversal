from standard_stat_sc import *
from definition import *
from func_standard import *

import pandas as pd, numpy as np, os, sys, platform, io
from pathlib import Path
from contextlib import redirect_stdout

# ────────────────────────────────────────────────────────────────────────────────
# Paramètres
# ────────────────────────────────────────────────────────────────────────────────
FILE_NAME = "Step5__150924_030425_bugFixTradeResult1_extractOnlyFullSession_OnlyShort_feat.csv"
ENV = detect_environment()

if ENV == "pycharm":
    base_dir = (
        Path("C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject")
        if platform.system() != "Darwin"
        else Path("/Users/aurelienlachaud/Documents/trading_local")
    )
    DIRECTORY_PATH = base_dir / "5_0_5TP_1SL_1" / "merge"
else:
    DIRECTORY_PATH = Path.cwd()

FILE_PATH = DIRECTORY_PATH / FILE_NAME

# ────────────────────────────────────────────────────────────────────────────────
# Chargement et pré‑traitement
# ────────────────────────────────────────────────────────────────────────────────
df_init_features, CUSTOM_SESSIONS = load_features_and_sections(FILE_PATH)

cats = [
    "Trades échoués short", "Trades échoués long",
    "Trades réussis short", "Trades réussis long"
]
df_analysis = df_init_features[df_init_features["trade_category"].isin(cats)].copy()
df_analysis["class"] = np.where(df_analysis["trade_category"].str.contains("échoués"), 0, 1)
df_analysis["pos_type"] = np.where(df_analysis["trade_category"].str.contains("short"), "short", "long")

# ────────────────────────────────────────────────────────────────────────────────
# Filtres
# ────────────────────────────────────────────────────────────────────────────────
features_conditions_algo0 = {
    'cumDOM_AskBid_pullStack_avgDiff_ratio': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0.3, 'max': 6, 'active': True}],

    'diffVolDelta_2_2Ratio': [
        {'type': 'greater_than_or_equal', 'threshold': 0.25, 'active': True},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': -0.25, 'max': 0.25, 'active': False}],

    'sc_reg_std_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.6, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0.4, 'max': 1.7, 'active': True}],

    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],
}

features_conditions_algo1 = {
    'finished_auction_low': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}  # Correction de la plage
    ],

    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0.2, 'max': 0.8, 'active': True}],

    'diffVolDelta_2_2Ratio': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': True},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': -0.25, 'max': 0.25, 'active': False}],

    'close_sma_zscore_6': [
        {'type': 'greater_than_or_equal', 'threshold': 0.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0.4, 'max': 3, 'active': True}],

    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],
}

features_conditions_algo2 = {
    'finished_auction_low': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}  # Correction de la plage
    ],

    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': -0.45, 'max': 0.55, 'active': True}],

    'ratio_delta_vol_VA6P': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0.25, 'max': 200, 'active': True}],
    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],
}

features_conditions_algo3 = {
    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': -0.45, 'max': 0.65, 'active': True}],

    'ratio_delta_vol_VA11P': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0.25, 'max': 200, 'active': True}],
    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],
}
features_conditions_algo4 = {
    'finished_auction_low': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}  # Correction de la plage
    ],

    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': -0.45, 'max': 0.65, 'active': True}],

    'is_williams_r_overbought': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],

    'VolPocVolCandleRatio': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0, 'max': 0.2, 'active': True}],

    'is_rangeSlope': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}],
    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],
}
features_conditions_algo5 = {
    'finished_auction_low': [
        {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 1, 'max': 1, 'active': True}  # Correction de la plage
    ],

    'sc_reg_slope_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
        {'type': 'between', 'min': 0.25, 'max': 0.8, 'active': True}],

    'close_sma_zscore_6': [
        {'type': 'greater_than_or_equal', 'threshold': 0.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0.4, 'max': 3, 'active': True}],

    'sc_reg_std_30P_2': [
        {'type': 'greater_than_or_equal', 'threshold': 0.6, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0.4, 'max': 1.7, 'active': True}],

    'VolPocVolCandleRatio': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 0, 'max': 0.25, 'active': True}],

    'candleDuration': [
        {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
        {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
        {'type': 'between', 'min': 1, 'max': 180, 'active': True}],
}
features_conditions_algo6 = {
'sc_reg_slope_30P_2': [
                {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
                {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
                {'type': 'between', 'min': 0.32, 'max': 0.8, 'active': True}],

        'ratio_volRevMove_volImpulsMove': [
                {'type': 'greater_than_or_equal', 'threshold': 10, 'active': False},
                {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
                {'type': 'between', 'min': 0.6, 'max':  0.9, 'active': True}],

        'candleDuration': [
                {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
                {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
                {'type': 'between', 'min': 1, 'max': 180, 'active': True}],
}
features_conditions_algo7 = {
     'imbType_contZone': [
             {'type': 'greater_than_or_equal', 'threshold': -5, 'active': False},
             {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
             {'type': 'between', 'min': 1, 'max': 2, 'active': True}],

     'ratio_volRevMoveZone1_volRevMoveExtrem_XRevZone': [
             {'type': 'greater_than_or_equal', 'threshold': 2.5, 'active': False},
             {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
             {'type': 'between', 'min': 2.5, 'max': 10, 'active': True}],

    'ratio_deltaRevMove_volRevMove': [
            {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
            {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
            {'type': 'between', 'min': -0.8, 'max': -0.25, 'active': True}],

    'bear_imbalance_high_1': [
            {'type': 'greater_than_or_equal', 'threshold': 3, 'active': False},
            {'type': 'less_than_or_equal', 'threshold': 1, 'active': False},
            {'type': 'between', 'min': 1.2, 'max': 100, 'active': True}],
    'sc_reg_slope_30P_2': [
            {'type': 'greater_than_or_equal', 'threshold': 0.2, 'active': False},
            {'type': 'less_than_or_equal', 'threshold': 5, 'active': False},
            {'type': 'between', 'min': -0.3, 'max': 0.8, 'active': True}],


    'candleDuration': [
            {'type': 'greater_than_or_equal', 'threshold': 1.3, 'active': False},
            {'type': 'less_than_or_equal', 'threshold': 0.8, 'active': False},
            {'type': 'between', 'min': 1, 'max': 180, 'active': True}],
}
algorithms = {
    "features_conditions_algo0": features_conditions_algo0,
    "features_conditions_algo1": features_conditions_algo1,
    "features_conditions_algo2": features_conditions_algo2,
    "features_conditions_algo3": features_conditions_algo3,
    "features_conditions_algo4": features_conditions_algo4,
    "features_conditions_algo5": features_conditions_algo5,
    "features_conditions_algo6": features_conditions_algo6,
    "features_conditions_algo7": features_conditions_algo7,

}

# ────────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ────────────────────────────────────────────────────────────────────────────────
def save_csv(df: pd.DataFrame, path: Path, sep=";") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=sep, index=False)
    print(f"✓ Fichier enregistré: {path}")
    return path

def header_print(before: dict, after: dict, name: str):
    with io.StringIO() as buf, redirect_stdout(buf):
        print_comparative_performance(before, after)
        out = buf.getvalue()
    for h in ("STATISTIQUES GLOBALES", "PERFORMANCE GLOBALE",
              "ANALYSE DES TRADES LONGS", "ANALYSE DES TRADES SHORTS"):
        out = out.replace(h, f"{h} - {name}")
    print(out)

# ────────────────────────────────────────────────────────────────────────────────
# Boucle principale
# ────────────────────────────────────────────────────────────────────────────────
results, to_save = {}, []          # NOTE 1 : `to_save` stocke les DF à persister en fin de script
metrics_before = calculate_performance_metrics(df_analysis)

for algo_name, cond in algorithms.items():
    print(f"\n{'='*80}\nÉVALUATION DE {algo_name}\n{'='*80}")
    df_filt = apply_feature_conditions(df_analysis, cond)
    df_full = preprocess_sessions_with_date(
        create_full_dataframe_with_filtered_pnl(df_init_features, df_filt)
    )

    metrics_after = calculate_performance_metrics(df_filt)
    header_print(metrics_before, metrics_after, algo_name)

    wins_b, wins_a = (df_analysis["class"] == 1).sum(), (df_filt["class"] == 1).sum()
    fails_b, fails_a = (df_analysis["class"] == 0).sum(), (df_filt["class"] == 0).sum()
    win_rate_a = wins_a / (wins_a + fails_a) * 100
    pnl_a = df_filt["trade_pnl"].sum()

    profits_a = df_filt.loc[df_filt["trade_pnl"] > 0, "trade_pnl"].sum()
    losses_a = abs(df_filt.loc[df_filt["trade_pnl"] < 0, "trade_pnl"].sum())
    pf_a = profits_a / losses_a if losses_a else 0

    print(f"Win Rate après : {win_rate_a:.2f}%  |  Net PnL : {pnl_a:.2f}  |  Profit Factor : {pf_a:.2f}")

    results[algo_name] = {
        "Nombre de trades": (df_full["PnlAfterFiltering"] != 0).sum(),
        "Net PnL": df_full["PnlAfterFiltering"].sum(),
        "Win Rate (%)": win_rate_a,
        "Profit Factor": pf_a,
    }
    to_save.append((algo_name, df_full))   # NOTE 2 : file d’attente pour la sauvegarde

# ────────────────────────────────────────────────────────────────────────────────
# Tableau comparatif
# ────────────────────────────────────────────────────────────────────────────────
comparison_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Algorithme"})
print("\nTABLEAU COMPARATIF DES ALGORITHMES\n", comparison_df.to_string(index=False))

save_csv(comparison_df, DIRECTORY_PATH / "algo_comparison.csv")

# ────────────────────────────────────────────────────────────────────────────────
# Sauvegardes différées
# ────────────────────────────────────────────────────────────────────────────────
for algo_name, df_full in to_save:         # NOTE 3 : exécution après tous les affichages
    save_csv(df_full, DIRECTORY_PATH / f"df_with_sessions_{algo_name}.csv")
    save_csv(df_full[df_full["PnlAfterFiltering"] != 0],
             DIRECTORY_PATH / f"trades_{algo_name}.csv")

print("\n" + "="*80 + "\nANALYSE TERMINÉE AVEC SUCCÈS\n" + "="*80)  # NOTE 4 : fin propre
