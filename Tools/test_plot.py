import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


def prepare_trading_data(df):
    """
    Prépare les données pour l'analyse et la visualisation
    """
    # DEBUG: Afficher les timestamps min et max avant traitement
    print(f"Timestamp min: {df['deltaTimestampOpeningSection1min'].min()}")
    print(f"Timestamp max: {df['deltaTimestampOpeningSection1min'].max()}")

    df_copy = df.copy()
    df_copy = df_copy[df_copy['class_binaire'].isin([0, 1])]
    df_copy.loc[:, 'normalized_timestamp'] = df_copy['deltaTimestampOpeningSection1min']

    # DEBUG: Vérifier le nombre de bins créés
    bins = range(0, 1381, 10)  # Devrait créer 138 bins (0 à 1380 par pas de 10)
    print(f"Nombre de bins créés: {len(list(bins)) - 1}")

    df_copy.loc[:, 'time_bin'] = pd.cut(df_copy['normalized_timestamp'],
                                        bins=bins,
                                        labels=range(0, 1380, 10))

    success_rate = df_copy.groupby('time_bin', observed=True)['class_binaire'].agg([
        ('mean', lambda x: (x == 1).mean() * 100),
        ('count', 'count')
    ]).reset_index()

    # DEBUG: Vérifier les bins présents
    print(f"Bins uniques dans success_rate: {success_rate['time_bin'].nunique()}")
    print(f"Premier bin: {success_rate['time_bin'].min()}")
    print(f"Dernier bin: {success_rate['time_bin'].max()}")

    # Ajouter tous les bins manquants
    all_bins = pd.Series(range(0, 1380, 10))
    missing_bins = set(all_bins) - set(success_rate['time_bin'])
    print(f"Bins manquants: {missing_bins}")

    if missing_bins:
        missing_df = pd.DataFrame({
            'time_bin': list(missing_bins),
            'mean': 0,
            'count': 0
        })
        success_rate = pd.concat([success_rate, missing_df])

    success_rate = success_rate.sort_values('time_bin').reset_index(drop=True)

    # DEBUG: Vérifier la structure finale
    print(f"Nombre final de bins: {len(success_rate)}")

    # Réorganiser les données pour avoir 22h-00h à gauche
    late_night = success_rate[success_rate['time_bin'] < 240].copy()
    next_day = success_rate[success_rate['time_bin'] >= 240].copy()

    final_data = pd.concat([late_night, next_day])

    # DEBUG: Vérifier la structure finale après réorganisation
    print(f"Structure finale des données:")
    print(f"Total rows: {len(final_data)}")
    print(f"Première heure: {final_data['time_bin'].iloc[0]}")
    print(f"Dernière heure: {final_data['time_bin'].iloc[-1]}")

    return final_data


def plot_trading_success(df):
    success_rate = prepare_trading_data(df)

    # DEBUG: Vérifier la structure des données avant le plotting
    print(f"\nDonnées pour le plotting:")
    print(f"Nombre de barres à afficher: {len(success_rate)}")
    print(f"Dernier timestamp: {success_rate['time_bin'].max()}")

    plt.figure(figsize=(15, 8))

    # Tracer les barres
    bars = plt.bar(range(len(success_rate)), success_rate['mean'])

    # Annotations sur les barres
    for idx, row in enumerate(success_rate.itertuples()):
        if row.count > 0:
            plt.text(idx, row.mean, f'n={int(row.count)}',
                     ha='center', va='bottom', rotation=90, fontsize=8)

    # Configuration des ticks de l'axe X
    hours = [h % 24 for h in range(22, 22 + 24)]  # 22h à 21h le lendemain
    tick_positions = [((h - 22) % 24) * 6 for h in hours]  # 6 bins par heure
    tick_labels = [f"{h:02d}:00" for h in hours]

    plt.xticks(tick_positions, tick_labels, rotation=45)

    # DEBUG: Vérifier les limites de l'axe X
    print(f"Positions des ticks: {tick_positions}")
    print(f"Labels des ticks: {tick_labels}")

    # Ligne de minuit
    plt.axvline(x=12, color='red', linestyle='--', alpha=0.5)

    plt.title('Taux de réussite moyen des trades par tranche de 10 minutes\n(sur tous les jours de trading)')
    plt.xlabel('Heure de la journée')
    plt.ylabel('Taux de réussite moyen (%)')
    plt.grid(True, alpha=0.3)

    # Force les limites de l'axe X pour inclure toutes les données
    plt.xlim(-0.5, len(success_rate) - 0.5)

    # Annotations des périodes
    ymin, ymax = plt.ylim()
    plt.text(12, ymax * 1.02, '22h-00h', ha='center')
    plt.text(len(success_rate) / 2 + 12, ymax * 1.02, '00h-21h', ha='center')

    valid_trades = df[df['class_binaire'].isin([0, 1])]
    stats_text = (
        f"Nombre total de trades valides: {len(valid_trades)}\n"
        f"Taux de réussite global: {(valid_trades['class_binaire'] == 1).mean() * 100:.2f}%"
    )
    plt.figtext(0.02, 0.02, stats_text, fontsize=8, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
    DIRECTORY_PATH_ = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_6TP_1SL\merge"
    FILE_PATH_ = os.path.join(DIRECTORY_PATH_, FILE_NAME_)

    df = pd.read_csv(FILE_PATH_, sep=';', encoding='iso-8859-1')
    plt=plot_trading_success(df)
    plt.show()