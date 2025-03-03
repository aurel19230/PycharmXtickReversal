import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def plot_distributions_short_long_grid(df, features, class_col='class'):
    """
    Crée une unique figure avec jusqu'à 24 features.
    - 4 features par ligne => 4 * 2 = 8 colonnes
    - nrows = nombre de lignes nécessaire pour afficher toutes les features
    - Pour chaque feature, on a 2 subplots contigus :
         - (Short)  => en bleu/orange (class=0 / class=1)
         - (Long)   => en bleu/orange (class=0 / class=1)
    """

    # 1) Limite à 24 features (optionnel si déjà fait plus haut)
    max_features = 24
    features = features[:max_features]
    n_features = len(features)

    # 2) On veut 4 features par ligne, chaque feature occupe 2 colonnes
    #    => ncols = 8
    #    => nrows = ceil(n_features / 4)
    ncols = 8
    nrows = int(np.ceil(n_features / 4))

    # 3) Préparer la figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(8 * 2, 4 * nrows),  # 8 colonnes * ~2 en largeur, 4 en hauteur par ligne
                             squeeze=False)

    # 4) Boucle sur chaque feature
    for i, feature in enumerate(features):
        # -- Calcul de la ligne et des 2 colonnes (short, long) --
        row = i // 4
        col_short = (i % 4) * 2
        col_long = col_short + 1

        # Subplots correspondants
        ax_short = axes[row, col_short]
        ax_long  = axes[row, col_long]

        # Filtrage pour short / long
        df_short = df[df['pos_type'] == 'short']
        df_long  = df[df['pos_type'] == 'long']

        # === Distribution SHORT ===
        sns.histplot(
            df_short[df_short[class_col] == 0][feature],
            color='blue',
            kde=True,
            label='Class 0',
            ax=ax_short
        )
        sns.histplot(
            df_short[df_short[class_col] == 1][feature],
            color='orange',
            kde=True,
            label='Class 1',
            ax=ax_short
        )
        ax_short.set_title(f"{feature} - SHORT")
        ax_short.set_xlabel('')
        ax_short.set_ylabel('Fréquence')
        ax_short.legend()

        # === Distribution LONG ===
        sns.histplot(
            df_long[df_long[class_col] == 0][feature],
            color='blue',
            kde=True,
            label='Class 0',
            ax=ax_long
        )
        sns.histplot(
            df_long[df_long[class_col] == 1][feature],
            color='orange',
            kde=True,
            label='Class 1',
            ax=ax_long
        )
        ax_long.set_title(f"{feature} - LONG")
        ax_long.set_xlabel('')
        ax_long.set_ylabel('Fréquence')
        ax_long.legend()

    # 5) Masquer les sous-graphiques inutilisés si < 24 features
    used_subplots = n_features * 2  # chaque feature utilise 2 subplots
    total_subplots = nrows * ncols
    for j in range(used_subplots, total_subplots):
        row_empty = j // ncols
        col_empty = j % ncols
        axes[row_empty, col_empty].axis('off')

    plt.tight_layout()
    plt.show()

# ======================
# 4. Définition des fonctions de traçage
# ======================
def plot_boxplots(df, features, category_col='trade_category', nrows=3, ncols=4):
    """
    Trace des sns.boxplot pour une liste de 'features' avec un ordre personnalisé.

    :param df: DataFrame contenant les données filtrées.
    :param features: Liste de features (colonnes) à tracer.
    :param category_col: Nom de la colonne catégorielle pour l'axe X (ex: 'trade_category').
    :param nrows: Nombre de lignes de subplots.
    :param ncols: Nombre de colonnes de subplots.
    """
    # Définir l'ordre des catégories
    custom_order = ["Trades réussis long", "Trades échoués long", "Trades réussis short", "Trades échoués short"]

    # Préparation de la figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

    # Boucle sur chaque feature
    for idx, feature in enumerate(features):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        sns.boxplot(
            data=df,
            x=category_col,
            y=feature,
            hue=category_col,
            order=custom_order,  # Appliquer l'ordre personnalisé
            ax=ax,
            showmeans=True,
            palette="Set2",
            dodge=False
        )

        # Supprime la légende si elle existe
        if ax.legend_ is not None:
            ax.legend_.remove()

        # Personnalisation
        ax.set_title(feature)
        ax.set_xlabel('')
        ax.set_ylabel('Valeur')
        ax.tick_params(axis='x', rotation=30)  # Inclinaison des labels en X

    # Masquer les axes vides si le nombre de features est inférieur à nrows*ncols
    total_plots = len(features)
    for idx_empty in range(total_plots, nrows * ncols):
        row = idx_empty // ncols
        col = idx_empty % ncols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()



# Fonction pour appliquer les conditions de filtrage
def apply_feature_conditions(df, features_conditions):
    mask = np.ones(len(df), dtype=bool)  # Initialisation du masque global à True

    for feature, conditions in features_conditions.items():
        # Extraire le nom de base de la feature (pour gérer les _range1, _range2, etc.)
        base_feature = feature.split('_range')[0]

        # Filtrer les conditions actives
        active_conditions = [cond for cond in conditions if cond.get('active', False)]
        if not active_conditions:
            continue  # Aucune condition active pour cette feature

        feature_mask = np.zeros(len(df), dtype=bool)  # Initialisation à False

        for condition in active_conditions:
            if condition['type'] == 'greater_than_or_equal':
                feature_mask |= df[base_feature].fillna(-np.inf) >= condition['threshold']
            elif condition['type'] == 'less_than_or_equal':
                feature_mask |= df[base_feature].fillna(np.inf) <= condition['threshold']
            elif condition['type'] == 'between':
                feature_mask |= df[base_feature].fillna(np.nan).between(
                    condition['min'], condition['max'], inclusive='both'
                )

        mask &= feature_mask  # Intersection avec le masque global

    return df[mask]


def find_consecutive_trades(df, trade_category):
    """
    Trouve les séquences consécutives pour une catégorie spécifique de trades,
    en tenant compte des interruptions par d'autres types de trades.
    """
    filtered_df = df.copy()  # On garde tous les trades pour voir les interruptions

    if filtered_df.empty:
        return 0, None, None

    # Conversion des dates
    if not pd.api.types.is_datetime64_any_dtype(filtered_df['date']):
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])

    # Trier par date
    filtered_df = filtered_df.sort_values('date')

    max_sequence = 0
    current_sequence = 0
    max_start_date = None
    max_end_date = None
    current_start_date = None

    dates = filtered_df['date'].dt.to_pydatetime()
    categories = filtered_df['trade_category'].values

    for i, (date, category) in enumerate(zip(dates, categories)):
        if category == trade_category:
            if current_sequence == 0:
                current_sequence = 1
                current_start_date = date
            else:
                # Vérifier si le trade précédent était de la même catégorie
                if categories[i - 1] == trade_category:
                    current_sequence += 1
                else:
                    # Réinitialiser si le trade précédent était différent
                    current_sequence = 1
                    current_start_date = date
        else:
            # Si on trouve la plus longue séquence jusqu'ici
            if current_sequence > max_sequence:
                max_sequence = current_sequence
                max_start_date = current_start_date
                max_end_date = dates[i - 1]
            current_sequence = 0

    # Vérifier la dernière séquence
    if current_sequence > max_sequence:
        max_sequence = current_sequence
        max_start_date = current_start_date
        max_end_date = dates[-1]

    return max_sequence, max_start_date, max_end_date
def calculate_performance_metrics(df):
    """
    Calcule les métriques complètes de performance de trading.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les données de trading avec les colonnes:
        - 'trade_pnl' : Profit/Perte de chaque trade
        - 'pos_type' : Type de position ('long' ou 'short')
        - index : dates des trades

    Returns:
    --------
    dict
        Dictionnaire structuré contenant toutes les métriques de performance
    """
    # 1. Métriques de base sur l'ensemble des trades
    total_trades = len(df)
    winning_trades = df[df['trade_pnl'] > 0]
    losing_trades = df[df['trade_pnl'] < 0]

    num_winning_trades = len(winning_trades)
    num_losing_trades = len(losing_trades)

    # 2. Séparation et analyse par direction
    # Trades longs
    long_trades = df[df['pos_type'] == 'long']
    winning_trades_long = winning_trades[winning_trades['pos_type'] == 'long']
    losing_trades_long = losing_trades[losing_trades['pos_type'] == 'long']

    num_winning_trades_long = len(winning_trades_long)
    num_losing_trades_long = len(losing_trades_long)
    total_trades_long = num_winning_trades_long + num_losing_trades_long

    # Trades shorts
    short_trades = df[df['pos_type'] == 'short']
    winning_trades_short = winning_trades[winning_trades['pos_type'] == 'short']
    losing_trades_short = losing_trades[losing_trades['pos_type'] == 'short']

    num_winning_trades_short = len(winning_trades_short)
    num_losing_trades_short = len(losing_trades_short)
    total_trades_short = num_winning_trades_short + num_losing_trades_short

    # 3. Calcul des profits et pertes
    gross_profit = winning_trades['trade_pnl'].sum() if not winning_trades.empty else 0
    gross_loss = losing_trades['trade_pnl'].sum() if not losing_trades.empty else 0
    net_pnl = gross_profit + gross_loss

    # Profit factor avec gestion de division par zéro
    profit_factor = gross_profit / abs(gross_loss) if abs(gross_loss) > 0 else np.inf if gross_profit > 0 else 0

    # 4. Calcul des win rates
    win_rate = (num_winning_trades / total_trades * 100) if total_trades > 0 else 0
    win_rate_long = (num_winning_trades_long / total_trades_long * 100) if total_trades_long > 0 else 0
    win_rate_short = (num_winning_trades_short / total_trades_short * 100) if total_trades_short > 0 else 0

    # 5. Calcul des moyennes de PnL
    expected_pnl_net = df['trade_pnl'].mean() if not df.empty else 0
    expected_pnl_long = long_trades['trade_pnl'].mean() if not long_trades.empty else 0
    expected_pnl_short = short_trades['trade_pnl'].mean() if not short_trades.empty else 0

    # Moyennes des profits et pertes
    avg_profit_per_win = winning_trades['trade_pnl'].mean() if not winning_trades.empty else 0
    avg_loss_per_loss = losing_trades['trade_pnl'].mean() if not losing_trades.empty else 0

    # 6. Analyse des séquences consécutives
    # Analyse des séquences consécutives pour les longs
    long_win_seq, long_win_start, long_win_end = find_consecutive_trades(df, "Trades réussis long")
    long_lose_seq, long_lose_start, long_lose_end = find_consecutive_trades(df, "Trades échoués long")

    # Analyse des séquences consécutives pour les shorts
    short_win_seq, short_win_start, short_win_end = find_consecutive_trades(df, "Trades réussis short")
    short_lose_seq, short_lose_start, short_lose_end = find_consecutive_trades(df, "Trades échoués short")

    # 7. Identification des meilleurs et pires trades
    def get_extreme_trade(trades_df, extreme_type='max'):
        if trades_df.empty:
            return {'PnL': 0, 'Date': None}
        if extreme_type == 'max':
            idx = trades_df['trade_pnl'].idxmax()
            pnl = trades_df['trade_pnl'].max()
        else:
            idx = trades_df['trade_pnl'].idxmin()
            pnl = trades_df['trade_pnl'].min()
        return {'PnL': pnl, 'Date': idx}

    # 8. Construction du dictionnaire de résultats
    return {
        "Total Trades": total_trades,
        "Trades Réussis": num_winning_trades,
        "Trades Échoués": num_losing_trades,

        "Trades Longs": {
            "Total": total_trades_long,
            "Réussis": num_winning_trades_long,
            "Échoués": num_losing_trades_long,
            "Win Rate": win_rate_long,
            "PnL Moyen": expected_pnl_long,
            "Meilleur Trade": get_extreme_trade(long_trades, 'max'),
            "Pire Trade": get_extreme_trade(long_trades, 'min'),
            "Séquences Consécutives": {
                "Max Trades Gagnants": {
                    "Nombre": long_win_seq,
                    "Date Début": long_win_start,
                    "Date Fin": long_win_end
                },
                "Max Trades Perdants": {
                    "Nombre": long_lose_seq,
                    "Date Début": long_lose_start,
                    "Date Fin": long_lose_end
                }
            }
        },

        "Trades Shorts": {
            "Total": total_trades_short,
            "Réussis": num_winning_trades_short,
            "Échoués": num_losing_trades_short,
            "Win Rate": win_rate_short,
            "PnL Moyen": expected_pnl_short,
            "Meilleur Trade": get_extreme_trade(short_trades, 'max'),
            "Pire Trade": get_extreme_trade(short_trades, 'min'),
            "Séquences Consécutives": {
                "Max Trades Gagnants": {
                    "Nombre": short_win_seq,
                    "Date Début": short_win_start,
                    "Date Fin": short_win_end
                },
                "Max Trades Perdants": {
                    "Nombre": short_lose_seq,
                    "Date Début": short_lose_start,
                    "Date Fin": short_lose_end
                }
            }
        },

        "Performance Globale": {
            "Win Rate Total": win_rate,
            "Gross Profit": gross_profit,
            "Gross Loss": gross_loss,
            "Net PnL": net_pnl,
            "Profit Factor": profit_factor,
            "PnL Moyen par Trade": expected_pnl_net,
            "Profit Moyen (Trades Gagnants)": avg_profit_per_win,
            "Perte Moyenne (Trades Perdants)": avg_loss_per_loss
        }
    }
def print_comparative_performance(metrics_before, metrics_after):
    """
    Affiche une comparaison détaillée et complète des performances avant et après filtrage.

    Parameters:
    -----------
    metrics_before : dict
        Métriques de performance avant filtrage
    metrics_after : dict
        Métriques de performance après filtrage
    """

    def calculate_change(before, after):
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            if before == 0:
                return "N/A" if after == 0 else "+∞" if after > 0 else "-∞"
            change = ((after - before) / abs(before)) * 100
            return f"{change:+.2f}%"
        return "N/A"

    print("\n═══ ANALYSE COMPARATIVE DES PERFORMANCES ═══")

    print("\n📊 STATISTIQUES GLOBALES")
    print("═" * 75)
    print(f"{'Métrique':<35} {'Avant':<15} {'Après':<15} {'Variation':<15}")
    print("─" * 75)

    # Statistiques de base
    basic_metrics = {
        'Nombre total de trades': 'Total Trades',
        'Trades Réussis': 'Trades Réussis',
        'Trades Échoués': 'Trades Échoués'
    }

    for label, key in basic_metrics.items():
        before_val = metrics_before[key]
        after_val = metrics_after[key]
        print(f"{label:<35} {before_val:<15} {after_val:<15} {calculate_change(before_val, after_val):<15}")

    print("\n📈 PERFORMANCE GLOBALE")
    print("═" * 75)

    global_metrics = {
        'Win Rate Total (%)': 'Win Rate Total',
        'Gross Profit': 'Gross Profit',
        'Gross Loss': 'Gross Loss',
        'Net PnL': 'Net PnL',
        'Profit Factor': 'Profit Factor',
        'PnL Moyen par Trade': 'PnL Moyen par Trade',
        'Profit Moyen (Trades Gagnants)': 'Profit Moyen (Trades Gagnants)',
        'Perte Moyenne (Trades Perdants)': 'Perte Moyenne (Trades Perdants)'
    }

    for label, key in global_metrics.items():
        before_val = metrics_before['Performance Globale'][key]
        after_val = metrics_after['Performance Globale'][key]
        print(f"{label:<35} {before_val:15.2f} {after_val:15.2f} {calculate_change(before_val, after_val):<15}")

    # Analyse détaillée par direction
    directions = ['Longs', 'Shorts']
    for direction in directions:
        print(f"\n📊 ANALYSE DES TRADES {direction.upper()}")
        print("═" * 75)

        direction_metrics = {
            'Nombre total': 'Total',
            'Trades Réussis': 'Réussis',
            'Trades Échoués': 'Échoués',
            'Win Rate (%)': 'Win Rate',
            'PnL Moyen': 'PnL Moyen'
        }

        for label, key in direction_metrics.items():
            before_val = metrics_before[f'Trades {direction}'][key]
            after_val = metrics_after[f'Trades {direction}'][key]
            print(f"{label:<35} {before_val:15.2f} {after_val:15.2f} {calculate_change(before_val, after_val):<15}")

        # Meilleurs et pires trades avec leurs dates
        print(f"\n🎯 TRADES EXTRÊMES {direction.upper()}")
        print("─" * 75)

        # Meilleur trade
        best_before = metrics_before[f'Trades {direction}']['Meilleur Trade']
        best_after = metrics_after[f'Trades {direction}']['Meilleur Trade']
        print(f"Meilleur trade avant: {best_before['PnL']:.2f} (Date: {best_before['Date']})")
        print(f"Meilleur trade après: {best_after['PnL']:.2f} (Date: {best_after['Date']})")

        # Pire trade
        worst_before = metrics_before[f'Trades {direction}']['Pire Trade']
        worst_after = metrics_after[f'Trades {direction}']['Pire Trade']
        print(f"Pire trade avant: {worst_before['PnL']:.2f} (Date: {worst_before['Date']})")
        print(f"Pire trade après: {worst_after['PnL']:.2f} (Date: {worst_after['Date']})")

    # Résumé de l'impact du filtrage
    print("\n📑 RÉSUMÉ DE L'IMPACT DU FILTRAGE")
    print("═" * 75)
    trades_removed = metrics_before['Total Trades'] - metrics_after['Total Trades']
    trades_removed_pct = (trades_removed / metrics_before['Total Trades']) * 100

    print(f"Trades filtrés: {trades_removed} ({trades_removed_pct:.2f}% du total)")

    wr_impact = metrics_after['Performance Globale']['Win Rate Total'] - metrics_before['Performance Globale'][
        'Win Rate Total']
    print(f"Impact sur le Win Rate: {wr_impact:+.2f}%")

    pnl_impact = metrics_after['Performance Globale']['Net PnL'] - metrics_before['Performance Globale']['Net PnL']
    print(f"Impact sur le Net PnL: {pnl_impact:+.2f}")

    pf_impact = metrics_after['Performance Globale']['Profit Factor'] - metrics_before['Performance Globale'][
        'Profit Factor']
    print(f"Impact sur le Profit Factor: {pf_impact:+.2f}")

    # Ajouter la section des séquences juste avant le résumé de l'impact
    for direction in ['Longs', 'Shorts']:
        print(f"\n📊 SÉQUENCES CONSÉCUTIVES {direction.upper()}")
        print("═" * 75)

        for period, metrics in [("Avant", metrics_before), ("Après", metrics_after)]:
            sequences = metrics[f"Trades {direction}"]["Séquences Consécutives"]
            print(f"\n{period}:")

            # Affichage des trades gagnants consécutifs
            win_seq = sequences["Max Trades Gagnants"]
            print(f"Trades gagnants consécutifs maximum : {win_seq['Nombre']}")
            if win_seq['Nombre'] > 0:
                print(f"  Période : du {win_seq['Date Début']} au {win_seq['Date Fin']}")

            # Affichage des trades perdants consécutifs
            lose_seq = sequences["Max Trades Perdants"]
            print(f"Trades perdants consécutifs maximum : {lose_seq['Nombre']}")
            if lose_seq['Nombre'] > 0:
                print(f"  Période : du {lose_seq['Date Début']} au {lose_seq['Date Fin']}")
            print()