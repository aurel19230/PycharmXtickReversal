
from standard_stat_sc import *
# =============================================================================
# 1. Chargement des données
# =============================================================================
from definition import *
from func_standard import *

FILE_NAME_ = "Step5__150924_030425_bugFixTradeResult1_extractOnlyFullSession_OnlyShort_feat.csv"
ENV = detect_environment()
if ENV == 'pycharm':
        if platform.system() != "Darwin":
                DIRECTORY_PATH = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\5_0_5TP_1SL_1\\\merge"

        else:
                DIRECTORY_PATH = "/Users/aurelienlachaud/Documents/trading_local/5_0_5TP_1SL_1/merge"

FILE_PATH = os.path.join(DIRECTORY_PATH, FILE_NAME_)

df_init_features, CUSTOM_SESSIONS = load_features_and_sections(FILE_PATH)
print(df_init_features["diffPriceClosePoc_0_0"].shape)

# =============================================================================
# 2. Filtrage des catégories d'intérêt
# =============================================================================
categories_of_interest = [
    "Trades échoués short",
    "Trades échoués long",
    "Trades réussis short",
    "Trades réussis long"
]

df_analysis = df_init_features[df_init_features['trade_category'].isin(categories_of_interest)].copy()
print(df_analysis["diffPriceClosePoc_0_0"].shape)

# =============================================================================
# 3. Ajout de deux colonnes pour simplifier le filtrage
#    - 'class' = 0 (échoués) ou 1 (réussis)
#    - 'pos_type' = 'short' ou 'long'
# =============================================================================
df_analysis['class'] = np.where(df_analysis['trade_category'].str.contains('échoués'), 0, 1)
df_analysis['pos_type'] = np.where(df_analysis['trade_category'].str.contains('short'), 'short', 'long')
print(df_analysis["diffPriceClosePoc_0_0"].shape)

#calcul des métric de performance avant filtrage
#YY


# Définition du dictionnaire des features et leurs conditions
features_conditions = {
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


# Application du filtrage comme dans votre code original
df_filtered = apply_feature_conditions(df_analysis, features_conditions)

# Création du DataFrame complet avec la colonne "PnlAfterFiltering"
df_full_afterFiltering = create_full_dataframe_with_filtered_pnl(df_init_features, df_filtered)

# 2. Prétraiter les sessions pour ajouter 'session_id' et 'session_date'
df_with_sessions_and_dates = preprocess_sessions_with_date(df_full_afterFiltering)

#fig = plot_trading_performance(df_with_sessions_and_dates)
#plt.show()

def save_dataframe_to_csv(df, directory_path, filename, sep=';', index=False):
        """
        Enregistre un DataFrame dans un fichier CSV avec le séparateur spécifié.

        Args:
            df (pd.DataFrame): Le DataFrame à enregistrer
            directory_path (str): Le chemin du répertoire où enregistrer le fichier
            filename (str): Le nom du fichier (sans le chemin)
            sep (str): Le séparateur à utiliser (';' par défaut)
            index (bool): Si True, inclut l'index dans le fichier CSV

        Returns:
            str: Le chemin complet du fichier créé
        """
        import os

        # Créer le répertoire s'il n'existe pas
        if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                print(f"Répertoire créé: {directory_path}")

        # Construire le chemin complet du fichier
        file_path = os.path.join(directory_path, filename)

        # Enregistrer le DataFrame
        df.to_csv(file_path, sep=sep, index=index)

        print(f"✓ Fichier enregistré: {file_path}")
        return file_path


# Utilisation pour votre cas spécifique
output_filename = "df_with_sessions_and_dates.csv"
saved_file_path = save_dataframe_to_csv(df_with_sessions_and_dates, DIRECTORY_PATH, output_filename)

# Vérification des résultats
print("Somme du PnL initial:", df_init_features['trade_pnl'].sum())
print("Somme du PnL après filtrage:", df_full_afterFiltering['PnlAfterFiltering'].sum())
print("Nombre de trades retenus après filtrage:", (df_full_afterFiltering['PnlAfterFiltering'] != 0).sum())


metrics_before = calculate_performance_metrics(df_analysis)
metrics_after = calculate_performance_metrics(df_filtered)
print_comparative_performance(metrics_before, metrics_after)
# =============================================================================
# 4. Sélection des features (jusqu'à 24 max)
# =============================================================================
#features = [
#        'reg_std_5P',
#    'reg_std_10P',
#    'reg_std_15P',
#    'reg_std_30P',

    # Ajoutez d'autres colonnes si désiré, jusqu'à 24...
#]
#features_to_plot = [f for f in features if f in df_analysis.columns][:24]

#plot_boxplots(df_analysis, features, category_col='trade_category', nrows=3, ncols=4)

#plot_distributions_short_long_grid(df_analysis, features_to_plot, class_col='class')
import os
import sys


def save_trades_to_csv(df, directory_path, sep=';', index=False):
        """
        Enregistre les données de trading dans un fichier CSV avec le nom du script Python actuel.

        Args:
            df (pd.DataFrame): Le DataFrame contenant les données de trading
            directory_path (str): Le chemin du répertoire où enregistrer le fichier
            sep (str): Le séparateur à utiliser (';' par défaut)
            index (bool): Si True, inclut l'index dans le fichier CSV

        Returns:
            str: Le chemin complet du fichier créé
        """
        # Obtenir le nom du script Python actuel (sans le chemin)
        script_name = os.path.basename(sys.argv[0])
        # Remplacer l'extension .py par .csv
        output_filename = os.path.splitext(script_name)[0] + ".csv"

        # Créer le répertoire s'il n'existe pas
        if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                print(f"Répertoire créé: {directory_path}")

        # Construire le chemin complet du fichier
        file_path = os.path.join(directory_path, output_filename)

        # Filtrer pour ne garder que les lignes qui ont généré des trades
        trades_df = df[df['PnlAfterFiltering'] != 0].copy()

        # Enregistrer le DataFrame
        trades_df.to_csv(file_path, sep=sep, index=index)

        print(f"✓ Fichier enregistré: {file_path}")
        print(f"  Nombre de trades enregistrés: {len(trades_df)}")
        return file_path

# Enregistrer uniquement les lignes qui ont généré des trades
saved_file_path = save_trades_to_csv(df_with_sessions_and_dates, DIRECTORY_PATH)

# Vérification des résultats
print("Somme du PnL initial:", df_init_features['trade_pnl'].sum())
print("Somme du PnL après filtrage:", df_full_afterFiltering['PnlAfterFiltering'].sum())
print("Nombre de trades retenus après filtrage:", (df_full_afterFiltering['PnlAfterFiltering'] != 0).sum())