import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.power import TTestIndPower
from func_standard import load_data
from sklearn.feature_selection import f_classif  # Test F (ANOVA)

# Chemin vers ton fichier
file_name = "Step5_5_0_5TP_1SL_150924_280225_bugFixTradeResult_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
directory_path = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\5_0_5TP_1SL\\\merge_I1_I2"
file_path = os.path.join(directory_path, file_name)

# Charger les données
df = load_data(file_path)


# ---- FONCTION D'ANALYSE AVEC TEST F (ANOVA) ---- #
def analyze_feature_power(df, feature_list, target_col='class_binaire', alpha=0.05, target_power=0.8, n_simulations=1000, sample_fraction=0.8):
    results = []
    power_analysis = TTestIndPower()

    # 🚀 Filtrer les colonnes constantes
    df = df.loc[:, df.nunique() > 1]

    for feature in feature_list:
        if feature in df.columns:
            data_filtered = df[[feature, target_col]].dropna()
            group0 = data_filtered[data_filtered[target_col] == 0][feature].values
            group1 = data_filtered[data_filtered[target_col] == 1][feature].values

            if len(group0) > 1 and len(group1) > 1:
                # --- TEST T (Welch) ---
                t_stat, p_value_t = stats.ttest_ind(group0, group1, equal_var=False)

                # --- TEST F (ANOVA) ---
                try:
                    f_stat, p_value_f = f_classif(data_filtered[[feature]], data_filtered[target_col])
                    p_value_f = p_value_f[0]  # Extraire la valeur du test F
                except Exception:
                    f_stat, p_value_f = np.nan, np.nan  # Gestion des erreurs

                # --- COHEN'S D ---
                mean_diff = np.mean(group1) - np.mean(group0)
                pooled_std = np.sqrt(((len(group0)-1)*np.std(group0, ddof=1)**2 + (len(group1)-1)*np.std(group1, ddof=1)**2) / (len(group0) + len(group1) - 2))
                effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

                # --- GESTION DU CAS où L’EFFET EST NUL ---
                if effect_size == 0:
                    required_n = np.nan  # Pas possible de calculer
                    power_analytical = 0
                    power_monte_carlo = 0
                    print(f"⚠️  Attention : Effet de taille nul pour {feature}, puissance non calculable.")
                else:
                    # --- PUISSANCE ANALYTIQUE ---
                    power_analytical = power_analysis.power(effect_size=effect_size, nobs1=len(group0), alpha=alpha, ratio=len(group1)/len(group0))

                    # --- PUISSANCE MONTE CARLO ---
                    significant_count = 0
                    for _ in range(n_simulations):
                        sample0 = np.random.choice(group0, size=int(len(group0) * sample_fraction), replace=False)
                        sample1 = np.random.choice(group1, size=int(len(group1) * sample_fraction), replace=False)
                        _, p_sim = stats.ttest_ind(sample0, sample1, equal_var=False)
                        if p_sim < alpha:
                            significant_count += 1
                    power_monte_carlo = significant_count / n_simulations

                    # --- TAILLE D'ÉCHANTILLON REQUISE ---
                    try:
                        required_n = power_analysis.solve_power(effect_size=effect_size, power=target_power, alpha=alpha, ratio=len(group1)/len(group0))
                    except ValueError:
                        required_n = np.nan  # Ne peut pas être calculé

                results.append({
                    'Feature': feature,
                    'Sample_Size': len(data_filtered),
                    'Effect_Size': effect_size,
                    'P-Value_T': p_value_t,  # Test t (Welch)
                    'P-Value_F': p_value_f,  # Test F (ANOVA)
                    'Power_Analytical': power_analytical,
                    'Power_MonteCarlo': power_monte_carlo,
                    'Required_N': np.ceil(required_n) if not np.isnan(required_n) else np.nan,
                    'Sufficient_Analytical': power_analytical >= target_power,
                    'Sufficient_MonteCarlo': power_monte_carlo >= target_power
                })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Power_MonteCarlo', ascending=False)
    return results_df


# ---- CHARGEMENT DES FEATURES À ANALYSER ---- #

if True:
    feature_list = [
        'diffLowPrice_0_1', 'diffVolDelta_2_2Ratio', 'ratio_deltaRevMoveExtrem_volRevMoveExtrem',
        'cumDOM_AskBid_pullStack_avgDiff_ratio', 'ratio_deltaRevMove_volRevMove', 'VolPocVolCandleRatio',
        'ratio_volRevZone_VolCandle', 'ratio_volRevMove_volImpulsMove', 'diffVolDelta_1_1Ratio',
        'pocDeltaPocVolRatio', 'ratio_delta_vol_VA6P', 'delta_impulsMove_XRevZone_bigStand_extrem',
        'ratio_volRevMoveZone1_volImpulsMoveExtrem_XRevZone'
    ]
else:
    # Colonnes à exclure explicitement (variables cibles, pnl, dates...)
    excluded_columns = [
        'class_binaire',
        'timestamp',
        'SessionStartEnd',
        'trade_pnl_theoric', 'trade_pnl', 'trade_category','sl_pnl',
        'tp3_pnl', 'tp2_pnl', 'tp1_pnl', 'tp1_pnl_theoric',
        'timeStampOpening', 'date', 'close', 'high', 'low', 'open'
    ]

    # ✅ Sélection correcte des colonnes numériques en excluant explicitement celles mentionnées
    feature_list = df.select_dtypes(include=[np.number]).columns.difference(excluded_columns).tolist()

# Vérification
print(f"📌 Nombre de features sélectionnées : {len(feature_list)}")
print(f"🔹 Liste des features :\n{feature_list}")

# Filtrage des données (classe binaire doit être 0 ou 1)
df_filtered = df[df['class_binaire'].isin([0, 1])]

# ---- EXPLICATION DES RÉSULTATS ---- #
explanation = """
🔍 **Explication des variables du tableau de résultats :**

- **Feature** : Nom de la feature analysée.
- **Sample_Size** : Nombre d’observations utilisées après filtrage des NaN.
- **Effect_Size (Cohen's d)** : Mesure de la séparation entre les deux classes.
  - **> 0.8** : Effet fort ✅
  - **0.5 - 0.8** : Effet moyen ⚠️
  - **< 0.5** : Effet faible ❌
- **P-Value_T (Test t de Welch)** : Vérifie si la moyenne des deux classes est différente (**p < 0.05** signifie significatif).
- **P-Value_F (ANOVA F-test)** : Mesure la variance entre les classes (**p < 0.05** indique une différence de variabilité importante).
- **Power_Analytical** : Puissance statistique basée sur une formule analytique.
- **Power_MonteCarlo** : Puissance statistique estimée via simulations.
- **Required_N** : Nombre d’observations nécessaires pour atteindre **Puissance = 0.8**.
- **Sufficient_Analytical** : L'échantillon actuel est-il suffisant selon l'analyse analytique ?
- **Sufficient_MonteCarlo** : L'échantillon actuel est-il suffisant selon les simulations Monte Carlo ?

🎯 **Interprétation des seuils** :
- ✅ **Puissance ≥ 0.8** : La feature a une distinction nette entre classes.
- ⚠️ **0.6 ≤ Puissance < 0.8** : Possible impact, mais incertain.
- ❌ **Puissance < 0.6** : Probablement inutile pour la classification.

---------------------------------------------
"""

print(explanation)

# ---- ANALYSE DE PUISSANCE ---- #
print("\n🔍 **Analyse de puissance statistique pour les features :**")
power_results = analyze_feature_power(df_filtered, feature_list)
print(power_results)


# ---- VISUALISATION ---- #
def plot_power_analysis(power_results):
    if power_results.empty:
        print("Pas de résultats d'analyse de puissance à visualiser.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.barplot(x='Feature', y='Power_MonteCarlo', data=power_results, ax=ax1, palette='viridis')
    ax1.axhline(y=0.8, color='r', linestyle='--', label='Puissance cible = 0.8')
    ax1.set_title('Puissance statistique par feature (Monte Carlo)')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    ax1.set_ylabel('Puissance')
    ax1.legend()
    plt.show()

# ---- FILTRAGE DES FEATURES PERTINENTES ---- #
powerful_features = power_results[power_results['Power_MonteCarlo'] >= 0.6]['Feature'].tolist()
print(f"\n✅ **Features avec une puissance suffisante (≥ 0.6) :** {len(powerful_features)}/{len(feature_list)}")
if powerful_features:
    print("\n".join(f"- {f}" for f in powerful_features))
plot_power_analysis(power_results)



print("\n\nCode pour les disctribution dans le code suive cette partie")

def plot_feature_distributions(df, feature_list, target_col='class_binaire', figsize=(16, 10)):
    from sklearn.feature_selection import f_classif
    from matplotlib.patches import Patch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    import warnings

    n_features = len(feature_list)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    # Création d'une figure avec GridSpec pour mieux contrôler les sous-graphiques
    fig = plt.figure(figsize=figsize)

    # Création d'un GridSpec avec une ligne supplémentaire pour la légende
    gs = GridSpec(n_rows + 1, n_cols, figure=fig, height_ratios=[*[1] * n_rows, 0.1])

    fig.suptitle('Distribution des features par classe', fontsize=16)

    sns.set_style("whitegrid")
    palette = {0: "royalblue", 1: "crimson"}

    axes = []
    for i in range(n_rows):
        for j in range(n_cols):
            if i * n_cols + j < n_features:
                # Ajouter les axes pour chaque graphique
                axes.append(fig.add_subplot(gs[i, j]))

    for i, feature in enumerate(feature_list):
        if i < len(axes):
            ax = axes[i]

            if feature in df.columns:
                # Ne pas afficher la légende pour chaque graphique
                sns.histplot(
                    data=df,
                    x=feature,
                    hue=target_col,
                    kde=True,
                    palette=palette,
                    alpha=0.6,
                    ax=ax,
                    legend=False  # Suppression de la légende individuelle
                )

                # Calculer les statistiques pour chaque classe
                class0 = df[df[target_col] == 0][feature]
                class1 = df[df[target_col] == 1][feature]

                # Vérifier s'il y a des valeurs dans chaque classe
                if len(class0) > 0 and len(class1) > 0:
                    # Calculer les moyennes et écarts-types en ignorant les NaN
                    mean0, std0 = class0.mean(skipna=True), class0.std(skipna=True)
                    mean1, std1 = class1.mean(skipna=True), class1.std(skipna=True)

                    # Vérifier si les valeurs sont valides (non NaN)
                    mean0 = mean0 if pd.notna(mean0) else 0
                    std0 = std0 if pd.notna(std0) else 0
                    mean1 = mean1 if pd.notna(mean1) else 0
                    std1 = std1 if pd.notna(std1) else 0

                    # Gestion des NaN pour f_classif
                    X = df[[feature]].copy()
                    y = df[target_col].copy()

                    # Filtrer les lignes sans NaN (pour les deux X et y)
                    mask = X[feature].notna() & y.notna()
                    X_filtered = X.loc[mask]
                    y_filtered = y.loc[mask]

                    # Calculer f_classif seulement s'il y a assez de données
                    if len(X_filtered) > 0 and len(np.unique(y_filtered)) > 1:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                f_scores, p_values = f_classif(X_filtered, y_filtered)
                                f_stat, p_val = f_scores[0], p_values[0]
                        except Exception as e:
                            print(f"Erreur lors du calcul f_classif pour {feature}: {e}")
                            f_stat, p_val = 0, 1
                    else:
                        f_stat, p_val = 0, 1

                    # Ajustement de l'annotation avec des valeurs sécurisées
                    ax.annotate(
                        f"Cl.0: µ={mean0:.2f}, σ={std0:.2f}\n"
                        f"Cl.1: µ={mean1:.2f}, σ={std1:.2f}\n"
                        f"F: {f_stat:.2f}, p: {p_val:.3f}",
                        xy=(0.98, 0.95),
                        xycoords='axes fraction',
                        ha='right',
                        va='top',
                        fontsize='small',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8)
                    )

                ax.set_title(f"{feature}", fontsize=12)
                ax.set_xlabel(feature, fontsize=10)
                ax.set_ylabel("Fréquence", fontsize=10)
            else:
                ax.text(0.5, 0.5, f"{feature} non trouvée",
                        ha='center', va='center', fontsize=10)
                ax.set_axis_off()

    # Création d'un subplot dédié pour la légende
    legend_ax = fig.add_subplot(gs[n_rows, :])
    legend_ax.axis('off')  # Masquer les axes

    # Création des handles pour la légende
    legend_handles = [
        Patch(facecolor="royalblue", edgecolor="black", label="Classe 0"),
        Patch(facecolor="crimson", edgecolor="black", label="Classe 1")
    ]

    # Ajout de la légende au subplot dédié
    legend = legend_ax.legend(
        handles=legend_handles,
        title='Classes',
        loc='center',
        ncol=2,
        fontsize='medium',
        framealpha=0.8
    )

    plt.tight_layout()
    plt.show()

    return fig

# Vérifier que le filtrage a fonctionné
#print(df_filtered['class_binaire'].unique())

# Utiliser le DataFrame filtré
plot_feature_distributions(df_filtered, feature_list)