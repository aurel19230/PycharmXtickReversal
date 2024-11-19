import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from standardFunc import timestamp_to_date_utc
from standardFunc import print_notification, load_data
import os

# Charger les données
# Nom du fichier
file_name = "Step3_4_0_6TP_1SL_080919_141024_extractOnlyFullSession.csv"
#file_name = "Step3_4_0_4TP_1SL_080919_091024_extractOnly220LastFullSession.csv"

# Chemin du répertoire
directory_path = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\4_0_6TP_1SL\\merge"

# Combiner le chemin du répertoire avec le nom du fichier
file_path = os.path.join(directory_path, file_name)

# Demander à l'utilisateur de choisir l'option
user_choice = input("Entrez 'Entrée' pour prendre en compte les shorts et longs, 's' pour les shorts uniquement ou 'l' pour les longs uniquement : ")

# Charger les données
df = load_data(file_path)

df['timeStampOpening'] = pd.to_numeric(df['timeStampOpening'], errors='coerce')

# Convertir la colonne timeStamp en datetime
df['formatted_date'] = timestamp_to_date_utc(df['timeStampOpening'])
df['date'] = pd.to_datetime(df['formatted_date'])

# Créer les colonnes 'class_binaire' et 'class_multi' en fonction du choix de l'utilisateur
if user_choice.lower() == 's':
    # Classification binaire pour les trades "short"
    df['class_binaire'] = np.select(
        [
            (df['tradeDir'] == -1) & (df['tradeResult'] == 1),
            (df['tradeDir'] == -1) & (df['tradeResult'] == -1)
        ],
        [1, 0],
        default=99
    )
    # Classification multi-classe pour les trades "short"
    df['class_multi'] = np.select(
        [
            (df['tradeDir'] == -1) & (df['tradeResult'] == 1),
            (df['tradeDir'] == -1) & (df['tradeResult'] == -1)
        ],
        [1, 0],
        default=99
    )
    output_file_suffix = "_OnlyShort"

elif user_choice.lower() == 'l':
    # Classification binaire pour les trades "long"
    df['class_binaire'] = np.select(
        [
            (df['tradeDir'] == 1) & (df['tradeResult'] == 1),
            (df['tradeDir'] == 1) & (df['tradeResult'] == -1)
        ],
        [1, 0],
        default=99
    )
    # Classification multi-classe pour les trades "long"
    df['class_multi'] = np.select(
        [
            (df['tradeDir'] == 1) & (df['tradeResult'] == 1),
            (df['tradeDir'] == 1) & (df['tradeResult'] == -1)
        ],
        [1, 0],
        default=99
    )
    output_file_suffix = "_OnlyLong"

else:
    # Classification binaire pour tous les trades
    df['class_binaire'] = np.select(
        [
            (df['tradeDir'] == 1) & (df['tradeResult'] == 1),
            (df['tradeDir'] == -1) & (df['tradeResult'] == 1),
            (df['tradeDir'] == 1) & (df['tradeResult'] == -1),
            (df['tradeDir'] == -1) & (df['tradeResult'] == -1)
        ],
        [1, 1, 0, 0],
        default=99
    )
    # Classification multi-classe pour tous les trades
    df['class_multi'] = np.select(
        [
            (df['tradeDir'] == 1) & (df['tradeResult'] == 1),
            (df['tradeDir'] == -1) & (df['tradeResult'] == 1),
            (df['tradeDir'] == 1) & (df['tradeResult'] == -1),
            (df['tradeDir'] == -1) & (df['tradeResult'] == -1)
        ],
        [0, 1, 2, 3],
        default=99
    )
    output_file_suffix = ""


# Filtrer les données pour exclure les valeurs 99 dans la colonne 'class'
filtered_class_counts = df[df['class_binaire'] != 99]['class_binaire'].value_counts()
filtered_class_percentages = filtered_class_counts / filtered_class_counts.sum() * 100

# Préparation des données pour le graphique de distribution mensuelle détaillée
df['month'] = df['date'].dt.strftime('%Y-%m')

# Set default value for trade_category
df['trade_category'] = 'Pas de trade'

# Create mask for rows where class is not 99
mask = df['class_binaire'] != 99

# Assign trade categories only to rows where class is not 99
df.loc[mask, 'trade_category'] = np.select(
    [
        (df.loc[mask, 'tradeDir'] == 1) & (df.loc[mask, 'tradeResult'] == 1),
        (df.loc[mask, 'tradeDir'] == 1) & (df.loc[mask, 'tradeResult'] == -1),
        (df.loc[mask, 'tradeDir'] == -1) & (df.loc[mask, 'tradeResult'] == 1),
        (df.loc[mask, 'tradeDir'] == -1) & (df.loc[mask, 'tradeResult'] == -1)
    ],
    [
        'Trades réussis long',
        'Trades échoués long',
        'Trades réussis short',
        'Trades échoués short'
    ],
    default='Pas de trade'
)

# Calculer les pourcentages pour le camembert
total_trades = len(df)
no_trade_count = df[df['tradeResult'] == 99].shape[0]
short_trade_count = df[df['tradeDir'] == -1].shape[0]
long_trade_count = df[df['tradeDir'] == 1].shape[0]

no_trade = no_trade_count / total_trades * 100
short_trade = short_trade_count / total_trades * 100
long_trade = long_trade_count / total_trades * 100

# Calculer les pourcentages pour le graphique à barres
trades = df[df['tradeResult'] != 99]
total_active_trades = len(trades)

short_trades = trades[trades['tradeDir'] == -1]
long_trades = trades[trades['tradeDir'] == 1]

short_percent = len(short_trades) / total_active_trades * 100
long_percent = len(long_trades) / total_active_trades * 100

short_success = short_trades[short_trades['tradeResult'] == 1].shape[0] / len(short_trades) * 100 if len(short_trades) > 0 else 0
short_fail = 100 - short_success

long_success = long_trades[long_trades['tradeResult'] == 1].shape[0] / len(long_trades) * 100 if len(long_trades) > 0 else 0
long_fail = 100 - long_success

monthly_distribution = df[df['tradeResult'] != 99].groupby(['month', 'trade_category']).size().unstack(fill_value=0)
monthly_distribution['Total'] = monthly_distribution.sum(axis=1)
monthly_distribution = monthly_distribution.div(monthly_distribution['Total'], axis=0) * 100

# Ensure all categories are present, fill with 0 if missing
all_categories = ['Trades réussis long', 'Trades réussis short', 'Trades échoués long', 'Trades échoués short']
for category in all_categories:
    if category not in monthly_distribution.columns:
        monthly_distribution[category] = 0

# Obtenir le nom du fichier d'entrée
nom_fichier_entree = os.path.basename(file_path)


# Obtenir le chemin complet du fichier de sortie
# Enlever l'extension .csv si elle existe
nom_fichier_entree_sans_extension = os.path.splitext(nom_fichier_entree)[0]
nom_fichier_entree_sans_extension = nom_fichier_entree_sans_extension.replace("Step3", "Step4")

# Rajouter le suffixe et ajouter .csv à la fin
nom_fichier_sortie = f"{nom_fichier_entree_sans_extension}{output_file_suffix}.csv"
chemin_sortie = os.path.join(os.path.dirname(file_path), nom_fichier_sortie)

# Sauvegarder le DataFrame
print_notification(f"Sauvegarde du DataFrame dans : {chemin_sortie}")
df.to_csv(chemin_sortie, sep=';', index=False, encoding='iso-8859-1')

# Afficher les informations dans la console
print("\n--- Informations des graphiques ---\n")

print("1. Répartition des positions:")
print(f"Total des trades: {total_trades}")
print(f"Aucun trade: {no_trade_count} ({no_trade:.1f}%)")
print(f"Trades short: {short_trade_count} ({short_trade:.1f}%)")
print(f"Trades long: {long_trade_count} ({long_trade:.1f}%)")

print("\n2. Répartition et résultats des trades actifs:")
print(f"Total des trades actifs: {total_active_trades}")
print(f"Trades short: {len(short_trades)} ({short_percent:.1f}%)")
print(f"  Réussis: {short_success:.1f}%")
print(f"  Échoués: {short_fail:.1f}%")
print(f"Trades long: {len(long_trades)} ({long_percent:.1f}%)")
print(f"  Réussis: {long_success:.1f}%")
print(f"  Échoués: {long_fail:.1f}%")

print("\n3. Distribution mensuelle détaillée des trades:")
print(monthly_distribution.to_string())

print("\n4. Répartition globale des trades réussis et échoués:")
successful_trades = filtered_class_counts[1]
failed_trades = filtered_class_counts[0]
print(f"Trades réussis: {successful_trades} ({filtered_class_percentages[1]:.1f}%)")
print(f"Trades échoués: {failed_trades} ({filtered_class_percentages[0]:.1f}%)")

print("\n--- Fin des informations ---")

# Création des graphiques
fig, axes = plt.subplots(2, 2, figsize=(20, 12))

# Graphique 1: Camembert de répartition des positions
ax1 = axes[0, 0]
sizes = [no_trade, short_trade, long_trade]
labels = [f'Aucun trade\n{no_trade_count}', f'Trade Shorts\n{short_trade_count}', f'Trade longs\n{long_trade_count}']
wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                                   pctdistance=0.85, labeldistance=1.05)
plt.setp(texts, size=10, weight="bold")
plt.setp(autotexts, size=9, weight="bold")
ax1.set_title(f'Répartition des positions\nTotal: {total_trades}')
ax1.axis('equal')

# Graphique 2: Barres de répartition et résultats des trades actifs
ax2 = axes[0, 1]
ax2.bar(['Short', 'Long'], [short_fail, long_fail], color=['red', 'red'], alpha=0.7, width=0.5, label='% trades échoués')
ax2.bar(['Short', 'Long'], [short_success, long_success], bottom=[short_fail, long_fail], color=['green', 'green'], alpha=0.7, width=0.5, label='% trades réussis')

for i, (success, fail) in enumerate(zip([short_success, long_success], [short_fail, long_fail])):
    ax2.text(i, fail/2, f'{fail:.1f}%', ha='center', va='center', color='white')
    ax2.text(i, fail + success/2, f'{success:.1f}%', ha='center', va='center', color='white')

ax2.set_xticks([0, 1])
ax2.set_xticklabels([f'Short\n{short_percent:.1f}%', f'Long\n{long_percent:.1f}%'])
ax2.set_ylim(0, 110)
ax2.set_ylabel('Pourcentage')
ax2.set_title('Répartition et résultats des trades actifs')
ax2.legend()

# Graphique 3: Aires empilées de distribution mensuelle détaillée
ax3 = axes[1, 0]
colors = ['darkgreen', 'lightgreen', 'darkred', 'salmon']
ax3.stackplot(monthly_distribution.index,
              [monthly_distribution[cat] for cat in all_categories],
              labels=all_categories,
              colors=colors)

ax3.set_ylabel('Pourcentage')
ax3.set_title('Distribution mensuelle détaillée des trades')
ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax3.set_ylim(0, 100)

dates = [pd.to_datetime(date) for date in monthly_distribution.index]
ax3.set_xticks(range(len(dates)))
ax3.set_xticklabels([date.strftime('%b. %y') for date in dates], rotation=90, fontsize=9)

# Graphique 4: Camembert de répartition globale des trades réussis et échoués
ax4 = axes[1, 1]

# Déterminer le titre en fonction du choix de l'utilisateur
if user_choice.lower() == 's':
    title = 'Répartition des trades Shorts réussis et échoués'
elif user_choice.lower() == 'l':
    title = 'Répartition des trades Longs réussis et échoués'
else:
    title = 'Répartition des trades réussis et échoués (Longs et Shorts)'

wedges, texts, autotexts = ax4.pie(filtered_class_percentages,
                                   labels=[f'Trades échoués\n{failed_trades}', f'Trades réussis\n{successful_trades}'],
                                   autopct='%1.1f%%',
                                   colors=['red', 'green'],
                                   pctdistance=0.85,
                                   labeldistance=1.05)

plt.setp(texts, size=10, weight="bold")
plt.setp(autotexts, size=9, weight="bold")

ax4.set_title(title)
ax4.axis('equal')

plt.tight_layout()
plt.show()