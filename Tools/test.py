import pandas as pd
import numpy as np
from itertools import combinations

# Charger les données
file_path = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_4TP_1SL\merge\\MergedAllFile_030619_300824_merged_extractOnlyFullSession.csv"
df = pd.read_csv(file_path, sep=';')

# Calculer la section de 30 minutes
minRange = 30
df['deltaTimestampOpeningSection30min'] = df['deltaTimestampOpening'].apply(
    lambda x: min(int(np.floor(x / minRange)) * minRange, 1350))

# Grouper par 'deltaTimestampOpeningSection30min' et sommer les volumes
volume_by_section = df.groupby('deltaTimestampOpeningSection30min')['volume'].sum().reset_index()
volume_by_section = volume_by_section.sort_values('deltaTimestampOpeningSection30min')


# Fonction pour calculer la variance des volumes entre les groupes
def calculate_variance(groups):
    volumes = [sum(volume_by_section.loc[start:end, 'volume']) for start, end in groups]
    return np.var(volumes)


# Fonction pour trouver la meilleure répartition
def find_best_grouping(min_groups=4, max_groups=8):
    best_grouping = None
    best_variance = float('inf')

    for n_groups in range(min_groups, max_groups + 1):
        # Générer toutes les combinaisons possibles de points de coupure
        possible_cuts = combinations(range(1, len(volume_by_section)), n_groups - 1)

        for cuts in possible_cuts:
            groups = [(0, cuts[0] - 1)]  # Premier groupe
            groups.extend((cuts[i], cuts[i + 1] - 1) for i in range(len(cuts) - 1))  # Groupes intermédiaires
            groups.append((cuts[-1], len(volume_by_section) - 1))  # Dernier groupe

            variance = calculate_variance(groups)

            if variance < best_variance:
                best_variance = variance
                best_grouping = groups

    return best_grouping


# Trouver la meilleure répartition
best_grouping = find_best_grouping()

# Afficher les résultats
print("Meilleure répartition des groupes :")
for i, (start, end) in enumerate(best_grouping, 1):
    start_time = volume_by_section.iloc[start]['deltaTimestampOpeningSection30min']
    end_time = volume_by_section.iloc[end]['deltaTimestampOpeningSection30min'] + 30
    group_volume = volume_by_section.iloc[start:end + 1]['volume'].sum()
    print(f"Groupe {i}: De {start_time} à {end_time} minutes, Volume total = {group_volume}")

# Calculer et afficher la variance des volumes entre les groupes
final_variance = calculate_variance(best_grouping)
print(f"\nVariance des volumes entre les groupes : {final_variance}")

# Afficher le volume total
total_volume = volume_by_section['volume'].sum()
print(f"Volume total sur toutes les tranches: {total_volume}")