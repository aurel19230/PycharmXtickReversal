import numpy as np


def calculate_weighted_score(original_scores, num_early_splits=0, early_weight=0.8, std_penalty_factor=1.0):
    """
    Calcule le score avec une pondération correcte de l'écart-type.

    Args:
        original_scores: Liste des scores originaux
        num_early_splits: Nombre de premiers splits à pondérer
        early_weight: Poids à appliquer aux premiers splits (0 < early_weight < 1)
        std_penalty_factor: Facteur de pénalité pour l'écart-type
    """
    scores = np.array(original_scores)
    weights = np.ones(len(scores))

    if num_early_splits > 0:
        weights[:num_early_splits] = early_weight

    # Moyenne pondérée
    weighted_mean = np.average(scores, weights=weights)

    # Écart-type pondé

    weighted_var = np.average((scores - weighted_mean) ** 2, weights=weights)
    weighted_std = np.sqrt(weighted_var)
    # Score final
    score_adjusted = weighted_mean - std_penalty_factor * weighted_std

    return score_adjusted, weighted_mean, weighted_std


# Test
scores = [-182.3, -343.9, -341.2, -256.5, -138.3, -305.1]
score_adjusted, mean, std = calculate_weighted_score(
    scores,
    num_early_splits=2,
    early_weight=0.8,
    std_penalty_factor=1.0
)
from colorama import Fore, Style, init

color_code = Fore.RED

print(f"{color_code}Moyenne pondérée: {mean:.2f}")
print(f"Écart-type pondéré: {std:.2f}")
print(f"Score ajusté final: {score_adjusted:.2f}")