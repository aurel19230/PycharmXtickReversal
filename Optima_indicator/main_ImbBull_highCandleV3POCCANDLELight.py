# -*- coding: utf-8 -*-
"""optuna_live_monitor.py – version 4
====================================
- Intègre **trois** conditions distinctes avec des plages de sampling différentes.
- Utilise **deux** datasets de validation pour une plus grande robustesse.
- Raccourci clavier « & » (utilisant pynput) pour déclencher un calcul immédiat
  sur le jeu TEST pendant l'optimisation.
- Affichage synthétique à chaque trial + rapport détaillé périodique / sur
  nouveau meilleur score.
- Affichage coloré des résultats avec colorama
- Ajout du nombre de sessions à côté du Total dans l'affichage des résultats
- Ajout d'un affichage détaillé des trades par catégorie exclusive pour chaque dataset
- Ajout d'une option de filtrage par POC (-1)
"""
from __future__ import annotations

# ─────────────────── CONFIG ──────────────────────────────────────
from pathlib import Path
import sys, time, math, optuna, pandas as pd, numpy as np
import threading

# Remplacer msvcrt par pynput
from pynput import keyboard

# Ajout de colorama pour les affichages colorés
from colorama import init, Fore, Back, Style

# Initialiser colorama (nécessaire pour Windows)
init(autoreset=True)

RANDOM_SEED = 42
DIR = Path(r"C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/"
           r"Sierra chart/xTickReversal/simu/5_0_5TP_6SL/merge")

CSV_TRAIN = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split1_01012024_01052024.csv"
CSV_TEST = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split2_01052024_01102024.csv"
CSV_VAL1 = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split3_01102024_28022025.csv"
CSV_VAL = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split4_02032025_14052025.csv"

# Configuration par mode
CONFIGS = {
    "light": {
        "WINRATE_MIN": 0.53,  # WR minimum acceptable
        "PCT_TRADE_MIN": 0.035,  # % de candles tradées minimum
        "ALPHA": 0.70,  # poids du WR dans le score
    },
    "aggressive": {
        "WINRATE_MIN": 0.55,  # WR minimum acceptable moins strict
        "PCT_TRADE_MIN": 0.02,  # % de candles tradées minimum plus élevé
        "ALPHA": 0.65,  # légèrement moins de poids sur le WR
    }
}
# *** BEST so far ▸ trial 1123  score=0.4277
#     [GLOBAL - Trades uniques]
#     TR WR 61.29% | pct 3.06% | ✓57 ✗36 Total=93 (sessions: 45)
#     V1 WR 61.51% | pct 5.13% | ✓310 ✗194 Total=504 (sessions: 47)
#     V2 WR 55.68% | pct 3.09% | ✓98 ✗78 Total=176 (sessions: 58)
#     Gap WR moyen: 3.88% | Gap PCT moyen: 1.38%
#
#     [DÉTAIL PAR CATÉGORIE DE TRADES - TRAIN]
#     Condition 1 uniquement : WR=65.38% | Total=52
#     Condition 2 uniquement : WR=0.00% | Total=0
#     Condition 3 uniquement : WR=63.64% | Total=11
#     Conditions 1+2 : WR=45.45% | Total=11
#     Conditions 1+3 : WR=58.33% | Total=12
#     Conditions 2+3 : WR=0.00% | Total=0
#     Toutes conditions (1+2+3) : WR=57.14% | Total=7
#     Vérification : 93 trades catégorisés vs 93 total global
#
#     [DÉTAIL PAR CATÉGORIE DE TRADES - VAL]
#     Condition 1 uniquement : WR=61.02% | Total=449
#     Condition 2 uniquement : WR=0.00% | Total=0
#     Condition 3 uniquement : WR=0.00% | Total=1
#     Conditions 1+2 : WR=63.41% | Total=41
#     Conditions 1+3 : WR=66.67% | Total=6
#     Conditions 2+3 : WR=0.00% | Total=0
#     Toutes conditions (1+2+3) : WR=85.71% | Total=7
#     Vérification : 504 trades catégorisés vs 504 total global
#
#     [DÉTAIL PAR CATÉGORIE DE TRADES - VAL1]
#     Condition 1 uniquement : WR=57.36% | Total=129
#     Condition 2 uniquement : WR=0.00% | Total=0
#     Condition 3 uniquement : WR=0.00% | Total=2
#     Conditions 1+2 : WR=60.00% | Total=25
#     Conditions 1+3 : WR=66.67% | Total=6
#     Conditions 2+3 : WR=0.00% | Total=0
#     Toutes conditions (1+2+3) : WR=35.71% | Total=14
#     Vérification : 176 trades catégorisés vs 176 total global
#
#     [Condition 1]
#     TR WR 60.98% | pct 2.70% | ✓50 ✗32 Total=82 (sessions: 39)
#     V1 WR 55.68% | pct 3.09% | ✓98 ✗78 Total=176 (sessions: 58)
#     V2 WR 56.32% | pct 3.05% | ✓98 ✗76 Total=174 (sessions: 58)
#
#     [Condition 2]
#     TR WR 50.00% | pct 0.59% | ✓9 ✗9 Total=18 (sessions: 16)
#     V1 WR 66.67% | pct 0.49% | ✓32 ✗16 Total=48 (sessions: 28)
#     V2 WR 51.28% | pct 0.68% | ✓20 ✗19 Total=39 (sessions: 27)
#
#     [Condition 3]
#     TR WR 60.00% | pct 0.99% | ✓18 ✗12 Total=30 (sessions: 23)
#     V1 WR 71.43% | pct 0.14% | ✓10 ✗4 Total=14 (sessions: 13)
#     V2 WR 40.91% | pct 0.39% | ✓9 ✗13 Total=22 (sessions: 20)
#
#     params ➜ {'bidVolHigh_1': 3, 'bull_imbalance_high_0': 2.557075830254644, 'bidVolHigh_1_2Cond': 18, 'bull_imbalance_high_0_2Cond': 3.638182553600235, 'bidVolHigh_1_3Cond': 55, 'bull_imbalance_high_0_3Cond': 2.2890964360654595, 'pos_poc_min': -1.0, 'pos_poc_max': 0.0}
#
# POC filtré entre -1.0 et 0.0 : TR 100.0%, V1 99.5%, V2 100.0%
#   1150 | TR 60.42%/ 3.16% | V1 61.42%/ 5.17% | V2 55.00%/ 3.16% ✔
#   1152 | TR 59.18%/ 3.22% | V1 59.90%/ 5.96% | V2 53.00%/ 3.51%
#   1153 | TR 60.00%/ 3.13% | V1 61.39%/ 5.14% | V2 55.62%/ 3.12% ✔
#   1154 | TR 59.22%/ 3.39% | V1 60.96%/ 5.52% | V2 54.01%/ 3.28%
#   1155 | TR 59.80%/ 3.36% | V1 61.18%/ 5.19% | V2 54.40%/ 3.19% ✔
#   1156 | TR 59.78%/ 3.03% | V1 60.96%/ 4.87% | V2 55.75%/ 3.05% ✔
#   1158 | TR 59.38%/ 3.16% | V1 61.05%/ 5.62% | V2 53.68%/ 3.33%
#   1161 | TR 58.51%/ 3.09% | V1 61.59%/ 5.01% | V2 55.31%/ 3.14% ✔
#   1162 | TR 60.00%/ 3.29% | V1 61.01%/ 5.45% | V2 54.10%/ 3.21% ✔
#   1164 | TR 59.38%/ 3.16% | V1 60.92%/ 4.84% | V2 56.07%/ 3.04% ✔
#   1165 | TR 60.61%/ 3.26% | V1 60.70%/ 5.85% | V2 53.81%/ 3.46%
#   1166 | TR 60.82%/ 3.19% | V1 61.42%/ 5.17% | V2 54.44%/ 3.16% ✔
#   1168 | TR 59.43%/ 3.49% | V1 61.04%/ 5.48% | V2 53.76%/ 3.26%
#   1171 | TR 58.76%/ 3.19% | V1 60.81%/ 4.80% | V2 55.23%/ 3.02% ✔
#   1173 | TR 60.00%/ 3.29% | V1 61.03%/ 5.53% | V2 54.05%/ 3.25% ✔
#   1176 | TR 59.43%/ 3.49% | V1 60.52%/ 5.85% | V2 53.03%/ 3.47%
#   1179 | TR 58.59%/ 3.26% | V1 61.57%/ 5.06% | V2 55.49%/ 3.19% ✔
#   1182 | TR 59.43%/ 3.49% | V1 60.92%/ 5.54% | V2 53.72%/ 3.30%
#   1184 | TR 60.82%/ 3.19% | V1 61.25%/ 5.20% | V2 54.70%/ 3.18% ✔
#   1185 | TR 59.79%/ 3.19% | V1 59.83%/ 5.95% | V2 53.00%/ 3.51%
#   1186 | TR 59.38%/ 3.16% | V1 60.76%/ 4.82% | V2 55.56%/ 3.00% ✔
#   1188 | TR 59.38%/ 3.16% | V1 61.05%/ 5.62% | V2 53.68%/ 3.33%
#   1189 | TR 60.00%/ 3.13% | V1 61.39%/ 5.14% | V2 55.62%/ 3.12% ✔
#   1191 | TR 58.76%/ 3.19% | V1 61.04%/ 4.88% | V2 55.37%/ 3.11%
#   1194 | TR 59.62%/ 3.42% | V1 60.83%/ 5.64% | V2 53.37%/ 3.39%
#   1195 | TR 61.54%/ 3.42% | V1 59.54%/ 6.19% | V2 53.40%/ 3.62% ✔
#   1196 | TR 61.05%/ 3.13% | V1 61.49%/ 5.18% | V2 54.75%/ 3.14% ✔
#   1197 | TR 60.61%/ 3.26% | V1 61.01%/ 5.45% | V2 54.10%/ 3.21%
#   1198 | TR 59.78%/ 3.03% | V1 61.16%/ 5.61% | V2 54.59%/ 3.25%
# Best trial 1123  value 0.4277
# 🧪  Test automatique (trial 1200)
#
# 🧮  Calcul sur DATASET TEST
#
# POC filtré entre -1.0 et 0.0 : TEST 100.0% (5520/5520)
# --- Détail par condition ---
# Condition 1: WR=56.22%  pct=3.35%  ✓104 ✗81  Total=185 (sessions: 63)
# Condition 2: WR=84.00%  pct=0.45%  ✓21 ✗4  Total=25 (sessions: 17)
# Condition 3: WR=65.38%  pct=0.47%  ✓17 ✗9  Total=26 (sessions: 22)
#
# --- Résultat combiné (union) ---
# Union: WR=55.50%  pct=3.46%  ✓106 ✗85  Total=191 (sessions: 63)
#
#     [DÉTAIL PAR CATÉGORIE DE TRADES - TEST]
#     Condition 1 uniquement : WR=51.35% | Total=148
#     Condition 2 uniquement : WR=0.00% | Total=0
#     Condition 3 uniquement : WR=33.33% | Total=6
#     Conditions 1+2 : WR=76.47% | Total=17
#     Conditions 1+3 : WR=58.33% | Total=12
#     Conditions 2+3 : WR=0.00% | Total=0
#     Toutes conditions (1+2+3) : WR=100.00% | Total=8
#     Vérification : 191 trades catégorisés vs 191 total global
# ✅ VALIDE
#
# --- Résultat combiné (union) ---
# Union: WR=54.50%  pct=3.42%  ✓103 ✗86  Total=189 (sessions: 67)
#
#     [DÉTAIL PAR CATÉGORIE DE TRADES - TEST]
#     Condition 1 uniquement : WR=50.88% | Total=114
#     Condition 2 uniquement : WR=0.00% | Total=0
#     Condition 3 uniquement : WR=51.11% | Total=45
#     Conditions 1+2 : WR=69.23% | Total=13
#     Conditions 1+3 : WR=55.56% | Total=9
#     Conditions 2+3 : WR=0.00% | Total=0
#     Toutes conditions (1+2+3) : WR=100.00% | Total=8
#     Vérification : 189 trades catégorisés vs 189 total global
# ✅ VALIDE
#
#
# Best trial 831  value 0.3982
#
# *** BEST so far ▸ trial 831  score=0.3982
#     [GLOBAL - Trades uniques]
#     TR WR 57.66% | pct 3.65% | ✓64 ✗47 Total=111 (sessions: 55)
#     V1 WR 59.34% | pct 4.30% | ✓251 ✗172 Total=423 (sessions: 48)
#     V2 WR 56.65% | pct 3.04% | ✓98 ✗75 Total=173 (sessions: 66)
#     Gap WR moyen: 1.79% | Gap PCT moyen: 0.84%
#
#     [DÉTAIL PAR CATÉGORIE DE TRADES - TRAIN]
#     Condition 1 uniquement : WR=62.79% | Total=43
#     Condition 2 uniquement : WR=0.00% | Total=0
#     Condition 3 uniquement : WR=56.86% | Total=51
#     Conditions 1+2 : WR=42.86% | Total=7
#     Conditions 1+3 : WR=50.00% | Total=6
#     Conditions 2+3 : WR=0.00% | Total=0
#     Toutes conditions (1+2+3) : WR=50.00% | Total=4
#     Vérification : 111 trades catégorisés vs 111 total global
#
#     [DÉTAIL PAR CATÉGORIE DE TRADES - VAL]
#     Condition 1 uniquement : WR=58.77% | Total=359
#     Condition 2 uniquement : WR=0.00% | Total=0
#     Condition 3 uniquement : WR=62.50% | Total=16
#     Conditions 1+2 : WR=60.53% | Total=38
#     Conditions 1+3 : WR=66.67% | Total=6
#     Conditions 2+3 : WR=0.00% | Total=0
#     Toutes conditions (1+2+3) : WR=75.00% | Total=4
#     Vérification : 423 trades catégorisés vs 423 total global
#
#     [DÉTAIL PAR CATÉGORIE DE TRADES - VAL1]
#     Condition 1 uniquement : WR=59.05% | Total=105
#     Condition 2 uniquement : WR=0.00% | Total=0
#     Condition 3 uniquement : WR=55.17% | Total=29
#     Conditions 1+2 : WR=60.00% | Total=20
#     Conditions 1+3 : WR=57.14% | Total=7
#     Conditions 2+3 : WR=0.00% | Total=0
#     Toutes conditions (1+2+3) : WR=33.33% | Total=12
#     Vérification : 173 trades catégorisés vs 173 total global
#
#     [Condition 1]
#     TR WR 58.33% | pct 1.97% | ✓35 ✗25 Total=60 (sessions: 33)
#     V1 WR 56.65% | pct 3.04% | ✓98 ✗75 Total=173 (sessions: 66)
#     V2 WR 56.94% | pct 2.53% | ✓82 ✗62 Total=144 (sessions: 56)
#
#     [Condition 2]
#     TR WR 45.45% | pct 0.36% | ✓5 ✗6 Total=11 (sessions: 10)
#     V1 WR 61.90% | pct 0.43% | ✓26 ✗16 Total=42 (sessions: 26)
#     V2 WR 50.00% | pct 0.56% | ✓16 ✗16 Total=32 (sessions: 24)
#
#     [Condition 3]
#     TR WR 55.74% | pct 2.01% | ✓34 ✗27 Total=61 (sessions: 41)
#     V1 WR 65.38% | pct 0.26% | ✓17 ✗9 Total=26 (sessions: 19)
#     V2 WR 50.00% | pct 0.84% | ✓24 ✗24 Total=48 (sessions: 35)
#
#     params ➜ {'bidVolHigh_1': 3, 'bull_imbalance_high_0': 2.859299964161862, 'bidVolHigh_1_2Cond': 18, 'bull_imbalance_high_0_2Cond': 4.06176028836505, 'bidVolHigh_1_3Cond': 52, 'bull_imbalance_high_0_3Cond': 1.7855243437036787, 'pos_poc_min': -1.0, 'pos_poc_max': 0.0}
#

choice = input(
    "Filtrage :\n"
    "  [Entrée] → light (meilleur scénario testé au170552025) pour l'ajouter au voting en en faisant un algo tiers\n"
    "  a        → agressif\n"
    "  z        → light + poc variable \n"
    "Choix : "
).strip().lower()

if choice == "a":
    cfg = CONFIGS["aggressive"]
    FILTER_POC = False
elif choice == "z":
    cfg = CONFIGS["light"]
    FILTER_POC = True
else:
    cfg = CONFIGS["light"]
    FILTER_POC = False

print(f"\n→ Mode : {'agressif' if choice == 'a' else 'light'}"
      f"{' + poc=-1' if FILTER_POC else ''}\n")

WINRATE_MIN = cfg["WINRATE_MIN"]
PCT_TRADE_MIN = cfg["PCT_TRADE_MIN"]
ALPHA = cfg["ALPHA"]

# Paramètres non modifiés par le choix utilisateur
N_TRIALS = 10_000
PRINT_EVERY = 50
FAILED_PENALTY = -0.001

# Gap penalties
LAMBDA_WR = 1  # 0.60
LAMBDA_PCT = 0  # 0.20

# ── Bornes par condition ────────────────────────────────────────
BID_MIN_1, BID_MAX_1 = 3, 10
BULL_MIN_1, BULL_MAX_1 = 2, 6

BID_MIN_2, BID_MAX_2 = 10, 23  # Chevauchement partiel avec Condition 1
BULL_MIN_2, BULL_MAX_2 = 2, 6  # Valeurs plus élevées, explorées différemment

BID_MIN_3, BID_MAX_3 = 23, 60  # Élargissement pour capturer plus de trades
BULL_MIN_3, BULL_MAX_3 = 1.5, 6  # Élargissement pour améliorer la stabilité

# Définissez les bornes pour les deux paramètres à optimiser
POS_POC_LOWER_BOUND_MIN, POS_POC_LOWER_BOUND_MAX = -1, 0  # Plage pour la borne inférieure
POS_POC_UPPER_BOUND_MIN, POS_POC_UPPER_BOUND_MAX = -1, 0  # Plage pour la borne supérieure
POS_POC_STEP = 0.25  # Pas d'incrémentation

import chardet


# ───────────────────── DATA LOADING ─────────────────────────────
def detect_file_encoding(file_path: str, sample_size: int = 100_000) -> str:
    with open(file_path, 'rb') as f:
        raw = f.read(sample_size)
    return chardet.detect(raw)['encoding']


def load_csv(path: str | Path) -> tuple[pd.DataFrame, int]:
    encoding = detect_file_encoding(str(path))
    if encoding.lower() == "ascii":
        encoding = "ISO-8859-1"

    print(f"{path.name} ➜ encodage détecté: {encoding}")

    # Chargement robuste
    df = pd.read_csv(path, sep=";", encoding=encoding, parse_dates=["date"], low_memory=False)

    # 🔧 Correction de SessionStartEnd
    df["SessionStartEnd"] = pd.to_numeric(df["SessionStartEnd"], errors="coerce")
    df = df.dropna(subset=["SessionStartEnd"])
    df["SessionStartEnd"] = df["SessionStartEnd"].astype(int)

    # 🔍 Vérif des valeurs possibles
    print(f"{path.name} ➜ uniques SessionStartEnd: {df['SessionStartEnd'].unique()}")

    # 📊 Compter les sessions avant filtre class_binaire
    nb_start = (df["SessionStartEnd"] == 10).sum()
    nb_end = (df["SessionStartEnd"] == 20).sum()
    nb_sessions = min(nb_start, nb_end)

    if nb_start != nb_end:
        print(
            f"{Fore.YELLOW}⚠️ Incohérence sessions: {nb_start} débuts vs {nb_end} fins dans {path.name}{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}✔ {nb_sessions} sessions complètes détectées dans {path.name}{Style.RESET_ALL}")

    # ✅ Numérotation des sessions (avant filtrage)
    df["session_id"] = (df["SessionStartEnd"] == 10).cumsum().astype("int32")

    # 🧼 Seulement maintenant : filtrage de la cible
    df = df[df["class_binaire"].isin([0, 1])].copy()
    df.reset_index(drop=True, inplace=True)

    return df, nb_sessions


# Chargement des datasets complets (une seule fois)
TRAIN, TRAIN_SESSIONS = load_csv(CSV_TRAIN)
VAL, VAL_SESSIONS = load_csv(CSV_VAL)
VAL1, VAL1_SESSIONS = load_csv(CSV_VAL1)
TEST, TEST_SESSIONS = load_csv(CSV_TEST)

# Affichage des informations sur les datasets
for lbl, d, sessions in zip(("TRAIN", "VAL", "VAL1", "TEST"),
                            (TRAIN, VAL, VAL1, TEST),
                            (TRAIN_SESSIONS, VAL_SESSIONS, VAL1_SESSIONS, TEST_SESSIONS)):
    print(f"{lbl:<5} | lignes={len(d):,}  WR brut={(d['class_binaire'] == 1).mean():.2%}  Sessions={sessions}")
print("—————————————————————————————————————————————————————————————\n")


# ───────────────────── MASK BUILDERS ────────────────────────────

def imbalance_high_rev(df: pd.DataFrame, *, bidVolHigh_1: float, bull_imbalance_high_0: float, **kwargs) -> pd.Series:
    """Condition 1: Ignore les paramètres supplémentaires"""
    return (df["bidVolHigh_1"] > bidVolHigh_1) & (df["bull_imbalance_high_0"] > bull_imbalance_high_0)


def imbalance_high_rev_2(df: pd.DataFrame, *, bidVolHigh_1_2Cond: float, bull_imbalance_high_0_2Cond: float,
                         **kwargs) -> pd.Series:
    """Condition 2: Ignore les paramètres supplémentaires"""
    return (df["bidVolHigh_1"] > bidVolHigh_1_2Cond) & (df["bull_imbalance_high_0"] > bull_imbalance_high_0_2Cond)


def imbalance_high_rev_3(df: pd.DataFrame, *, bidVolHigh_1_3Cond: float, bull_imbalance_high_0_3Cond: float,
                         **kwargs) -> pd.Series:
    """Condition 3: Ignore les paramètres supplémentaires"""
    return (df["bidVolHigh_1"] > bidVolHigh_1_3Cond) & (df["bull_imbalance_high_0"] > bull_imbalance_high_0_3Cond)


# ───────────────────── METRICS HELPERS ──────────────────────────

def _metrics(df: pd.DataFrame, mask: pd.Series, original_len: int = None) -> tuple[float, float, int, int, int]:
    """Calcule les métriques avec le nombre de sessions couvertes"""
    sub = df.loc[mask]
    if sub.empty:
        return 0.0, 0.0, 0, 0, 0

    # Remplacer ceci:
    # wins = int((sub["class_binaire"] == 0.75).sum())

    # Par ceci:
    wins = int((sub["class_binaire"] == 1).sum())

    total = len(sub)

    # Utiliser la longueur originale du dataset si fournie, sinon utiliser len(df)
    base_len = original_len if original_len is not None else len(df)

    # Calculer le pourcentage par rapport au dataset original, pas le filtré
    pct_trade = total / base_len

    # Calculer le nombre de sessions uniques où il y a des trades
    sessions_covered = sub["session_id"].nunique()

    return wins / total, pct_trade, wins, total - wins, sessions_covered


def _metrics_combined(df: pd.DataFrame, m1: pd.Series, m2: pd.Series, m3: pd.Series, original_len: int = None):
    """Calcule les métriques combinées avec le nombre de sessions couvertes"""
    m_u = m1 | m2 | m3
    m_12 = m1 & m2
    m_13 = m1 & m3
    m_23 = m2 & m3
    m_123 = m1 & m2 & m3
    return _metrics(df, m_u, original_len) + _metrics(df, m_12, original_len) + _metrics(df, m_13,
                                                                                         original_len) + _metrics(df,
                                                                                                                  m_23,
                                                                                                                  original_len) + _metrics(
        df, m_123, original_len)


def _metrics_exclusive(df: pd.DataFrame, m1: pd.Series, m2: pd.Series, m3: pd.Series, original_len: int = None):
    """Calcule des métriques détaillées montrant les trades uniques et les chevauchements"""
    # Catégories exclusives
    m1_only = m1 & ~m2 & ~m3  # Uniquement condition 1
    m2_only = ~m1 & m2 & ~m3  # Uniquement condition 2
    m3_only = ~m1 & ~m2 & m3  # Uniquement condition 3
    m12_only = m1 & m2 & ~m3  # Conditions 1 et 2 seulement
    m13_only = m1 & ~m2 & m3  # Conditions 1 et 3 seulement
    m23_only = ~m1 & m2 & m3  # Conditions 2 et 3 seulement
    m123 = m1 & m2 & m3  # Toutes les conditions

    # Global (union)
    m_u = m1 | m2 | m3

    # Calcul des métriques pour chaque catégorie
    metrics_global = _metrics(df, m_u, original_len)
    metrics_1_only = _metrics(df, m1_only, original_len)
    metrics_2_only = _metrics(df, m2_only, original_len)
    metrics_3_only = _metrics(df, m3_only, original_len)
    metrics_12_only = _metrics(df, m12_only, original_len)
    metrics_13_only = _metrics(df, m13_only, original_len)
    metrics_23_only = _metrics(df, m23_only, original_len)
    metrics_123 = _metrics(df, m123, original_len)

    return {
        "global": metrics_global,
        "cond1_only": metrics_1_only,
        "cond2_only": metrics_2_only,
        "cond3_only": metrics_3_only,
        "cond12": metrics_12_only,
        "cond13": metrics_13_only,
        "cond23": metrics_23_only,
        "cond123": metrics_123
    }


# ───────────────────── OPTUNA OBJECTIVE ─────────────────────────
# Initialisation de best_trial avec des valeurs par défaut
best_trial = {
    "score": -math.inf,
    "number": None,  # Initialisation de number à None
    "score_old": -math.inf,
    # Autres champs initialisés à des valeurs par défaut
    "wr_t": 0.0, "pct_t": 0.0, "suc_t": 0, "fail_t": 0, "sess_t": 0,
    "wr_v": 0.0, "pct_v": 0.0, "suc_v": 0, "fail_v": 0, "sess_v": 0,
    "wr_v1": 0.0, "pct_v1": 0.0, "suc_v1": 0, "fail_v1": 0, "sess_v1": 0,

    "wr_t1": 0.0, "pct_t1": 0.0, "suc_t1": 0, "fail_t1": 0, "sess_t1": 0,
    "wr_t2": 0.0, "pct_t2": 0.0, "suc_t2": 0, "fail_t2": 0, "sess_t2": 0,
    "wr_t3": 0.0, "pct_t3": 0.0, "suc_t3": 0, "fail_t3": 0, "sess_t3": 0,

    "wr_v1": 0.0, "pct_v1": 0.0, "suc_v1": 0, "fail_v1": 0, "sess_v1": 0,
    "wr_v2": 0.0, "pct_v2": 0.0, "suc_v2": 0, "fail_v2": 0, "sess_v2": 0,
    "wr_v3": 0.0, "pct_v3": 0.0, "suc_v3": 0, "fail_v3": 0, "sess_v3": 0,

    "wr_v1_1": 0.0, "pct_v1_1": 0.0, "suc_v1_1": 0, "fail_v1_1": 0, "sess_v1_1": 0,
    "wr_v1_2": 0.0, "pct_v1_2": 0.0, "suc_v1_2": 0, "fail_v1_2": 0, "sess_v1_2": 0,
    "wr_v1_3": 0.0, "pct_v1_3": 0.0, "suc_v1_3": 0, "fail_v1_3": 0, "sess_v1_3": 0,

    "avg_gap_wr": 0.0,
    "avg_gap_pct": 0.0,

    # Nouvelles métriques détaillées
    "metrics_detail_train": None,
    "metrics_detail_val": None,
    "metrics_detail_val1": None,

    "params": {}
}


def objective(trial: optuna.trial.Trial) -> float:
    # Paramètres spécifiques aux trois conditions
    p = {
        "bidVolHigh_1": trial.suggest_int("bidVolHigh_1", BID_MIN_1, BID_MAX_1),
        "bull_imbalance_high_0": trial.suggest_float("bull_imbalance_high_0", BULL_MIN_1, BULL_MAX_1),
        "bidVolHigh_1_2Cond": trial.suggest_int("bidVolHigh_1_2Cond", BID_MIN_2, BID_MAX_2),
        "bull_imbalance_high_0_2Cond": trial.suggest_float("bull_imbalance_high_0_2Cond", BULL_MIN_2, BULL_MAX_2),
        "bidVolHigh_1_3Cond": trial.suggest_int("bidVolHigh_1_3Cond", BID_MIN_3, BID_MAX_3),
        "bull_imbalance_high_0_3Cond": trial.suggest_float("bull_imbalance_high_0_3Cond", BULL_MIN_3, BULL_MAX_3),
    }

    # Conserver les tailles originales des datasets
    train_len = len(TRAIN)
    val_len = len(VAL)
    val1_len = len(VAL1)

    # Si FILTER_POC est activé, suggérer les bornes à optimiser avec pas de 0.25
    if FILTER_POC:
        # Suggérer la borne inférieure avec un pas de 0.25
        min_value = trial.suggest_float("pos_poc_min", POS_POC_LOWER_BOUND_MIN, POS_POC_LOWER_BOUND_MAX,
                                        step=POS_POC_STEP)

        # Suggérer la borne supérieure avec un pas de 0.25
        max_value = trial.suggest_float("pos_poc_max", POS_POC_UPPER_BOUND_MIN, POS_POC_UPPER_BOUND_MAX,
                                        step=POS_POC_STEP)

        # S'assurer que min_value <= max_value
        p["pos_poc_min"] = min(min_value, max_value)
        p["pos_poc_max"] = max(min_value, max_value)

    # Appliquer le filtrage POC si nécessaire
    train_df = TRAIN.copy()
    val_df = VAL.copy()
    val1_df = VAL1.copy()

    if FILTER_POC:
        poc_min = p["pos_poc_min"]
        poc_max = p["pos_poc_max"]

        # Filtre à la volée avec condition double
        train_df = train_df[
            (train_df["diffPriceClosePoc_0_0"] >= poc_min) & (train_df["diffPriceClosePoc_0_0"] <= poc_max)]
        val_df = val_df[(val_df["diffPriceClosePoc_0_0"] >= poc_min) & (val_df["diffPriceClosePoc_0_0"] <= poc_max)]
        val1_df = val1_df[(val1_df["diffPriceClosePoc_0_0"] >= poc_min) & (val1_df["diffPriceClosePoc_0_0"] <= poc_max)]

        # Affichage du filtrage
        if trial.number % PRINT_EVERY == 0:
            print(f"{Fore.CYAN}POC filtré entre {poc_min} et {poc_max} : "
                  f"TR {len(train_df) / len(TRAIN):.1%}, "
                  f"V1 {len(val_df) / len(VAL):.1%}, "
                  f"V2 {len(val1_df) / len(VAL1):.1%}{Style.RESET_ALL}")

    # Masks pour les datasets filtrés
    m1_t = imbalance_high_rev(train_df, **p)
    m2_t = imbalance_high_rev_2(train_df, **p)
    m3_t = imbalance_high_rev_3(train_df, **p)

    m1_v = imbalance_high_rev(val_df, **p)
    m2_v = imbalance_high_rev_2(val_df, **p)
    m3_v = imbalance_high_rev_3(val_df, **p)

    m1_v1 = imbalance_high_rev(val1_df, **p)
    m2_v1 = imbalance_high_rev_2(val1_df, **p)
    m3_v1 = imbalance_high_rev_3(val1_df, **p)

    # ────────── Métriques TRAIN ──────────────
    # Métriques individuelles par condition avec sessions
    wr_t1, pct_t1, suc_t1, fail_t1, sess_t1 = _metrics(train_df, m1_t, train_len)
    wr_t2, pct_t2, suc_t2, fail_t2, sess_t2 = _metrics(train_df, m2_t, train_len)
    wr_t3, pct_t3, suc_t3, fail_t3, sess_t3 = _metrics(train_df, m3_t, train_len)

    # Métrique combinée avec sessions
    wr_t, pct_t, suc_t, fail_t, sess_t, *_ = _metrics_combined(train_df, m1_t, m2_t, m3_t, train_len)

    # Métriques détaillées par catégorie
    metrics_detail_train = _metrics_exclusive(train_df, m1_t, m2_t, m3_t, train_len)

    # ────────── Métriques VAL (Validation 1) ──────────────
    # Métriques individuelles par condition avec sessions
    wr_v1, pct_v1, suc_v1, fail_v1, sess_v1 = _metrics(val_df, m1_v, val_len)
    wr_v2, pct_v2, suc_v2, fail_v2, sess_v2 = _metrics(val_df, m2_v, val_len)
    wr_v3, pct_v3, suc_v3, fail_v3, sess_v3 = _metrics(val_df, m3_v, val_len)

    # Métrique combinée avec sessions
    wr_v, pct_v, suc_v, fail_v, sess_v, *_ = _metrics_combined(val_df, m1_v, m2_v, m3_v, val_len)

    # Métriques détaillées par catégorie
    metrics_detail_val = _metrics_exclusive(val_df, m1_v, m2_v, m3_v, val_len)

    # ────────── Métriques VAL1 (Validation 2) ──────────────
    # Métriques individuelles par condition avec sessions
    wr_v1_1, pct_v1_1, suc_v1_1, fail_v1_1, sess_v1_1 = _metrics(val1_df, m1_v1, val1_len)
    wr_v1_2, pct_v1_2, suc_v1_2, fail_v1_2, sess_v1_2 = _metrics(val1_df, m2_v1, val1_len)
    wr_v1_3, pct_v1_3, suc_v1_3, fail_v1_3, sess_v1_3 = _metrics(val1_df, m3_v1, val1_len)

    # Métrique combinée avec sessions
    wr_v1, pct_v1, suc_v1, fail_v1, sess_v1, *_ = _metrics_combined(val1_df, m1_v1, m2_v1, m3_v1, val1_len)

    # Métriques détaillées par catégorie
    metrics_detail_val1 = _metrics_exclusive(val1_df, m1_v1, m2_v1, m3_v1, val1_len)

    # Quick threshold veto - Vérification des seuils sur les trois datasets
    if (wr_t < WINRATE_MIN or pct_t < PCT_TRADE_MIN or
            wr_v < WINRATE_MIN or pct_v < PCT_TRADE_MIN or
            wr_v1 < WINRATE_MIN or pct_v1 < PCT_TRADE_MIN):
        return FAILED_PENALTY

    # Calcul des écarts entre les jeux de données
    gap_wr_tv = abs(wr_t - wr_v)
    gap_pct_tv = abs(pct_t - pct_v)

    gap_wr_tv1 = abs(wr_t - wr_v1)
    gap_pct_tv1 = abs(pct_t - pct_v1)

    gap_wr_vv1 = abs(wr_v - wr_v1)
    gap_pct_vv1 = abs(pct_v - pct_v1)

    # Moyenne des écarts
    avg_gap_wr = (gap_wr_tv + gap_wr_tv1 + gap_wr_vv1) / 3
    avg_gap_pct = (gap_pct_tv + gap_pct_tv1 + gap_pct_vv1) / 3

    # Score qui considère les trois datasets et les écarts moyens
    score = (ALPHA * (wr_t + wr_v + wr_v1) / 3 +
             (1 - ALPHA) * (pct_t + pct_v + pct_v1) / 3 -
             LAMBDA_WR * avg_gap_wr -
             LAMBDA_PCT * avg_gap_pct)

    global best_trial
    if score > best_trial["score"]:
        best_trial = {
            "number": trial.number,
            "score": score,
            # Métriques combinées - TRAIN
            "wr_t": wr_t, "pct_t": pct_t, "suc_t": suc_t, "fail_t": fail_t, "sess_t": sess_t,
            # Métriques combinées - VAL
            "wr_v": wr_v, "pct_v": pct_v, "suc_v": suc_v, "fail_v": fail_v, "sess_v": sess_v,
            # Métriques combinées - VAL1
            "wr_v1": wr_v1, "pct_v1": pct_v1, "suc_v1": suc_v1, "fail_v1": fail_v1, "sess_v1": sess_v1,

            # Métriques par condition - TRAIN
            "wr_t1": wr_t1, "pct_t1": pct_t1, "suc_t1": suc_t1, "fail_t1": fail_t1, "sess_t1": sess_t1,
            "wr_t2": wr_t2, "pct_t2": pct_t2, "suc_t2": suc_t2, "fail_t2": fail_t2, "sess_t2": sess_t2,
            "wr_t3": wr_t3, "pct_t3": pct_t3, "suc_t3": suc_t3, "fail_t3": fail_t3, "sess_t3": sess_t3,

            # Métriques par condition - VAL
            "wr_v1": wr_v1, "pct_v1": pct_v1, "suc_v1": suc_v1, "fail_v1": fail_v1, "sess_v1": sess_v1,
            "wr_v2": wr_v2, "pct_v2": pct_v2, "suc_v2": suc_v2, "fail_v2": fail_v2, "sess_v2": sess_v2,
            "wr_v3": wr_v3, "pct_v3": pct_v3, "suc_v3": suc_v3, "fail_v3": fail_v3, "sess_v3": sess_v3,

            # Métriques par condition - VAL1
            "wr_v1_1": wr_v1_1, "pct_v1_1": pct_v1_1, "suc_v1_1": suc_v1_1, "fail_v1_1": fail_v1_1,
            "sess_v1_1": sess_v1_1,
            "wr_v1_2": wr_v1_2, "pct_v1_2": pct_v1_2, "suc_v1_2": suc_v1_2, "fail_v1_2": fail_v1_2,
            "sess_v1_2": sess_v1_2,
            "wr_v1_3": wr_v1_3, "pct_v1_3": pct_v1_3, "suc_v1_3": suc_v1_3, "fail_v1_3": fail_v1_3,
            "sess_v1_3": sess_v1_3,

            # Écarts moyens
            "avg_gap_wr": avg_gap_wr,
            "avg_gap_pct": avg_gap_pct,

            # Métriques détaillées
            "metrics_detail_train": metrics_detail_train,
            "metrics_detail_val": metrics_detail_val,
            "metrics_detail_val1": metrics_detail_val1,

            "params": p
        }

    # Live print avec les trois datasets
    print(f"{trial.number:>6} | "
          f"TR {Fore.GREEN}{wr_t:6.2%}{Style.RESET_ALL}/{pct_t:6.2%} | "
          f"V1 {Fore.GREEN}{wr_v:6.2%}{Style.RESET_ALL}/{pct_v:6.2%} | "
          f"V2 {Fore.GREEN}{wr_v1:6.2%}{Style.RESET_ALL}/{pct_v1:6.2%}",
          f"{Fore.GREEN}✔{Style.RESET_ALL}" if score > best_trial.get("score_old", -math.inf) else "")

    best_trial["score_old"] = score  # helper for symbol
    return score


# Fonction pour afficher les métriques détaillées
def print_detailed_metrics(dataset_name, metrics_detail):
    """Affiche les métriques détaillées par catégorie de trades"""

    wr_g, pct_g, suc_g, fail_g, sess_g = metrics_detail["global"]
    wr_1, pct_1, suc_1, fail_1, sess_1 = metrics_detail["cond1_only"]
    wr_2, pct_2, suc_2, fail_2, sess_2 = metrics_detail["cond2_only"]
    wr_3, pct_3, suc_3, fail_3, sess_3 = metrics_detail["cond3_only"]
    wr_12, pct_12, suc_12, fail_12, sess_12 = metrics_detail["cond12"]
    wr_13, pct_13, suc_13, fail_13, sess_13 = metrics_detail["cond13"]
    wr_23, pct_23, suc_23, fail_23, sess_23 = metrics_detail["cond23"]
    wr_123, pct_123, suc_123, fail_123, sess_123 = metrics_detail["cond123"]

    # Calculer les totaux pour chaque catégorie
    total_g = suc_g + fail_g
    total_1 = suc_1 + fail_1
    total_2 = suc_2 + fail_2
    total_3 = suc_3 + fail_3
    total_12 = suc_12 + fail_12
    total_13 = suc_13 + fail_13
    total_23 = suc_23 + fail_23
    total_123 = suc_123 + fail_123

    # Vérification de la somme
    total_details = total_1 + total_2 + total_3 + total_12 + total_13 + total_23 + total_123

    print(f"\n    {Fore.CYAN}[DÉTAIL PAR CATÉGORIE DE TRADES - {dataset_name}]{Style.RESET_ALL}")
    print(f"    Condition 1 uniquement : WR={Fore.GREEN}{wr_1:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_1}{Style.RESET_ALL}")
    print(f"    Condition 2 uniquement : WR={Fore.GREEN}{wr_2:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_2}{Style.RESET_ALL}")
    print(f"    Condition 3 uniquement : WR={Fore.GREEN}{wr_3:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_3}{Style.RESET_ALL}")
    print(f"    Conditions 1+2 : WR={Fore.GREEN}{wr_12:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_12}{Style.RESET_ALL}")
    print(f"    Conditions 1+3 : WR={Fore.GREEN}{wr_13:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_13}{Style.RESET_ALL}")
    print(f"    Conditions 2+3 : WR={Fore.GREEN}{wr_23:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_23}{Style.RESET_ALL}")
    print(f"    Toutes conditions (1+2+3) : WR={Fore.GREEN}{wr_123:.2%}{Style.RESET_ALL} | "
          f"Total={Fore.CYAN}{total_123}{Style.RESET_ALL}")

    # Vérification
    print(
        f"    {Fore.YELLOW}Vérification : {total_details} trades catégorisés vs {total_g} total global{Style.RESET_ALL}")
    if total_details != total_g:
        print(f"    {Fore.RED}⚠️ Anomalie détectée: La somme des détails ({total_details}) "
              f"ne correspond pas au total global ({total_g}){Style.RESET_ALL}")



# ───────────────────── HOLD‑OUT TEST ────────────────────────────
def calculate_test_metrics(params: dict):
    print(f"\n{Fore.CYAN}🧮  Calcul sur DATASET TEST{Style.RESET_ALL}\n")

    # Conserver la taille originale du dataset TEST
    test_len = len(TEST)

    # Appliquer le filtrage POC à la volée si nécessaire
    test_df = TEST.copy()
    if FILTER_POC and "pos_poc_min" in params and "pos_poc_max" in params:
        poc_min = params["pos_poc_min"]
        poc_max = params["pos_poc_max"]

        # S'assurer que min <= max
        poc_min, poc_max = min(poc_min, poc_max), max(poc_min, poc_max)

        # Filtre avec plage de valeurs
        test_df = test_df[(test_df["diffPriceClosePoc_0_0"] >= poc_min) & (test_df["diffPriceClosePoc_0_0"] <= poc_max)]

        print(f"{Fore.CYAN}POC filtré entre {poc_min} et {poc_max} : "
              f"TEST {len(test_df) / len(TEST):.1%} ({len(test_df)}/{len(TEST)}){Style.RESET_ALL}")

    # Créer les masques sur le dataframe filtré
    m1 = imbalance_high_rev(test_df, **params)
    m2 = imbalance_high_rev_2(test_df, **params)
    m3 = imbalance_high_rev_3(test_df, **params)

    # IMPORTANT: Utiliser test_df au lieu de TEST pour toutes les fonctions de métriques!
    # Calcul des métriques par condition avec sessions, en passant la longueur originale
    wr_1, pct_1, suc_1, fail_1, sess_1 = _metrics(test_df, m1, test_len)
    wr_2, pct_2, suc_2, fail_2, sess_2 = _metrics(test_df, m2, test_len)
    wr_3, pct_3, suc_3, fail_3, sess_3 = _metrics(test_df, m3, test_len)

    # Calcul des métriques combinées avec sessions
    wr_u, pct_u, suc_u, fail_u, sess_u, *_ = _metrics_combined(test_df, m1, m2, m3, test_len)

    # Calcul des métriques détaillées par catégorie
    metrics_detail_test = _metrics_exclusive(test_df, m1, m2, m3, test_len)

    # Affichage détaillé par condition
    print(f"{Fore.YELLOW}--- Détail par condition ---{Style.RESET_ALL}")
    print(f"Condition 1: WR={Fore.GREEN}{wr_1:.2%}{Style.RESET_ALL}  pct={pct_1:.2%}  "
          f"✓{Fore.GREEN}{suc_1}{Style.RESET_ALL} ✗{Fore.RED}{fail_1}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_1 + fail_1}{Style.RESET_ALL} (sessions: {sess_1})")
    print(f"Condition 2: WR={Fore.GREEN}{wr_2:.2%}{Style.RESET_ALL}  pct={pct_2:.2%}  "
          f"✓{Fore.GREEN}{suc_2}{Style.RESET_ALL} ✗{Fore.RED}{fail_2}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_2 + fail_2}{Style.RESET_ALL} (sessions: {sess_2})")
    print(f"Condition 3: WR={Fore.GREEN}{wr_3:.2%}{Style.RESET_ALL}  pct={pct_3:.2%}  "
          f"✓{Fore.GREEN}{suc_3}{Style.RESET_ALL} ✗{Fore.RED}{fail_3}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_3 + fail_3}{Style.RESET_ALL} (sessions: {sess_3})")

    # Affichage résultat combiné
    print(f"\n{Fore.YELLOW}--- Résultat combiné (union) ---{Style.RESET_ALL}")
    print(f"Union: WR={Fore.GREEN}{wr_u:.2%}{Style.RESET_ALL}  pct={pct_u:.2%}  "
          f"✓{Fore.GREEN}{suc_u}{Style.RESET_ALL} ✗{Fore.RED}{fail_u}{Style.RESET_ALL}  "
          f"Total={Fore.CYAN}{suc_u + fail_u}{Style.RESET_ALL} (sessions: {sess_u})")

    # Affichage détaillé par catégorie de trades
    print_detailed_metrics("TEST", metrics_detail_test)

    # Notez ici que nous utilisons les seuils sur les métriques calculées avec le dataframe filtré
    is_valid = (wr_u >= WINRATE_MIN and pct_u >= PCT_TRADE_MIN)
    if is_valid:
        print(f"{Fore.GREEN}✅ VALIDE{Style.RESET_ALL}\n\n")
    else:
        print(f"{Fore.RED}❌ REJET{Style.RESET_ALL}")

    return wr_u, pct_u, suc_u, fail_u, sess_u


# ───────────────────── KEYBOARD LISTENING ───────────────────────
RUN_TEST = False


def on_press(key):
    global RUN_TEST
    try:
        if key.char == '&':
            print(f"\n{Fore.YELLOW}🧪  Test demandé via '&'{Style.RESET_ALL}")
            RUN_TEST = True
    except AttributeError:
        # Touche spéciale sans caractère
        pass


# Démarrer listener dans un thread séparé
def start_keyboard_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True  # Le thread sera automatiquement terminé quand le programme principal se termine
    listener.start()
    return listener


# ───────────────────── MAIN LOOP ────────────────────────────────
def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    last_best_shown = None

    # Démarrer le listener clavier
    listener = start_keyboard_listener()
    print(
        f"{Fore.CYAN}Écouteur clavier démarré - appuyez sur '&' à tout moment pour tester sur le dataset TEST{Style.RESET_ALL}")

    # NOUVEAU CODE
    for done in range(1, N_TRIALS + 1):
        study.optimize(objective, n_trials=1)

        # Lancer le test automatiquement si la touche "&" a été pressée
        # ou si nous sommes à un multiple de PRINT_EVERY
        if RUN_TEST or done % PRINT_EVERY == 0:
            globals()["RUN_TEST"] = False  # Réinitialiser le flag
            # Si c'est un test automatique périodique, l'indiquer
            if done % PRINT_EVERY == 0 and not RUN_TEST:
                print(f"\n{Fore.YELLOW}🧪  Test automatique (trial {done}){Style.RESET_ALL}")
            calculate_test_metrics(study.best_params)

        if best_trial.get("number") is not None:
            print(f"Best trial {best_trial['number']}  value {Fore.GREEN}{best_trial['score']:.4f}{Style.RESET_ALL}",
                  end="\r")

        if (done % PRINT_EVERY == 0 or best_trial.get("number") != last_best_shown):
            bt = best_trial
            print(
                f"\n\n{Fore.YELLOW}*** BEST so far ▸ trial {bt['number']}  score={Fore.GREEN}{bt['score']:.4f}{Style.RESET_ALL}")

            # Affichage global avec trades réussis/échoués/totaux et sessions
            print(f"    {Fore.CYAN}[GLOBAL - Trades uniques]{Style.RESET_ALL}")
            print(f"    TR WR {Fore.GREEN}{bt['wr_t']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_t'] + bt['fail_t']}{Style.RESET_ALL} (sessions: {bt['sess_t']})")

            print(f"    V1 WR {Fore.GREEN}{bt['wr_v']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v'] + bt['fail_v']}{Style.RESET_ALL} (sessions: {bt['sess_v']})")

            print(f"    V2 WR {Fore.GREEN}{bt['wr_v1']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v1']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v1']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v1']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v1'] + bt['fail_v1']}{Style.RESET_ALL} (sessions: {bt['sess_v1']})")

            # Affichage des écarts moyens
            print(f"    Gap WR moyen: {Fore.YELLOW}{bt['avg_gap_wr']:.2%}{Style.RESET_ALL} | "
                  f"Gap PCT moyen: {Fore.YELLOW}{bt['avg_gap_pct']:.2%}{Style.RESET_ALL}")

            # Affichage détaillé par catégorie de trades pour chaque dataset
            if bt['metrics_detail_train']:
                print_detailed_metrics("TRAIN", bt['metrics_detail_train'])

            if bt['metrics_detail_val']:
                print_detailed_metrics("VAL", bt['metrics_detail_val'])

            if bt['metrics_detail_val1']:
                print_detailed_metrics("VAL1", bt['metrics_detail_val1'])

            # Détail par condition (affichage original)
            print(f"\n    {Fore.CYAN}[Condition 1]{Style.RESET_ALL}")
            print(f"    TR WR {Fore.GREEN}{bt['wr_t1']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t1']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t1']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t1']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_t1'] + bt['fail_t1']}{Style.RESET_ALL} (sessions: {bt['sess_t1']})")

            print(f"    V1 WR {Fore.GREEN}{bt['wr_v1']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v1']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v1']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v1']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v1'] + bt['fail_v1']}{Style.RESET_ALL} (sessions: {bt['sess_v1']})")

            print(f"    V2 WR {Fore.GREEN}{bt['wr_v1_1']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v1_1']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v1_1']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v1_1']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v1_1'] + bt['fail_v1_1']}{Style.RESET_ALL} (sessions: {bt['sess_v1_1']})")

            print(f"\n    {Fore.CYAN}[Condition 2]{Style.RESET_ALL}")
            print(f"    TR WR {Fore.GREEN}{bt['wr_t2']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t2']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t2']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t2']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_t2'] + bt['fail_t2']}{Style.RESET_ALL} (sessions: {bt['sess_t2']})")

            print(f"    V1 WR {Fore.GREEN}{bt['wr_v2']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v2']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v2']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v2']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v2'] + bt['fail_v2']}{Style.RESET_ALL} (sessions: {bt['sess_v2']})")

            print(f"    V2 WR {Fore.GREEN}{bt['wr_v1_2']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v1_2']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v1_2']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v1_2']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v1_2'] + bt['fail_v1_2']}{Style.RESET_ALL} (sessions: {bt['sess_v1_2']})")

            print(f"\n    {Fore.CYAN}[Condition 3]{Style.RESET_ALL}")
            print(f"    TR WR {Fore.GREEN}{bt['wr_t3']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_t3']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_t3']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_t3']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_t3'] + bt['fail_t3']}{Style.RESET_ALL} (sessions: {bt['sess_t3']})")

            print(f"    V1 WR {Fore.GREEN}{bt['wr_v3']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v3']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v3']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v3']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v3'] + bt['fail_v3']}{Style.RESET_ALL} (sessions: {bt['sess_v3']})")

            print(f"    V2 WR {Fore.GREEN}{bt['wr_v1_3']:.2%}{Style.RESET_ALL} | "
                  f"pct {bt['pct_v1_3']:.2%} | "
                  f"✓{Fore.GREEN}{bt['suc_v1_3']}{Style.RESET_ALL} "
                  f"✗{Fore.RED}{bt['fail_v1_3']}{Style.RESET_ALL} "
                  f"Total={Fore.CYAN}{bt['suc_v1_3'] + bt['fail_v1_3']}{Style.RESET_ALL} (sessions: {bt['sess_v1_3']})")

            print(f"\n    params ➜ {Fore.MAGENTA}{bt['params']}{Style.RESET_ALL}\n")

            last_best_shown = best_trial["number"]

    # Ces deux lignes doivent être alignées avec la définition de la boucle for,
    # pas avec le contenu de la boucle
    print(f"\n{Fore.YELLOW}🔚  Fin des essais Optuna.{Style.RESET_ALL}")
    calculate_test_metrics(study.best_params)


if __name__ == "__main__":
    main()