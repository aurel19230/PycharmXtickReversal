"""
Optuna + VWAP-Reversal-Pro (full-candle optimisation)
====================================================

▪️ Le signal est calculé sur *toutes* les bougies d’un split (df_full)
▪️ Les métriques win-rate & % trades sont mesurées uniquement sur
   les bougies étiquetées 0/1 (df_lab)
▪️ TRAIN, VAL, VAL1, TEST sont chacun chargés avec cette double vue
▪️ Clavier : « & » → test immédiat | « ² » → stop propre puis test
"""

from __future__ import annotations

# ────────── Imports ────────────────────────────────────────────
from pathlib import Path
from typing import Tuple

import optuna
import pandas as pd
from colorama import Fore, Style
from pynput import keyboard

from func_standard import vwap_reversal_pro, metrics_vwap_premmium

# ────────── Config ─────────────────────────────────────────────
DIR = Path(r"C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/simu/5_0_5TP_6SL/merge")
CSV_TRAIN = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split1_01012024_01052024.csv"
CSV_TEST  = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split2_01052024_01102024.csv"
CSV_VAL1  = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split3_01102024_28022025.csv"
CSV_VAL   = DIR / "Step5_5_0_5TP_6SL_010124_150525_extractOnlyFullSession_OnlyShort_feat__split4_02032025_14052025.csv"

WINRATE_MIN   = 0.54
PCT_TRADE_MIN = 0.06
ALPHA         = 0.7     # poids win-rate dans le score
LAMBDA_WR     = 1.0     # pénalité écart WR
LAMBDA_PCT    = 0.0     # pénalité écart % trades
N_TRIALS      = 10_000
PRINT_EVERY   = 20
FAILED_PENALTY = -1e-3
RANDOM_SEED    = 42

# ────────── Key flags ─────────────────────────────────────────
STOP_OPTIMIZATION   = False  # touche ²
DF_TEST_CALCULATION = False  # touche &

def _on_press(key):
    global STOP_OPTIMIZATION, DF_TEST_CALCULATION
    if hasattr(key, 'char'):
        if key.char == '²':
            print("\n🛑  Stop requested (²)")
            STOP_OPTIMIZATION = True
        elif key.char == '&':
            print("\n🧪  Test requested (&) – will run TEST after this trial")
            DF_TEST_CALCULATION = True

keyboard.Listener(on_press=_on_press, daemon=True).start()

# ────────── Data loader ───────────────────────────────────────

def load_csv(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return (df_full, df_lab, nb_sessions)."""
    df_full = pd.read_csv(path, sep=';', encoding='ISO-8859-1',
                          parse_dates=['date'], low_memory=False)

    # Correction / comptage sessions
    df_full['SessionStartEnd'] = pd.to_numeric(df_full['SessionStartEnd'],
                                               errors='coerce')
    df_full.dropna(subset=['SessionStartEnd'], inplace=True)
    df_full['SessionStartEnd'] = df_full['SessionStartEnd'].astype(int)

    nb_start = (df_full['SessionStartEnd'] == 10).sum()
    nb_end   = (df_full['SessionStartEnd'] == 20).sum()
    nb_sessions = min(nb_start, nb_end)

    if nb_start != nb_end:
        print(f"{Fore.YELLOW}⚠️  Sessions mismatch {nb_start}/{nb_end} "
              f"in {Path(path).name}{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}✔ {nb_sessions} sessions in "
              f"{Path(path).name}{Style.RESET_ALL}")

    # session_id avant filtrage
    df_full['session_id'] = (df_full['SessionStartEnd'] == 10).cumsum().astype('int32')

    # df_lab : uniquement bougies 0/1
    df_lab = df_full[df_full['class_binaire'].isin([0, 1])].copy()
    #df_lab.reset_index(drop=True, inplace=True)

    return df_full, df_lab, nb_sessions

TR_FULL , TR_LAB , _ = load_csv(CSV_TRAIN)
VA_FULL , VA_LAB , _ = load_csv(CSV_VAL  )
VA1_FULL, VA1_LAB, _ = load_csv(CSV_VAL1 )
TE_FULL , TE_LAB , _ = load_csv(CSV_TEST )

for lbl, d_lab in zip(("TR","VA","VA1","TE"), (TR_LAB, VA_LAB, VA1_LAB, TE_LAB)):
    print(f"{lbl:<4} | rows={len(d_lab):,} | brute WR={(d_lab['class_binaire']==1).mean():.2%}")
print("────────────────────────────────────────────\n")

# ────────── Helpers ───────────────────────────────────────────

def perf(df_full: pd.DataFrame, df_lab: pd.DataFrame, params: dict) -> tuple[float, float]:
    sig_full, _ = vwap_reversal_pro(df_full, **params)
    sig_lab = sig_full.loc[df_lab.index]
    return metrics_vwap_premmium(df_lab, sig_lab)

best_trial: dict = {"score": -1}

# ────────── Objective ─────────────────────────────────────────

def objective(trial: optuna.trial.Trial) -> float:
    global best_trial

    params = dict(
        lookback      = trial.suggest_int   ("lookback"     , 25, 30),
        momentum      = trial.suggest_int   ("momentum"     , 6, 10),
        z_window      = trial.suggest_int   ("z_window"     , 36, 44),
        atr_period    = trial.suggest_int   ("atr_period"   , 15, 25),
        atr_mult      = trial.suggest_float ("atr_mult"     , 2, 3),
        ema_filter    = trial.suggest_int   ("ema_filter"   , 35, 70),
        vol_lookback  = trial.suggest_int   ("vol_lookback" , 5, 10),
        vol_ratio_min = trial.suggest_float ("vol_ratio_min", 0.1, 0.5),
    )

    wr_t , pct_t  = perf(TR_FULL , TR_LAB , params)
    wr_v , pct_v  = perf(VA_FULL , VA_LAB , params)
    wr_v1, pct_v1 = perf(VA1_FULL, VA1_LAB, params)

    print(f"{trial.number:>6} | TR {wr_t:6.2%}/{pct_t:6.2%} | "
          f"VA {wr_v:6.2%}/{pct_v:6.2%} | VA1 {wr_v1:6.2%}/{pct_v1:6.2%}", end=' ')

    # contraintes hard
    if any([
        wr_t  < WINRATE_MIN, pct_t  < PCT_TRADE_MIN,
        wr_v  < WINRATE_MIN, pct_v  < PCT_TRADE_MIN,
        wr_v1 < WINRATE_MIN, pct_v1 < PCT_TRADE_MIN]):
        print("✘ KO")
        return FAILED_PENALTY

    gap_wr  = max(abs(wr_t-wr_v), abs(wr_t-wr_v1), abs(wr_v-wr_v1))
    gap_pct = max(abs(pct_t-pct_v), abs(pct_t-pct_v1), abs(pct_v-pct_v1))

    avg_wr  = (wr_t + wr_v + wr_v1) / 3
    avg_pct = (pct_t + pct_v + pct_v1) / 3

    score = ALPHA*avg_wr + (1-ALPHA)*avg_pct - LAMBDA_WR*gap_wr - LAMBDA_PCT*gap_pct
    print(f"✔ {score:.4f} ΔWR={gap_wr:.2%}")

    if score > best_trial.get("score", -1):
        best_trial = dict(score=score, number=trial.number,
                          wr_t=wr_t, pct_t=pct_t,
                          wr_v=wr_v, pct_v=pct_v,
                          wr_v1=wr_v1, pct_v1=pct_v1,
                          params=params)
    return score

# ────────── Optuna run ────────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))

def cb_stop(study, trial):
    if STOP_OPTIMIZATION:
        raise optuna.exceptions.TrialPruned("manual stop")

for i in range(N_TRIALS):
    if STOP_OPTIMIZATION:
        break
    study.optimize(objective, n_trials=1, callbacks=[cb_stop], show_progress_bar=False)

    # ➜ Affichage BEST toutes PRINT_EVERY itérations
    if i % PRINT_EVERY == PRINT_EVERY - 1 and best_trial.get("number") is not None:
        bt = best_trial
        print(f"\n*** BEST so far ▸ trial {bt['number']} score={bt['score']:.4f}")
        for lbl, wr, pct, lab_df in (
            ('TR' , bt['wr_t'] , bt['pct_t'] , TR_LAB ),
            ('VA' , bt['wr_v'] , bt['pct_v'] , VA_LAB ),
            ('VA1', bt['wr_v1'], bt['pct_v1'], VA1_LAB)):
            n_tot   = len(lab_df)
            n_trade = int(pct * n_tot)
            print(f"  {lbl:<3} | rows {n_tot:,} | trades {n_trade:,} ({pct:.2%}) | WR {wr:.2%}")
        print("      params →", bt['params'], "\n")
        # Ajoutez cette ligne pour déclencher le test après l'affichage PRINT_EVERY
        DF_TEST_CALCULATION = True
    # ➜ Test à la volée (&) avec *meilleurs* paramètres
    # --- dans le bloc « test à la volée (&) » -----------------------
    # ➜ Test à la volée (&) avec *meilleurs* paramètres
    if DF_TEST_CALCULATION:
        DF_TEST_CALCULATION = False
        best_params = best_trial.get('params', study.best_params)

        # Afficher les paramètres utilisés
        print("🧪 TEST with BEST params:")
        print("🧪 Parameters used:")
        for key, value in best_params.items():
            print(f"🧪    {key}: {value}")

        sig_full, _ = vwap_reversal_pro(TE_FULL, **best_params)
        wr, pct = metrics_vwap_premmium(TE_LAB, sig_full.loc[TE_LAB.index])

        n_tot = len(TE_LAB)
        n_trade = int(pct * n_tot)

        print(f"🧪 TEST now | rows {n_tot:,} | trades {n_trade:,} ({pct:.2%}) | WR {wr:.2%}\n")

print("\n🔚 Optuna finished")

# ────────── Hold-out TEST final ───────────────────────────────
if best_trial.get("params"):
    best_params = best_trial['params']
else:
    best_params = study.best_params  # fallback (rare)

sig_full, _ = vwap_reversal_pro(TE_FULL, **best_params)
wr, pct = metrics_vwap_premmium(TE_LAB, sig_full.loc[TE_LAB.index])
n_tot, n_trade = len(TE_LAB), int(pct * n_tot)
print("========= HOLD-OUT TEST =========")
print(f"rows {n_tot:,} | trades {n_trade:,} ({pct:.2%}) | WR {wr:.2%}")
print("params →", best_params)
