"""
Optuna + VWAP Reversal Pro
==========================

・ Affiche les métriques TRAIN / VAL pour chaque trial
・ Toutes les 50 trials : récap du meilleur score + paramètres
・ Touche « & » : calcule la perf sur TEST sans stopper Optuna
・ Touche « ² » : stoppe Optuna après la trial courante puis calcule TEST
"""

# ─────────────────── CONFIG ───────────────────────────────────────
DIR = R"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject" \
      R"\Sierra chart\xTickReversal\simu\5_0_5TP_6SL\merge"

CSV_TRAIN = DIR + R"\Step5_5_0_5TP_6SL_020924_150525_extractOnlyFullSession_OnlyShort_feat__split1_02092024_10032025.csv"
CSV_VAL = DIR + R"\Step5_5_0_5TP_6SL_020924_150525_extractOnlyFullSession_OnlyShort_feat__split2_10032025_14052025.csv"
# Assurez-vous d'avoir un fichier TEST distinct, sinon utilisez VAL
CSV_TEST = DIR + R"\Step5_5_0_5TP_6SL_020924_150525_extractOnlyFullSession_OnlyShort_feat__split2_10032025_14052025.csv"

WINRATE_MIN = 0.532
PCT_TRADE_MIN = 0.07
ALPHA = 0.7  # poids win-rate
LAMBDA_WR = 0  # pénalité continue gap win-rate (utilisé seulement si GAP_SEUIL=0)
LAMBDA_PCT = 0  # pénalité continue gap % trades
GAP_SEUIL = 0.014  # seuil maximal pour le gap de win-rate (0 pour désactiver)
N_TRIALS = 10_000
PRINT_EVERY = 50
FAILED_PENALTY = -0.001
RANDOM_SEED = 42

# ─────────────────── IMPORTS ──────────────────────────────────────
import pandas as pd, numpy as np, optuna
from ta.volatility import AverageTrueRange
from pynput import keyboard  # gestion clavier asynchrone

# ───────────── FLAGS CLAVIER ─────────────────────────────────────
STOP_OPTIMIZATION = False  # touche ²
DF_TEST_CALCULATION = False  # touche &


def on_press(key):
    global STOP_OPTIMIZATION, DF_TEST_CALCULATION
    if hasattr(key, 'char'):
        if key.char == '²':
            print("\n🛑  Stop requested (²)")
            STOP_OPTIMIZATION = True
        elif key.char == '&':
            print("\n🧪  Test requested (&) – will run TEST after this trial")
            DF_TEST_CALCULATION = True


keyboard.Listener(on_press=on_press, daemon=True).start()


# ─────────────────── DATA LOAD ───────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="iso-8859-1",
                     parse_dates=["date"])
    if "Volume" in df.columns and "volume" not in df.columns:
        df.rename(columns={"Volume": "volume"}, inplace=True)
    need = {"close", "high", "low", "VWAP", "class_binaire", "SessionStartEnd"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{path} ➜ colonnes manquantes : {miss}")
    df = df[df["class_binaire"].isin([0, 1])].copy()
    df.reset_index(drop=True, inplace=True)
    df["session_id"] = (df["SessionStartEnd"] == 10).cumsum().astype("int32")
    return df


TRAIN, VAL, TEST = map(load_csv, (CSV_TRAIN, CSV_VAL, CSV_TEST))

for lbl, d in zip(("TRAIN", "VAL", "TEST"), (TRAIN, VAL, TEST)):
    print(f"{lbl:<5} | lignes={len(d):,}  winrate brut={(d['class_binaire'] == 1).mean():.2%}")
print("============================================================\n")


# ────────── INDICATEUR VWAP REVERSAL PRO ─────────────────────────
def vwap_reversal_pro(df, *, lookback, momentum,
                      z_window, atr_period, atr_mult,
                      ema_filter, vol_lookback, vol_ratio_max):
    w = df.copy()
    w["distance"] = w["close"] - w["VWAP"]
    g = w.groupby("session_id", group_keys=False)

    def _z(s):
        m = s.rolling(z_window, z_window).mean()
        sd = s.rolling(z_window, z_window).std()
        return ((s - m) / sd).replace([np.inf, -np.inf], 0).fillna(0)

    w["z_dist"] = g["distance"].transform(_z)

    w["speed"] = g["z_dist"].transform(lambda s: s.diff(lookback)).fillna(0)
    w["mom"] = g["speed"].transform(lambda s: s.diff(momentum)).fillna(0)

    atr_parts = []
    for _, blk in g:
        atr = AverageTrueRange(blk["high"], blk["low"], blk["close"],
                               atr_period, fillna=True).average_true_range()
        atr.index = blk.index
        atr_parts.append(atr)
    w["atr"] = pd.concat(atr_parts).sort_index()
    w["dyn_th"] = w["atr"] * atr_mult

    w["ema"] = g["close"].transform(
        lambda s: s.ewm(span=ema_filter, adjust=False).mean())
    w["trend_ok"] = g["ema"].transform(lambda s: s.diff() < 0)

    if "volume" in w.columns:
        w["vol_ma"] = g["volume"].transform(
            lambda s: s.rolling(vol_lookback, vol_lookback).mean())
        w["vol_ok"] = (w["volume"] / w["vol_ma"]) < vol_ratio_max
        w["vol_ok"] = w["vol_ok"].fillna(False)
    else:
        w["vol_ok"] = True

    return (
            (w["z_dist"] > 0) &
            (w["speed"] > 0) &
            ((w["mom"] < -w["dyn_th"]) | w["trend_ok"]) &
            w["vol_ok"]
    )


# ────────── MÉTRIQUES ─────────────────────────────────────────────
def metrics(df, mask):
    sub = df[mask.values]
    win_rate = (sub["class_binaire"] == 1).mean() if not sub.empty else 0.0
    pct_trades = len(sub) / len(df)

    # Calcul du nombre de trades réussis et échoués
    success = (sub["class_binaire"] == 1).sum() if not sub.empty else 0
    failed = len(sub) - success if not sub.empty else 0

    return (win_rate, pct_trades, success, failed)


# ────────── FUNCTION POUR CALCULER TEST ────────────────────────────
def calculate_test_metrics(study):
    print("\n🧮  Calcul sur DATASET TEST\n")
    sig_test = vwap_reversal_pro(TEST, **study.best_params)
    wr_test, pct_test, success_test, failed_test = metrics(TEST, sig_test)
    print("========= HOLD-OUT TEST =========")
    print(f"Best params : {study.best_params}")
    print(f"Win-rate              : {wr_test:.2%}")
    print(f"% échantillons tradés : {pct_test:.2%}")
    print(f"Trades réussis        : {success_test}")
    print(f"Trades échoués        : {failed_test}")
    print(f"Total trades          : {success_test + failed_test}")
    print("✅  VALIDE" if (wr_test >= WINRATE_MIN and pct_test >= PCT_TRADE_MIN)
          else "❌  REJET")
    return wr_test, pct_test, success_test, failed_test


# ────────── OBJECTIVE OPTUNA ─────────────────────────────────────
best_trial = {"score": -1}


def objective(trial):
    params = dict(
        lookback=trial.suggest_int("lookback", 3, 10),
        momentum=trial.suggest_int("momentum", 1, 4),
        z_window=trial.suggest_int("z_window", 4, 20, step=2),
        atr_period=trial.suggest_int("atr_period", 4, 30),
        atr_mult=trial.suggest_float("atr_mult", 0.5, 1.9),
        ema_filter=trial.suggest_int("ema_filter", 10, 450, step=10),
        vol_lookback=trial.suggest_int("vol_lookback", 2, 10),
        vol_ratio_max=trial.suggest_float("vol_ratio_max", 0.9, 1.8),
    )

    wr_t, pct_t, success_t, failed_t = metrics(TRAIN, vwap_reversal_pro(TRAIN, **params))
    wr_v, pct_v, success_v, failed_v = metrics(VAL, vwap_reversal_pro(VAL, **params))

    # log toujours
    print(f"{trial.number:>6} | "
          f"TR {wr_t:6.2%}/{pct_t:6.2%} | "
          f"VA {wr_v:6.2%}/{pct_v:6.2%}", end=' ')

    # contraintes minima
    if (wr_t < WINRATE_MIN or pct_t < PCT_TRADE_MIN
            or wr_v < WINRATE_MIN or pct_v < PCT_TRADE_MIN):
        print("✘ seuils KO")
        return FAILED_PENALTY

    # calcul du gap win-rate
    gap_wr = abs(wr_t - wr_v)
    gap_pct = abs(pct_t - pct_v)

    # vérification du seuil de gap si activé
    if GAP_SEUIL > 0 and gap_wr > GAP_SEUIL:
        print(f"✘ gap={gap_wr:.2%} > seuil={GAP_SEUIL:.2%}")
        return FAILED_PENALTY

    # score avec pénalité continue (uniquement si GAP_SEUIL = 0)
    lambda_wr_effective = 0 if GAP_SEUIL > 0 else LAMBDA_WR

    score = (
            ALPHA * (wr_t + wr_v) / 2
            + (1 - ALPHA) * (pct_t + pct_v) / 2
            - lambda_wr_effective * gap_wr
            - LAMBDA_PCT * gap_pct
    )
    print(f"✔ score={score:.4f} gap={gap_wr:.2%}")

    if score > best_trial.get("score", -1):
        best_trial.update(score=score, number=trial.number,
                          wr_t=wr_t, pct_t=pct_t, success_t=success_t, failed_t=failed_t,
                          wr_v=wr_v, pct_v=pct_v, success_v=success_v, failed_v=failed_v,
                          params=params)

    # 7)  résumé toutes les PRINT_EVERY iterations
    if trial.number % PRINT_EVERY == 0 and best_trial["number"] is not None:
        bt = best_trial
        print(f"\n*** BEST so far ▸ trial {bt['number']}  "
              f"score={bt['score']:.4f} | "
              f"TR {bt['wr_t']:.2%}/{bt['pct_t']:.2%} (✓{bt['success_t']}/✗{bt['failed_t']}) | "
              f"VA {bt['wr_v']:.2%}/{bt['pct_v']:.2%} (✓{bt['success_v']}/✗{bt['failed_v']})\n"
              f"      params ➜ {bt['params']}\n")
    return score


# ────────── CALLBACK POUR ARRÊT (²) ET TEST (&) ───────────────────
def callback_optuna_stop(study, trial):
    global DF_TEST_CALCULATION

    # Pour le test avec & sans arrêter l'optimisation
    if DF_TEST_CALCULATION:
        print("\n🧪  Exécution du test sur TEST demandé via &")
        calculate_test_metrics(study)
        print("\n⏩ Reprise de l'optimisation...\n")
        # Réinitialiser le flag après avoir exécuté le test
        DF_TEST_CALCULATION = False

    # Pour l'arrêt complet avec ²
    if STOP_OPTIMIZATION:
        raise optuna.exceptions.TrialPruned("Stopped by user (²)")


# ────────── LANCEMENT OPTUNA ─────────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))

study.optimize(objective,
               n_trials=N_TRIALS,
               callbacks=[callback_optuna_stop],
               show_progress_bar=True)

print("\n🔚 Optuna finished.")

# ────────── TEST À LA FIN ────────────────────────────────────────
if STOP_OPTIMIZATION:
    calculate_test_metrics(study)