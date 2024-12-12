from enum import Enum

class cv_config(Enum):
    TIME_SERIE_SPLIT = 0
    TIME_SERIE_SPLIT_NON_ANCHORED = 1
    TIMESERIES_SPLIT_BY_ID = 2
    K_FOLD = 3
    K_FOLD_SHUFFLE = 4

class modeleType (Enum):
    XGB=0
    CATBOOT=1

class optuna_doubleMetrics(Enum):
    DISABLE = 0
    USE_DIST_TO_IDEAL = 1
    USE_WEIGHTED_AVG = 2



class xgb_metric(Enum):
    XGB_METRIC_ROCAUC = 1
    XGB_METRIC_AUCPR = 2
    XGB_METRIC_F1 = 4
    XGB_METRIC_PRECISION = 5
    XGB_METRIC_RECALL = 6
    XGB_METRIC_MCC = 7
    XGB_METRIC_YOUDEN_J = 8
    XGB_METRIC_SHARPE_RATIO = 9
    XGB_METRIC_CUSTOM_METRIC_PROFITBASED = 10
    XGB_METRIC_CUSTOM_METRIC_TP_FP = 11

class scalerChoice(Enum):
    SCALER_DISABLE = 0
    SCALER_STANDARD = 1
    SCALER_ROBUST = 2
    SCALER_MINMAX = 3  # Nouveau : échelle [0,1]
    SCALER_MAXABS = 4  # Nouveau : échelle [-1,1]


class ScalerMode(Enum):
    FIT_TRANSFORM = 0  # Pour l'entraînement : fit + transform
    TRANSFORM = 1
     # Pour le test : transform uniquement
