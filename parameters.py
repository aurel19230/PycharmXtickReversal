from definition import *
import os
import numpy as np

def get_path():
    from func_standard import detect_environment

    ENV = detect_environment()
    if ENV == 'pycharm':
        base_results_path = r"C:/Users/aulac/OneDrive/Documents/Trading/PyCharmProject/MLStrategy/data_preprocessing/results_optim/"

        FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorized_oldclean.csv"
        FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
        # FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorizedScaledWithNanVal.csv"
        # FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
        FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
        FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
        FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnly220LastFullSession_OnlyShort_feat_winsorized.csv"
        FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnly220LastFullSession_OnlyShort_feat_winsorized.csv"
        FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnly900LastFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
        DIRECTORY_PATH = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\4_0_5TP_1SL_newBB\merge"
        FILE_PATH = os.path.join(DIRECTORY_PATH, FILE_NAME_)
    else:  # collab
        base_results_path = r"/content/drive/MyDrive/Colab_Notebooks/xtickReversal/results_optim/"
    return DIRECTORY_PATH,FILE_PATH,base_results_path

def get_model_param_range(model_type):
    """Retourne la configuration des paramètres selon le type de modèle"""
    if model_type == modeleType.XGB:
        return {
            'num_boost_round': {'min': 400, 'max': 1100},
            'max_depth': {'min': 3, 'max': 9},
            'learning_rate': {'min': 0.0009, 'max': 0.01, 'log': True},
            'min_child_weight': {'min': 1, 'max': 5},
            'subsample': {'min': 0.45, 'max': 0.80},
            'colsample_bytree': {'min': 0.6, 'max': 0.80},
            'colsample_bylevel': {'min': 0.4, 'max': 0.6},
            'colsample_bynode': {'min': 0.65, 'max': 0.95},
            'gamma': {'min': 5, 'max': 13},
            'reg_alpha': {'min': 1, 'max': 2, 'log': True},
            'reg_lambda': {'min': 0.1, 'max': 0.9, 'log': True}
        }
    elif model_type == modeleType.CATBOOT:
        return {
            'iterations': {'min': 500, 'max': 1500},
            'depth': {'min': 6, 'max': 12},
            'learning_rate': {'min': 0.001, 'max': 0.03, 'log': True},

            # Paramètres de sous-échantillonnage
            'min_child_samples': {'min': 10, 'max': 50},
            'subsample': {'min': 0.6, 'max': 0.8},
            'colsample_ratio': {'min': 0.6, 'max': 0.9},

            # Paramètres de régularisation
            'l2_leaf_reg': {'min': 2.0, 'max': 15.0, 'log': True},
            'random_strength': {'min': 0.05, 'max': 0.5, 'log': True},
            'bagging_temperature': {'min': 0.2, 'max': 0.8},

            # Paramètres temporels
            'fold_permutation_block': {'min': 10, 'max': 50},
            'leaf_estimation_iterations': {'min': 5, 'max': 15},

            # Paramètres catégoriels
            'leaf_estimation_method': {'values': ['Newton']},
            'grow_policy': {'values': ['Depthwise']},
            'bootstrap_type': {'values': ['Bayesian']}
        }

def get_weight_param():

    weight_param = {
        'threshold': {'min': 0.51, 'max': 0.67},  # total_trades_val = tp + fp
        'w_p': {'min': 0.8, 'max': 2},  # poid pour la class 1 dans objective
        'w_n': {'min': 0.7, 'max': 1.5},  # poid pour la class 0 dans objective
        'profit_per_tp': {'min': 1.25, 'max': 1.25},  # fixe, dépend des profits par trade
        'loss_per_fp': {'min': -1.25, 'max': -1.25},  # fixe, dépend des pertes par trade
        'penalty_per_fn': {'min': 0, 'max': 0},
        'weight_split': {'min': 0.65, 'max': 0.65},
        'nb_split_weight': {'min': 0, 'max': 0},  # si 0, pas d'utilisation de weight_split
        'std_penalty_factor': {'min': 0, 'max': 0}
    }
    return weight_param

def get_config():
    # Configuration
    config = {
        'target_directory': "",
        'xgb_metric_custom': xgb_metric.XGB_METRIC_CUSTOM_METRIC_PROFITBASED,
        'device_': 'cuda',
        'n_trials_optuna': 5000,
        'nb_split_tscv_': 6,
        'test_size_ratio': 0.2,
        'nanvalue_to_newval_': np.nan,
        'random_state_seed': 35,
        'early_stopping_rounds': 60,
        # 'use_shapeImportance_file': r"C:\Users\aulac\OneDrive\Documents\Trading\PyCharmProject\MLStrategy\data_preprocessing\shap_dependencies_results\shap_values_Training_Set.csv",
        'results_directory': "",
        'cv_method': cv_config.TIME_SERIE_SPLIT,
        # cv_config.K_FOLD, #,  TIME_SERIE_SPLIT TIMESERIES_SPLIT_BY_ID TIME_SERIE_SPLIT_NON_ANCHORED
        'non_acnhored_val_ratio': 0.5,
        'weightPareto_pnl_val': 0.4,
        'weightPareto_pnl_diff': 0.6,
        'use_of_rfe_in_optuna': rfe_param.NO_RFE,
        'min_features_if_RFE_AUTO': 3,
        'optuna_objective_type': optuna_doubleMetrics.DISABLE,  # USE_DIST_TO_IDEAL,
        'use_optuna_constraints_func': True,
        'constraint_min_trades_threshold_by_Fold': 25,
        'constraint_ecart_train_val': 0.3,
        'constraint_winrates_by_fold': 0.53,
        'use_imbalance_penalty': False,
        'is_log_enabled': False,
        'enable_vif_corr_mi': True,
        'vif_threshold': 15,
        'corr_threshold': 1,
        'mi_threshold': 0.001,
        'scaler_choice': scalerChoice.SCALER_ROBUST,  # ou  ou SCALER_DISABLE SCALER_ROBUST SCALER_STANDARD
        'model_type': modeleType.XGB
    }
    return config