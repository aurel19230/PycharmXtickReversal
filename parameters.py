from definition import *
import os
import numpy as np

def get_path():
    from func_standard import detect_environment
    FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorized_oldclean.csv"
    FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
    # FILE_NAME_ = "Step5_4_0_6TP_1SL_080919_141024_extractOnlyFullSession_OnlyShort_feat_winsorizedScaledWithNanVal.csv"
    # FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
    FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
    FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnlyFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
    FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnly220LastFullSession_OnlyShort_feat_winsorized.csv"
    FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnly220LastFullSession_OnlyShort_feat_winsorized.csv"
    FILE_NAME_ = "Step5_4_0_5TP_1SL_newBB_080919_281124_extractOnly900LastFullSession_OnlyShort_feat_winsorized_MorningasieEurope.csv"
    FILE_NAME_ = "Step5_5_0_5TP_1SL_150924_280225_bugFixTradeResult_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"
    FILE_NAME_ = "Step5_1erAu15Nov_311024_151124_bugFixTradeResult1_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"


    ENV = detect_environment()
    if ENV == 'pycharm':
        if platform.system() != "Darwin":
            base_results_path = r"C:/Users/aulac/OneDrive/Documents/Trading/PyCharmProject/MLStrategy/data_preprocessing/results_optim/"
            DIRECTORY_PATH = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\merge_I1_I2"
            DIRECTORY_PATH = r"C:\Users\aulac\OneDrive\Documents\Trading\VisualStudioProject\Sierra chart\xTickReversal\simu\5_0_5TP_1SL\UnerAu15dec22h\merge"

        else:
            base_results_path = "/Users/aurelienlachaud/Documents/trading_local/data_preprocessing/results_optim/"
            DIRECTORY_PATH ="/Users/aurelienlachaud/Documents/trading_local/"
            #DIRECTORY_PATH = "/Users/aurelienlachaud/Library/CloudStorage/OneDrive-Personal/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/simu/4_0_5TP_1SL_newBB/merge"
    else:  # collab
        DIRECTORY_PATH =r"/content/drive/MyDrive/testFile/"
        base_results_path = r"/content/drive/MyDrive/Colab_Notebooks/xtickReversal/results_optim/"
    FILE_PATH = os.path.join(DIRECTORY_PATH, FILE_NAME_)

    return DIRECTORY_PATH,FILE_PATH,base_results_path

def get_model_param_range(model_type):
    """Retourne la configuration des paramètres selon le type de modèle"""
    if model_type == modelType.XGB:
        return {
            # Nombre total d'arbres à construire
            'num_boost_round': {'min': 250, 'max': 1200},

            # Profondeur maximale de chaque arbre (contrôle la complexité)
            'max_depth': {'min': 2, 'max': 15},

            # Taux d'apprentissage (plus petit = plus robuste mais plus lent)
            'learning_rate': {'min': 0.001, 'max': 0.15, 'log': True},

            # Poids minimum nécessaire pour créer un nouveau nœud enfant
            'min_child_weight': {'min': 1, 'max': 10},

            # Fraction des observations utilisées pour construire chaque arbre
            'subsample': {'min': 0.5, 'max': 0.90},

            # Fraction des features utilisées pour construire chaque arbre
            'colsample_bytree': {'min': 0.5, 'max': 0.90},

            # Fraction des features utilisées à chaque niveau de l'arbre
            'colsample_bylevel': {'min': 0.5, 'max': 0.9},

            # Fraction des features utilisées pour chaque nœud
            'colsample_bynode': {'min': 0.5, 'max': 0.9},

            # Réduction minimale de perte requise pour faire une séparation
            'gamma': {'min': 1, 'max': 8},

            # Terme de régularisation L1 sur les poids
            'reg_alpha': {'min': 1, 'max': 20, 'log': True},

            # Terme de régularisation L2 sur les poids
            'reg_lambda': {'min': 0.1, 'max': 2, 'log': True},

            # Paramètres supplémentaires
           # 'max_leaves': {'min': 0, 'max': 8},  # Nombre maximum de feuilles dans l'arbre
            #'min_split_loss': {'min': 0, 'max': 10},  # Perte minimale pour faire un split
            #'grow_policy': {'values': ['depthwise', 'lossguide']},  # Stratégie de croissance de l'arbre
            #'tree_method': {'values': ['auto', 'exact', 'approx', 'hist']}  # Méthode de construction des arbres
        }
    elif model_type == modelType.LGBM:
        return {
            # Plus de feuilles par arbre -> arbres plus complexes -> log odds plus étendus
            'num_leaves': {'min': 40, 'max': 100},

            # Taux plus élevé = mises à jour plus agressives -> log odds plus extrêmes
            'learning_rate': {'min': 0.005, 'max': 0.2, 'log': True},

            # Plus petit nombre d'échantillons par feuille -> splits plus fins -> log odds plus dispersés
            'min_child_samples': {'min': 40, 'max': 180},

            # Plus grande fraction des données -> arbres plus similaires -> log odds moins moyennés
            'bagging_fraction': {'min': 0.5, 'max': 0.95},

            # Plus de features par arbre -> décisions plus tranchées -> log odds plus étendus
            'feature_fraction': {'min': 0.6, 'max': 0.95},

            # Plus de features par niveau -> splits plus discriminants -> log odds plus contrastés
            'feature_fraction_bynode': {'min': 0.65, 'max': 0.9},

            # Seuil de gain plus bas -> plus de splits autorisés -> log odds plus dispersés
            'min_split_gain': {'min': 0.8, 'max': 5},

            # Moins de régularisation L1 -> poids moins contraints -> log odds plus extrêmes
            'lambda_l1': {'min': 0.1, 'max':4.5, 'log': True},

            # Moins de régularisation L2 -> poids moins lissés -> log odds plus étendus
            'lambda_l2': {'min': 0.1, 'max': 3.5, 'log': True},

            # Bagging plus fréquent (valeurs plus petites) -> plus de moyennage -> log odds plus resserrés
            # Bagging moins fréquent (valeurs plus grandes) -> moins de moyennage -> log odds plus dispersés
            'bagging_freq': {'min': 1, 'max': 12}
        }



    elif model_type == modelType.CATBOOST:
        return {
            # Équivalent à XGB num_boost_round : Nombre total d'arbres
            'iterations': {'min': 500, 'max': 1500},

            # Équivalent à XGB max_depth : Profondeur maximale de chaque arbre
            'depth': {'min': 6, 'max': 15},

            # Équivalent à XGB learning_rate : Taux d'apprentissage
            'learning_rate': {'min': 0.007, 'max': 0.06, 'log': True},

            # Paramètres de sous-échantillonnage
            # Équivalent à XGB min_child_weight : Nombre minimum d'échantillons dans une feuille
            'min_child_samples': {'min': 10, 'max': 50},

            # Équivalent à XGB subsample : Fraction des observations pour chaque arbre
            'subsample': {'min': 0.6, 'max': 0.8},

            # Équivalent à XGB colsample_bytree : Fraction des features pour chaque arbre
            'colsample_ratio': {'min': 0.6, 'max': 0.9},

            # Paramètres de régularisation
            # Équivalent à XGB reg_lambda : Régularisation L2
            'l2_leaf_reg': {'min': 2.0, 'max': 15.0, 'log': True},

            # Spécifique à CatBoost : Force de la randomisation dans la construction des arbres
            'random_strength': {'min': 0.05, 'max': 0.5, 'log': True},

            # Spécifique à CatBoost : Contrôle l'intensité du bagging bayésien
            'bagging_temperature': {'min': 0.2, 'max': 0.8},

            # Paramètres temporels
            # Spécifique à CatBoost : Taille des blocs pour la permutation des données
            'fold_permutation_block': {'min': 10, 'max': 50},

            # Spécifique à CatBoost : Nombre d'itérations pour l'estimation des valeurs des feuilles
            'leaf_estimation_iterations': {'min': 5, 'max': 15},

            # Paramètres catégoriels
            # Spécifique à CatBoost : Méthode d'estimation des valeurs des feuilles
            'leaf_estimation_method': {'values': ['Newton', 'Gradient']},

            # Spécifique à CatBoost : Stratégie de croissance des arbres
            'grow_policy': {'values': ['SymmetricTree', 'Depthwise', 'Lossguide']},

            # Spécifique à CatBoost : Type de bootstrap utilisé
            'bootstrap_type': {'values': ['Bayesian', 'Bernoulli', 'MVS']},

            # Paramètres supplémentaires
            # Paramètres d'arrêt précoce
            'od_type': {'values': ['IncToDec', 'Iter']},
            'od_wait': {'min': 20, 'max': 50},

            # Paramètres de gestion des variables catégorielles
            'max_ctr_complexity': {'min': 1, 'max': 4},
            'ctr_leaf_count_limit': {'min': 10, 'max': 100}
        }

def get_weight_param():

    weight_param = {
        # Nombre d'itérations de boosting (équivalent à num_boost_round dans XGB)
        'num_boost_round': {'min': 400, 'max': 1200},
        'threshold': {'min': 0.45, 'max': 0.55},  # total_trades_val = tp + fp
        'w_p': {'min': 1, 'max': 1},  # car déja pris en compte dans le weigh des data
        'w_n': {'min': 1, 'max': 1},  # car déja pris en compte dans le weigh des data

        'penalty_per_fn': {'min': 0, 'max': 0},
        'weight_split': {'min': 0.65, 'max': 0.65},
        'nb_split_weight': {'min': 0, 'max': 0},  # si 0, pas d'utilisation de weight_split
        'std_penalty_factor': {'min': 0, 'max': 0}
    }
    return weight_param

def get_config():
    # Configuration
    config = {
        'boosting_type':'gbdt', #dart
        'target_directory': "",
        'device_': 'cpu',
        'n_trials_optuna': 100000,
        'nb_split_tscv_': 4,
        'test_size_ratio': 0.2,
        'nanvalue_to_newval_': np.nan,
        'random_state_seed': 35,
        'early_stopping_rounds': 150,
        #'profit_per_tp':1.25,
        #'loss_per_fp': -1.25,
        # 'use_shapeImportance_file': r"C:\Users\aulac\OneDrive\Documents\Trading\PyCharmProject\MLStrategy\data_preprocessing\shap_dependencies_results\shap_values_Training_Set.csv",
        'results_directory': "",

        'enable_script_CUSTOM_SESSIONS':False,
        'non_acnhored_val_ratio': 0.7,
        'weightPareto_pnl_val': 0.65,
        'weightPareto_pnl_diff': 0.35,
        'use_of_rfe_in_optuna': rfe_param.NO_RFE,
        'min_features_if_RFE_AUTO': 3,
        'optuna_objective_type': optuna_doubleMetrics.DISABLE,  # USE_DIST_TO_IDEAL, DISABLE
        'use_optuna_constraints_func': True,
        'config_constraint_min_trades_threshold_by_Fold': 25,
        'config_constraint_ratioWinrate_train_val': 25,
        'config_constraint_winrates_val_by_fold': 56.5,
        'use_imbalance_penalty': False, #pour prendre en compte les différence de trade entre les folds
        'is_log_enabled': False,
        'remove_inf_nan_afterFeaturesSelections': True,
        'auto_filtering_mode': AutoFilteringOptions.ENABLE_VIF_CORR_MI , #ENABLE_MRMR #DISPLAY_MODE_NOFILTERING ENABLE_VIF_CORR_MI ENABLE_FISHER
        'compute_vif':True,
        'vif_threshold': 15,
        'corr_threshold': 1.5,
        'mi_threshold': 0.01,
        #"fisher_score_threshold": 10.0,  # exemple de seuil Fisher (optionnel)
        "fisher_pvalue_threshold": 0.05,  # seuil classique p-value (optionnel)
        "fisher_top_n_features": 150,  # nombre maximal de features (optionnel) Prioritaire si activé
        #"mrmr_n_features": 30, # par défaut X.shape[1] ( quand commenté) ,  taille totale des features
        #"mrmr_score_threshold": 0.0, # par défaut -np.inf ( quand commenté) , ce qui signifie pas de seuil
        'use_pnl_theoric':True,
        'scaler_choice': scalerChoice.SCALER_STANDARD,  # ou  ou SCALER_DISABLE SCALER_ROBUST SCALER_STANDARD SCALER_ROBUST SCALER_STANDARD SCALER_MINMAX SCALER_MAXABS
        'cv_method': cv_config.TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVTRAIN,# TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVTRAIN,
        # TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVTRAIN TIME_SERIE_SPLIT
        # cv_config.K_FOLD, #,  TIME_SERIE_SPLIT TIMESERIES_SPLIT_BY_ID TIME_SERIE_SPLIT_NON_ANCHORED_AFTER_PREVVAL
        #'reinsert_nan_inf_afterScaling':False, ne fonctionne pas à date
        'model_type': modelType.LGBM, #XGB LGBM
        'custom_objective_lossFct': model_custom_objective.LGB_CUSTOM_OBJECTIVE_PROFITBASED  ,#LGB_CUSTOM_OBJECTIVE_PROFITBASED XGB_CUSTOM_OBJECTIVE_PROFITBASED LGB_CUSTOM_OBJECTIVE_CROSS_ENTROPY LGB_CUSTOM_OBJECTIVE_PROFITBASED
        'custom_metric_eval': model_custom_metric.LGB_CUSTOM_METRIC_PNL, #LGB_CUSTOM_METRIC_PNL XGB_CUSTOM_METRIC_PNL
         #'model_type': modelType.XGB, XGB_CUSTOM_METRIC_PNL
         #'custom_objective_lossFct': xgb_metric.XGB_METRIC_CUSTOM_METRIC_PROFITBASED, LGB_CUSTOM_METRIC_PNL

    }
    return config