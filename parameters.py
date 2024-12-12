from definition import *

def get_model_param_range(model_type):
    """Retourne la configuration des paramètres selon le type de modèle"""
    if model_type == modeleType.XGB:
        return {
            'num_boost_round': {'min': 600, 'max': 1000},
            'max_depth': {'min': 7, 'max': 9},
            'learning_rate': {'min': 0.001, 'max': 0.009, 'log': True},
            'min_child_weight': {'min': 1, 'max': 4},
            'subsample': {'min': 0.45, 'max': 0.75},
            'colsample_bytree': {'min': 0.6, 'max': 0.80},
            'colsample_bylevel': {'min': 0.4, 'max': 0.6},
            'colsample_bynode': {'min': 0.65, 'max': 0.95},
            'gamma': {'min': 5, 'max': 13},
            'reg_alpha': {'min': 1, 'max': 2, 'log': True},
            'reg_lambda': {'min': 0.1, 'max': 0.9, 'log': True}
        }
    elif model_type == modeleType.CATBOOT:
        return {
            'iterations': {'min': 500, 'max': 1100},
            'depth': {'min': 5, 'max': 10},
            'learning_rate': {'min': 0.001, 'max': 0.05, 'log': True},
            'min_child_samples': {'min': 5, 'max': 20},
            'subsample': {'min': 0.55, 'max': 0.75},
            'colsample_ratio': {'min': 0.50, 'max': 0.80},
            'l2_leaf_reg': {'min': 1.0, 'max': 10.0, 'log': True},
            'random_strength': {'min': 0.1, 'max': 1.0, 'log': True},
            'bagging_temperature': {'min': 0.0, 'max': 1.0},
            'fold_permutation_block': {'min': 1, 'max': 5},
            'leaf_estimation_iterations': {'min': 1, 'max': 10},
            'leaf_estimation_method': {'values': ['Newton']},
            'grow_policy': {'values': ['Depthwise']},
            'bootstrap_type': {'values': ['Bayesian', 'MVS']}
        }