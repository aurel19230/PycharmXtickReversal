import xgboost as xgb
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cupy as cp

class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, learning_rate=None, subsample=None, colsample_bytree=None, n_estimators=None, **params):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_estimators = n_estimators
        self.params = {
            'objective': 'multi:softprob',
            'num_class': 7,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'device': 'gpu',
            'random_state': 42,
            **params
        }

    def set_params(self, **params):
        if 'max_depth' in params:
            self.max_depth = params['max_depth']
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        if 'subsample' in params:
            self.subsample = params['subsample']
        if 'colsample_bytree' in params:
            self.colsample_bytree = params['colsample_bytree']
        if 'n_estimators' in params:
            self.n_estimators = params['n_estimators']
        self.params.update(params)
        return self

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params.update({
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'n_estimators': self.n_estimators
        })
        return params

    def fit(self, X, y):
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'objective': 'multi:softprob',
            'num_class': 7,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'device': 'gpu',
            'random_state': 42
        }
        params.update(self.params)

        X = cp.asnumpy(X)  # Convert X from cupy array to numpy array
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(params, dtrain, num_boost_round=self.n_estimators)
        return self

    def predict(self, X):
        X = cp.array(X)  # Convertir X_test en array cupy

       # dtest = xgb.DMatrix(X)
        preds = self.model.predict(X)
        return np.argmax(preds, axis=1)  # Convert probabilities to class labels

# Charger le dataset
data = fetch_covtype()
X, y = data.data, data.target

# Ajuster les étiquettes pour qu'elles soient dans la plage [0, 6]
y = y - 1

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Définir l'espace de recherche des hyperparamètres
param_grid = {
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.6, 0.9],
    'colsample_bytree': [0.6, 0.8],
    'n_estimators': [200]
}

print("XGBClassifier")

# Configurer la recherche par grille
grid_search = GridSearchCV(estimator=XGBoostClassifier(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)


# Effectuer la recherche par grille
grid_search.fit(X_train, y_train)

# Afficher les meilleurs hyperparamètres
print("Meilleurs hyperparamètres : ", grid_search.best_params_)

# Faire des prédictions avec le meilleur modèle
best_model = grid_search.best_estimator_
preds = best_model.predict(X_test)

# Calculer les métriques
accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average=None, zero_division=0)
recall = recall_score(y_test, preds, average=None, zero_division=0)
f1 = f1_score(y_test, preds, average=None, zero_division=0)

# Afficher les métriques pour chaque classe
print(f'Test Accuracy: {accuracy:.4f}')
for cls in range(7):
    print(f'Test Precision for class {cls}: {precision[cls]:.4f}')
    print(f'Test Recall for class {cls}: {recall[cls]:.4f}')
    print(f'Test F1 Score for class {cls}: {f1[cls]:.4f}')