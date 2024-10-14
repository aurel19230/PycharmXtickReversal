import xgboost as xgb
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt


def generate_data(n_samples=1000, n_features=100):
    np.random.seed(42)
    X = pd.DataFrame()

    for i in range(n_features):
        if i % 10 == 0:
            X[f'feature_{i}'] = np.random.normal(0, 1000, n_samples)  # Large scale
        elif i % 10 == 1:
            X[f'feature_{i}'] = np.random.normal(0, 0.01, n_samples)  # Small scale
        elif i % 10 == 2:
            X[f'feature_{i}'] = np.random.exponential(1, n_samples)
        elif i % 10 == 3:
            X[f'feature_{i}'] = np.random.uniform(-1, 1, n_samples)
        elif i % 10 == 4:
            X[f'feature_{i}'] = np.random.chisquare(5, n_samples)
        elif i % 10 == 5:
            X[f'feature_{i}'] = np.random.poisson(3, n_samples)
        elif i % 10 == 6:
            X[f'feature_{i}'] = np.random.beta(2, 5, n_samples)
        elif i % 10 == 7:
            X[f'feature_{i}'] = np.random.gamma(2, 2, n_samples)
        elif i % 10 == 8:
            X[f'feature_{i}'] = np.random.lognormal(0, 1, n_samples)
        else:
            X[f'feature_{i}'] = np.random.normal(0, 1, n_samples)

    y = np.zeros(n_samples)
    for i in range(0, n_features, 10):
        y += X[f'feature_{i}'] / 1000  # Large scale features
        y += X[f'feature_{i + 1}'] * 100  # Small scale features
        y += np.log1p(X[f'feature_{i + 2}'])
        y -= X[f'feature_{i + 3}'] ** 2
        y += np.sqrt(np.abs(X[f'feature_{i + 4}']))
        y += X[f'feature_{i + 5}'] / 10
        y += X[f'feature_{i + 6}'] * 5
        y -= np.sin(X[f'feature_{i + 7}'])
        y += np.log(np.abs(X[f'feature_{i + 8}']) + 1)
        y += X[f'feature_{i + 9}']

    y += np.random.normal(0, 0.1, n_samples)
    y = y > np.median(y)  # Convert to binary

    return X, y


def train_and_evaluate_shap(X, y, preprocessing=None):
    X_processed = X.copy()
    if preprocessing:
        X_processed = pd.DataFrame(preprocessing.fit_transform(X_processed), columns=X.columns)

    dtrain = xgb.DMatrix(X_processed, label=y)
    params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda'  # Utiliser CUDA pour GPU
    }

    model = xgb.train(params, dtrain, num_boost_round=100)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)

    feature_importance = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({'feature': X_processed.columns, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False)

    return importance_df, shap_values


# Génération des données
X, y = generate_data(n_samples=1000, n_features=100)

# Analyse sans prétraitement
importance_raw, shap_values_raw = train_and_evaluate_shap(X, y)
print("Top 20 features importance sans prétraitement (SHAP):")
print(importance_raw.head(20))

# Analyse avec StandardScaler
scaler = StandardScaler()
importance_scaled, shap_values_scaled = train_and_evaluate_shap(X, y, scaler)
print("\nTop 20 features importance avec StandardScaler (SHAP):")
print(importance_scaled.head(20))

# Analyse avec RobustScaler
robust_scaler = RobustScaler()
importance_robust, shap_values_robust = train_and_evaluate_shap(X, y, robust_scaler)
print("\nTop 20 features importance avec RobustScaler (SHAP):")
print(importance_robust.head(20))

# Visualisation
plt.figure(figsize=(14, 10))
shap.summary_plot(shap_values_raw, X, plot_type="bar", max_display=20, show=False)
plt.title("SHAP Feature Importance (Sans prétraitement)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
shap.summary_plot(shap_values_scaled, X, plot_type="bar", max_display=20, show=False)
plt.title("SHAP Feature Importance (Avec StandardScaler)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
shap.summa