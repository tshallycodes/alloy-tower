import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import shap
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE, '.env'))

mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', f'sqlite:///{os.path.join(BASE, "src", "mlflow.db")}')
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment('alloytower-predictions')
print(f'MLflow URI: {mlflow_uri}')

df = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'ft_eng.csv'))
print(f'Loaded: {df.shape}')

with open(os.path.join(BASE, 'models', 'feature_columns.json')) as f:
    feature_columns = json.load(f)
print(f'Features ({len(feature_columns)}): {feature_columns}')

X = df[feature_columns]
y = df['last_sale_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train: {X_train.shape}  Test: {X_test.shape}')


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        'R2':    round(float(r2_score(y_test, preds)), 4),
        'RMSE':  round(float(np.sqrt(mean_squared_error(y_test, preds))), 2),
        'MedAE': round(float(median_absolute_error(y_test, preds)), 2),
    }


def log_shap(explainer, X_sample, name):
    shap_vals = explainer.shap_values(X_sample)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_sample, show=False)
    plt.tight_layout()
    path = os.path.abspath(f'shap_summary_{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(path)
    os.remove(path)
    print(f'  SHAP saved: shap_summary_{name}.png')


# ── Random Forest ────────────────────────────────────────────────────
print('\n--- Training Random Forest ---')
param_dist_rf = {
    'n_estimators':      [200, 300, 500],
    'max_depth':         [None, 15, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'max_features':      ['sqrt', 'log2', 0.5],
}

with mlflow.start_run(run_name='RandomForest'):
    rs_rf = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_dist_rf, n_iter=50, cv=5, scoring='r2', random_state=42, n_jobs=-1
    )
    rs_rf.fit(X_train, y_train)
    best_rf = rs_rf.best_estimator_
    rf_metrics = evaluate(best_rf, X_test, y_test)
    mlflow.log_params(rs_rf.best_params_)
    mlflow.log_metrics(rf_metrics)
    X_shap_rf = X_test.sample(min(200, len(X_test)), random_state=42)
    log_shap(shap.TreeExplainer(best_rf), X_shap_rf, 'RandomForest')
    joblib.dump(best_rf, os.path.join(BASE, 'models', 'rf_model.pkl'))
    mlflow.sklearn.log_model(best_rf, name='rf_model')
    print(f'  Best params: {rs_rf.best_params_}')
    print(f'  Metrics: {rf_metrics}')

# ── XGBoost ──────────────────────────────────────────────────────────
print('\n--- Training XGBoost ---')
param_dist_xgb = {
    'n_estimators':     [200, 300, 500],
    'max_depth':        [3, 5, 7, 9],
    'learning_rate':    [0.01, 0.05, 0.1, 0.2],
    'subsample':        [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma':            [0, 0.1, 0.2],
    'reg_alpha':        [0, 0.1, 1.0],
    'reg_lambda':       [1, 1.5, 2],
}

with mlflow.start_run(run_name='XGBoost'):
    rs_xgb = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
        param_dist_xgb, n_iter=50, cv=5, scoring='r2', random_state=42, n_jobs=1
    )
    rs_xgb.fit(X_train, y_train)
    best_xgb = rs_xgb.best_estimator_
    xgb_metrics = evaluate(best_xgb, X_test, y_test)
    mlflow.log_params(rs_xgb.best_params_)
    mlflow.log_metrics(xgb_metrics)
    log_shap(shap.TreeExplainer(best_xgb), X_test, 'XGBoost')
    joblib.dump(best_xgb, os.path.join(BASE, 'models', 'gb_model.pkl'))
    mlflow.sklearn.log_model(best_xgb, name='gb_model')
    print(f'  Best params: {rs_xgb.best_params_}')
    print(f'  Metrics: {xgb_metrics}')

# ── Comparison ───────────────────────────────────────────────────────
results = pd.DataFrame({'RandomForest': rf_metrics, 'XGBoost': xgb_metrics}).T
results = results.sort_values('R2', ascending=False)
print('\n=== Model Comparison ===')
print(results.to_string())
print(f'\nBest: {results.index[0]}')
