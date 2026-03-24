import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def train_model(input_path, model_path, params_path=None):
    df = pd.read_csv(input_path)
    
    X = df.drop('price', axis=1)
    y = df['price']
    
    print(f"Размер данных: {X.shape}")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Оценка
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'rmse_train': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'rmse_test': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'mae_train': float(mean_absolute_error(y_train, y_pred_train)),
        'mae_test': float(mean_absolute_error(y_test, y_pred_test)),
        'r2_train': float(r2_score(y_train, y_pred_train)),
        'r2_test': float(r2_score(y_test, y_pred_test))
    }
    
    print(f"\n Метрики:")
    print(f"   RMSE train: {metrics['rmse_train']:.2f}")
    print(f"   RMSE test: {metrics['rmse_test']:.2f}")
    print(f"   R² test: {metrics['r2_test']:.4f}")
    
    # Сохраняем модель
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    # Сохраняем метрики
    if params_path:
        with open(params_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return model, metrics

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/features.csv"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/model.pkl"
    metrics_path = sys.argv[3] if len(sys.argv) > 3 else "metrics/metrics.json"
    train_model(input_path, model_path, metrics_path)