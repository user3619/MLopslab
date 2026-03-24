import pandas as pd
import numpy as np
import json
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(model_path, test_data_path, metrics_path):
    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)
    
    X = df.drop('price', axis=1)
    y = df['price']
    
    y_pred = model.predict(X)
    
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        'mae': float(mean_absolute_error(y, y_pred)),
        'r2': float(r2_score(y, y_pred))
    }
    
    print(f"\n Итоговые метрики на тесте:")
    print(f"   RMSE: {metrics['rmse']:.2f}")
    print(f"   MAE: {metrics['mae']:.2f}")
    print(f"   R²: {metrics['r2']:.4f}")
    
    # Сохраняем метрики
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/model.pkl"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/features.csv"
    metrics_path = sys.argv[3] if len(sys.argv) > 3 else "metrics/evaluation.json"
    evaluate(model_path, test_path, metrics_path)