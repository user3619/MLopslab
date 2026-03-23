import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import mlflow

def train_model():
    df = pd.read_csv('data/diamonds_processed.csv')

    X = df.drop('price', axis=1)
    y = df['price']

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
    

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/diamonds_model.pkl')
    

    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_name="random_forest_diamonds"):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 12)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "diamonds_model")
        
        # Сохраняем модель
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="diamonds_model"
        )
        
        # Получаем URI модели
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/diamonds_model"
        
        # Сохраняем URI в файл
        with open('model_uri.txt', 'w') as f:
            f.write(model_uri)

 
    model_path = 'models/diamonds_model.pkl'

    print(model_path)

    return model

if __name__ == "__main__":
    train_model()