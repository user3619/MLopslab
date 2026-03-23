import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def download_diamonds():
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv')
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/diamonds_raw.csv', index=False)
    
    print(f"Загружено {len(df)} записей")
    print(f"Колонки: {list(df.columns)}")
    
    return df


def preprocess_diamonds():
    # Загружаем данные
    df = pd.read_csv('data/diamonds_raw.csv')
    print(f"Исходный размер: {df.shape}")
    
    # Целевая переменная
    target = 'price'
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Категориальные и числовые признаки
    cat_features = ['cut', 'color', 'clarity']
    num_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
    
    # Создаём пайплайн предобработки
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
    ])
    
    # Применяем предобработку
    X_processed = preprocessor.fit_transform(X)
    
    # Сохраняем препроцессор
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    
    # Сохраняем обработанные данные
    processed_df = pd.DataFrame(X_processed)
    processed_df[target] = y.values
    processed_df.to_csv('data/diamonds_processed.csv', index=False)
    
    print(f"После предобработки: {X_processed.shape}")
    print(f"Количество признаков после One-Hot: {X_processed.shape[1]}")
    
    return X_processed, y

if __name__ == "__main__":
    download_diamonds()
    preprocess_diamonds()