# src/feature_engineering.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_features(input_path, output_path, preprocessor_path=None):
    """
    Генерация признаков и предобработка
    """
    print(" Загрузка очищенных данных...")
    df = pd.read_csv(input_path)
    
    # Целевая переменная
    target = 'price'
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Проверяем и заменяем нулевые значения в числовых колонках
    print(" Проверка нулевых значений...")
    for col in ['y', 'z']:
        if col in X.columns:
            zeros = (X[col] == 0).sum()
            if zeros > 0:
                print(f"   В колонке {col} найдено {zeros} нулевых значений. Заменяем на 0.001")
                X[col] = X[col].replace(0, 0.001)
    
    # Категориальные и числовые признаки
    cat_features = ['cut', 'color', 'clarity']
    num_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
    
    # Создание новых признаков (только числовые)
    print(" Создание новых признаков...")
    
    # Соотношение сторон (с защитой от деления на 0)
    X['aspect_ratio'] = X['x'] / X['y']
    X['aspect_ratio'] = X['aspect_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Объём (приблизительный)
    X['volume'] = X['x'] * X['y'] * X['z']
    
    # Площадь стола
    X['table_area'] = (X['table'] / 100) * (X['x'] * X['y'])
    
    # Соотношение глубины (с защитой от деления на 0)
    X['depth_ratio'] = X['depth'] / X['z']
    X['depth_ratio'] = X['depth_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Проверка на бесконечности ТОЛЬКО в числовых колонках
    print(" Проверка на бесконечности...")
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(X[col]).sum()
        if inf_count > 0:
            print(f"   В колонке {col} найдено {inf_count} бесконечных значений. Заменяем на 0")
            X[col] = X[col].replace([np.inf, -np.inf], 0)
    
    # Заполняем NaN (если остались) только в числовых
    X[numeric_cols] = X[numeric_cols].fillna(0)
    
    # Обновляем списки признаков
    new_features = ['aspect_ratio', 'volume', 'table_area', 'depth_ratio']
    num_features = num_features + new_features
    
    print(f" Числовые признаки: {num_features}")
    print(f" Категориальные признаки: {cat_features}")
    
    # Предобработка
    print(" Предобработка признаков...")
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    
    # Сохраняем препроцессор
    if preprocessor_path:
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
        print(f" Препроцессор сохранён в {preprocessor_path}")
    
    # Получаем имена колонок после One-Hot Encoding
    print(" Получение имён колонок...")
    
    # Имена числовых колонок
    all_columns = num_features.copy()
    
    # Имена категориальных колонок (после One-Hot)
    try:
        encoder = preprocessor.named_transformers_['cat']
        for i, cat in enumerate(cat_features):
            categories = encoder.categories_[i]
            # Если drop='first', пропускаем первую категорию
            start_idx = 1 if encoder.drop == 'first' else 0
            for category in categories[start_idx:]:
                all_columns.append(f"{cat}_{category}")
    except Exception as e:
        print(f"   Не удалось получить имена категорий: {e}")
        # Если не получилось, используем простые имена
        all_columns = [f"feature_{i}" for i in range(X_processed.shape[1])]
    
    print(f"   Всего колонок: {len(all_columns)}")
    print(f"   X_processed shape: {X_processed.shape}")
    
    # Проверяем соответствие размеров
    if X_processed.shape[1] != len(all_columns):
        print(f" ВНИМАНИЕ: Несоответствие размеров!")
        print(f"   X_processed имеет {X_processed.shape[1]} колонок")
        print(f"   all_columns имеет {len(all_columns)} колонок")
        
        # Если не совпадает, используем простые имена
        print("   Используем простые имена колонок...")
        all_columns = [f"feature_{i}" for i in range(X_processed.shape[1])]
    
    # Создаём DataFrame
    X_processed_df = pd.DataFrame(X_processed, columns=all_columns)
    X_processed_df[target] = y.values
    
    # Сохраняем
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_processed_df.to_csv(output_path, index=False)
    
    print(f" Признаки созданы. Размер: {X_processed_df.shape}")
    print(f"   Колонки: {list(X_processed_df.columns[:5])}...")
    
    return X_processed_df

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/cleaned.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/features.csv"
    preprocessor_path = sys.argv[3] if len(sys.argv) > 3 else "models/preprocessor.pkl"
    create_features(input_path, output_path, preprocessor_path)