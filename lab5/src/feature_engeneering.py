import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def create_features(input_path, output_path, preprocessor_path=None):
    """
    Генерация признаков и предобработка
    """
    print("🔄 Загрузка очищенных данных...")
    df = pd.read_csv(input_path)
    
    # Целевая переменная
    target = 'price'
    X = df.drop(target, axis=1)
    y = df[target]
    
    # Категориальные и числовые признаки
    cat_features = ['cut', 'color', 'clarity']
    num_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
    
    # Создание новых признаков
    print("➕ Создание новых признаков...")
    
    # Соотношение сторон
    X['aspect_ratio'] = X['x'] / X['y']
    
    # Объём (приблизительный)
    X['volume'] = X['x'] * X['y'] * X['z']
    
    # Площадь стола
    X['table_area'] = (X['table'] / 100) * (X['x'] * X['y'])
    
    # Соотношение глубины
    X['depth_ratio'] = X['depth'] / X['z']
    
    # Обновляем списки признаков
    num_features = num_features + ['aspect_ratio', 'volume', 'table_area', 'depth_ratio']
    
    # Предобработка
    print("⚙️ Предобработка признаков...")
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    
    # Сохраняем препроцессор
    if preprocessor_path:
        os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
        joblib.dump(preprocessor, preprocessor_path)
        print(f"✅ Препроцессор сохранён в {preprocessor_path}")
    
    # Создаём DataFrame с обработанными данными
    cat_columns = []
    for cat in cat_features:
        categories = preprocessor.named_transformers_['cat'].categories_[0][1:]
        cat_columns.extend([f"{cat}_{c}" for c in categories])
    
    all_columns = num_features + cat_columns
    X_processed_df = pd.DataFrame(X_processed, columns=all_columns)
    X_processed_df[target] = y.values
    
    # Сохраняем
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X_processed_df.to_csv(output_path, index=False)
    
    print(f" Признаки созданы. Размер: {X_processed_df.shape}")
    print(f"   Числовые признаки: {len(num_features)}")
    print(f"   Категориальные признаки: {len(cat_columns)}")
    print(f"   Всего: {X_processed_df.shape[1]}")
    
    return X_processed_df

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/cleaned.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/features.csv"
    preprocessor_path = sys.argv[3] if len(sys.argv) > 3 else "models/preprocessor.pkl"
    create_features(input_path, output_path, preprocessor_path)