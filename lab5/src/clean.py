import pandas as pd
import os

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    print(f"Исходный размер: {df.shape}")
    
    # Удаление дубликатов
    initial_count = len(df)
    df = df.drop_duplicates()
    print(f"Удалено дубликатов: {initial_count - len(df)}")
    
    # Удаление пропусков
    initial_count = len(df)
    df = df.dropna()
    print(f"Удалено строк с пропусками: {initial_count - len(df)}")
    
    # Удаление выбросов
    # Цена не может быть отрицательной
    df = df[df['price'] > 0]
    
    # Ограничение веса
    df = df[df['carat'] < 5]
    
    print(f"Размер после очистки: {df.shape}")
    
    # Сохранение
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df

if __name__ == "__main__":
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/diamonds.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/cleaned.csv"
    clean_data(input_path, output_path)