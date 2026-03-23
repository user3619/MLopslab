import pandas as pd
import json

# Загружаем обработанные данные
df = pd.read_csv('data/diamonds_processed.csv')
X = df.drop('price', axis=1)

# Берём первый пример
sample = X.iloc[0].tolist()

# Создаём колонки от 0 до 22
columns = [str(i) for i in range(len(sample))]

print(f"Количество признаков: {len(sample)}")
print(f"Пример: {sample}")

# Сохраняем в JSON
with open('sample_request.json', 'w') as f:
    json.dump({
        'dataframe_split': {
            'columns': columns,
            'data': [sample]
        }
    }, f, indent=2)

print("sample_request.json создан")