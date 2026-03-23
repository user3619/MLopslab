import pandas as pd
import os

def download_diamonds():
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv')
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/diamonds_raw.csv', index=False)
    
    print(f"Загружено {len(df)} записей")
    print(f"Колонки: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    download_diamonds()