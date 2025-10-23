import pandas as pd
import numpy as np
import random, os

np.random.seed(42)
random.seed(42)

# Параметры датасета
n_records = 2000

# Генерируем данные
data = {
    'brand': np.random.choice(['Samsung', 'Apple', 'Xiaomi', 'OnePlus', 'Google', 'Huawei'], 
                             n_records, p=[0.25, 0.2, 0.18, 0.15, 0.12, 0.1]),
    'model_year': np.random.randint(2018, 2024, n_records),
    'ram_gb': np.random.choice([4, 6, 8, 12, 16], n_records, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
    'storage_gb': np.random.choice([64, 128, 256, 512], n_records, p=[0.2, 0.5, 0.25, 0.05]),
    'screen_size': np.round(np.random.normal(6.3, 0.3, n_records), 1),
    'battery_mah': np.random.randint(3000, 6000, n_records),
    'camera_mp': np.random.choice([12, 48, 64, 108, 200], n_records, p=[0.3, 0.4, 0.2, 0.08, 0.02]),
    'price_usd': np.random.normal(650, 250, n_records),
    'performance_score': np.random.normal(75, 15, n_records)
}

# Создаем DataFrame
df = pd.DataFrame(data)

# Добавляем некоторые реалистичные корреляции и аномалии
df.loc[df['brand'] == 'Apple', 'price_usd'] += 150
df.loc[df['brand'] == 'Samsung', 'price_usd'] += 50
df.loc[df['ram_gb'] >= 12, 'performance_score'] += 10
df.loc[df['storage_gb'] == 512, 'price_usd'] += 100

# Добавляем несколько выбросов
outlier_indices = random.sample(range(n_records), 8)
df.loc[outlier_indices[:4], 'price_usd'] *= 2.5  # Дорогие выбросы
df.loc[outlier_indices[4:], 'performance_score'] *= 0.5  # Низкая производительность

# Округляем числовые значения
df['price_usd'] = np.round(df['price_usd'], 2)
df['performance_score'] = np.round(df['performance_score'], 1)

# Убедимся, что нет отрицательных значений
df['price_usd'] = df['price_usd'].clip(lower=100)
df['performance_score'] = df['performance_score'].clip(lower=20)

# Сохраняем исходный датасет
df.to_csv(os.path.join('lab3','smartphones_performance.csv'), index=False)

print(f"Размер датасета: {df.shape}")
print("\nПервые 5 строк датасета:")
print(df.head())
print("\nОсновная информация о датасете:")
print(df.info())
print("\nОписательная статистика:")
print(df.describe())