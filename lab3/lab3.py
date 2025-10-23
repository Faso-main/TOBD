# Лабораторная работа №3
# «Основные статистические показатели»

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# Загрузка датасета
df = pd.read_csv(os.path.join('lab3','smartphones_performance.csv'))

# Выбираем числовую колонку для анализа (performance_score)
data = df['performance_score']


print("\n1. Основные статистические показатели")

# Среднее арифметическое
mean_value = np.mean(data)
print(f"Среднее арифметическое: {mean_value:.2f}")

# Медиана
median_value = np.median(data)
print(f"Медиана: {median_value:.2f}")

# Минимум и максимум
min_value = np.min(data)
max_value = np.max(data)
print(f"Минимум: {min_value:.2f}")
print(f"Максимум: {max_value:.2f}")

# Мода
mode_result = stats.mode(data)
print(f"Мода: {mode_result.mode:.2f}")

# Стандартное отклонение
std_dev = np.std(data, ddof=1)
print(f"Стандартное отклонение: {std_dev:.2f}")

# Дисперсия
variance = np.var(data, ddof=1)
print(f"Дисперсия: {variance:.2f}")

# Математическое ожидание
expected_value = np.mean(data)
print(f"Математическое ожидание: {expected_value:.2f}")

print("\n2. Проверка гипотезы о нормальном распределении")

# Тест Шапиро-Уилка
shapiro_stat, shapiro_p = stats.shapiro(data)
print(f"Тест Шапиро-Уилка:")
print(f"  Статистика: {shapiro_stat:.4f}")
print(f"  p-значение: {shapiro_p:.4f}")

# Тест Колмогорова-Смирнова
ks_stat, ks_p = stats.kstest(data, 'norm', args=(mean_value, std_dev))
print(f"Тест Колмогорова-Смирнова:")
print(f"  Статистика: {ks_stat:.4f}")
print(f"  p-значение: {ks_p:.4f}")

# Интерпретация результатов
alpha = 0.05
print(f"\nИнтерпретация (уровень значимости α = {alpha}):")

if shapiro_p > alpha: print("  По тесту Шапиро-Уилка: распределение не отличается от нормального")
else: print("  По тесту Шапиро-Уилка: распределение отличается от нормального")

if ks_p > alpha: print("  По тесту Колмогорова-Смирнова: распределение не отличается от нормального")
else:  print("  По тесту Колмогорова-Смирнова: распределение отличается от нормального")

print("\n3. Визуализация распределения данных")

# Создаем графики
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Гистограмма с кривой нормального распределения
axes[0, 0].hist(data, bins=15, density=True, alpha=0.7, color='lightblue', edgecolor='black')
xmin, xmax = axes[0, 0].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean_value, std_dev)
axes[0, 0].plot(x, p, 'k', linewidth=2, label='Нормальное распределение')
axes[0, 0].axvline(mean_value, color='red', linestyle='--', label=f'Среднее = {mean_value:.1f}')
axes[0, 0].set_title('Гистограмма распределения')
axes[0, 0].set_xlabel('Производительность (баллы)')
axes[0, 0].set_ylabel('Плотность вероятности')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Box plot
axes[0, 1].boxplot(data, vert=False)
axes[0, 1].set_title('Box plot')
axes[0, 1].set_xlabel('Производительность (баллы)')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(data, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q plot')
axes[1, 0].grid(True, alpha=0.3)

# Эмпирическая функция распределения
from statsmodels.distributions.empirical_distribution import ECDF
ecdf = ECDF(data)
x = np.linspace(min_value, max_value, 100)
axes[1, 1].plot(x, ecdf(x), label='Эмпирическая ФР')
axes[1, 1].plot(x, stats.norm.cdf(x, mean_value, std_dev), 'r-', label='Теоретическая нормальная ФР')
axes[1, 1].set_title('Эмпирическая функция распределения')
axes[1, 1].set_xlabel('Производительность (баллы)')
axes[1, 1].set_ylabel('Вероятность')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
print("\n4. Сводная таблица результатов")


summary_data = {
    'Показатель': [
        'Среднее арифметическое',
        'Медиана', 
        'Стандартное отклонение',
        'Дисперсия',
        'Минимум',
        'Максимум',
        'Мода',
        'Объем выборки'
    ],
    'Значение': [
        f"{mean_value:.2f}",
        f"{median_value:.2f}", 
        f"{std_dev:.2f}",
        f"{variance:.2f}",
        f"{min_value:.2f}",
        f"{max_value:.2f}",
        f"{mode_result.mode:.2f}",
        f"{len(data)}"
    ]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))


print("Распределение показателей производительности смартфонов:")
print(f"- Среднее значение: {mean_value:.1f} баллов")
print(f"- Стандартное отклонение: {std_dev:.1f} баллов")

if shapiro_p > alpha and ks_p > alpha: print("- Распределение соответствует нормальному")
else: print("- Распределение не соответствует нормальному")

# Анализ выбросов
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = data[(data < lower_bound) | (data > upper_bound)]

print(f"- Обнаружено выбросов: {len(outliers)}")
print(f"- Диапазон нормальных значений: [{lower_bound:.1f}, {upper_bound:.1f}]")

plt.figure(figsize=(10, 6))
plt.scatter(df['performance_score'], df['price_usd'], alpha=0.6, color='blue')
plt.xlabel('Производительность (баллы)')
plt.ylabel('Цена (USD)')
plt.title('Зависимость цены смартфона от производительности')
plt.grid(True, alpha=0.3)

# Добавляем линию тренда
z = np.polyfit(df['performance_score'], df['price_usd'], 1)
p = np.poly1d(z)
plt.plot(df['performance_score'], p(df['performance_score']), "r--", alpha=0.8, 
         label=f'Линия тренда: y = {z[0]:.2f}x + {z[1]:.2f}')

plt.legend()
plt.tight_layout()
plt.show()

# Вычисляем коэффициент корреляции
correlation = np.corrcoef(df['performance_score'], df['price_usd'])[0, 1]
print(f"Коэффициент корреляции между производительностью и ценой: {correlation:.3f}")