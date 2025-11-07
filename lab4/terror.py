import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

sns.set_style("whitegrid")

# Загрузка датасета
df = pd.read_csv(os.path.join('lab4', 'globalterrorismdb_0718dist.csv'), encoding='ISO-8859-1', low_memory=False)
print("Датасет успешно загружен.")
print(f"Размер датасета: {df.shape}")

# Выбор нужных столбцов для географической кластеризации
geo_data = df[['latitude', 'longitude']].copy()
geo_data.dropna(inplace=True)
print(f"Осталось {len(geo_data)} записей после удаления пропусков в координатах.")

# Анализ распределения координат
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.histplot(geo_data['latitude'], kde=True)
plt.title('Распределение широт')

plt.subplot(1, 2, 2)
sns.histplot(geo_data['longitude'], kde=True)
plt.title('Распределение долгот')

plt.tight_layout()
plt.show()

# Боксплоты координат
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=geo_data['latitude'])
plt.title('Боксплот широт')

plt.subplot(1, 2, 2)
sns.boxplot(y=geo_data['longitude'])
plt.title('Боксплот долгот')

plt.tight_layout()
plt.show()

# Масштабирование данных
scaler = StandardScaler()
scaled_geo_data = scaler.fit_transform(geo_data)

# Метод локтя для определения оптимального числа кластеров
inertia = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(scaled_geo_data)
    inertia.append(kmeans_model.inertia_)
    
    # Вычисляем silhouette score
    from sklearn.metrics import silhouette_score
    if k > 1:  # silhouette score требует минимум 2 кластера
        silhouette_scores.append(silhouette_score(scaled_geo_data, cluster_labels))

# График метода локтя
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Инерция (WCSS)')
plt.title('Метод "локтя" для определения оптимального k')
plt.xticks(K_range)

# График silhouette scores
plt.subplot(1, 2, 2)
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score для разных k')
plt.xticks(range(2, 11))

plt.tight_layout()
plt.show()

# Кластеризация с оптимальным k
OPTIMAL_K = 6
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_geo_data)
geo_data['cluster'] = clusters
print(f"Кластеризация с k={OPTIMAL_K} завершена.")

# 2D визуализация с разными палитрами
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Первый вариант визуализации
sns.scatterplot(
    x='longitude', 
    y='latitude', 
    hue='cluster', 
    data=geo_data, 
    palette='viridis',
    alpha=0.6,
    legend='full',
    ax=axes[0]
)
axes[0].set_xlim(-180, 180)
axes[0].set_ylim(-90, 90)
axes[0].set_title(f'Географическая кластеризация терактов ({OPTIMAL_K} кластеров)')
axes[0].set_xlabel('Долгота (Longitude)')
axes[0].set_ylabel('Широта (Latitude)')

# Второй вариант визуализации
sns.scatterplot(
    x='longitude', 
    y='latitude', 
    hue='cluster', 
    data=geo_data, 
    palette='tab10',
    alpha=0.7,
    legend='full',
    ax=axes[1]
)
axes[1].set_xlim(-180, 180)
axes[1].set_ylim(-90, 90)
axes[1].set_title(f'Альтернативная визуализация кластеров')
axes[1].set_xlabel('Долгота (Longitude)')
axes[1].set_ylabel('Широта (Latitude)')

plt.tight_layout()
plt.show()

# 3D визуализация
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

colors = ['purple', 'orange', 'green', 'blue', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']

for cluster_num in range(OPTIMAL_K):
    cluster_data = geo_data[geo_data['cluster'] == cluster_num]
    ax.scatter(
        cluster_data['longitude'], 
        cluster_data['latitude'], 
        np.ones(len(cluster_data)) * cluster_num,
        s=20, 
        c=colors[cluster_num % len(colors)], 
        label=f'Кластер {cluster_num}',
        alpha=0.6,
        depthshade=True
    )

# Центроиды кластеров
centroids_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)
for i, centroid in enumerate(centroids_original_scale):
    ax.scatter(
        centroid[1],
        centroid[0],
        i,
        s=200, 
        c='red', 
        marker='X', 
        label=f'Центроид {i}' if i == 0 else "",
        edgecolors='black',
        linewidth=2
    )

ax.set_xlabel('Долгота (Longitude)')
ax.set_ylabel('Широта (Latitude)')
ax.set_zlabel('Кластер')
ax.set_title(f'3D визуализация географических кластеров терактов\n{OPTIMAL_K} кластеров')
plt.legend()
plt.show()

# Дополнительная 3D визуализация с плотностью
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Создаем сетку для подсчета плотности
lon_bins = np.linspace(-180, 180, 30)
lat_bins = np.linspace(-90, 90, 30)

# Считаем плотность в каждой ячейке сетки
density, x_edges, y_edges = np.histogram2d(
    geo_data['longitude'], 
    geo_data['latitude'], 
    bins=[lon_bins, lat_bins]
)

# Создаем координаты для поверхности
X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])

# Создаем поверхность плотности
surf = ax.plot_surface(
    X, Y, density.T, 
    cmap='hot', 
    alpha=0.6,
    linewidth=0, 
    antialiased=True
)

# Добавляем точки терактов
for cluster_num in range(OPTIMAL_K):
    cluster_data = geo_data[geo_data['cluster'] == cluster_num]
    sample_data = cluster_data.sample(n=min(1000, len(cluster_data)), random_state=42)
    
    ax.scatter(
        sample_data['longitude'], 
        sample_data['latitude'], 
        np.zeros(len(sample_data)),
        s=10, 
        c=colors[cluster_num % len(colors)], 
        label=f'Кластер {cluster_num}',
        alpha=0.7,
        depthshade=True
    )

ax.set_xlabel('Долгота (Longitude)')
ax.set_ylabel('Широта (Latitude)')
ax.set_zlabel('Плотность терактов')
ax.set_title('3D карта плотности терактов с кластерами')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20, label='Плотность терактов')
plt.legend()
plt.show()

# Статистика по кластерам
print("\n" + "="*50)
print("СТАТИСТИКА ПО КЛАСТЕРАМ:")
print("="*50)

cluster_stats = geo_data['cluster'].value_counts().sort_index()
total_attacks = len(geo_data)

for cluster_id, count in cluster_stats.items():
    cluster_points = geo_data[geo_data['cluster'] == cluster_id]
    center_lat = cluster_points['latitude'].mean()
    center_lon = cluster_points['longitude'].mean()
    percentage = (count / total_attacks) * 100
    
    print(f"Кластер {cluster_id}:")
    print(f"  • Количество терактов: {count} ({percentage:.1f}% от общего числа)")
    print(f"  • Географический центр: ({center_lat:.2f}°N, {center_lon:.2f}°E)")
    print(f"  • Диапазон широт: [{cluster_points['latitude'].min():.2f}, {cluster_points['latitude'].max():.2f}]")
    print(f"  • Диапазон долгот: [{cluster_points['longitude'].min():.2f}, {cluster_points['longitude'].max():.2f}]")
    print()

print(f"ИТОГО: {total_attacks} терактов сгруппированы в {OPTIMAL_K} географических кластеров")

# Картографическая визуализация
plt.figure(figsize=(16, 10))
scatter = plt.scatter(
    geo_data['longitude'], 
    geo_data['latitude'], 
    c=geo_data['cluster'],
    cmap='viridis',
    alpha=0.6,
    s=10
)
plt.colorbar(scatter, label='Кластер')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.xlabel('Долгота (Longitude)')
plt.ylabel('Широта (Latitude)')
plt.title('Картографическое представление кластеров терактов')
plt.grid(True, alpha=0.3)

# Добавляем центроиды
plt.scatter(
    centroids_original_scale[:, 1],
    centroids_original_scale[:, 0],
    s=200,
    c='red',
    label='Центроиды',
    edgecolors='black',
    linewidth=2
)
plt.legend()
plt.show()