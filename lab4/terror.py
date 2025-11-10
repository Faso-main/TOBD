import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os, time

sns.set_style("whitegrid")

df = pd.read_csv(os.path.join('lab4', 'globalterrorismdb_0718dist.csv'), encoding='ISO-8859-1', low_memory=False)
print(f"Датасет загружен. Размер: {df.shape}")

analysis_data = df[['latitude', 'longitude', 'nkill', 'nwound', 'attacktype1']].copy()
analysis_data.dropna(inplace=True)

SAMPLE_SIZE = 5000
if len(analysis_data) > SAMPLE_SIZE:
    analysis_data = analysis_data.sample(n=SAMPLE_SIZE, random_state=42)

geo_data = analysis_data[['latitude', 'longitude']].copy()

scaler = StandardScaler()
scaled_geo_data = scaler.fit_transform(geo_data)

OPTIMAL_K = 4
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=5)
clusters = kmeans.fit_predict(scaled_geo_data)
analysis_data['cluster'] = clusters

scatter_sample = analysis_data.sample(n=min(1000, len(analysis_data)), random_state=42)

sns.pairplot(scatter_sample, 
             hue='cluster',
             palette='viridis',
             diag_kind='hist', 
             plot_kws={'alpha': 0.7, 's': 30},
             diag_kws={'alpha': 0.8, 'bins': 20})
plt.suptitle('Матрица диаграмм рассеяния: взаимосвязи признаков терактов (цвета = кластеры)', 
             y=1.02, fontsize=14)
plt.show()

sns.pairplot(scatter_sample, 
             hue='cluster',
             palette='tab10',
             diag_kind='kde',
             plot_kws={'alpha': 0.6, 's': 25},
             diag_kws={'alpha': 0.7})
plt.suptitle('Матрица диаграмм рассеяния (KDE диагональ)', 
             y=1.02, fontsize=14)
plt.show()

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.violinplot(x='cluster', y='nkill', data=analysis_data, palette='viridis')
plt.title('Распределение количества погибших по кластерам')
plt.xlabel('Кластер')
plt.ylabel('Количество погибших')

plt.subplot(2, 2, 2)
sns.violinplot(x='cluster', y='nwound', data=analysis_data, palette='viridis')
plt.title('Распределение количества раненых по кластерам')
plt.xlabel('Кластер')
plt.ylabel('Количество раненых')

plt.subplot(2, 2, 3)
sns.violinplot(x='cluster', y='latitude', data=analysis_data, palette='viridis')
plt.title('Географическое распределение по кластерам (широта)')
plt.xlabel('Кластер')
plt.ylabel('Широта')

plt.subplot(2, 2, 4)
top_attack_types = analysis_data['attacktype1'].value_counts().head(6).index
filtered_data = analysis_data[analysis_data['attacktype1'].isin(top_attack_types)]

sns.violinplot(x='cluster', y='attacktype1', data=filtered_data, palette='viridis')
plt.title('Распределение типов атак по кластерам')
plt.xlabel('Кластер')
plt.ylabel('Тип атаки')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
scatter = plt.scatter(
    analysis_data['longitude'], 
    analysis_data['latitude'], 
    c=analysis_data['cluster'],
    cmap='viridis',
    alpha=0.6,
    s=20
)
plt.colorbar(scatter, label='Кластер')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.xlabel('Долгота (Longitude)')
plt.ylabel('Широта (Latitude)')
plt.title('Географическое распределение кластеров')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter = plt.scatter(
    analysis_data['longitude'], 
    analysis_data['latitude'], 
    c=analysis_data['cluster'],
    cmap='viridis',
    alpha=0.6,
    s=analysis_data['nkill']*2 + 10
)
plt.colorbar(scatter, label='Кластер')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.xlabel('Долгота (Longitude)')
plt.ylabel('Широта (Latitude)')
plt.title('Кластеры (размер точек = количество погибших)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

colors = ['purple', 'orange', 'green', 'blue', 'brown', 'pink']

plot_sample = analysis_data.sample(n=min(1500, len(analysis_data)), random_state=42)

for cluster_num in range(OPTIMAL_K):
    cluster_data = plot_sample[plot_sample['cluster'] == cluster_num]
    ax.scatter(
        cluster_data['longitude'], 
        cluster_data['latitude'], 
        cluster_data['nkill'],
        s=cluster_data['nwound']*2 + 20,
        c=colors[cluster_num % len(colors)], 
        label=f'Кластер {cluster_num}',
        alpha=0.7,
        depthshade=True
    )

ax.set_xlabel('Долгота')
ax.set_ylabel('Широта')
ax.set_zlabel('Количество погибших')
ax.set_title('3D визуализация: География и интенсивность терактов\n(Размер точек = количество раненых)')
plt.legend()
plt.show()

geo_features = analysis_data[['latitude', 'longitude', 'nkill', 'nwound', 'cluster']].copy()

plt.figure(figsize=(12, 10))
sns.pairplot(geo_features, 
             hue='cluster',
             palette='tab10',
             plot_kws={'alpha': 0.7, 's': 40},
             diag_kind='hist')
plt.suptitle('Матрица рассеяния: География и последствия терактов', 
             y=1.02, fontsize=16)
plt.show()

cluster_stats = analysis_data['cluster'].value_counts().sort_index()
total_attacks = len(analysis_data)

for cluster_id, count in cluster_stats.items():
    cluster_points = analysis_data[analysis_data['cluster'] == cluster_id]
    center_lat = cluster_points['latitude'].mean()
    center_lon = cluster_points['longitude'].mean()
    avg_kills = cluster_points['nkill'].mean()
    avg_wounds = cluster_points['nwound'].mean()
    percentage = (count / total_attacks) * 100
    
    print(f"Кластер {cluster_id}:")
    print(f"  • Терактов: {count} ({percentage:.1f}%)")
    print(f"  • Географический центр: ({center_lat:.2f}°N, {center_lon:.2f}°E)")
    print(f"  • Среднее погибших: {avg_kills:.1f}")
    print(f"  • Среднее раненых: {avg_wounds:.1f}")
    print(f"  • Географический охват: {cluster_points['latitude'].min():.1f}°-{cluster_points['latitude'].max():.1f}°N, "
          f"{cluster_points['longitude'].min():.1f}°-{cluster_points['longitude'].max():.1f}°E")
    print()

print(f"ИТОГО: {total_attacks} терактов, {OPTIMAL_K} географических кластеров")
