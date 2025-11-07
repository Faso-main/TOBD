import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import os, logging

DATASET_PATH=os.path.join('lab4','IRIS.csv')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Загрузка данных
iris = pd.read_csv(DATASET_PATH)
x = iris.iloc[:, [0, 1, 2, 3]].values

# Анализ данных
logger.info(iris.head(5))
logger.info(iris.shape)
logger.info(iris.dtypes)
logger.info(iris.columns)
logger.info(iris.info())
logger.info(iris.describe())
logger.info(iris[0:10])
logger.info("\nРаспределение по видам:")
iris_outcome = pd.crosstab(index=iris["species"], columns="count")
logger.info(iris_outcome)

# Визуализация распределения признаков
sns.FacetGrid(iris, hue="species", height=3).map(sns.distplot, "petal_length").add_legend()
sns.FacetGrid(iris, hue="species", height=3).map(sns.distplot, "petal_width").add_legend()
sns.FacetGrid(iris, hue="species", height=3).map(sns.distplot, "sepal_length").add_legend()
plt.show()

# Box plot и violin plot
sns.boxplot(x="species", y="petal_length", data=iris)
plt.show()
sns.violinplot(x="species", y="petal_length", data=iris)
plt.show()

# Scatter plot матрица
sns.set_style("whitegrid")
sns.pairplot(iris, hue="species", height=3)
plt.show()

# Определение оптимального числа кластеров
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Метод локтя для определения оптимального числа кластеров')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()

# Кластеризация K-means
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='purple', label='Кластер 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='orange', label='Кластер 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Кластер 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Центроиды')
plt.title('Кластеризация K-means ирисов Фишера')
plt.xlabel('Длина чашелистика')
plt.ylabel('Ширина чашелистика')
plt.legend()
plt.show()