# 1. Імпорт бібліотек
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 2. Завантаження датасету
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)

# 3. Використання методу ліктя для визначення оптимальної кількості кластерів
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# 4. Побудова графіку методу ліктя
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Кількість кластерів (k)')
plt.ylabel('Inertia (внутрішньокластерна сума квадратів)')
plt.title('Метод ліктя для визначення оптимальної кількості кластерів')
plt.grid(True)
plt.show()

# 5. Проведення кластеризації (наприклад, виберемо k=3 за методом ліктя)
k_optimal = 3
kmeans_final = KMeans(n_clusters=k_optimal, random_state=42)
clusters = kmeans_final.fit_predict(X)

# Додавання інформації про кластери до датафрейму
X['Cluster'] = clusters

# 6. Побудова графіку кластерів (по перших двох ознаках для наочності)
plt.figure(figsize=(8, 6))
for cluster in range(k_optimal):
    cluster_points = X[X['Cluster'] == cluster]
    plt.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], label=f'Кластер {cluster}')
    
plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title('Візуалізація кластерів (перші дві ознаки)')
plt.legend()
plt.grid(True)
plt.show()

# 7. Висновок
print("\n--- Висновок ---")
print("Оптимальна кількість кластерів згідно методу ліктя — 3.")
print("Кластери добре розділилися за ознаками датасету Iris.")
