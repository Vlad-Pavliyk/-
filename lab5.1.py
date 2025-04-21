import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# === 1. Завантаження і первинний аналіз ===
df = pd.read_csv("diabetes_prediction_dataset.csv")

print("Перші 5 рядків:\n", df.head())
print("\nРозмір датасету:", df.shape)
print("\nТипи даних:\n", df.dtypes)

# Пропущені значення
print("\nПропущені значення:\n", df.isnull().sum())
df.fillna(df.mean(numeric_only=True), inplace=True)

# Дублікати
print("\nКількість дублікатів:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Описова статистика
print("\nОписова статистика:\n", df.describe())

# Кодування категоріальних змінних
df = pd.get_dummies(df, drop_first=True)

# === 2. Класифікація ===
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'LogisticRegression': LogisticRegression(),
    'RidgeClassifier': RidgeClassifier(),
    'SGDClassifier': SGDClassifier(),
    'SVC': SVC()
}

param_grid = {
    'LogisticRegression': {'C': [0.01, 0.1, 1, 10]},
    'RidgeClassifier': {'alpha': [0.01, 0.1, 1, 10]},
    'SGDClassifier': {'alpha': [0.0001, 0.001, 0.01], 'loss': ['hinge', 'log_loss']},
    'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

best_models = {}

for name in models:
    print(f"\n🔍 Підбір параметрів для {name}...")
    search = HalvingGridSearchCV(models[name], param_grid[name], cv=5, n_jobs=-1)
    search.fit(X_train, y_train)
    best_models[name] = search.best_estimator_
    print(f"✅ Найкращі параметри для {name}: {search.best_params_}")

for name, model in best_models.items():
    print(f"\n=== 📊 Класифікаційний звіт для {name} ===")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Матриця плутанини:\n", confusion_matrix(y_test, y_pred))

print("\n🎯 Випадкові 10 прогнозів (LogisticRegression):")
random_indices = np.random.choice(len(X_test), size=10, replace=False)
for i in random_indices:
    x_row = X_test[i].reshape(1, -1)
    true_val = y_test.iloc[i]
    pred_val = best_models['LogisticRegression'].predict(x_row)[0]
    print(f"Індекс {i} — Справжній: {true_val}, Прогноз: {pred_val}")

# === 3. Кластеризація ===
X_clust = df.drop("diabetes", axis=1)

# Метод ліктя
inertia = []
range_n = range(2, 11)
for k in range_n:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_clust)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range_n, inertia, marker='o')
plt.title('Метод ліктя')
plt.xlabel('Кількість кластерів')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Метод силуетів
silhouette_scores = []
for k in range_n:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_clust)
    score = silhouette_score(X_clust, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(range_n, silhouette_scores, marker='s', color='green')
plt.title('Метод силуетів')
plt.xlabel('Кількість кластерів')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Візуалізація кластерів (через PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clust)

kmeans = KMeans(n_clusters=3, random_state=42)  # Або іншу оптимальну k
labels = kmeans.fit_predict(X_clust)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=40)
plt.title("Візуалізація кластерів (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
