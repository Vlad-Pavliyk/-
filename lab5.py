import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# 1. Завантаження і попередній аналіз
df = pd.read_csv(r"D:\labl ME\diabetes_prediction_dataset.csv")



print("Пропущені значення:\n", df.isnull().sum())

# Кодування категоріальних змінних
df = pd.get_dummies(df, drop_first=True)

# 2. Розділення на ознаки та цільову змінну
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# 3. Розділення на тренувальні і тестові вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Масштабування
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Побудова моделей
models = {
    'LogisticRegression': LogisticRegression(),
    'RidgeClassifier': RidgeClassifier(),
    'SGDClassifier': SGDClassifier(),
    'SVC': SVC()
}

# 6. Підбір параметрів HalvingGridSearchCV
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

# 7. Оцінка моделей
for name, model in best_models.items():
    print(f"\n=== 📊 Класифікаційний звіт для {name} ===")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Матриця плутанини:\n", confusion_matrix(y_test, y_pred))

# 8. Випадкові 10 записів з передбаченням
print("\n🎯 10 випадкових прогнозів:")
random_indices = np.random.choice(len(X_test), size=10, replace=False)
for i in random_indices:
    x_row = X_test[i].reshape(1, -1)
    true_val = y_test.iloc[i]
    pred_val = best_models['LogisticRegression'].predict(x_row)[0]
    print(f"Індекс {i} — Справжній: {true_val}, Прогнозований (LogisticRegression): {pred_val}")
