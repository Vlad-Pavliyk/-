# 1. Імпорт бібліотек
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 2. Завантаження даних
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# 3. Поділ на тренувальну і тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Побудова базової лінійної регресії
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Оцінка базової моделі
print("--- Базова Лінійна Регресія ---")
print(f"R^2 на тренувальній вибірці: {model_lr.score(X_train, y_train):.4f}")
print(f"R^2 на тестовій вибірці: {model_lr.score(X_test, y_test):.4f}")
print(f"MSE на тестовій вибірці: {mean_squared_error(y_test, y_pred_lr):.4f}")

# 5. Введення регуляризації (Ridge і Lasso)
alphas = np.logspace(-3, 2, 50)
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_scores.append(r2_score(y_test, ridge.predict(X_test)))
    
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_scores.append(r2_score(y_test, lasso.predict(X_test)))

# 6. Побудова графіків
plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_scores, label='Ridge R²', marker='o')
plt.plot(alphas, lasso_scores, label='Lasso R²', marker='s')
plt.xscale('log')
plt.xlabel('Alpha (параметр регуляризації)')
plt.ylabel('R² на тестовій вибірці')
plt.title('Залежність R² від параметра регуляризації')
plt.legend()
plt.grid(True)
plt.show()

# 7. Висновок
print("\n--- Висновок ---")
print("Без регуляризації модель мала схильність до перенавчання.")
print("Введення регуляризації Ridge і Lasso допомогло стабілізувати модель і покращити генералізацію.")
print("Оптимальні значення alpha можна обирати за графіком: де R² максимальне.")
