import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

def recommend_mnk(user_id, ratings_matrix):
    # Отримуємо рейтинги користувача
    target_user_ratings = ratings_matrix.loc[user_id]
    
    X = []  # Оцінки інших користувачів
    y = []  # Оцінки цільового користувача для спільних фільмів

    # Перебираємо всіх інших користувачів
    for other_id in ratings_matrix.index:
        if other_id == user_id:
            continue

        other_user_ratings = ratings_matrix.loc[other_id]

        # Знаходимо спільні фільми, які обидва користувачі оцінили
        common_movies = (target_user_ratings > 0) & (other_user_ratings > 0)

        if common_movies.sum() >= 5:  # Перевірка, чи є хоча б 5 спільних оцінок
            X.append(other_user_ratings[common_movies].values)  # Оцінки інших користувачів
            y.append(target_user_ratings[common_movies].values)  # Оцінки цільового користувача

    # Якщо немає достатньо спільних оцінок
    if not X:
        return "Недостатньо спільних оцінок для побудови моделі"

    # Перевірка X та y
    print("X:", X)
    print("y:", y)

    # Перетворення списків у масиви для моделювання
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    
    # Створення та навчання моделі лінійної регресії
    model = LinearRegression().fit(X, y)

    # Передбачаємо рейтинги для фільмів, які користувач не оцінив
    predictions = {}
    for movie_id in ratings_matrix.columns:
        if ratings_matrix.loc[user_id, movie_id] == 0:  # Перевірка, чи не оцінив користувач цей фільм
            other_ratings = ratings_matrix[movie_id][ratings_matrix[movie_id] > 0]
            if len(other_ratings) > 0:  # Якщо є оцінки інших користувачів
                avg_rating = other_ratings.mean()  # Середній рейтинг фільму серед інших користувачів
                predicted = model.predict([[avg_rating]])[0]  # Прогнозуємо рейтинг для цього фільму
                predictions[movie_id] = predicted  # Зберігаємо прогноз

    # Перевірка результатів предсказання
    print("Predictions:", predictions)

    # Повертаємо топ-5 рекомендацій
    top = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
    return top


# Тестування функції з прикладом даних
ratings_data = {
    'Movie1': [5, 3, 0, 4],
    'Movie2': [0, 2, 5, 3],
    'Movie3': [4, 0, 4, 5],
    'Movie4': [3, 0, 4, 0]
}

ratings_matrix = pd.DataFrame(ratings_data, index=[1, 2, 3, 4])

# Приклад виклику функції
user_id = 1
recommendations = recommend_mnk(user_id, ratings_matrix)
print("Top 5 recommendations:", recommendations)
