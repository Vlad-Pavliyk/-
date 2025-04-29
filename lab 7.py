import pandas as pd
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# Завантаження даних
data = Dataset.load_builtin('ml-100k')

# Розділення на тренувальний та тестовий набори
trainset, testset = train_test_split(data, test_size=0.25, random_state=42)

# Алгоритм 1: SVD (матричне розкладання)
svd = SVD()
svd.fit(trainset)
svd_predictions = svd.test(testset)

# Алгоритм 2: KNNBasic (k-найближчих сусідів)
knn = KNNBasic()
knn.fit(trainset)
knn_predictions = knn.test(testset)

# Оцінка якості
print("SVD RMSE:", accuracy.rmse(svd_predictions))
print("KNN RMSE:", accuracy.rmse(knn_predictions))

# Приклад рекомендацій для користувача з id=1
user_id = "1"

# Отримуємо список усіх фільмів
movies = data.build_full_trainset().all_items()
movies = [data.build_full_trainset().to_raw_iid(i) for i in movies]

# Прогнозуємо рейтинги для фільмів, які користувач не бачив
user_ratings = []
for movie_id in movies:
    if not trainset.knows_user(trainset.to_inner_uid(user_id)) or not trainset.knows_item(trainset.to_inner_iid(movie_id)):
        continue
    pred = svd.predict(user_id, movie_id)
    user_ratings.append((movie_id, pred.est))

# Сортуємо за рейтингом (найвищі спочатку)
top_recommendations = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:10]

print("\nТоп-10 рекомендацій для користувача", user_id, ":")
for movie_id, rating in top_recommendations:
    print(f"Фільм ID: {movie_id}, Прогнозований рейтинг: {rating:.2f}")