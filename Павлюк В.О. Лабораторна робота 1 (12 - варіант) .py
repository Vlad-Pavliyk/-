import numpy as np

# Завдання 1
# Створіть дві матриці розміром 2x2, заповніть їх випадковими цілими числами в діапазоні від 1 до 10 та знайдіть їхню суму.
A1 = np.random.randint(1, 11, (2, 2))
B1 = np.random.randint(1, 11, (2, 2))
result1 = A1 + B1
print("Завдання 1: Сума матриць 2x2:")
print(result1)

# Завдання 2
A2 = np.random.randint(1, 11, (3, 3))
B2 = np.random.randint(1, 11, (3, 3))
result2 = A2 - B2
print("\nЗавдання 2: Різниця матриць 3x3:")
print(result2)

# Завдання 3
A3 = np.random.randint(1, 11, (4, 4))
B3 = np.random.randint(1, 11, (4, 4))
result3 = A3 + B3
print("\nЗавдання 3: Сума матриць 4x4:")
print(result3)

# Завдання 4
A4 = np.random.randint(1, 11, (5, 5))
B4 = np.random.randint(1, 11, (5, 5))
result4 = A4 / B4
print("\nЗавдання 4: Поелементне ділення матриць 5x5:")
print(result4)

# Завдання 5
A5 = np.random.randint(1, 11, (6, 6))
B5 = np.random.randint(1, 11, (6, 6))
result5 = A5 + B5
print("\nЗавдання 5: Сума матриць 6x6:")
print(result5)

# Завдання 6
A6 = np.random.randint(1, 11, (7, 7))
B6 = np.random.randint(1, 11, (7, 7))
result6 = A6 - B6
print("\nЗавдання 6: Різниця матриць 7x7:")
print(result6)

# Завдання 7
A7 = np.random.randint(1, 11, (8, 8))
B7 = np.random.randint(1, 11, (8, 8))
result7 = A7 + B7
print("\nЗавдання 7: Сума матриць 8x8:")
print(result7)

# Завдання 8
A8 = np.random.randint(1, 11, (9, 9))
B8 = np.random.randint(1, 11, (9, 9))
result8 = B8 / A8
print("\nЗавдання 8: Поелементне ділення елементів другої матриці на першу матрицю 9x9:")
print(result8)

# Завдання 9
A9 = np.random.randint(1, 11, (2, 2))
B9 = np.random.randint(1, 11, (2, 2))
sum_elements9 = A9.sum() + B9.sum()
print("\nЗавдання 9: Сума всіх елементів обох матриць 2x2:")
print(sum_elements9)

# Завдання 10
A10 = np.random.randint(1, 11, (3, 3))
B10 = np.random.randint(1, 11, (3, 3))
sum_elements10 = A10.sum() + B10.sum()
print("\nЗавдання 10: Сума всіх елементів обох матриць 3x3:")
print(sum_elements10)

# Завдання 11
A11 = np.random.randint(1, 11, (4, 4))
B11 = np.random.randint(1, 11, (4, 4))
diff_elements11 = A11.sum() - B11.sum()
print("\nЗавдання 11: Різниця всіх елементів обох матриць 4x4:")
print(diff_elements11)

# Завдання 12
A12 = np.random.randint(1, 11, (5, 5))
B12 = np.random.randint(1, 11, (5, 5))
result12 = A12 / B12
print("\nЗавдання 12: Поелементне ділення елементів матриць 5x5:")
print(result12)

# Висновок
# У ході виконання лабораторної роботи ми створили матриці різних розмірів,
# виконали операції додавання, віднімання, поелементного ділення
# та обчислили суму і різницю елементів. Це дозволило закріпити основи лінійної алгебри
# та роботу з бібліотекою NumPy у Python.
