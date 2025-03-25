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

# 13. Сума двох матриць 6x6 з випадковими числами від 1 до 10
matrix1 = np.random.randint(1, 11, (6, 6))
matrix2 = np.random.randint(1, 11, (6, 6))
print("13. Сума матриць 6x6:\n", matrix1 + matrix2)

# 14. Різниця двох матриць 7x7 з випадковими числами від 1 до 10
matrix1 = np.random.randint(1, 11, (7, 7))
matrix2 = np.random.randint(1, 11, (7, 7))
print("14. Різниця матриць 7x7:\n", matrix1 - matrix2)

# 15. Сума двох матриць 8x8 з випадковими числами від 1 до 10
matrix1 = np.random.randint(1, 11, (8, 8))
matrix2 = np.random.randint(1, 11, (8, 8))
print("15. Сума матриць 8x8:\n", matrix1 + matrix2)

# 16. Відношення другого стовпця другої матриці до першої (9x9)
matrix1 = np.random.randint(1, 11, (9, 9))
matrix2 = np.random.randint(1, 11, (9, 9))
print("16. Відношення другого стовпця:\n", matrix2[:, 1] / matrix1[:, 1])

# 17. Сума діагоналей двох матриць 2x2
matrix1 = np.random.randint(1, 11, (2, 2))
matrix2 = np.random.randint(1, 11, (2, 2))
print("17. Сума діагоналей 2x2:", np.trace(matrix1), np.trace(matrix2))

# 18. Сума діагоналей двох матриць 3x3
matrix1 = np.random.randint(1, 11, (3, 3))
matrix2 = np.random.randint(1, 11, (3, 3))
print("18. Сума діагоналей 3x3:", np.trace(matrix1), np.trace(matrix2))

# 19. Добуток елементів діагоналей двох матриць 3x3
print("19. Добуток діагоналей 3x3:", np.prod(np.diag(matrix1)), np.prod(np.diag(matrix2)))

# 20. Добуток діагоналі другої матриці 3x3
print("20. Добуток діагоналі другої матриці 3x3:", np.prod(np.diag(matrix2)))

# 21. Сума всіх елементів першої матриці 3x3
print("21. Сума елементів першої матриці 3x3:", np.sum(matrix1))

# 22. Сума всіх елементів другої матриці 3x3
print("22. Сума елементів другої матриці 3x3:", np.sum(matrix2))

# 23. Добуток всіх елементів першої матриці 3x3
print("23. Добуток елементів першої матриці 3x3:", np.prod(matrix1))

# 24. Добуток всіх елементів другої матриці 2x2 (від -5 до 3)
matrix2 = np.random.randint(-5, 4, (2, 2))
print("24. Добуток елементів другої матриці 2x2:", np.prod(matrix2))

# 25. Сума всіх елементів обох матриць 2x2 (від -2 до 2)
matrix1 = np.random.randint(-2, 3, (2, 2))
matrix2 = np.random.randint(-2, 3, (2, 2))
print("25. Сума елементів обох матриць 2x2:", np.sum(matrix1) + np.sum(matrix2))

# 26. Добуток всіх елементів обох матриць 3x3 (від 1 до 3)
matrix1 = np.random.randint(1, 4, (3, 3))
matrix2 = np.random.randint(1, 4, (3, 3))
print("26. Добуток елементів обох матриць 3x3:", np.prod(matrix1) * np.prod(matrix2))

# 27. Різниця всіх елементів першої матриці 3x3 (від -1 до 5)
matrix1 = np.random.randint(-1, 6, (3, 3))
print("27. Різниця елементів першої матриці 3x3:", np.sum(matrix1) - np.sum(matrix2))

# 28. Різниця елементів двох матриць 3x3 (від 1 до 3)
matrix1 = np.random.randint(1, 4, (3, 3))
matrix2 = np.random.randint(1, 4, (3, 3))
print("28. Різниця елементів двох матриць 3x3:\n", matrix1 - matrix2)

# 29. Відношення елементів першої матриці до другої 3x3 (від 1 до 5)
matrix1 = np.random.randint(1, 6, (3, 3))
matrix2 = np.random.randint(1, 6, (3, 3))
print("29. Відношення елементів першої матриці до другої:\n", matrix1 / matrix2)

# 30. Відношення елементів другої матриці до першої 3x3 (від -3 до 3)
matrix1 = np.random.randint(-3, 4, (3, 3))
matrix2 = np.random.randint(-3, 4, (3, 3))
print("30. Відношення елементів другої матриці до першої:\n", matrix2 / matrix1)
# Висновок
# У ході виконання лабораторної роботи ми створили матриці різних розмірів,
# виконали операції додавання, віднімання, поелементного ділення
# та обчислили суму і різницю елементів. Це дозволило закріпити основи лінійної алгебри
# та роботу з бібліотекою NumPy у Python.
