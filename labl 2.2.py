import pandas as pd
import numpy as np

# 1. Завантаження датасету Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Перші 5 рядків
print("Перші 5 рядків:\n", df.head())

# 3. Розмір
print("\nРозмір:", df.shape)

# 4. Назви стовпців
print("\nСтовпці:", df.columns.tolist())

# 5. Типи даних
print("\nТипи даних:\n", df.dtypes)

# 6. Перевірка на NaN
print("\nNaN до заповнення:\n", df.isna().sum())

# 7. Заповнення: Age – середнє, Embarked – мода
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode().iloc[0], inplace=True)

# 8. Повторна перевірка
print("\nNaN після заповнення:\n", df.isna().sum())

# 9. Видалення дублікатів
print("\nДублікатів:", df.duplicated().sum())
df = df.drop_duplicates()

# 10. Описова статистика
print("\nОписова статистика:\n", df.describe(include='all'))

# 11. Кількість пасажирів за класами
print("\nКількість пасажирів за класами:\n", df['Pclass'].value_counts())

# 12. Відсоток виживших
survived_pct = df['Survived'].mean() * 100
print(f"\nВижили: {survived_pct:.2f}%")

# 13. Середній вік за статтю
print("\nСередній вік за статтю:\n", df.groupby('Sex')['Age'].mean())

# 14. Виживання за класами
print("\nВиживання за класами:\n", df.groupby('Pclass')['Survived'].mean())

# 15. Виживання за статтю
print("\nВиживання за статтю:\n", df.groupby('Sex')['Survived'].mean())

# 16. Середній тариф за класами
print("\nСередній тариф за класами:\n", df.groupby('Pclass')['Fare'].mean())

# 17. Ім’я (first name) з колонки Name
def extract_first_name(name):
    try:
        return name.split(',')[1].strip().split(' ')[1]
    except IndexError:
        return ''

df['FirstName'] = df['Name'].apply(extract_first_name)
duplicates = df[df.duplicated('FirstName', keep=False)]
print("\nПасажири з однаковим ім’ям (приклад):\n", duplicates[['Name', 'FirstName']].head())

# 18. Пасажири з родиною
with_family = df[(df['SibSp'] + df['Parch']) > 0]
print(f"\nКількість пасажирів з родиною: {len(with_family)}")

# 19. Пасажири без родини
alone = df[(df['SibSp'] + df['Parch']) == 0]
print(f"\nКількість пасажирів без родини: {len(alone)}")

# 20. Зміна типу Survived на bool
df['Survived'] = df['Survived'].astype(bool)
print("\nНовий тип 'Survived':", df['Survived'].dtype)

# 21. Порт посадки
print("\nКількість за портом посадки:\n", df['Embarked'].value_counts())

# 22. Вікові групи
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100],
                        labels=['Дитина', 'Підліток', 'Молодий', 'Дорослий', 'Похилого віку'])
print("\nВиживання за віковими групами:\n", df.groupby('AgeGroup')['Survived'].mean())

# 23. Пасажир з найбільшим тарифом
max_fare_row = df[df['Fare'] == df['Fare'].max()]
print("\nПасажир з найбільшим тарифом:\n", max_fare_row[['Name', 'Fare', 'Pclass', 'Cabin']])

# 24. Топ-5 найдорожчих тарифів
top5_fares = df.sort_values(by='Fare', ascending=False).head(5)
print("\nТоп-5 найдорожчих тарифів:\n", top5_fares[['Name', 'Fare', 'Pclass', 'Survived']])

# 25. Збереження у CSV
df.to_csv("cleaned_titanic.csv", index=False)
print("\nОброблений датасет збережено у файл cleaned_titanic.csv")