import pandas as pd
import numpy as np

# Завантаження таблиці з Вікіпедії
url = 'https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)'
tables = pd.read_html(url)

# Пошук таблиці з потрібними стовпцями
for table in tables:
    if len(table.columns) >= 4 and 'Country' in str(table.columns[0]):
        df = table
        break

# Вивід кількості колонок та їх назв
print("Колонки в таблиці:", df.columns.tolist())

# Вибір тільки потрібних колонок (названия могут отличаться, поэтому — адаптация)
df = df.iloc[:, [0, 1, 2, 3]]  # Берём только первые 4 колонки

# Переименование
df.columns = ['Country', 'MVF_2024', 'WorldBank_2023', 'OON_2022']

# 1. Вивід перших 5 рядків
print("Перші 5 рядків:\n", df.head())

# 2. Розмір
print("\nРозмір датасету:", df.shape)

# 3. Кількість стовпців
print("\nСтовпці:", df.columns.tolist())

# 4. Типи даних
print("\nТипи даних:\n", df.dtypes)

# 5. Заміна "—" на NaN, перетворення типів
df.replace("—", np.nan, inplace=True)
df[['MVF_2024', 'WorldBank_2023', 'OON_2022']] = (
    df[['MVF_2024', 'WorldBank_2023', 'OON_2022']]
    .replace({',': '', '−': '-'}, regex=True)
    .apply(pd.to_numeric, errors='coerce')
)

# 6. Перевірка пропущених значень
print("\nКількість NaN до заповнення:\n", df.isna().sum())

# 7. Заповнення середнім
df.fillna(df.mean(numeric_only=True), inplace=True)

# 8. Перевірка повторно
print("\nNaN після заповнення:\n", df.isna().sum())

# 9. Перевірка дублікатів
print("\nДублікатів:", df.duplicated().sum())
df = df.drop_duplicates()

# 10. Описова статистика
print("\nОписова статистика:\n", df.describe())

# 11. Відхилення між MVF та WorldBank
df['Difference_MVF_WB'] = abs(df['MVF_2024'] - df['WorldBank_2023'])
max_diff_countries = df.sort_values(by='Difference_MVF_WB', ascending=False).head(5)
print("\nНайбільші відмінності між MVF та WB:\n", max_diff_countries[['Country', 'Difference_MVF_WB']])

# 12. Кореляція
print("\nКореляція:\n", df[['MVF_2024', 'WorldBank_2023', 'OON_2022']].corr())

# 13. Середні значення
print("\nСередні значення:\n", df[['MVF_2024', 'WorldBank_2023', 'OON_2022']].mean())

# 14. Стандартне відхилення між роками
df['STD'] = df[['MVF_2024', 'WorldBank_2023', 'OON_2022']].std(axis=1)
most_var = df.loc[df['STD'].idxmax()]
print("\nКраїна з найвищою варіативністю:\n", most_var[['Country', 'STD']])

# 15. Країни з найвищим/найнижчим показником кожного року
for col in ['MVF_2024', 'WorldBank_2023', 'OON_2022']:
    max_row = df.loc[df[col].idxmax()]
    min_row = df.loc[df[col].idxmin()]
    print(f"\n{col}: max — {max_row['Country']}, min — {min_row['Country']}")

# 16. Частка кожної країни в загальному значенні
for col in ['MVF_2024', 'WorldBank_2023', 'OON_2022']:
    df[f'Share_{col}'] = df[col] / df[col].sum()

# 17. Різниця часток
df['Share_change'] = df['Share_MVF_2024'] - df['Share_OON_2022']
top_change = df.sort_values(by='Share_change', ascending=False).head(5)
print("\nКраїни з найбільшим зростанням частки:\n", top_change[['Country', 'Share_change']])