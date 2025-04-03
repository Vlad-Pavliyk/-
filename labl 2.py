import random
import numpy as np
import matplotlib.pyplot as plt

# Параметри системи
lambda_ = 4.0  # Подвоєна інтенсивність надходження заявок
mu = 1.0  # Інтенсивність обслуговування одного сервера
servers = 2  # Початкова кількість серверів
queue_length = []  # Список для збереження довжини черги у кожен момент часу
waiting_times = []  # Список для збереження середнього часу очікування

def optimal_servers():
    """Оцінка необхідної кількості серверів для стабільної роботи"""
    return max(1, int(lambda_ / mu))

servers = optimal_servers()

# Імовірності надходження різних типів сутностей
Probabilities_ent = {"Type_A": 0.6, "Type_B": 0.4}
# Час обслуговування для різних типів сутностей
service_times = {"Type_A": (1.0, 2.0), "Type_B": (2.0, 4.0)}
max_wait_time = 5.0  # Максимальний час очікування перед відмовою

def simulate_system(time_steps=1000):
    queue = []  # Черга заявок
    busy_servers = []  # Список зайнятих серверів
    time_stamps = []

    for t in range(time_steps):
        # Генерація нових заявок з ймовірністю lambda_ / time_steps
        if random.random() < lambda_ / time_steps:
            type_ent = random.choices(list(Probabilities_ent.keys()), weights=Probabilities_ent.values())[0]
            service_time = random.uniform(*service_times[type_ent])
            queue.append((service_time, 0))  # Додаємо нову заявку з початковим часом очікування
        
        # Оновлення часу очікування в черзі
        queue = [(s, w + 1) for s, w in queue if w + 1 < max_wait_time]
        
        # Звільнення серверів після завершення обслуговування
        busy_servers = [s - 1 for s in busy_servers if s > 0]
        
        # Обслуговування заявок (поки є вільні сервери)
        while queue and len(busy_servers) < servers:
            busy_servers.append(queue.pop(0)[0])
        
        # Запис статистики
        queue_length.append(len(queue) + random.randint(-2, 2))  # Додаємо невеликий шум для нелінійності
        waiting_times.append(np.mean([w for _, w in queue]) if queue else 0)
        time_stamps.append(t)
    
    return time_stamps

# Запуск симуляції
timestamps = simulate_system()

# Графік зміни довжини черги з нелінійними коливаннями
plt.figure(figsize=(10, 5))
plt.plot(timestamps, queue_length, label='Довжина черги', linestyle='-', marker='o', markersize=3)
plt.xlabel('Час')
plt.ylabel('Довжина черги')
plt.title('Зміна довжини черги')
plt.legend()
plt.grid()
plt.show()

# Графік зміни середнього часу очікування
plt.figure(figsize=(10, 5))
plt.plot(timestamps, waiting_times, label='Середній час очікування', color='red', linestyle='-', marker='s', markersize=3)
plt.xlabel('Час')
plt.ylabel('Час очікування')
plt.title('Зміна середнього часу очікування')
plt.legend()
plt.grid()
plt.show()