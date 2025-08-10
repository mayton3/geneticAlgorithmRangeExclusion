import numpy as np
import os
import sys
import datetime
import pytz
from typing import Optional, Tuple

# Подключение Google Диска (запускать в Colab)
from google.colab import drive
drive.mount('/content/drive')

# Путь для сохранения на Google Диске — папка GA_backup
backup_folder = '/content/drive/MyDrive/GA_backup'
os.makedirs(backup_folder, exist_ok=True)

old_range = (-5_000_000, 5_000_000)
x_range = (-5_000_001, 5_000_001)  # расширенный диапазон на 1 единицу

def gen_valid_gene(low_new, high_new, low_old, high_old) -> int:
    """Генерируем случайное целое в [low_new, high_new], 
    повторяем, если полученное число внутри старого диапазона [low_old, high_old]."""
    while True:
        val = np.random.randint(low_new, high_new + 1)
        if val < low_old or val > high_old:
            return val

def fitness(population: np.ndarray) -> np.ndarray:
    """Вычисляем fitness, штрафуем особей с координатами внутри старого диапазона."""
    mask_inside_old_range = np.logical_and(population >= old_range[0], population <= old_range[1])
    mask_inside_old_range_any = mask_inside_old_range.any(axis=1)

    cubes = population.astype(np.float64) ** 3
    fitness_values = np.abs(np.sum(cubes, axis=1))

    fitness_values[mask_inside_old_range_any] = np.inf

    return fitness_values

def generate_population(size: int, x_range: Tuple[int, int], old_range: Tuple[int, int]) -> np.ndarray:
    population = np.empty((size, 3), dtype=int)
    for i in range(size):
        for j in range(3):
            population[i, j] = gen_valid_gene(x_range[0], x_range[1], old_range[0], old_range[1])
    return population

def crossover(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    point = np.random.randint(1, 3)
    child = np.empty(3, dtype=int)
    child[:point] = parent1[:point]
    child[point:] = parent2[point:]
    return child

def mutate(individual: np.ndarray, x_range: Tuple[int, int], old_range: Tuple[int, int], mutation_rate: float = 0.1) -> np.ndarray:
    for i in range(3):
        if np.random.rand() < mutation_rate:
            individual[i] = gen_valid_gene(x_range[0], x_range[1], old_range[0], old_range[1])
    return individual

def save_population_to_csv(population: np.ndarray, filename: str = "population_backup.csv") -> None:
    path = os.path.join(backup_folder, filename)
    np.savetxt(path, population, delimiter=",", fmt='%d')
    print(f"\nТекущее состояние популяции сохранено в файл '{path}'.")

def load_population_from_csv(filename: str = "population_backup.csv") -> Optional[np.ndarray]:
    path = os.path.join(backup_folder, filename)
    if os.path.exists(path):
        population = np.loadtxt(path, delimiter=",", dtype=int)
        if population.ndim == 1:
            population = population.reshape(1, -1)
        print(f"Загружена популяция из файла '{path}', размер: {population.shape}.")
        return population
    else:
        return None

def save_generation_to_csv(generation: int, filename: str = "generation_backup.csv") -> None:
    path = os.path.join(backup_folder, filename)
    np.savetxt(path, np.array([generation]), fmt='%d')
    print(f"Номер поколения {generation} сохранён в файл '{path}'.")

def load_generation_from_csv(filename: str = "generation_backup.csv") -> int:
    path = os.path.join(backup_folder, filename)
    if os.path.exists(path):
        gen_arr = np.loadtxt(path, dtype=int)
        if gen_arr.ndim == 0:
            generation = int(gen_arr)
        else:
            generation = int(gen_arr[0])
        print(f"Загружен номер поколения: {generation} из файла '{path}'.")
        return generation
    else:
        return 0

def genetic_algorithm(
    fitness,
    generate_population,
    crossover,
    mutate,
    population_size: int,
    x_range: Tuple[int, int],
    old_range: Tuple[int, int],
    generations: int,
    mutation_rate: float = 0.1
) -> np.ndarray:
    population = load_population_from_csv()
    start_gen = load_generation_from_csv()

    if population is None:
        print("Файл с сохранённой популяцией не найден. Создаём новую популяцию.")
        population = generate_population(population_size, x_range, old_range)
        start_gen = 0
    else:
        print(f"Продолжаем работу с загруженной популяцией размером {population.shape[0]} и поколением {start_gen}")
        if population.shape[0] < population_size:
            needed = population_size - population.shape[0]
            new_individuals = generate_population(needed, x_range, old_range)
            population = np.vstack([population, new_individuals])

    timezone = pytz.timezone('Europe/Kiev')  # поменяйте по необходимости

    try:
        for gen in range(start_gen, generations):
            fitness_values = fitness(population)
            best_idx = np.argmin(fitness_values)
            best_fitness = fitness_values[best_idx]

            current_time = datetime.datetime.now(pytz.utc).astimezone(timezone).strftime("%Y-%m-%d %H:%M:%S")

            # Формат вывода без 'e' с разделением тысяч и округлением до целого
            best_fitness_str = f"{best_fitness:,.0f}" if best_fitness != np.inf else "inf"

            print(f"[{current_time}] Поколение {gen + 1}, лучшая оценка: {best_fitness_str}")

            if best_fitness == 0:
                print("Найдено точное решение!")
                return population[best_idx]

            sorted_indices = np.argsort(fitness_values)
            population = population[sorted_indices]

            next_generation = list(population[:2].copy())  # элита

            while len(next_generation) < population_size:
                parent_pool_size = min(10, len(population))
                parents_idx = np.random.choice(parent_pool_size, size=2, replace=False)
                parent1 = population[parents_idx[0]]
                parent2 = population[parents_idx[1]]

                child = crossover(parent1, parent2)
                child = mutate(child, x_range, old_range, mutation_rate)
                next_generation.append(child)

            population = np.array(next_generation)

            if (gen + 1) % 3 == 0:
                save_population_to_csv(population)
                save_generation_to_csv(gen + 1)

    except KeyboardInterrupt:
        print("\nПолучен сигнал прерывания (Ctrl+C). Сохраняем состояние и завершаем работу...")
        save_population_to_csv(population)
        save_generation_to_csv(gen)
        sys.exit(0)

    fitness_values = fitness(population)
    best_idx = np.argmin(fitness_values)
    return population[best_idx]

if __name__ == "__main__":
    population_size = 50000000
    generations = 1000
    mutation_rate = 0.1

    solution = genetic_algorithm(
        fitness,
        generate_population,
        crossover,
        mutate,
        population_size,
        x_range,
        old_range,
        generations,
        mutation_rate
    )

    print(f"Найденное решение: a={solution[0]}, b={solution[1]}, c={solution[2]}")
    value = abs(solution[0]**3 + solution[1]**3 + solution[2]**3)
    print(f"Значение функции оценки: {value:,.0f}")
