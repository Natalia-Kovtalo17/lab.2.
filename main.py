
import numpy as np
import matplotlib.pyplot as plt
import csv


#зчитування даних з CSV
def read_data(filename):
    x = []
    y = []
    try:
        with open(filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:

                x.append(float(row['n']))
                y.append(float(row['t']))
        return np.array(x), np.array(y)
    except FileNotFoundError:
        print(f"Помилка: Файл {filename} не знайдено.")
        return None, None

def omega_k(x, x_nodes, k):
    """Обчислення значення функції omega_k(x)"""
    res = 1.0
    for i in range(k):
        res *= (x - x_nodes[i])
    return res


def divided_diff(x_nodes, y_nodes):
    """Обчислення розділених різниць"""
    n = len(y_nodes)
    coef = np.copy(y_nodes)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x_nodes[i] - x_nodes[i - j])
    return coef

def newton_interpolation(x, x_nodes, coefs):
    """Знаходження значення інтерполяційного многочлена"""
    n = len(coefs)
    res = coefs[0]
    for k in range(1, n):
        res += coefs[k] * omega_k(x, x_nodes, k)
    return res


def create_test_csv(n_points=20):
    with open('data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 't'])

        x_vals = np.linspace(10000, 160000, n_points)
        for x in x_vals:

            x_scaled = (x - 85000) / 75000
            t = (1 / (1 + 25 * x_scaled ** 2)) * 400
            writer.writerow([x, t])

#створення данних з n вузлами
create_test_csv(n_points=10)

x_nodes, y_nodes = read_data("data.csv")

if x_nodes is not None:

    coefs = divided_diff(x_nodes, y_nodes)

    a, b = x_nodes[0], x_nodes[-1]
    n = len(x_nodes) - 1
    h_tab = (b - a) / (20 * n)
    x_range = np.arange(a, b + h_tab, h_tab)

    y_newton = [newton_interpolation(val, x_nodes, coefs) for val in x_range]

    #побудова графіка
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_newton, label=f'Поліном Ньютона (n={n + 1})', color='blue')
    plt.scatter(x_nodes, y_nodes, color='red', s=20, label='Вузли з CSV')

    plt.title('Інтерполяція: Дослідження ефекту Рунге')
    plt.xlabel('Розмір датасету (n)')
    plt.ylabel('Час тренування (t)')
    plt.legend()
    plt.grid(True)

    plt.ylim(min(y_nodes) - 200, max(y_nodes) + 200)

    plt.show()

    prediction = newton_interpolation(120000, x_nodes, coefs)
    print(f"Прогноз для x=120000: {prediction:.2f}")