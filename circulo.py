from Topologia_comp import *
import numpy as np

def generate_points(num_points, radius=1, max_error=0.3):
    # Generar ángulos aleatorios
    angles = np.random.uniform(0, 2 * np.pi, num_points)

    # Generar radios con un poco de ruido
    true_radii = np.ones(num_points) * radius
    noisy_radii = true_radii + np.random.uniform(-max_error, max_error, num_points)

    # Convertir coordenadas polares a cartesianas
    x = noisy_radii * np.cos(angles)
    y = noisy_radii * np.sin(angles)

    return np.array([[x[i], y[i]] for i in range(num_points)])

# Generar 100 puntos cercanos a un círculo de radio 1 con un error máximo de 0.3
num_points = 50
points = generate_points(num_points, radius=1, max_error=0.25)

alpha = AlphaComplex(points)
alpha.analiza()