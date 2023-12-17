from Topologia_comp import *
import numpy as np

def generate_points(num_points, centro=(0, 0), radio=1, max_error=0.25):
    # Generar ángulos aleatorios
    angles = np.random.uniform(0, 2 * np.pi, num_points)

    # Generar radios con un poco de ruido
    true_radii = np.ones(num_points) * radio
    noisy_radii = true_radii + np.random.uniform(-max_error, max_error, num_points)

    # Convertir coordenadas polares a cartesianas
    x = centro[0] + noisy_radii*np.cos(angles)
    y = centro[1] + noisy_radii*np.sin(angles)

    return np.array(list(zip(x, y)))

alpha = AlphaComplex(np.concatenate((generate_points(30, centro=(1,1)), generate_points(30, centro=(-1,-1))), axis=0))
alpha.analiza()