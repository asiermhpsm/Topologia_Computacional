from Topologia_comp import *
import numpy as np

def generate_points(num_points, petalos=6,  radio=5, max_error=0.3):
    if petalos%2 != 0:
        raise ValueError("El numero de petalos debe ser un numero estrictamente positivo y par")
    # Generar ángulos aleatorios
    true_angles = np.linspace(0, 2 * np.pi, num_points)
    noisy_angles = true_angles + np.random.uniform(-0.1, 0.1, num_points)

    # Generar radios con un poco de ruido
    true_radii = radio * np.abs(np.cos(petalos/2 * true_angles))
    noisy_radii = true_radii + np.random.uniform(-max_error, max_error, num_points)

    # Convertir coordenadas polares a cartesianas
    x = noisy_radii * np.cos(noisy_angles)
    y = noisy_radii * np.sin(noisy_angles)

    return np.array(list(zip(x, y)))


alpha = AlphaComplex(generate_points(100, petalos=4, max_error=0.3))
alpha.analiza()