from Topologia_comp import *
import numpy as np

def generate_points(num_points, petalos=3,  radio=5, max_error=0.3):
    # Generar ángulos aleatorios
    true_angles = np.linspace(0, 2 * np.pi, num_points)
    noisy_angles = true_angles + np.random.uniform(-0.1, 0.1, num_points)

    # Generar radios con un poco de ruido
    true_radii = radio * np.abs(np.cos(petalos * true_angles))
    noisy_radii = true_radii + np.random.uniform(-max_error, max_error, num_points)

    # Convertir coordenadas polares a cartesianas
    x = noisy_radii * np.cos(noisy_angles)
    y = noisy_radii * np.sin(noisy_angles)

    return np.array([[x[i], y[i]] for i in range(num_points)])


alpha = AlphaComplex(generate_points(100, petalos=3, max_error=0.3))
alpha.analiza()