import matplotlib.pyplot as plt

# Datos de ejemplo
categorias = ['A', 'B', 'C', 'D']
inicio_barras = [2, 5, 1, 3]  # Valores de inicio de las barras
longitud_barras = [10, 15, 8, 12]  # Longitud de las barras
espacio_entre_lineas = 1  # Espacio entre cada línea

fig, ax = plt.subplots()

# Crear las barras como líneas finas de color rojo con espacio entre ellas
for cat, inicio, longitud in zip(categorias, inicio_barras, longitud_barras):
    ax.plot([inicio, inicio + longitud], [cat, cat], color='red', linewidth=1)
    ax.plot([inicio + longitud, inicio + longitud + 3], [cat, cat], color='white', linewidth=0)  # Línea blanca para el espacio

ax.set_xlabel('Longitud')
ax.set_ylabel('Categorías')
ax.set_title('Barras Finas de Color Rojo con Espacio')
plt.yticks([])  # Oculta las etiquetas del eje y

plt.show()
