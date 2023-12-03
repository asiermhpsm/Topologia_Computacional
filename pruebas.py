import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = 1 / (1 + np.exp(-x))
y4 = np.exp(x)

fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(x, y1)
ax[0, 0].set_title("Sine function")
ax[0, 0].get_xaxis().set_visible(False)
ax[0, 0].get_yaxis().set_visible(False)

ax[0, 1].plot(x, y2)
ax[0, 1].set_title("Cosine function")
ax[0, 1].get_xaxis().set_visible(False)
ax[0, 1].get_yaxis().set_visible(False)


ax[1, 0].plot(x, y3)
ax[1, 0].set_title("Sigmoid function")
ax[1, 0].get_xaxis().set_visible(False)
ax[1, 0].get_yaxis().set_visible(False)


ax[1, 1].plot(x, y4)
ax[1, 1].set_title("Exponential function")
ax[1, 1].get_xaxis().set_visible(False)
ax[1, 1].get_yaxis().set_visible(False)

fig.tight_layout()
plt.show()