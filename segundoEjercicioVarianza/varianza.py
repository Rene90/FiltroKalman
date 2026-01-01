import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import sqrt, pi, exp

# ----------------------------
# Datos
# ----------------------------
x = np.array([1, 2, 2, 3, 3, 3, 4, 4, 5])
n = len(x)

# ----------------------------
# Conjunto de valores distintos
# ----------------------------
D = np.unique(x)

# ----------------------------
# Frecuencias y PDF discreta
# ----------------------------
frecuencias = Counter(x)
P = np.array([frecuencias[di] / n for di in D])

# ----------------------------
# Media y varianza (definición probabilística)
# ----------------------------
media = np.sum(D * P)
varianza = np.sum((D - media)**2 * P)
sigma = sqrt(varianza)

# ----------------------------
# Mostrar resultados
# ----------------------------
print("Valores distintos D:", D)
print("PDF:", P)
print(f"\nMedia = {media:.3f}")
print(f"Varianza = {varianza:.3f}")
print(f"Desviación estándar = {sigma:.3f}")

# ----------------------------
# Gaussiana teórica
# ----------------------------
x_cont = np.linspace(min(D)-1, max(D)+1, 300)
gaussiana = (1 / (sigma * sqrt(2*pi))) * np.exp(-0.5 * ((x_cont - media)/sigma)**2)

# ----------------------------
# Gráfica PDF + Gaussiana
# ----------------------------
plt.figure()
plt.bar(D, P, width=0.4, label='PDF discreta')
plt.plot(x_cont, gaussiana, label='Gaussiana teórica', linewidth=2)
plt.axvline(media, linestyle='--', label='Media')
plt.xlabel('x')
plt.ylabel('Probabilidad')
plt.title('PDF discreta, media y Gaussiana teórica')
plt.legend()
plt.show()

# ----------------------------
# Gráfica de varianza (visual)
# ----------------------------
plt.figure()
plt.plot(D, (D - media)**2 * P, 'o-', label='Contribución a la varianza')
plt.xlabel('d_i')
plt.ylabel('$(d_i - M)^2 P(d_i)$')
plt.title('Contribuciones individuales a la varianza')
plt.legend()
plt.show()
