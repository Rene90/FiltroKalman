import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ----------------------------
# Datos
# ----------------------------
x = np.array([2, 3, 2, 4, 3, 2, 5, 3, 4, 2])
n = len(x)

# ----------------------------
# Conjunto de valores distintos D
# ----------------------------
D = np.unique(x)

# ----------------------------
# Frecuencias
# ----------------------------
frecuencias = Counter(x)

# ----------------------------
# PDF discreta
# ----------------------------
P = np.array([frecuencias[di] / n for di in D])

# ----------------------------
# CDF discreta
# ----------------------------
CDF = np.cumsum(P)

# ----------------------------
# Mostrar resultados
# ----------------------------
print("Conjunto D:", D)
print("\nTabla PDF y CDF:")
for di, pi, ci in zip(D, P, CDF):
    print(f"d = {di}, P(d_i) = {pi:.2f}, CDF(d_i) = {ci:.2f}")

print("\nSuma de probabilidades:", np.sum(P))

# ----------------------------
# Gráfica PDF (diagrama de barras)
# ----------------------------
plt.figure()
plt.bar(D, P)
plt.xlabel('Valores $d_i$')
plt.ylabel('Probabilidad $P(d_i)$')
plt.title('PDF discreta (diagrama de barras)')
plt.show()

# ----------------------------
# Gráfica CDF
# ----------------------------
plt.figure()
plt.step(D, CDF, where='post')
plt.xlabel('Valores $d_i$')
plt.ylabel('Probabilidad acumulada')
plt.title('CDF discreta')
plt.ylim(0, 1.05)
plt.show()
