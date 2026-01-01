import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# Observaciones (n x 2)
# --------------------------------
X = np.array([
    [10.9, 3.1],
    [9.8, 4.9],
    [10.2, 5.3],
    [10.0, 4.8],
    [10.1, 5.0]
])

n, s = X.shape

# --------------------------------
# Media multivariada (no ponderada)
# --------------------------------
mu_mean = np.mean(X, axis=0)

# --------------------------------
# Pesos de las observaciones
# (observaciones con distinta calidad)
# --------------------------------
# Mayor peso = mayor confianza
weights = np.array([5, 1, 3, 1, 5])

# Construcción de matriz de pesos P
P = np.diag(np.repeat(weights, s))

# --------------------------------
# Formulación WLS: l = A * mu
# --------------------------------
A = np.vstack([np.eye(s) for _ in range(n)])
l = X.reshape(-1, 1)

# --------------------------------
# Mínimos cuadrados ponderados
# --------------------------------
mu_wls = np.linalg.inv(A.T @ P @ A) @ A.T @ P @ l
mu_wls = mu_wls.flatten()

# --------------------------------
# Resultados numéricos
# --------------------------------
print("Media multivariada:")
print(mu_mean)

print("\nEstimación por mínimos cuadrados ponderados:")
print(mu_wls)

print("\nDiferencia (WLS - Media):")
print(mu_wls - mu_mean)

# --------------------------------
# Gráfica
# --------------------------------
plt.figure(figsize=(7, 6))

# Observaciones (tamaño proporcional al peso)
plt.scatter(
    X[:, 0], X[:, 1],
    s=weights * 80,
    alpha=0.7,
    label="Observaciones (peso)"
)

# Media
plt.scatter(
    mu_mean[0], mu_mean[1],
    marker="o",
    s=150,
    facecolors="none",
    color="red",
    label="Media"
)

# WLS
plt.scatter(
    mu_wls[0], mu_wls[1],
    marker="X",
    s=180,
    label="WLS"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Media vs Mínimos Cuadrados Ponderados")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
