import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Estado verdadero (desconocido)
# -----------------------------
x_true = np.array([10.0, 5.0])  # posición real (x, y)

# -----------------------------
# Covarianza del ruido
# -----------------------------
Sigma = np.array([[1.0, 0.3],
                  [0.3, 1.5]])

# -----------------------------
# Generación de mediciones
# -----------------------------
np.random.seed(42)
n = 50  # número de observaciones

noise = np.random.multivariate_normal(
    mean=[0, 0],
    cov=Sigma,
    size=n
)

measurements = x_true + noise

# -----------------------------
# Estimación: media multivariada
# -----------------------------
x_hat = np.mean(measurements, axis=0)

# -----------------------------
# Covarianza muestral
# -----------------------------
Sigma_hat = np.cov(measurements.T)

# -----------------------------
# Resultados numéricos
# -----------------------------
print("Estado verdadero:")
print(x_true)

print("\nEstimación (media multivariada):")
print(x_hat)

print("\nCovarianza muestral estimada:")
print(Sigma_hat)

# -----------------------------
# Gráfica
# -----------------------------
plt.figure(figsize=(7, 6))

# Mediciones
plt.scatter(
    measurements[:, 0],
    measurements[:, 1],
    alpha=0.6,
    label="Mediciones"
)

# Estado verdadero
plt.scatter(
    x_true[0],
    x_true[1],
    marker="*",
    s=200,
    label="Estado verdadero"
)

# Estimación (media)
plt.scatter(
    x_hat[0],
    x_hat[1],
    marker="X",
    s=150,
    label="Estimación (media)"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Estimación del vector de estado por media multivariada")
plt.legend()
plt.grid(True)
plt.axis("equal")

plt.show()
