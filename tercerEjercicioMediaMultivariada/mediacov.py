import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# -----------------------------
# Estado verdadero (desconocido)
# -----------------------------
x_true = np.array([10.0, 5.0])

# -----------------------------
# Covarianza del ruido
# -----------------------------
Sigma = np.array([[1.0, 0.3],
                  [0.3, 1.5]])

# -----------------------------
# Generación de mediciones
# -----------------------------
np.random.seed(42)
n = 50

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
# Autovalores y autovectores
# -----------------------------
eigvals, eigvecs = np.linalg.eig(Sigma_hat)

# Ordenar de mayor a menor
order = eigvals.argsort()[::-1]
eigvals = eigvals[order]
eigvecs = eigvecs[:, order]

# -----------------------------
# Parámetros de la elipse 2-sigma
# -----------------------------
chi_square_val = 4.0  # 2-sigma en 2D (aprox)
width = 2 * np.sqrt(chi_square_val * eigvals[0])
height = 2 * np.sqrt(chi_square_val * eigvals[1])

angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

# -----------------------------
# Gráfica
# -----------------------------
plt.figure(figsize=(7, 6))

plt.scatter(measurements[:, 0], measurements[:, 1],
            alpha=0.6, label="Mediciones")

plt.scatter(x_true[0], x_true[1],
            marker="*", s=200, label="Estado verdadero")

plt.scatter(x_hat[0], x_hat[1],
            marker="X", s=150, label="Estimación (media)")

# Elipse de error
ellipse = Ellipse(
    xy=x_hat,
    width=width,
    height=height,
    angle=angle,
    edgecolor="red",
    facecolor="none",
    linewidth=2,
    label="Elipse de error 2σ"
)

plt.gca().add_patch(ellipse)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Estimación multivariada y elipse de error 2σ")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
