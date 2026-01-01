import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# --------------------------------
# Función para dibujar el elipse
# --------------------------------
def plot_error_ellipse(mean, cov, ax, n_sigma=2, **kwargs):
    # Autovalores y autovectores
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Ordenar de mayor a menor
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Ángulo del elipse
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Semiejes
    width, height = 2 * n_sigma * np.sqrt(eigvals)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        **kwargs
    )
    ax.add_patch(ellipse)

# --------------------------------
# Observaciones (n x 2)
# --------------------------------
X = np.array([
    [10.5, 5.1],
    [9.8, 4.9],
    [10.2, 5.3],
    [10.0, 4.8],
    [10.1, 5.0]
])

n, s = X.shape

# --------------------------------
# Media multivariada
# --------------------------------
mu_mean = np.mean(X, axis=0)
Sigma_mean = np.cov(X.T)

# --------------------------------
# Pesos
# --------------------------------
weights = np.array([5, 1, 3, 1, 5])
P = np.diag(np.repeat(weights, s))

# --------------------------------
# Formulación WLS
# --------------------------------
A = np.vstack([np.eye(s) for _ in range(n)])
l = X.reshape(-1, 1)

mu_wls = np.linalg.inv(A.T @ P @ A) @ A.T @ P @ l
mu_wls = mu_wls.flatten()

Sigma_wls = np.linalg.inv(A.T @ P @ A)

# --------------------------------
# Resultados
# --------------------------------
print("Media multivariada:")
print(mu_mean)
print("\nCovarianza (media):")
print(Sigma_mean)

print("\nEstimación WLS:")
print(mu_wls)
print("\nCovarianza (WLS):")
print(Sigma_wls)

# --------------------------------
# Gráfica
# --------------------------------
fig, ax = plt.subplots(figsize=(7, 6))

# Observaciones
ax.scatter(
    X[:, 0], X[:, 1],
    s=weights * 80,
    alpha=0.7,
    label="Observaciones (peso)"
)

# Media
ax.scatter(
    mu_mean[0], mu_mean[1],
    marker="o",
    s=150,
    facecolors="none",
    color="red",
    label="Media"
)

# WLS
ax.scatter(
    mu_wls[0], mu_wls[1],
    marker="X",
    s=180,
    color="blue",
    label="WLS"
)

# Elipses de error (2σ)
plot_error_ellipse(
    mu_mean, Sigma_mean, ax,
    n_sigma=2,
    edgecolor="red",
    linestyle="--",
    linewidth=2,
    label="Elipse 2σ (Media)"
)

plot_error_ellipse(
    mu_wls, Sigma_wls, ax,
    n_sigma=2,
    edgecolor="blue",
    linewidth=2,
    label="Elipse 2σ (WLS)"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Media vs WLS con elipses de error (2σ)")
ax.legend()
ax.grid(True)
ax.axis("equal")

plt.show()
