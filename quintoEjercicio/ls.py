import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Posiciones de los satélites
# -----------------------------
sat_pos = np.array([
    [0.0, 0.0],
    [10.0, 0.0],
    [10.0, 10.0],
    [0.0, 10.0]
])

# -----------------------------
# Posición verdadera del receptor
# -----------------------------
x_true = np.array([4.5, 6.0])

# -----------------------------
# Generación de pseudodistancias
# -----------------------------
np.random.seed(1)
sigma = 0.2  # desviación estándar
distances = np.linalg.norm(sat_pos - x_true, axis=1)
noise = np.random.normal(0, sigma, size=4)
d_obs = distances + noise

# -----------------------------
# Aproximación inicial
# -----------------------------
x_hat = np.array([5.0, 5.0])

# -----------------------------
# Iteraciones LS
# -----------------------------
for _ in range(5):
    A = []
    l = []

    for i in range(4):
        dx = x_hat[0] - sat_pos[i, 0]
        dy = x_hat[1] - sat_pos[i, 1]
        r = np.sqrt(dx**2 + dy**2)

        A.append([dx / r, dy / r])
        l.append(d_obs[i] - r)

    A = np.array(A)
    l = np.array(l).reshape(-1, 1)

    dx = np.linalg.inv(A.T @ A) @ A.T @ l
    x_hat = x_hat + dx.flatten()

# -----------------------------
# Covarianza estimada
# -----------------------------
P = sigma**2 * np.linalg.inv(A.T @ A)

# -----------------------------
# Resultados
# -----------------------------
print("Posición verdadera:", x_true)
print("Estimación LS:", x_hat)
print("Covarianza estimada:\n", P)

# -----------------------------
# Elipse de error (2 sigma)
# -----------------------------
eigvals, eigvecs = np.linalg.eig(P)
theta = np.linspace(0, 2*np.pi, 100)
ellipse = np.array([
    2*np.sqrt(eigvals[0])*np.cos(theta),
    2*np.sqrt(eigvals[1])*np.sin(theta)
])
ellipse = eigvecs @ ellipse

# -----------------------------
# Gráfica
# -----------------------------
plt.figure(figsize=(7, 7))

# Satélites
plt.scatter(sat_pos[:, 0], sat_pos[:, 1], marker='^', s=120, label='Satélites')

# Receptor verdadero
plt.scatter(x_true[0], x_true[1], marker='*', s=200, label='Receptor verdadero')

# Estimación
plt.scatter(x_hat[0], x_hat[1], marker='X', s=150, label='Estimación LS')

# Elipse
plt.plot(
    ellipse[0] + x_hat[0],
    ellipse[1] + x_hat[1],
    label='Elipse de error 2σ'
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trilateración GNSS 2D por Mínimos Cuadrados")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
