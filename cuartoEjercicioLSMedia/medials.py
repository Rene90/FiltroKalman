import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# Observaciones (n x s)
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
# Formulación de mínimos cuadrados
# l = A * mu + v
# --------------------------------
A = np.vstack([np.eye(s) for _ in range(n)])
print ("Matriz A: " , A)
l = X.reshape(-1, 1)
print ("Matriz l: ",l)
# --------------------------------
# Estimación por mínimos cuadrados
# --------------------------------
mu_hat_ls = np.linalg.inv(A.T @ A) @ A.T @ l
mu_hat_ls = mu_hat_ls.flatten()

# --------------------------------
# Media directa
# --------------------------------
mu_hat_mean = np.mean(X, axis=0)

# --------------------------------
# Covarianza muestral
# --------------------------------
residuals = X - mu_hat_ls
Sigma_hat = (residuals.T @ residuals) / (n - 1)

# --------------------------------
# Impresión de resultados
# --------------------------------
print("Resultado por mínimos cuadrados:", mu_hat_ls)
print("Resultado por media multivariada:", mu_hat_mean)
print("Diferencia:", mu_hat_ls - mu_hat_mean)
print("\nCovarianza estimada:\n", Sigma_hat)

# --------------------------------
# Gráfica
# --------------------------------
plt.figure(figsize=(7, 6))

# Observaciones
plt.scatter(
    X[:, 0], X[:, 1],
    color="blue",
    alpha=0.7,
    label="Observaciones"
)

# Estimación por mínimos cuadrados
plt.scatter(
    mu_hat_ls[0], mu_hat_ls[1],
    color="red",
    marker="X",
    s=150,
    label="Estimación LS"
)

# Media multivariada
plt.scatter(
    mu_hat_mean[0], mu_hat_mean[1],
    color="green",
    marker="o",
    s=120,
    facecolors="none",
    label="Media multivariada"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Estimación del vector de estado: LS vs Media")
plt.legend()
plt.grid(True)
plt.axis("equal")

plt.show()
