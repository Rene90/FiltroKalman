import numpy as np
import matplotlib.pyplot as plt

# satélites
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
sigma = 0.2  # desviación estándar base
distances = np.linalg.norm(sat_pos - x_true, axis=1)
noise = np.random.normal(0, sigma, size=4)
d_obs = distances + noise

# -----------------------------
# Aproximación inicial
# -----------------------------
x_hat = np.array([5.0, 5.0])

# -----------------------------
# Configuración para convergencia
# -----------------------------
max_iter = 20
convergence_threshold = 1e-6
converged = False

# Arreglos para almacenar información de cada iteración
x_history = [x_hat.copy()]
P_history = []  # Para guardar matrices de covarianza
dx_norm_history = []  # Para guardar norma de la corrección

# -----------------------------
# Iteraciones LS con criterio de convergencia
# -----------------------------

for iter_num in range(max_iter):
    
    A = []
    l = []
    weights = []  # Pesos basados en distancia inversa
    
    for i in range(4):
        dx = x_hat[0] - sat_pos[i, 0]
        dy = x_hat[1] - sat_pos[i, 1]
        r = np.sqrt(dx**2 + dy**2)
        
        # Matriz de diseño (Jacobiano)
        A.append([dx / r, dy / r])
        
        # Término de observación
        l.append(d_obs[i] - r)
        
        # Peso inversamente proporcional a la distancia
        # (simula mayor incertidumbre en satélites lejanos)
        weight = 1.0 / (r + 0.1)  # +0.1 para evitar división por cero
        weights.append(weight)
    
    A = np.array(A)
    l = np.array(l).reshape(-1, 1)
    
    # Matriz de pesos W (diagonal con pesos)
    W = np.diag(weights)
    
    # Solución con pesos: Δx = (AᵀWA)⁻¹AᵀWl
    try:
        dx = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ l
    except np.linalg.LinAlgError:
        # Si la matriz es singular, usar mínimos cuadrados sin pesos
        dx = np.linalg.pinv(A) @ l
    
    dx_norm = np.linalg.norm(dx)
    dx_norm_history.append(dx_norm)
    
    # Actualizar posición
    x_hat = x_hat + dx.flatten()
    x_history.append(x_hat.copy())
    
    # Calcular matriz de covarianza para esta iteración
    # Covarianza ponderada: P = σ²(AᵀWA)⁻¹
    # Usamos un sigma promedio ponderado
    sigma_weighted = sigma * np.mean(weights)
    try:
        P = sigma_weighted**2 * np.linalg.inv(A.T @ W @ A)
    except np.linalg.LinAlgError:
        P = np.eye(2) * 100  # Valor grande si no es invertible
    
    P_history.append(P.copy())
    
    # Verificar convergencia
    if dx_norm < convergence_threshold:
        converged = True
        print(f"Convergió en {iter_num + 1} iteraciones")
        break

if not converged:
    print(f"No convergió después de {max_iter} iteraciones")

# -----------------------------
# Resultados finales
# -----------------------------
print("\n" + "="*50)
print("RESULTADOS FINALES")
print("="*50)
print("Posición verdadera:", x_true)
print("Estimación LS:", x_hat)
print("Error absoluto:", np.linalg.norm(x_hat - x_true))
print("Matriz de covarianza final:\n", P)

# -----------------------------
# Gráfica 1: Posición y elipse de error final
# -----------------------------
plt.figure(figsize=(14, 6))

# Subplot 1: Posición y elipse
plt.subplot(1, 1, 1)

# Satélites
plt.scatter(sat_pos[:, 0], sat_pos[:, 1], marker='^', s=120, 
            label='Satélites', color='blue', zorder=5)

# Receptor verdadero
plt.scatter(x_true[0], x_true[1], marker='*', s=200, 
            label='Receptor verdadero', color='green', zorder=5)

# Estimación final
plt.scatter(x_hat[0], x_hat[1], marker='X', s=150, 
            label='Estimación final', color='red', zorder=5)

# Trayectoria de convergencia
x_history = np.array(x_history)
plt.plot(x_history[:, 0], x_history[:, 1], 'r--', alpha=0.5, 
         label='Trayectoria iteraciones', linewidth=1)

# Puntos de cada iteración
plt.scatter(x_history[1:, 0], x_history[1:, 1], c=range(len(x_history)-1), 
            cmap='viridis', s=50, alpha=0.7, label='Iteraciones', zorder=4)

# Elipse de error final (2 sigma)
if len(P_history) > 0:
    P_final = P_history[-1]
    eigvals, eigvecs = np.linalg.eig(P_final)
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse = np.array([
        2*np.sqrt(eigvals[0])*np.cos(theta),
        2*np.sqrt(eigvals[1])*np.sin(theta)
    ])
    ellipse = eigvecs @ ellipse
    plt.plot(ellipse[0] + x_hat[0], ellipse[1] + x_hat[1], 
             'orange', linewidth=2, label='Elipse de error 2σ (final)')

plt.xlabel("X [unidades]")
plt.ylabel("Y [unidades]")
plt.title("Trilateración GNSS 2D - Mínimos Cuadrados Ponderados")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.axis("equal")

## Gráfica 2: Dos subplots separados
plt.figure(figsize=(12, 5))
# Extraer varianzas de cada iteración
var_x = [P[0, 0] for P in P_history]
var_y = [P[1, 1] for P in P_history]
cov_xy = [P[0, 1] for P in P_history]  # Covarianza cruzada
iter_num = list(range(1, len(P_history) + 1))
# Subplot izquierdo: Varianzas
"""plt.subplot(1, 2, 1)
plt.plot(iter_num, var_x, 'b-o', label='Varianza X (σ²ₓ)', linewidth=2, markersize=6)
plt.plot(iter_num, var_y, 'r-s', label='Varianza Y (σ²ᵧ)', linewidth=2, markersize=6)
plt.plot(iter_num, cov_xy, 'g-^', label='Covarianza XY (σₓᵧ)', linewidth=2, markersize=6, alpha=0.7)
plt.xlabel("Número de iteración")
plt.ylabel("Valor de varianza/covarianza")
plt.title("Convergencia de Incertidumbres")
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.legend(loc='upper right')"""

# Subplot derecho: Norma de corrección
plt.subplot(1, 1, 1)
plt.plot(iter_num, dx_norm_history, 'k--*', label='||Δx||', linewidth=2, markersize=8)
plt.axhline(y=convergence_threshold, color='r', linestyle=':', 
            alpha=0.8, label=f'Umbral ({convergence_threshold:.0e})')
plt.xlabel("Número de iteración")
plt.ylabel('Norma de corrección ||Δx||')
plt.title("Convergencia de la Corrección")
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.legend(loc='upper right')

# Añadir valor final como texto
plt.text(iter_num[-1], dx_norm_history[-1], 
         f'  Final: {dx_norm_history[-1]:.2e}', 
         verticalalignment='center', fontsize=10)

plt.tight_layout()
plt.show()
# -----------------------------
# Gráfica 3: Elipses de error por iteración
# -----------------------------
plt.figure(figsize=(10, 8))

# Configurar colores para las elipses (transparencia decreciente)
colors = plt.cm.viridis(np.linspace(0.2, 1, len(P_history)))

for i, (P_i, x_i, color) in enumerate(zip(P_history, x_history[1:], colors)):
    # Calcular elipse para esta iteración
    eigvals, eigvecs = np.linalg.eig(P_i)
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse = np.array([
        2*np.sqrt(eigvals[0])*np.cos(theta),
        2*np.sqrt(eigvals[1])*np.sin(theta)
    ])
    ellipse = eigvecs @ ellipse
    
    # Dibujar elipse
    plt.plot(ellipse[0] + x_i[0], ellipse[1] + x_i[1], 
             color=color, linewidth=1.5, alpha=0.7,
             label=f'Iter {i+1}' if i % max(1, len(P_history)//5) == 0 else None)
    
    # Punto central de la iteración
    plt.scatter(x_i[0], x_i[1], color=color, s=30, alpha=0.7)

# Elementos de referencia
plt.scatter(sat_pos[:, 0], sat_pos[:, 1], marker='^', s=120, 
            label='Satélites', color='blue', zorder=10)
plt.scatter(x_true[0], x_true[1], marker='*', s=200, 
            label='Posición verdadera', color='green', zorder=10)
plt.scatter(x_history[-1, 0], x_history[-1, 1], marker='X', s=150, 
            label='Estimación final', color='red', zorder=10)

plt.xlabel("X [unidades]")
plt.ylabel("Y [unidades]")
plt.title("Evolución de Elipses de Error 2σ por Iteración")
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.axis("equal")

plt.tight_layout()
plt.show()

# -----------------------------
# Información adicional de convergencia
# -----------------------------
print("\n" + "="*50)
print("ANÁLISIS DE CONVERGENCIA")
print("="*50)
print(f"Número de iteraciones: {len(P_history)}")
print(f"Norma de última corrección: {dx_norm_history[-1]:.2e}")
print("\nEvolución de varianzas:")
print(f"  Varianza X inicial: {var_x[0]:.4f}, final: {var_x[-1]:.4f}")
print(f"  Varianza Y inicial: {var_y[0]:.4f}, final: {var_y[-1]:.4f}")
print(f"  Reducción factor X: {var_x[0]/var_x[-1]:.2f}x")
print(f"  Reducción factor Y: {var_y[0]/var_y[-1]:.2f}x")
