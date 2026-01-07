"""
FILTRO DE KALMAN EXTENDIDO (EKF) CON REDUNDANCIA
Ejemplo: Modelo de Velocidad Constante (CV) con observación de rangos a 5 sensoress
Observación NO LINEAL: distancia a 5 sensoress conocidas
Redundancia: 5 sensores para mejor estimación
Menos ruido: σ = 0.5 m para evitar divergencia
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ExtendedKalmanFilter5sensores:
    """
    Implementación del Filtro de Kalman Extendido (EKF)
    para modelo CV con observación de rangos a 5 sensoress
    """
    
    def __init__(self, dt=1.0, process_noise_pos=0.01, process_noise_vel=0.001,
                 measurement_noise=0.25):  # σ = 0.5 m → var = 0.25 m²
        """
        Inicializa el filtro EKF con 5 sensoress
        
        Parámetros:
        -----------
        dt : float
            Paso de tiempo (s)
        process_noise_pos : float
            Ruido del proceso en posición (m²)
        process_noise_vel : float
            Ruido del proceso en velocidad (m²/s²)
        measurement_noise : float
            Varianza del ruido de medición de rango (m²)
        """
        self.dt = dt
        
        # Posiciones de las 5 sensoress (conocidas)
        # Distribuidas estratégicamente para buena cobertura
        self.sensores = np.array([
            [10.0, 0.0],     # Beacon 1: Este
            [0.0, 10.0],     # Beacon 2: Norte
            [-10.0, 0.0],    # Beacon 3: Oeste
            [0.0, -10.0],    # Beacon 4: Sur
            [7.0, 7.0]       # Beacon 5: Noreste (diagonal)
        ])
        
        self.n_sensores = len(self.sensores)
        
        # Modelo dinámico (LINEAL): Velocidad Constante
        # Estado: [pos_x, vel_x, pos_y, vel_y]ᵀ
        self.F = np.array([[1, dt, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]])
        
        # Ruido del proceso (Q) - más conservador
        self.Q = np.array([[process_noise_pos, 0, 0, 0],
                           [0, process_noise_vel, 0, 0],
                           [0, 0, process_noise_pos, 0],
                           [0, 0, 0, process_noise_vel]])
        
        # Ruido de medición (R) - 5 sensoress, menos ruido
        self.R = np.eye(self.n_sensores) * measurement_noise
        
        # Estado inicial y covarianza
        self.x = None
        self.P = None
        
        # Historial
        self.history = {
            'state': [],
            'covariance': [],
            'measurements': [],
            'predicted_measurements': [],
            'kalman_gain_norm': [],
            'innovation_norm': []
        }
    
    def initialize(self, initial_state, initial_covariance):
        """Inicializa el estado y covarianza del filtro"""
        self.x = initial_state.copy()
        self.P = initial_covariance.copy()
        
        # Guardar estado inicial
        self.history['state'].append(self.x.copy())
        self.history['covariance'].append(self.P.copy())
    
    def h(self, x):
        """
        Función de observación NO LINEAL
        Calcula distancia a cada una de las 5 sensoress
        
        Parámetros:
        -----------
        x : array (4,)
            Vector de estado [px, vx, py, vy]ᵀ
            
        Retorna:
        --------
        z_pred : array (5,)
            Rangos predichos a las 5 sensoress
        """
        px, vx, py, vy = x
        
        # Calcular distancia a cada sensores
        ranges = np.zeros(self.n_sensores)
        for i in range(self.n_sensores):
            dx = px - self.sensores[i, 0]
            dy = py - self.sensores[i, 1]
            ranges[i] = np.sqrt(dx**2 + dy**2)
        
        return ranges
    
    def H_jacobian(self, x):
        """
        Jacobiano de la función de observación h(x) para 5 sensoress
        
        Parámetros:
        -----------
        x : array (4,)
            Vector de estado
            
        Retorna:
        --------
        H : array (5, 4)
            Matriz Jacobiana ∂h/∂x
        """
        px, vx, py, vy = x
        
        # Calcular rangos actuales
        ranges = self.h(x)
        
        # Inicializar Jacobiano
        H = np.zeros((self.n_sensores, 4))
        
        # Derivadas para cada sensores
        for i in range(self.n_sensores):
            r = ranges[i]
            if r > 1e-6:  # Evitar división por cero
                H[i, 0] = (px - self.sensores[i, 0]) / r  # ∂r_i/∂px
                H[i, 2] = (py - self.sensores[i, 1]) / r  # ∂r_i/∂py
            # ∂r_i/∂vx = 0, ∂r_i/∂vy = 0 (las mediciones no dependen de velocidad)
        
        return H
    
    def predict(self):
        """Paso de predicción"""
        # Predicción del estado
        self.x = self.F @ self.x
        
        # Predicción de la covarianza
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy()
    
    def update(self, z_measured):
        """
        Paso de actualización con mediciones de 5 sensoress
        
        Parámetros:
        -----------
        z_measured : array (5,)
            Mediciones de rango a las 5 sensoress
        """
        # 1. Predicción de la medición usando función no lineal
        z_pred = self.h(self.x)
        
        # 2. Calcular Jacobiano en el estado predicho
        H = self.H_jacobian(self.x)
        
        # 3. Innovación (error entre medición y predicción)
        innovation = z_measured - z_pred
        
        # 4. Covarianza de innovación
        S = H @ self.P @ H.T + self.R
        
        # 5. Ganancia de Kalman
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Si S es singular, usar pseudoinversa
            K = self.P @ H.T @ np.linalg.pinv(S)
        
        # 6. Actualización del estado
        self.x = self.x + K @ innovation
        
        # 7. Actualización de la covarianza (forma de Joseph)
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        
        # Guardar histórico
        self.history['state'].append(self.x.copy())
        self.history['covariance'].append(self.P.copy())
        self.history['measurements'].append(z_measured.copy())
        self.history['predicted_measurements'].append(z_pred.copy())
        self.history['kalman_gain_norm'].append(np.linalg.norm(K))
        self.history['innovation_norm'].append(np.linalg.norm(innovation))
        
        return self.x.copy()

def simulate_true_trajectory(dt=1.0, total_time=100.0, initial_state=None):
    """
    Simula la trayectoria real del objeto
    
    Parámetros:
    -----------
    dt : float
        Paso de tiempo (s)
    total_time : float
        Tiempo total de simulación (s)
    initial_state : array (4,) o None
        Estado inicial [px0, vx0, py0, vy0]ᵀ
        
    Retorna:
    --------
    true_states : array (n_steps, 4)
        Estados reales en cada paso
    time : array (n_steps,)
        Vector de tiempo
    """
    n_steps = int(total_time / dt)
    time = np.arange(0, total_time, dt)
    
    if initial_state is None:
        # Estado inicial por defecto
        initial_state = np.array([0.0, 1.0, 0.0, 0.5])  # [px, vx, py, vy]
    
    true_states = np.zeros((n_steps, 4))
    true_states[0] = initial_state
    
    # Pequeño ruido en la dinámica real
    process_noise_real = np.array([0.001, 0.0001, 0.001, 0.0001])  # Menos ruido
    
    for k in range(1, n_steps):
        # Dinámica real con pequeño ruido
        F_real = np.array([[1, dt, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, dt],
                          [0, 0, 0, 1]])
        
        noise = np.random.normal(0, np.sqrt(process_noise_real))
        true_states[k] = F_real @ true_states[k-1] + noise
    
    return true_states, time

def generate_measurements_5sensores(true_states, sensores, measurement_noise_std=0.5):
    """
    Genera mediciones de rango ruidosas para 5 sensoress
    
    Parámetros:
    -----------
    true_states : array (n_steps, 4)
        Estados reales
    sensores : array (5, 2)
        Posiciones de las 5 sensoress
    measurement_noise_std : float
        Desviación estándar del ruido de medición (m)
        
    Retorna:
    --------
    measurements : array (n_steps, 5)
        Mediciones de rango a cada sensores
    """
    n_steps = true_states.shape[0]
    n_sensores = len(sensores)
    measurements = np.zeros((n_steps, n_sensores))
    
    for k in range(n_steps):
        px, _, py, _ = true_states[k]
        
        # Calcular rangos reales a cada sensores
        for i in range(n_sensores):
            dx = px - sensores[i, 0]
            dy = py - sensores[i, 1]
            r_true = np.sqrt(dx**2 + dy**2)
            
            # Añadir ruido Gaussiano (menos ruido)
            measurement_noise_std=0.5
            noise = np.random.normal(0, measurement_noise_std)
            measurements[k, i] = r_true + noise
    
    return measurements

def run_ekf_simulation_5sensores():
    """Ejecuta la simulación completa del EKF con 5 sensoress"""
    
    # ============================================
    # 1. CONFIGURACIÓN DE PARÁMETROS MEJORADOS
    # ============================================
    np.random.seed(42)  # Para reproducibilidad
    
    dt = 1.0  # Paso de tiempo (s)
    total_time = 100.0  # Tiempo total (s)
    n_steps = int(total_time / dt)
    
    # Parámetros del filtro - más conservadores para estabilidad
    process_noise_pos = .001 # m² 
    process_noise_vel = .0001 # m²/s² 
    measurement_noise_std = 0.4 # m 
    measurement_noise_var = measurement_noise_std**2  # m²
    
    print(f"Ruido de medición: σ = {measurement_noise_std:.2f} m")
    print(f"Varianza de medición: R = {measurement_noise_var:.3f} m²")
    
    # Estado inicial real
    true_initial_state = np.array([0.0, 1.0, 0.0, 0.5])  # [px, vx, py, vy]
    
    # Estado inicial para el filtro - más cercano al real
    filter_initial_state = np.array([0.5, 0.9, 0.3, 0.4])  # Pequeño error inicial
    
    # Covarianza inicial del filtro - más realista
    initial_covariance = np.diag([2.0, 0.5, 2.0, 0.5])  # Menos incertidumbre inicial
    
    # ============================================
    # 2. SIMULACIÓN DE TRAYECTORIA REAL
    # ============================================
    print("\n1. Simulando trayectoria real...")
    true_states, time = simulate_true_trajectory(dt, total_time, true_initial_state)
    
    # Posiciones de las 5 sensoress
    sensores = np.array([
        [10.0, 0.0],
        [0.0, 10.0],
        [-10.0, 0.0],
        [0.0, -10.0],
        [7.0, 7.0]
    ])
    
    print(f"sensoress configuradas: {len(sensores)} en total")
    for i, beacon in enumerate(sensores):
        print(f"  sensores {i+1}: ({beacon[0]:.1f}, {beacon[1]:.1f}) m")
    
    # ============================================
    # 3. GENERACIÓN DE MEDICIONES CON 5 sensoresS
    # ============================================
    print("\n2. Generando mediciones ruidosas de 5 sensoress...")
    measurements = generate_measurements_5sensores(true_states, sensores, measurement_noise_std)
    
    # ============================================
    # 4. INICIALIZACIÓN Y EJECUCIÓN DEL EKF
    # ============================================
    print("\n3. Ejecutando Filtro de Kalman Extendido...")
    
    # Crear filtro EKF con 5 sensoress
    ekf = ExtendedKalmanFilter5sensores(
        dt=dt,
        process_noise_pos=process_noise_pos,
        process_noise_vel=process_noise_vel,
        measurement_noise=measurement_noise_var
    )
    
    # Inicializar filtro
    ekf.initialize(filter_initial_state, initial_covariance)
    
    # Arrays para guardar resultados
    estimated_states = np.zeros((n_steps, 4))
    estimated_states[0] = filter_initial_state
    
    # Ejecutar filtro para cada paso de tiempo
    for k in range(1, n_steps):
        # 1. Predicción
        ekf.predict()
        
        # 2. Actualización con mediciones de las 5 sensoress
        estimated_states[k] = ekf.update(measurements[k])
    
    # Convertir historial a arrays
    history_states = np.array(ekf.history['state'])
    history_cov = np.array(ekf.history['covariance'])
    history_measurements = np.array(ekf.history['measurements'])
    
    # ============================================
    # 5. ANÁLISIS DE RESULTADOS
    # ============================================
    print("\n" + "="*70)
    print("ANÁLISIS DE RESULTADOS - EKF CON 5 sensoresS")
    print("="*70)
    
    # Calcular errores
    position_errors = np.sqrt((true_states[:, 0] - estimated_states[:, 0])**2 + 
                             (true_states[:, 2] - estimated_states[:, 2])**2)
    
    velocity_errors = np.sqrt((true_states[:, 1] - estimated_states[:, 1])**2 + 
                             (true_states[:, 3] - estimated_states[:, 3])**2)
    
    # Estadísticas
    print(f"\n--- ERRORES DE ESTIMACIÓN ---")
    print(f"Posición:")
    print(f"  • Error RMS: {np.sqrt(np.mean(position_errors**2)):.3f} m")
    print(f"  • Error máximo: {np.max(position_errors):.3f} m")
    print(f"  • Error medio: {np.mean(position_errors):.3f} m")
    print(f"  • Error inicial (primeros 10s): {np.mean(position_errors[:10]):.3f} m")
    print(f"  • Error final (últimos 10s): {np.mean(position_errors[-10:]):.3f} m")
    
    print(f"\nVelocidad:")
    print(f"  • Error RMS: {np.sqrt(np.mean(velocity_errors**2)):.3f} m/s")
    print(f"  • Error máximo: {np.max(velocity_errors):.3f} m/s")
    print(f"  • Error medio: {np.mean(velocity_errors):.3f} m/s")
    
    print(f"\n--- CONVERGENCIA DEL FILTRO ---")
    print(f"P inicial (diagonal): [{initial_covariance[0,0]:.2f}, {initial_covariance[1,1]:.2f}, "
          f"{initial_covariance[2,2]:.2f}, {initial_covariance[3,3]:.2f}]")
    print(f"P final (diagonal): [{history_cov[-1,0,0]:.4f}, {history_cov[-1,1,1]:.4f}, "
          f"{history_cov[-1,2,2]:.4f}, {history_cov[-1,3,3]:.4f}]")
    print(f"Reducción incertidumbre posición: {initial_covariance[0,0]/history_cov[-1,0,0]:.1f}x")
    
    print(f"\n--- INFORMACIÓN DE MEDICIONES ---")
    print(f"Número de sensoress: {len(sensores)}")
    print(f"Ruido de medición (σ): {measurement_noise_std:.2f} m")
    print(f"Varianza de medición (R): diag({measurement_noise_var:.3f})")
    print(f"Redundancia: {len(sensores)} mediciones por paso")
    print(f"Frecuencia de muestreo: {1/dt:.1f} Hz")
    
    print(f"\n--- INNOVACIÓN ---")
    innovation_norms = np.array(ekf.history['innovation_norm'])
    print(f"Norma media de innovación: {np.mean(innovation_norms):.3f} m")
    print(f"¿Innovación blanca? Media ≈ 0: {np.abs(np.mean(innovation_norms)) < 0.1}")
    
    return {
        'time': time,
        'true_states': true_states,
        'estimated_states': estimated_states,
        'measurements': measurements,
        'history_states': history_states,
        'history_cov': history_cov,
        'position_errors': position_errors,
        'velocity_errors': velocity_errors,
        'sensores': sensores,
        'ekf': ekf,
        'params': {
            'dt': dt,
            'measurement_noise_std': measurement_noise_std,
            'n_sensores': len(sensores)
        }
    }

def plot_results_5sensores(results):
    """Genera todas las gráficas para la presentación con 5 sensoress"""
    
    # Extraer datos de resultados
    time = results['time']
    true_states = results['true_states']
    estimated_states = results['estimated_states']
    measurements = results['measurements']
    history_cov = results['history_cov']
    position_errors = results['position_errors']
    sensores = results['sensores']
    ekf = results['ekf']
    params = results['params']
    
    # Colores para las 5 sensoress
    beacon_colors = ['red', 'green', 'blue', 'orange', 'purple']
    beacon_labels = ['sensores 1 (Este)', 'sensores 2 (Norte)', 'sensores 3 (Oeste)', 
                     'sensores 4 (Sur)', 'sensores 5 (Noreste)']
    
    # ============================================
    # GRÁFICA 1: Trayectoria y estimación con 5 sensoress
    # ============================================
    fig1, ax1 = plt.subplots(1, 2, figsize=(16, 7))
    
    # Subplot izquierdo: Trayectoria completa
    ax1[0].plot(true_states[:, 0], true_states[:, 2], 'k-', linewidth=3, 
                label='Trayectoria real', alpha=0.8)
    ax1[0].plot(estimated_states[:, 0], estimated_states[:, 2], 'b-', linewidth=2, 
                label='Estimación EKF', alpha=0.8)
    
    # Marcar las 5 sensoress
    for i, beacon in enumerate(sensores):
        ax1[0].scatter(beacon[0], beacon[1], s=250, marker='^', 
                      color=beacon_colors[i], label=beacon_labels[i], zorder=5)
        # Círculos de cobertura
        circle = Circle((beacon[0], beacon[1]), 15, fill=False, 
                       linestyle='--', alpha=0.2, color=beacon_colors[i])
        ax1[0].add_patch(circle)
    
    ax1[0].set_xlabel('Posición X (m)', fontsize=12)
    ax1[0].set_ylabel('Posición Y (m)', fontsize=12)
    ax1[0].set_title(f'EKF con {len(sensores)} sensoress - Trayectoria y Estimación', 
                    fontsize=14, fontweight='bold')
    ax1[0].legend(loc='upper left', fontsize=9)
    ax1[0].grid(True, alpha=0.3)
    ax1[0].axis('equal')
    ax1[0].set_xlim(-12, 12)
    ax1[0].set_ylim(-12, 12)
    
    # Subplot derecho: Error de posición vs tiempo
    ax1[1].plot(time, position_errors, 'r-', linewidth=2, label='Error de posición')
    ax1[1].axhline(y=np.mean(position_errors), color='blue', linestyle='--', 
                  label=f'Error medio: {np.mean(position_errors):.3f} m')
    ax1[1].fill_between(time, 0, position_errors, alpha=0.3, color='red')
    
    ax1[1].set_xlabel('Tiempo (s)', fontsize=12)
    ax1[1].set_ylabel('Error de posición (m)', fontsize=12)
    ax1[1].set_title('Error de Estimación vs Tiempo', fontsize=14, fontweight='bold')
    ax1[1].legend(loc='upper right', fontsize=10)
    ax1[1].grid(True, alpha=0.3)
    
    # Añadir estadísticas
    stats_text = f'5 sensoress, σ={params["measurement_noise_std"]:.2f} m\n'
    stats_text += f'Error RMS: {np.sqrt(np.mean(position_errors**2)):.3f} m\n'
    stats_text += f'Error máximo: {np.max(position_errors):.3f} m'
    ax1[1].text(0.02, 0.98, stats_text, transform=ax1[1].transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('ekf_5sensores_trajectory_error.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ============================================
    # GRÁFICA 2: Evolución de la incertidumbre (covarianza)
    # ============================================
    fig2, ax2 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Incertidumbre en posición X
    ax2[0, 0].plot(time, history_cov[:, 0, 0], 'b-', linewidth=2)
    ax2[0, 0].set_xlabel('Tiempo (s)', fontsize=10)
    ax2[0, 0].set_ylabel('Varianza X (m²)', fontsize=10)
    ax2[0, 0].set_title('Incertidumbre en Posición X', fontsize=12, fontweight='bold')
    ax2[0, 0].grid(True, alpha=0.3)
    ax2[0, 0].set_yscale('log')
    
    # Subplot 2: Incertidumbre en velocidad X
    ax2[0, 1].plot(time, history_cov[:, 1, 1], 'g-', linewidth=2)
    ax2[0, 1].set_xlabel('Tiempo (s)', fontsize=10)
    ax2[0, 1].set_ylabel('Varianza Vx (m²/s²)', fontsize=10)
    ax2[0, 1].set_title('Incertidumbre en Velocidad X', fontsize=12, fontweight='bold')
    ax2[0, 1].grid(True, alpha=0.3)
    ax2[0, 1].set_yscale('log')
    
    # Subplot 3: Incertidumbre en posición Y
    ax2[1, 0].plot(time, history_cov[:, 2, 2], 'b-', linewidth=2)
    ax2[1, 0].set_xlabel('Tiempo (s)', fontsize=10)
    ax2[1, 0].set_ylabel('Varianza Y (m²)', fontsize=10)
    ax2[1, 0].set_title('Incertidumbre en Posición Y', fontsize=12, fontweight='bold')
    ax2[1, 0].grid(True, alpha=0.3)
    ax2[1, 0].set_yscale('log')
    
    # Subplot 4: Incertidumbre en velocidad Y
    ax2[1, 1].plot(time, history_cov[:, 3, 3], 'g-', linewidth=2)
    ax2[1, 1].set_xlabel('Tiempo (s)', fontsize=10)
    ax2[1, 1].set_ylabel('Varianza Vy (m²/s²)', fontsize=10)
    ax2[1, 1].set_title('Incertidumbre en Velocidad Y', fontsize=12, fontweight='bold')
    ax2[1, 1].grid(True, alpha=0.3)
    ax2[1, 1].set_yscale('log')
    
    fig2.suptitle(f'Evolución de la Incertidumbre - {len(sensores)} sensoress', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('ekf_5sensores_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ============================================
    # GRÁFICA 3: Mediciones y predicciones para las 5 sensoress
    # ============================================
    fig3, ax3 = plt.subplots(3, 2, figsize=(14, 12))
    ax3 = ax3.flatten()
    
    predicted_measurements = np.array(ekf.history['predicted_measurements'])
    
    for i in range(len(sensores)):
        if i < len(ax3):
            ax3[i].plot(time, measurements[:, i], 'o', markersize=3, alpha=0.6, 
                       color=beacon_colors[i], label='Mediciones')
            ax3[i].plot(time[1:], predicted_measurements[:, i], '-', linewidth=1.5, 
                       color='black', label='Predicciones EKF')
            ax3[i].set_xlabel('Tiempo (s)', fontsize=9)
            ax3[i].set_ylabel('Rango (m)', fontsize=9)
            ax3[i].set_title(f'{beacon_labels[i]}', fontsize=11, fontweight='bold')
            ax3[i].legend(fontsize=8)
            ax3[i].grid(True, alpha=0.3)
    
    # Ocultar el último subplot si es necesario
    if len(sensores) < len(ax3):
        ax3[-1].axis('off')
    
    fig3.suptitle('Mediciones vs Predicciones para Cada sensores', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('ekf_5sensores_measurements.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ============================================
    # GRÁFICA 4: Innovación y ganancia de Kalman
    # ============================================
    fig4, ax4 = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calcular innovación para cada sensores
    innovation = measurements[1:,:] - predicted_measurements
    
    # Subplot izquierdo: Innovación para todas las sensoress
    for i in range(len(sensores)):
        ax4[0].plot(time[1:], innovation[:, i], '-', alpha=0.6, linewidth=1,
                   color=beacon_colors[i], label=f'sensores {i+1}')
    
    ax4[0].axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax4[0].axhline(y=2*params['measurement_noise_std'], color='k', linestyle='--', alpha=0.5)
    ax4[0].axhline(y=-2*params['measurement_noise_std'], color='k', linestyle='--', alpha=0.5)
    
    ax4[0].set_xlabel('Tiempo (s)', fontsize=12)
    ax4[0].set_ylabel('Innovación (m)', fontsize=12)
    ax4[0].set_title('Innovación por sensores', fontsize=14, fontweight='bold')
    ax4[0].legend(fontsize=9, ncol=2)
    ax4[0].grid(True, alpha=0.3)
    
    # Subplot derecho: Norma de la ganancia de Kalman
    kalman_gain_norm = np.array(ekf.history['kalman_gain_norm'])
    ax4[1].plot(time[1:], kalman_gain_norm, 'purple', linewidth=2)
    ax4[1].set_xlabel('Tiempo (s)', fontsize=12)
    ax4[1].set_ylabel('Norma de K', fontsize=12)
    ax4[1].set_title('Norma de la Ganancia de Kalman', fontsize=14, fontweight='bold')
    ax4[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ekf_5sensores_innovation_gain.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ============================================
    
    fig6, ax6 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

	# Errores
    error_x = true_states[:, 0] - estimated_states[:, 0]
    error_y = true_states[:, 2] - estimated_states[:, 2]

	# Desviaciones estándar (σ)
    sigma_x = np.sqrt(history_cov[:, 0, 0])
    sigma_y = np.sqrt(history_cov[:, 2, 2])
    # ---------- Error en X ----------
    ax6[0].plot(time, error_x, 'r-', linewidth=2, label='Error X')
    ax6[0].fill_between(time, -2*sigma_x, 2*sigma_x,color='gray', alpha=0.3, label='±2σ (95%)')
    ax6[0].axhline(0, color='k', linestyle='--')
    ax6[0].set_ylabel('Error X (m)')
    ax6[0].set_title('Error de Posición en X con Envoltura ±2σ')
    ax6[0].legend()
    ax6[0].grid(True, alpha=0.3)
    ax6[1].plot(time, error_y, 'b-', linewidth=2, label='Error Y')
    ax6[1].fill_between(time, -2*sigma_y, 2*sigma_y,color='gray', alpha=0.3, label='±2σ (95%)')
    ax6[1].axhline(0, color='k', linestyle='--')
    ax6[1].set_xlabel('Tiempo (s)')
    ax6[1].set_ylabel('Error Y (m)')
    ax6[1].set_title('Error de Posición en Y con Envoltura ±2σ')
    ax6[1].legend()
    ax6[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ekf_5sensores_error_xy_2sigma.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    
    fig7, ax7 = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

	# Errores
    error_x = true_states[:, 1] - estimated_states[:, 1]
    error_y = true_states[:, 3] - estimated_states[:, 3]

	# Desviaciones estándar (σ)
    sigma_x = np.sqrt(history_cov[:, 1, 1])
    sigma_y = np.sqrt(history_cov[:, 3, 3])
    # ---------- Error en X ----------
    ax7[0].plot(time, error_x, 'r-', linewidth=2, label='Error VX')
    ax7[0].fill_between(time, -2*sigma_x, 2*sigma_x,color='gray', alpha=0.3, label='±2σ (95%)')
    ax7[0].axhline(0, color='k', linestyle='--')
    ax7[0].set_ylabel('Error VX (m s)')
    ax7[0].set_title('Error de VElocidad en X con Envoltura ±2σ')
    ax7[0].legend()
    ax7[0].grid(True, alpha=0.3)
    ax7[1].plot(time, error_y, 'b-', linewidth=2, label='Error VY')
    ax7[1].fill_between(time, -2*sigma_y, 2*sigma_y,color='gray', alpha=0.3, label='±2σ (95%)')
    ax7[1].axhline(0, color='k', linestyle='--')
    ax7[1].set_xlabel('Tiempo (s)')
    ax7[1].set_ylabel('Error VY (m s)')
    ax7[1].set_title('Error de Velocdiad en Y con Envoltura ±2σ')
    ax7[1].legend()
    ax7[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ekf_5sensores_error_xy_2sigma.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n¡Gráficas generadas y guardadas exitosamente!")
    print("="*60)
    print("ARCHIVOS CREADOS:")
    print("1. ekf_5sensores_trajectory_error.png - Trayectoria y error")
    print("2. ekf_5sensores_uncertainty.png - Evolución de incertidumbre")
    print("3. ekf_5sensores_measurements.png - Mediciones vs predicciones")
    print("4. ekf_5sensores_innovation_gain.png - Innovación y ganancia")
    print("5. ekf_5sensores_comparison.png - Comparación configuraciones")
    print("="*60)

# ============================================
# EJECUCIÓN PRINCIPAL
# ============================================
if __name__ == "__main__":
    print("="*70)
    print("FILTRO DE KALMAN EXTENDIDO - 5 sensoresS CON REDUNDANCIA")
    print("="*70)
    print("Configuración mejorada:")
    print("• 5 sensoress estratégicamente distribuidas")
    print("• Ruido de medición reducido: σ = 0.5 m")
    print("• Redundancia de mediciones para mayor estabilidad")
    print("• Modelo CV (posición + velocidad)")
    print("="*70)

    # Ejecutar simulación EKF
    results = run_ekf_simulation_5sensores()

    # Graficar resultados
    plot_results_5sensores(results)

    print("\nSimulación finalizada correctamente.")
    print("Resultados listos para análisis y presentación.")

