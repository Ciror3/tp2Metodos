import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos
g = 9.81
l = 1.0   # Longitud de la vara del péndulo (metros)
m = 1.0   # Masa de la partícula (kg)

# Función que define las ecuaciones diferenciales del péndulo
def f(theta_omega):
    theta, omega = theta_omega
    omega0_squared = g / l
    dtheta_dt = omega
    domega_dt = -omega0_squared * np.sin(theta)
    return [dtheta_dt, domega_dt]


# Método de Runge-Kutta de orden 4
def rungeKuttaOrden4(a, b, N, alpha, f):
    h = (b - a) / N
    t = a
    w = alpha

    # Listas para almacenar los valores de tiempo, ángulo (theta), energía total (E), energía cinética (T) y energía potencial (V)
    time_values = []
    theta_values = []
    E_values = []
    T_values = []
    V_values = []

    for i in range(N):
        K1 = h * np.array(f(w))
        K2 = h * np.array(f(w + K1 / 2))
        K3 = h * np.array(f(w + K2 / 2))
        K4 = h * np.array(f(w + K3))

        w = w + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        t = a + i * h

        # Calcular la energía total (E), energía cinética (T) y energía potencial (V)
        theta, omega = w
        T = 0.5 * m * l**2 * omega**2
        V = -m * g * l * np.cos(theta) + m*g*l
        E = T + V

        # Almacenar los valores en las listas
        time_values.append(t)
        theta_values.append(theta)
        E_values.append(E)
        T_values.append(T)
        V_values.append(V)

    return time_values, theta_values, E_values, T_values, V_values

def euler_explicit(a, b, N, alpha, f):
    h = (b - a) / N
    t = a
    theta, omega = alpha

    time_values = []
    theta_values = []
    E_values = []
    T_values = []
    V_values = []

    for i in range(N):
        dtheta_dt, domega_dt = f([theta, omega])

        # Actualizar θ y ω utilizando el método de Euler
        theta = theta + h * dtheta_dt
        omega = omega + h * domega_dt
        t = a + i * h

        # Calcular la energía total (E), energía cinética (T) y energía potencial (V)
        T = 0.5 * m * l**2 * omega**2
        V = -m * g * l * np.cos(theta)+ m*g*l
        E = T + V

        time_values.append(t)
        theta_values.append(theta)
        E_values.append(E)
        T_values.append(T)
        V_values.append(V)

    return time_values, theta_values, E_values, T_values, V_values

# Parámetros de la simulación
a = 0.0       # Tiempo inicial
b = 10.0      # Tiempo final
N = 1000      # Número de pasos de tiempo
alpha = [np.pi / 6, 0]  # Condiciones iniciales: ángulo inicial y velocidad angular inicial

# Ejecutar el método de Runge-Kutta
timeR, thetaR, E, T, V = rungeKuttaOrden4(a, b, N, alpha, f)

# Llama a la función euler_explicit
timeE, thetaE, Ee, Te, Ve = euler_explicit(a, b, N, alpha, f)

# Graficar las trayectorias de ángulo theta y energía
plt.figure(figsize=(12, 6))

# Trayectoria del ángulo theta para Runge-Kutta
plt.subplot(2, 1, 1)
plt.plot(timeR, thetaR, label='Runge-Kutta')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (θ)')
plt.title('Trayectoria del Péndulo Simple')
plt.legend()

# Trayectoria del ángulo theta para Euler explícito
plt.subplot(2, 1, 2)
plt.plot(timeE, thetaE, label='Euler Explícito', color='orange')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (θ)')
plt.title('Trayectoria del Péndulo Simple')
plt.legend()

plt.tight_layout()
plt.show()

# Graficar la evolución de la energía para Runge-Kutta
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(timeR, E, label='Energía Total (E)')
plt.plot(timeR, T, label='Energía Cinética (T)')
plt.plot(timeR, V, label='Energía Potencial (V)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Energía')
plt.title('Evolución de la Energía en el Péndulo Simple (Runge-Kutta)')
plt.legend()

# Graficar la evolución de la energía para Euler explícito
plt.subplot(2, 1, 2)
plt.plot(timeE, Ee, label='Energía Total (E)', color='blue')
plt.plot(timeE, Te, label='Energía Cinética (T)', color='orange')
plt.plot(timeE, Ve, label='Energía Potencial (V)', color='green')
plt.xlabel('Tiempo (s)')
plt.ylabel('Energía')
plt.title('Evolución de la Energía en el Péndulo Simple (Euler Explícito)')
plt.legend()

plt.tight_layout()
plt.show()

