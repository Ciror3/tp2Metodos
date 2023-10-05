import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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

def linearized_pendulum(t, state):
    theta, omega = state
    omega0_squared = g / l
    dtheta_dt = omega
    domega_dt = -omega0_squared * theta
    return [dtheta_dt, domega_dt]

# Método de Runge-Kutta de orden 4
def rungeKuttaOrden4(a, b, N, alpha, f):
    h = (b - a) / N
    t = a
    w = alpha

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

initial_thetas = [np.pi / 6, np.pi / 4, np.pi / 3]
theta_labels = [r'$\frac{\pi}{6}$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{3}$']

# Simulation parameters
a = 0.0
b = 10.0
N = 1000

# Store the results for each set of initial conditions
results_euler = []
results_rk4 = []

for initial_theta in initial_thetas:
    # Initial conditions for Euler and Runge-Kutta (θ(t = 0) and θ̇(t = 0) = 0)
    alpha = [initial_theta, 0]

    # Solve with Euler
    timeE, thetaE, Ee, Te, Ve = euler_explicit(a, b, N, alpha, f)
    results_euler.append((timeE, thetaE, Ee, Te, Ve))

    # Solve with Runge-Kutta
    timeR, thetaR, Er, Tr, Vr = rungeKuttaOrden4(a, b, N, alpha, f)
    results_rk4.append((timeR, thetaR, Er, Tr, Vr))


fig_traj, axes_traj = plt.subplots(len(initial_thetas), 1, figsize=(10, 8))
fig_traj.suptitle('Comparison of Pendulum Trajectories (Runge-Kutta vs. Euler)')
theta0 = 0.1  # Valor inicial de theta
omega0 = 0.0
t_span = (0, 10)  # Intervalo de tiempo de 0 a 10 segundos
t_eval = np.linspace(*t_span, 1000)  # Puntos de evaluación

# Resuelve el sistema de ecuaciones diferenciales
sol = solve_ivp(linearized_pendulum, t_span, [theta0, omega0], t_eval=t_eval)

# Grafica los resultados
plt.figure(figsize=(10, 4))
plt.plot(sol.t, sol.y[0], label='Theta(t)')
plt.plot(sol.t, sol.y[1], label='Omega(t)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Valor')
plt.legend()
plt.title('Evolución de Theta y Omega en un péndulo linealizado')
plt.grid(True)

# Loop through each initial condition
for i, (initial_theta, label) in enumerate(zip(initial_thetas, theta_labels)):
    # Plot the pendulum trajectory for Runge-Kutta
    axes_traj[i].plot(results_rk4[i][0], results_rk4[i][1], label=f'Runge-Kutta', color='red')

    # Plot the pendulum trajectory for Euler
    axes_traj[i].plot(results_euler[i][0], results_euler[i][1], label=f'Euler', color='blue')

    # Set labels and titles for each subplot
    axes_traj[i].set_xlabel('Time (s)')
    axes_traj[i].set_ylabel('Angle (θ)')
    axes_traj[i].set_title(f'Pendulum Trajectory (θ(t=0)={label})')
    axes_traj[i].legend()




# Create subplots for energy comparison (total, kinetic, and potential) for different initial angles
fig, axs_energy = plt.subplots(len(initial_thetas), 1, figsize=(10, 8), sharex=True)

# Plot the energy evolution for each initial condition (Runge-Kutta)
for i, (initial_theta, label) in enumerate(zip(initial_thetas, theta_labels)):
    # Plot the energy evolution for each initial condition (Euler)
    axs_energy[i].plot(results_euler[i][0], results_euler[i][3], label=f'Kinetic',color = 'green')
    axs_energy[i].plot(results_euler[i][0], results_euler[i][2], label=f'Total',color = 'blue')
    axs_energy[i].plot(results_euler[i][0], results_euler[i][4], label=f'Potential',color = 'orange')

    # Set labels and titles for each energy subplot
    axs_energy[i].set_ylabel('Energy')
    axs_energy[i].set_title(f'Energy Comparison (θ(t=0)={label})')
    axs_energy[i].legend()

fig, axs_energyR = plt.subplots(len(initial_thetas), 1, figsize=(10, 8), sharex=True)

for i, (initial_theta, label) in enumerate(zip(initial_thetas, theta_labels)):
    axs_energyR[i].plot(results_rk4[i][0], results_rk4[i][3], label=f'Kinetic', color = 'green')
    axs_energyR[i].plot(results_rk4[i][0], results_rk4[i][2], label=f'Total', color = 'blue')
    axs_energyR[i].plot(results_rk4[i][0], results_rk4[i][4], label=f'Potential', color = 'orange')

    # Set labels and titles for each energy subplot
    axs_energyR[i].set_ylabel('Energy')
    axs_energyR[i].set_title(f'Energy Comparison (θ(t=0)={label})')
    axs_energyR[i].legend()

# Set a common xlabel for all subplots
axs_energy[-1].set_xlabel('Time (s)')

plt.tight_layout()

# Show the plots
plt.show()