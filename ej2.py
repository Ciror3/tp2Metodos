
import numpy as np
import matplotlib.pyplot as plt

def fDinamicaPoblacional(N,r,K,A):#k=cant maxima de indiviudos que se pueden sostener con los recursos, A = tamno minimo de poblacion requerida para sobrevivir
    return r * N * (1 - N / K) * (N / A - 1)

def euler(f, N0, r, K, A, h, num_steps):
    N_values = []
    t_values = []

    for i in range(num_steps):
        N_new = N0 + h * f(N0, r, K, A)
        t_new = (i + 1) * h

        N_values.append(N_new)
        t_values.append(t_new)

        N0 = N_new

    return t_values, N_values

def runge_kutta(f, N0, r, K, A, h, num_steps):
    N_values = []
    t_values = []

    for i in range(num_steps):
        k1 = h * f(N0, r, K, A)
        k2 = h * f(N0 + 0.5 * k1, r, K, A)
        k3 = h * f(N0 + 0.5 * k2, r, K, A)
        k4 = h * f(N0 + k3, r, K, A)

        N_new = N0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t_new = (i + 1) * h

        N_values.append(N_new)
        t_values.append(t_new)

        N0 = N_new

    return t_values, N_values

N0 = 10  # Condición inicial
r = 0.1  # Tasa de crecimiento
K = 100  # Capacidad de carga
A = 5    # Tamaño mínimo de población requerida para sobrevivir
h = 0.1  # Tamaño del paso
num_steps = 100  # Número de pasos de tiempo

# Resolución usando el método de Runge-Kutta
t_rk, N_rk = runge_kutta(fDinamicaPoblacional, N0, r, K, A, h, num_steps)

plt.figure(figsize=(10, 4))
plt.plot(t_rk, N_rk, label='Runge-Kutta')

plt.xlabel('Tiempo')
plt.ylabel('Población')
plt.legend()
plt.title('Evolución de la población a lo largo del tiempo')
plt.grid(True)

plt.show()

