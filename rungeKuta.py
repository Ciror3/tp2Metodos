import numpy as np
import matplotlib.pyplot as plt

def rungeKuttaOrden4(a, b, N, alpha, f):
    h = (b - a) / N
    t = a
    w = alpha

    for i in range(1, N+1):
        K1 = h * f(t, w)
        K2 = h * f(t + h/2, w + K1/2)
        K3 = h * f(t + h/2, w + K2/2)
        K4 = h * f(t + h, w + K3)
        
        w = w + (K1 + 2*K2 + 2*K3 + K4) / 6
        t = a + i * h
    
    return t, w

def f(t, w):
    return t * w

a = 0
b = 1
N = 10
alpha = 1

# Call the function to solve the differential equation
t_final, w_final = rungeKuttaOrden4(a, b, N, alpha, f)

print(f"t_final = {t_final}, w_final = {w_final}")
