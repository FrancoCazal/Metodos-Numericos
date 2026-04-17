# ## Problema de Optimización de Suministros

# Una empresa dedicada a la venta de helados y ensaladas de frutas necesita adquirir frutas de diferentes proveedores.
# Cada proveedor ofrece un porcentaje específico de frutas. La empresa requiere ciertas cantidades de ingredientes para
# su producción total: **52 kg de fresas, 56 kg de plátanos, 34 kg de manzanas y 59 kg de kiwis.**

# Determina la cantidad de kilogramos que debe adquirir de cada proveedor para obtener los ingredientes necesarios.

# ### Tabla de Composición por Proveedor

# | Proveedor | Fresas (%) | Plátanos (%) | Manzanas (%) | Kiwis (%) |
# | --- | --- | --- | --- | --- |
# | **A** | 14 | 11 | 60 | 15 |
# | **B** | 65 | 17 | 8 | 10 |
# | **C** | 15 | 60 | 19 | 6 |
# | **D** | 12 | 12 | 11 | 65 |

# Sistema: A.x = B

import numpy as np

A = np.array([[14,65,15,12],
              [11,17,60,12],
              [60,8,19,11],
              [15,10,6,65]], dtype=float)

B = np.array([[5200],
              [5600],
              [3400],
              [5900]], dtype=float)

C = np.hstack((A, B))

delta = 0.01
p=6

np.set_printoptions(precision=p)
if np.abs(np.linalg.det(A)) < delta:
        print('el sistema no tiene solucion o tiene infinitas soluciones')
else:
    n=A.shape[0]
    j = -1
    i = n - 1
    while j < n-2:
        if i == n - 1:
            j += 1
            i = j
            subcol = np.abs(C[j:, j])
            t = np.argmax(subcol)
            pibot = subcol[t]
            if delta > pibot:
                print('el sistema no tiene solucion o tiene infinitas soluciones')
            else:
                aux=C[j,:].copy()
                C[j,:]=C[t+j,:]
                C[t+j,:]=aux.copy()
        i += 1
        k = C[i, j] / C[j, j]
        C[i, j:] -= k * C[j, j:]
X=np.zeros((n,1),dtype=float)
X[n-1] = C[-1, -1] / C[n-1, n-1]
for i in range(n-2, -1, -1):
    X[i] = (C[i, -1] - C[i, i+1:n].dot(X[i+1:n])) / C[i, i]

print(X)
