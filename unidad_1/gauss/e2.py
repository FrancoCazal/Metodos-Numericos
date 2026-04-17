# Problema 4 - Solución mediante Eliminación Gaussiana
# Un ingeniero eléctrico supervisa la producción de tres tipos de componentes eléctricos. Se requiere determinar la
# cantidad óptima de producción basándose en la disponibilidad de materiales.

# Tabla de Requerimientos (Gramos por Componente)
# Componente	Metal [g/comp]	Plástico [g/comp]	Caucho [g/comp]
# 1	7	3	35
# 2	12	12	12
# 3	26	4	15

# Disponibilidad Diaria(Convertida a gramos para consistencia en los cálculos)
# Metal: 9.57 kg = 9570 g
# Plástico: 5.71 kg = 5710 g
# Caucho: 15.14 kg = 15140 g

# Pregunta:¿Cuántos componentes de cada tipo se puede producirse por día?

import numpy as np

A = np.array([[7,12,26],
              [3,12,4],
              [35,12,15]], dtype=float)

B = np.array([[9570],
              [5710],
              [15140]], dtype=float)

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

