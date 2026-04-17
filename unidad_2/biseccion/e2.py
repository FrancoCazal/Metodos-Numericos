import jax.numpy as jnp
import os
import matplotlib.pyplot as plt
from jax import grad

# Limpia la consola
os.system('cls')

# Define la funcion f(r) que modela el volumen/costo de un recipiente cilindrico
# con tapa, donde r es el radio. Se busca la raiz de esta funcion (f(r)=0).
f= lambda r:2*jnp.pi*(r+0.25)**2*0.25+(r+0.25)**2*1000/r**2-1000

# Calcula la derivada de f de forma automatica usando diferenciacion automatica de JAX
df=grad(f)

# Genera un vector de valores de r en el intervalo [2.1, 7.0) con paso 0.1
x=jnp.arange(2.1,7.0,0.1)

# Grafico superior: dibuja f(r) para visualizar donde cruza el eje x (raices)
plt.subplot(2,1,1)
plt.plot(x,f(x))

# Grafico inferior: dibuja f'(r) para ver el comportamiento de la derivada
plt.subplot(2,1,2)

# Evalua la derivada en cada punto del vector x
# (grad de JAX solo opera sobre escalares, por eso se usa un bucle)
ydf=x.copy()
for i in range(len(x)):
    ydf=ydf.at[i].set(df(x[i]))

# Grafica la derivada en rojo y una linea horizontal en y=0 en negro como referencia
plt.plot(x,ydf,'r',[2,7], [0,0],'k')
plt.show()