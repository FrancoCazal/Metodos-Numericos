import numpy as np

A=np.array([[2,1,1],[4,3,3],[8,7,9]])
B=np.array([[2],[4],[6]])
C=np.hstack((A,B))
D=np.zeros((3,4))

print(A, A.shape)
print(A.dtype)
print(B, B.shape)
print(C)
print(D)