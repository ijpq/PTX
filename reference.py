import numpy as np

a = np.arange(0,128).reshape((8,16))
b = np.arange(-1, -129, -1)
b = np.reshape(b, (16,8), 'F')
print(a)
print(b)

dst = np.matmul(a,b)
print(dst)
np.savetxt('array.txt', dst, fmt="%d")

print(a[0])
print(b[:, 0])
dst = np.matmul(a[0], b[:, 0])
print(dst)
