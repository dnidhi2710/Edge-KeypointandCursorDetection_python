import numpy as np

rx = [[1, 0, 0], [0, 0.15, 0.99], [0, -0.99, 0.15]]
ry = [[0.96, 0, -0.26], [0, 1, 0], [0.26, 0, 0.96]]
rz = [[0.63, -0.77, 0], [0.77, 0.63, 0], [0, 0, 1]]

a = np.matmul(ry, rx)
op = np.matmul(rz, a)
#op = np.dot(rz,np.dot(ry,rx))

print("rotation matrix:")
print(op)

ext = [[0.6048,  0.046,   -0.79,  600],
       [0.739, 0.292, 0.594, 300],
       [0.26, -0.9504,  0.144, 200],
       [0, 0, 0,  1]]

rt = np.linalg.inv(ext)

print(rt)
print("translation vector:")
print(np.matrix.round(rt))

wx = [1249, 227, -67]

wc = np.matmul(op, wx)

print(wc)

K = [[100, 0.5, 200, 0], [0, 100, 150, 0], [0, 0, 1, 0]]
ip = [60, 80, 100, 1]

op1 = np.matmul(K, ip)
print(op1)
