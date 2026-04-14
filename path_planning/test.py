import numpy as np
# from matplotlib import pyplot as plt
# from scipy.interpolate import splprep, splev


# x = [-44.88929310563101, -52.59911875073924, -52.69975808361432, -52.8003974164894, -52.901036749364465, -53.00167608223955, -53.10231541511462,
#      -53.20295474798969, -53.30359408086477, -53.35383347759522, -54.14915523633018, -54.19947490276771, -54.249714299498166, -55.29382495067249]
# y = [-0.47691940581777514, 0.3921607738265719, 0.4931211855300348,  0.5940815972334906,  0.6950420089369465, 0.7960024206404024, 0.8969628323438582,
#      0.997923244047314, 1.09888365575077, 1.1997637977471374,8.156239301020014, 8.206719506871744, 8.307599648868106, 17.280473946461786 ]

# # Define knot vector, coefficients, and degree
# tck, u = splprep([x, y], s=0.2, k=3)

# u_eval = np.linspace(-40, -60, 60)
# px, py = splev(u, tck)
# print(u_eval)
# dx, dy = splev(u_eval, tck, der=1)
# # print(px,py)

# fig, ax = plt.subplots()
# # ax.plot(x, y, 'ro')
# ax.plot(px, py, 'r-')
# plt.show()

# point1 = np.array([1.0,2.0])
# point2 = np.array([1.0,1.0])
# print(point1-point2)

dis = [7,2,5,4]
p = dis.index(min(dis))
print(p)