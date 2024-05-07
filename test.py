from src.utils import Mobility, Distribution
import yaml
import numpy as np

# print(np.random.choice([-1,1],p=[0.3,0.7]))
# for i in range(10):
#     print(sum(np.random.uniform(3000,15000,[100,1]))/100)


import matplotlib.pyplot as plt


# total = 5
# points = np.random.randint(low = -20, high = 20, size = [total,2])

# px, py = points.T

# minx = min(px)
# miny = min(py)

# maxx = max(px)
# maxy = max(py)

# avgx = sum(px)/total
# avgy = sum(py)/total

# x1 = (minx+avgx)/2
# y1 = (miny+avgy)/2

# x2 = (maxx+avgx)/2
# y2 = (maxy+avgy)/2

# plt.scatter(px,py)
# plt.plot(avgx,avgy,'ro')

# plt.plot(x1,y1,'gx')
# plt.plot(x2,y2,'gx')

# plt.show()