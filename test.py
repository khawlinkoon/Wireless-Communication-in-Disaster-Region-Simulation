from src.utils import Mobility, Distribution
import yaml
import numpy as np

m = Mobility("car")

# for i in range(10):
#     print(m.changePos())
for i in range(10):
    print(np.random.choice([-1,1],p=[0.3,0.7]))

# for i in range(10):
#     print(Distribution().getDistribution("normal",[0,1,1]))

# with open("config.yml") as f:
#     config = yaml.safe_load(f)

# import os

# # print(os.path.exists("map.csv"))

# # os.system("python ./src/GenerateMapHeight.py")

# t = np.loadtxt("map.csv", delimiter=",", dtype=int)
# print(t.max())

