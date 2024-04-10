from src.utils import Mobility, Distribution
import yaml

# m = Mobility("car")
# print(m.get())

# for i in range(10):
#     print(Distribution().getDistribution("normal",[0,1,1]))

# with open("config.yml") as f:
#     config = yaml.safe_load(f)

import os

print(os.path.exists("map.csv"))

os.system("python ./src/GenerateMapHeight.py")