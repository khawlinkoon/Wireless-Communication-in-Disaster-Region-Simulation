from src.utils import Mobility, Distribution
import yaml

# m = Mobility("car")
# print(m.get())

# for i in range(10):
#     print(Distribution().getDistribution("normal",[0,1,1]))

for i in range(10):
    print(Distribution().getDistribution("normal", [0,5,2]))