from src.utils import Mobility

m = Mobility("car")
print(m.get())

for i in range(10):
    print(m.changePos())