import numpy as np
from scipy import special as sp
import yaml

class Mobility:
    def __init__(
        self,
        name: str,
    ) -> None:
        self.read()
        self.name = name

        self.mobility = next(item for item in self.full_mobility if item["name"] == self.name)
    
    def changePos(self) -> float:
        output = Distribution().getDistribution(
            self.mobility['distribution'], 
            self.mobility['param'])
        return output

    def get(self) -> str:
        return self.name

    def read(self):
        with open("mobility.yml") as f:
            self.full_mobility = yaml.safe_load(f)['types']

class Distribution:
    def __init__(
        self,
    ) -> None:

        self.distribution = {
            "poisson": np.random.poisson,       # mean, total, ratio
            "uniform": np.random.uniform,       # min, max, total
            "choice": np.random.choice,         # choice, prob
            "normal": np.random.normal,         # mean, sd, total
            "randint": np.random.randint,       # min, max, total
            "rayleigh": np.random.rayleigh      # scale, total
        }

    def getDistribution(
        self,
        name: str,
        args: list
    ) -> list:
        match name:
            case "poisson":
                return self.distribution[name](np.array(args[0])/args[2], args[1])*args[2]
            case "uniform":
                return self.distribution[name](args[0], args[1], args[2])
            case "choice":
                return self.distribution[name](args[0], args[1])
            case "normal":
                return self.distribution[name](args[0], args[1], args[2])
            case "randint":
                return self.distribution[name](low=args[0], high=args[1], size=args[2])
            case "rayleigh":
                return self.distribution[name](args[0], args[1])
            case _:
                return NotImplementedError(f"{name} distribution not added")