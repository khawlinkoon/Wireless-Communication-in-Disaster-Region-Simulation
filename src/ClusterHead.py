import numpy as np

class ClusterHead:
    def __init__(
        self,
        current_position: np.array, #(x,y,z)
        end_position: np.array,
        speed: float,
        max_range: float,
        current_range: float,
        idle_energy: float,
        move_energy: float,
    ) -> None:
        self.current_position = current_position
        self.end_position = end_position
        self.speed = speed
        self.max_range = max_range
        self.current_range = current_range
        self.idle_energy = idle_energy
        self.move_energy = move_energy
        self.energy_usage = 0
    


class ClusterHeadStats:
    def __init__(self, cluster_head: list,) -> None:
        self.cluster_head = cluster_head
    
    def getCurrentPosition(self) -> np.array:
        return np.array([item.current_position for item in self.cluster_head])
    
    def getCurrentRange(self) -> list:
        return [item.current_range for item in self.cluster_head]

    def getCurrentEnergy(self) -> list:
        return [item.energy_usage for item in self.clsuter_head]

    def updatePosition(self) -> None:
        for ind in range(len(self.cluster_head)):
            distance = np.linalg.norm(self.cluster_head[ind].end_position - self.cluster_head[ind].current_position)
            if distance < 1:
                self.updateEnergyUsage(1)
            else:
                self.updateEnergyUsage(2)
                if distance < self.cluster_head[ind].speed:
                    self.cluster_head[ind].current_position = self.cluster_head[ind].end_position
                else:
                    ratio = self.cluster_head[ind].speed/distance
                    self.cluster_head[ind].current_position = [
                        (1-ratio)*self.cluster_head[ind].current_position[0] + ratio*self.cluster_head[ind].end_position[0],
                        (1-ratio)*self.cluster_head[ind].current_position[1] + ratio*self.cluster_head[ind].end_position[1],
                        (1-ratio)*self.cluster_head[ind].current_position[2] + ratio*self.cluster_head[ind].end_position[2],
                    ]
                    # self.cluster_head[ind].current_position += (self.cluster_head[ind].end_position - self.cluster_head[ind].current_position)/self.cluster_head[ind].speed

    def updateEndPosition(self, position: list, height: float) -> None:
        # print(self.cluster_head)
        for ind in range(len(self.cluster_head)):
            self.cluster_head[ind].end_position = position[ind]
            self.cluster_head[ind].end_position[2] = height
    
    def updateCurrentRange(self, range: list) -> None:
        for ind in range(len(self.cluster_head)):
            self.cluster_head[ind].current_range = min(range[ind],self.cluster_head[ind].max_range)

    def updateEnergyUsage(self, typ: bool) -> None:
        # false = idle, true = move
        for ind in range(len(self.cluster_head)):
            if typ:
                self.cluster_head[ind].energy_usage += self.cluster_head[ind].move_energy
            else:
                self.cluster_head[ind].energy_usage += self.cluster_head[ind].idle_energy
