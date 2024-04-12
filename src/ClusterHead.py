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
        self.operational = True
    

class ClusterHeadStats:
    def __init__(self, cluster_head: dict,) -> None:
        self.cluster_head = cluster_head
        self.cluster_head_total = len(cluster_head)
        self.id_num = list(cluster_head.keys())
        print(self.id_num)
        self.cluster_head_value = list(cluster_head.values())
    
    def getCurrentPosition(self) -> np.array:
        return np.array([item.current_position for item in self.cluster_head_value])
    
    def getCurrentRange(self) -> list:
        return [item.current_range for item in self.cluster_head_value]

    def getCurrentEnergy(self) -> list:
        return [item.energy_usage for item in self.cluster_head_value]

    def updatePosition(self) -> None:
        for ind in range(len(self.cluster_head_value)):
            distance = np.linalg.norm(self.cluster_head_value[ind].end_position - self.cluster_head_value[ind].current_position)
            if distance < 1:
                self.updateEnergyUsage(1)
            else:
                self.updateEnergyUsage(2)
                if distance < self.cluster_head_value[ind].speed:
                    self.cluster_head_value[ind].current_position = np.array(self.cluster_head_value[ind].end_position)
                else:
                    ratio = self.cluster_head_value[ind].speed/distance
                    self.cluster_head_value[ind].current_position = np.array([
                        (1-ratio)*self.cluster_head_value[ind].current_position[0] + ratio*self.cluster_head_value[ind].end_position[0],
                        (1-ratio)*self.cluster_head_value[ind].current_position[1] + ratio*self.cluster_head_value[ind].end_position[1],
                        (1-ratio)*self.cluster_head_value[ind].current_position[2] + ratio*self.cluster_head_value[ind].end_position[2],
                    ])
                    # self.cluster_head_value[ind].current_position += (self.cluster_head_value[ind].end_position - self.cluster_head_value[ind].current_position)/self.clustcluster_head_valueer_head[ind].speed

    def updateEndPosition(self, total: int, position: list, height: float) -> None:
        # print(self.cluster_head_value)
        for ind in range(min(total, len(self.cluster_head_value))):
            self.cluster_head_value[ind].end_position = np.array(position[ind])
            self.cluster_head_value[ind].end_position[2] = height
    
    def updateCurrentRange(self, rnge: list) -> None:
        for ind in range(len(self.cluster_head_value)):
            self.cluster_head_value[ind].current_range = min(rnge[ind],self.cluster_head_value[ind].max_range)

    def updateEnergyUsage(self, typ: bool) -> None:
        # false = idle, true = move
        for ind in range(len(self.cluster_head_value)):
            if typ:
                self.cluster_head_value[ind].energy_usage += self.cluster_head_value[ind].move_energy
            else:
                self.cluster_head_value[ind].energy_usage += self.cluster_head_value[ind].idle_energy
