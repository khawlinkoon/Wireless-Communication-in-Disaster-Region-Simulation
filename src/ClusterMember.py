from .utils import *

class ClusterMember:
    def __init__(
        self,
        position: tuple[float], #(x,y,z)
        mobility: Mobility,
        energy: int,
    ) -> None:
        self.position = position
        self.mobility = mobility
        self.energy = energy


        self.base_station = -1
        self.connected = False

    
    def printPosition(self) -> str:
        return self.position

    def printMobility(self) -> str:
        return self.mobility.get()

class ClusterMemberStats:
    def __init__(
        self,
        cluster_member: list,
    ) -> None:
        self.cluster_member = cluster_member

    def getPosition(self) -> list:
        return [item.position for item in self.cluster_member]
    
    def getMobility(self) -> list:
        return [item.mobility for item in self.cluster_member]
        
    def getEnergy(self) -> list:
        return [item.energy for item in self.cluster_member]
    
    def setBaseStation(self, labels: list) -> None:
        for ind,label in enumerate(labels):
            self.cluster_member[ind].base_station = label
    
    def maxDistance(self, centers: list) -> list:
        output = [0 for i in range(len(centers))]
        for cluster in self.cluster_member:
            current_group = cluster.base_station
            output[current_group] = max(output[current_group],np.linalg.norm(centers[current_group] - cluster.position))
        return output

    def getConnectivity(self, cluster_heads:list) -> list:
        total = [0 for head in cluster_heads]
        connected = [0 for head in cluster_heads]
        for cluster in self.cluster_member:
            if cluster.base_station == -1:
                continue
            current_head = cluster_heads[cluster.base_station]
            total[cluster.base_station] += 1
            distance = np.linalg.norm(current_head.position-cluster.position)
            if distance <= current_head.current_range:
                connected[cluster.base_station] += 1
        output = []
        for ind, tot in enumerate(total):
            output.append(connected[ind]/tot*100 if tot != 0 else 0)
        return output
    
    # TODO change this to path loss
    def getProbability(self, cluster_heads:list) -> list:
        total = [[] for head in cluster_heads]

        for cluster in self.cluster_member:
            if cluster.base_station == -1:
                continue
            current_head = cluster_heads[cluster.base_station]

            def uplinkProb(current_head_position, cluster_member_position):
                a = 9.6
                b = 0.16

                distance = np.linalg.norm(current_head_position-cluster_member_position)
                height = abs(current_head_position[2]-cluster_member_position[2])

                # print(distance, height)
                if distance == 0:
                    return 1
                
                theta = np.arcsin(height/distance)*180/np.pi
                prob = 1/(1 + a * np.exp(1)**(-1*b*(theta - a)))

                return prob*100
            
            prob = uplinkProb(current_head.position, cluster.position)            
            total[cluster.base_station].append(prob)
        
        for ind, item in enumerate(total):
            if len(item) > 0:
                total[ind] = sum(item)/len(item)
            else:
                total[ind] = 0
        
        return total

    def updatePosition(self, cluster_heads:list) -> None:
        for ind in range(len(self.cluster_member)):
            delta = self.cluster_member[ind].mobility.changePos()
            
            current_cluster_head = cluster_heads[self.cluster_member[ind].base_station]
            distance = np.linalg.norm(self.cluster_member[ind].position[0:2]-current_cluster_head.position[0:2])
            if distance < current_cluster_head.current_range:
                self.cluster_member[ind].position[0] += delta[0]
                self.cluster_member[ind].position[1] += delta[1]
            else:
                x1, y1, z1 = self.cluster_member[ind].position
                x2, y2, z2 = current_cluster_head.position
                if x1 < x2:
                    self.cluster_member[ind].position[0] = x1+abs(delta[0])
                else:
                    self.cluster_member[ind].position[0] = x1-abs(delta[0])
                if y1 < y2:
                    self.cluster_member[ind].position[1] = y1+abs(delta[1])
                else:
                    self.cluster_member[ind].position[1] = y1-abs(delta[1])
                
            
