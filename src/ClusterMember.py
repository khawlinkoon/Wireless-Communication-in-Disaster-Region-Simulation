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
        self.leach_status = False

    
    def printPosition(self) -> str:
        return self.position

    def printMobility(self) -> str:
        return self.mobility.get()

class ClusterMemberStats:
    def __init__(self, cluster_member: list,) -> None:
        with open("pathloss.yml") as f:
            self.pathloss_config = yaml.safe_load(f)["types"]
        self.cluster_member = cluster_member
        self.connected_total = []

    def getPosition(self) -> np.array:
        return np.array([item.position for item in self.cluster_member])
    
    def getMobility(self) -> list:
        return [item.mobility for item in self.cluster_member]
        
    def getEnergy(self) -> list:
        return [item.energy[0] for item in self.cluster_member]

    def getLeachStatus(self) -> list:
        return [item.leach_status for item in self.cluster_member]

    def updateEnergySingle(self, ind: int, energy: float) -> None:
        self.cluster_member[ind].energy -= energy

    def setMobility(self, cycle: int) -> None:
        if cycle < 2 and cycle != 0:
            prob = 0.2
        elif cycle < 4:
            prob = 0.4
        elif cycle < 6:
            prob = 0.6
        else:
            prob = 0.3
        for cluster in self.cluster_member:
            choice = np.random.choice([True,False],p=[prob,1-prob])
            cluster.mobility = Mobility("walking") if choice else Mobility("car")
        
    
    def setBaseStation(self, labels: list) -> None:
        for ind,label in enumerate(labels):
            self.cluster_member[ind].base_station = label
    
    def maxDistance(self, centers: list) -> list:
        output = [0 for i in range(len(centers))]
        for cluster in self.cluster_member:
            current_group = cluster.base_station
            output[current_group] = max(output[current_group],np.linalg.norm(centers[current_group] - cluster.position))
        return output
    
    def getClusterHead(self, centers: list) -> list:
        cluster_head = [[] for i in range(len(centers))]
        for ind, cluster in enumerate(self.cluster_member):
            current_group = cluster.base_station
            x = np.array([centers[current_group][0], centers[current_group][1]])
            y = np.array([cluster.position[0], cluster.position[1]])
            distance = np.linalg.norm(x - y)
            cluster_head[current_group].append([distance, -1*cluster.energy[0], ind])
        for ind in range(len(cluster_head)):
            cluster_head[ind] = sorted(cluster_head[ind], key=lambda x: [x[0], x[1]])
        return [x[0][2] for x in cluster_head]
    
    def getLeachCh(self, centers: list, prob: float, rnd: int) -> list:
        clusterTrue = [[] for i in range(len(centers))]
        clusterFalse = [[] for i in range(len(centers))]
        for ind, cluster in enumerate(self.cluster_member):
            current_group = cluster.base_station
            if cluster.leach_status:
                clusterTrue[current_group].append(ind)
            else:
                clusterFalse[current_group].append(ind)
        selected = []
        # # prob = prob/(1-prob*(rnd%(1/prob)))
        # print(prob)
        for i in range(len(centers)):
            if len(clusterFalse[i]) < 10:
                swap(clusterFalse, clusterTrue)
            for ch in clusterFalse[i]:
                choice = np.random.choice([True,False],p=[prob,1-prob])
                if choice:
                    selected.append(ch)
                    self.cluster_member[i].leach_status = not self.cluster_member[i].leach_status
        return selected

    def updatePosition(self, setpoint:list, energy:float, towards:bool = False) -> None:
        for ind in range(len(self.cluster_member)):
            self.cluster_member[ind].energy -= energy
            delta = self.cluster_member[ind].mobility.changePos()
            
            distance = 999999999999999999
            current_cluster_head = setpoint

            distance = np.linalg.norm(self.cluster_member[ind].position[0:2]-setpoint[0:2])
            
            fixed_distance = 100
            if distance >= 300 and towards:
                x1, y1, z1 = self.cluster_member[ind].position
                x2, y2, z2 = current_cluster_head
                if x1 < x2 - fixed_distance:
                    self.cluster_member[ind].position[0] = x1+abs(delta[0])*np.random.choice([-1,1],p=[0.3,0.7])*0.6
                elif x1 > x2 + fixed_distance:
                    self.cluster_member[ind].position[0] = x1-abs(delta[0])*np.random.choice([-1,1],p=[0.3,0.7])*0.6
                else:
                    self.cluster_member[ind].position[0] += delta[0]
                if y1 < y2 - fixed_distance:
                    self.cluster_member[ind].position[1] = y1+abs(delta[1])*np.random.choice([-1,1],p=[0.3,0.7])*0.4
                elif y1 > y2 + fixed_distance:
                    self.cluster_member[ind].position[1] = y1-abs(delta[1])*np.random.choice([-1,1],p=[0.3,0.7])*0.4
                else:
                    self.cluster_member[ind].position[1] += delta[1]
            else:
                self.cluster_member[ind].position[0] += delta[0]
                self.cluster_member[ind].position[1] += delta[1]
    
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

                if distance == 0:
                    return 1
                
                theta = np.arcsin(height/distance)*180/np.pi
                prob = 1/(1 + a * np.exp(1)**(-1*b*(theta - a)))

                return prob*100
            
            prob = uplinkProb(current_head.current_position, cluster.position)            
            total[cluster.base_station].append(prob)
        
        for ind, item in enumerate(total):
            if len(item) > 0:
                total[ind] = sum(item)/len(item)
            else:
                total[ind] = 0
        
        return total

    def getPathLoss(self, cluster_heads: list, main_position: np.array, specific_nodes: list, location: int = 0)-> list:
        total1 = [[] for head in cluster_heads]
        total2 = [[] for head in cluster_heads]

        specific = False if len(specific_nodes)  == 0 else True

        for ind, cluster in enumerate(self.cluster_member):
            if cluster.base_station == -1:
                continue
            
            if specific:
                if ind not in specific_nodes:
                    continue
            
            # print(ind)
            current_head = cluster_heads[cluster.base_station]
            current_head_position = current_head.current_position
            cluster_member_position = cluster.position
    
            current_pathloss_config = list(filter(lambda x:x["name"]=="los/nlos",self.pathloss_config))[0]
            a = current_pathloss_config['a'][location]
            b = current_pathloss_config['b'][location]
            freq = current_pathloss_config['fc']
            eta_los = current_pathloss_config['eta_los'][location]
            eta_nlos = current_pathloss_config['eta_nlos'][location]
            g0 = current_pathloss_config['g0'] # TODO: check source

            # https://www.researchgate.net/publication/338998664_Air-to-Ground_channel_model_for_UAVs_in_Dense_UrbanEnvironments
            distance = np.linalg.norm(current_head_position-cluster_member_position)
            height = abs(current_head_position[2]-cluster_member_position[2])
            theta = np.arcsin(height/distance)*180/np.pi

            free_space_pl = 20*np.log10(distance*freq) - 147.55

            d2 = current_head.getDistance(main_position)
            if d2 == 0:
                d2 = 0.1
            excess_pl = height*np.sqrt(2/(3*10**8/freq)*(1/distance+1/d2))

            prob_los = 1/(1 + a*np.exp(1)**(-1*b*(theta-a)))
            prob_nlos = 1 - prob_los

            theta3 = 107.6*10**(-0.1*g0)
            gain = g0
            if theta <= theta3:
                gain -= 12*(theta/theta3)**2
            else:
                gain -= 12 + 10*np.log10(theta/theta3)
            antenna_loss = -2*(gain)
            difrac_loss = 0
            mean_pl = free_space_pl + antenna_loss + difrac_loss + 20*np.log10(1-prob_los)
            
            total1[cluster.base_station].append(mean_pl)

            def qfunc(x):
                return 0.5-0.5*sp.erf(x/np.sqrt(2))
            
            def dbmToDb(x):
                return 10*np.log10(10**(x/10-3))

            g_db = 3 #dB
            p_min = dbmToDb(20)
            p_transmit = dbmToDb(50)
            noise = dbmToDb(-174)
            
            connectivity = prob_los*qfunc((p_min+mean_pl-p_transmit-g_db+eta_los)/noise) + prob_nlos*qfunc((p_min+mean_pl-p_transmit-g_db+eta_nlos)/noise)
            total2[cluster.base_station].append(connectivity)
        
        for ind,item in enumerate(total1):
            if len(item) > 0:
                total1[ind] = sum(item)/len(item)
            else:
                total1[ind] = 0

        for ind,item in enumerate(total2):
            if len(item) > 0:
                total2[ind] = sum(item)/len(item)*100
            else:
                total2[ind] = 0
        
        self.connected_total = total2

        return total1

    def getConnectivity(self, cluster_heads: list, main_position: np.array, specific_nodes: list) -> list:
        total = [[] for head in cluster_heads]
        for cluster in self.cluster_member:
            if cluster.base_station == -1:
                continue
            
            current_head = cluster_heads[cluster.base_station]
            current_head_position = current_head.current_position
            current_head_range = current_head.current_range
            cluster_member_position = cluster.position

            distance = np.linalg.norm(current_head_position-cluster_member_position)
            if current_head_range >= distance:
                total[cluster.base_station].append(1)
            elif current_head_range*1.5 >= distance:
                total[cluster.base_station].append(0.5)
            else:
                total[cluster.base_station].append(0)
        
        for ind,item in enumerate(total):
            if len(item) > 0:
                total[ind] = sum(item)/len(item)*100
            else:
                total[ind] = 0
        
        return total


