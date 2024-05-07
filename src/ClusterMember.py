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
    def __init__(self, cluster_member: list,) -> None:
        with open("pathloss.yml") as f:
            self.pathloss_config = yaml.safe_load(f)["types"]
        self.cluster_member = cluster_member
        self.connected_total = []

    def getPosition(self) -> list:
        return [item.position for item in self.cluster_member]
    
    def getMobility(self) -> list:
        return [item.mobility for item in self.cluster_member]
        
    def getEnergy(self) -> list:
        return [item.energy[0] for item in self.cluster_member]

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
            # print(cluster_head[ind])
            cluster_head[ind] = sorted(cluster_head[ind], key=lambda x: [x[0], x[1]])
            # print(cluster_head[ind])
        return [x[0][2] for x in cluster_head]
        # return cluster_head

    def updatePosition(self, setpoint:list, towards:bool = False) -> None:
        for ind in range(len(self.cluster_member)):
            delta = self.cluster_member[ind].mobility.changePos()
            
            distance = 999999999999999999
            current_cluster_head = setpoint

            # for i,head in enumerate(setpoint):
            #     temp  = np.linalg.norm(self.cluster_member[ind].position[0:2]-head[0:2])
            #     if distance != min(distance,temp):
            #         distance = temp
            #         current_cluster_head = setpoint[i]
            distance = np.linalg.norm(self.cluster_member[ind].position[0:2]-setpoint[0:2])
            
            # distance = np.linalg.norm(self.cluster_member[ind].position[0:2]-current_cluster_head[0:2])
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
                    # *np.random.choice([-1,1],p=[0.3,0.7])
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

                # print(distance, height)
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

    # def getPathLoss(self, cluster_heads:list, location:int = 0) -> list:
    #     total1 = [[] for head in cluster_heads]
    #     total2 = [[] for head in cluster_heads]
    #     ch_pos = [x.current_position for x in cluster_heads]

    #     for cluster in self.cluster_member:
    #         if cluster.base_station == -1:
    #             continue
            
    #         current_head = cluster_heads[cluster.base_station]

    #         def los_nlosPathLoss(current_head_position, cluster_member_position, location):
    #             def dbmToDb(x):
    #                 out = 10**(x/10-3)
    #                 return 10*np.log10(out)
    #             current_pathloss_config = list(filter(lambda x:x["name"]=="los/nlos",self.pathloss_config))[0]
    #             a = current_pathloss_config['a'][location]
    #             b = current_pathloss_config['b'][location]
    #             fc = current_pathloss_config['fc']
    #             eta_los = current_pathloss_config['eta_los'][location]
    #             eta_nlos = current_pathloss_config['eta_nlos'][location]

    #             c = 300000000

    #             distance = np.linalg.norm(current_head_position-cluster_member_position)
    #             height = abs(current_head_position[2]-cluster_member_position[2])
                
    #             theta = np.arcsin(height/distance)*180/np.pi
                
    #             prob_los = 1/(1 + a * np.exp(1)**(-1*b*(theta - a)))
    #             prob_nlos = 1 - prob_los

    #             loss = 20*np.log10(4*np.pi*fc*distance/c) + eta_los*prob_los + eta_nlos*prob_nlos
    #             # loss = 20*np.log10(4*np.pi*fc*distance/c) + eta_los   
                
    #             g_db = 3 #dB
    #             p_min = dbmToDb(20)
    #             p_transmit = dbmToDb(50)
    #             # p_min = 20
    #             # p_transmit = 50
    #             noise = dbmToDb(-174) #dBm/Hz

    #             def qfunc(x):
    #                 return 0.5-0.5*sp.erf(x/np.sqrt(2))
                    
    #             connectivity = prob_los*qfunc((p_min+loss-p_transmit-g_db+eta_los)/noise) + prob_nlos*qfunc((p_min+loss-p_transmit-g_db+eta_nlos)/noise)

    #             return loss, connectivity*100
            
    #         # print(ch_pos)
    #         # print(cluster.base_station)
    #         # print(cluster.position)
    #         loss, connectivity = los_nlosPathLoss(current_head.current_position, cluster.position, location)
    #         total1[cluster.base_station].append(loss)
    #         total2[cluster.base_station].append(connectivity)

    #     for ind, item in enumerate(total1):
    #         if len(item) > 0:
    #             total1[ind] = sum(item)/len(item)
    #         else:
    #             total1[ind] = 0

    #     for ind, item in enumerate(total2):
    #         if len(item) > 0:
    #             total2[ind] = sum(item)/len(item)
    #         else:
    #             total2[ind] = 0
        
    #     self.connected_total = total2
        
    #     return total1

    def getPathLoss(self, cluster_heads:list, main_position:np.array, location:int = 0)-> list:
        total1 = [[] for head in cluster_heads]
        total2 = [[] for head in cluster_heads]

        for cluster in self.cluster_member:
            if cluster.base_station == -1:
                continue

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
            difrac_loss = -20*np.log10(0.225/excess_pl) if excess_pl>0 else 0
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

    def getConnectivity(self, cluster_heads:list, main_position:np.array) -> list:
        self.getPathLoss(cluster_heads, main_position)
        return self.connected_total


