import json
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation

from src.utils import Mobility, Distribution
from src.ClusterAlgo import Kmeans, DensityBased
from src.ClusterMember import ClusterMember, ClusterMemberStats
from src.ClusterHead import ClusterHead

class Simulation:
    def __init__(
        self,
    ) -> None:
        np.random.seed(1234)
        self.initialize()
    
    def getConfig(self) -> None:
        with open("config.json") as f:
            self.config = json.load(f)

    def initialize(self) -> None:
        self.getConfig()
        self.dist_obj = Distribution()
        self.cluster_member = initializeClusterMembers(
            self.config["length"],
            self.config["width"],
            self.config["random_nodes"],
            self.config["cluster_nodes"],
            self.config["energy"],
            self.dist_obj
        )
        self.cluster_head = []
        self.cluster_member_stats = ClusterMemberStats(self.cluster_member)
    
    def getClusteringAlgo(self):
        currentAlgorithm = self.config["algorithm"].lower()

        if currentAlgorithm == "kmeans":
            algo = Kmeans()
            algo.setData(self.cluster_member_stats)
            number_of_cluster = 5
            self.clustering = algo.generateModel(n_clusters = number_of_cluster)
            
            self.center_x, self.center_y, self.center_z = self.clustering.cluster_centers_.T
            max_distance = self.cluster_member_stats.maxDistance(self.clustering.cluster_centers_)
            self.cluster_head = []
            for ind in range(number_of_cluster):
                head = ClusterHead(
                    id = ind,
                    position = [self.center_x[ind],self.center_y[ind],100],
                    speed = self.config["uav_speed"],
                    max_range = self.config["uav_range"],
                    current_range = min(max_distance[ind],self.config["uav_range"]),
                    energy = self.config["uav_range"]
                )
                self.cluster_head.append(head)
        elif currentAlgorithm == "density-based":
            algo = DensityBased()
            algo.setData(self.cluster_member_stats)
            # self.clustering = algo.generateModel(eps = self.config['uav_range'])
            self.clustering = algo.generateModel(eps = 100)

            temp_center_x = [[] for ind in range(len(np.unique(self.clustering.labels_)))]
            temp_center_y = [[] for ind in range(len(np.unique(self.clustering.labels_)))]
            temp_center_z = [[] for ind in range(len(np.unique(self.clustering.labels_)))]
            
            for cluster in self.cluster_member:
                temp_center_x[cluster.base_station].append(cluster.position[0])
                temp_center_y[cluster.base_station].append(cluster.position[1])
                temp_center_z[cluster.base_station].append(cluster.position[2])
            
            self.center_x = [0 for ind in range(len(np.unique(self.clustering.labels_)))]
            self.center_y = [0 for ind in range(len(np.unique(self.clustering.labels_)))]
            self.center_z = [0 for ind in range(len(np.unique(self.clustering.labels_)))]
            for ind in range(len(np.unique(self.clustering.labels_))):
                self.center_x[ind] = sum(temp_center_x[ind])/len(temp_center_x[ind]) if len(temp_center_x[ind]) != 0 else 0
                self.center_y[ind] = sum(temp_center_y[ind])/len(temp_center_y[ind]) if len(temp_center_y[ind]) != 0 else 0
                self.center_z[ind] = sum(temp_center_z[ind])/len(temp_center_z[ind]) if len(temp_center_z[ind]) != 0 else 0
            
            max_distance = self.cluster_member_stats.maxDistance(np.array([self.center_x,self.center_y,self.center_z]).T)
            self.cluster_head = []
            for ind in range(len(np.unique(self.clustering.labels_))):
                head = ClusterHead(
                    id = ind,
                    position = [self.center_x[ind],self.center_y[ind],100],
                    speed = self.config["uav_speed"],
                    max_range = self.config["uav_range"],
                    current_range = min(max_distance[ind],self.config["uav_range"]),
                    energy = self.config["uav_range"]
                )
                self.cluster_head.append(head)

        else:
            return NotImplementedError
    
    def run(self) -> None:
        self.setFigure()
        self.graph1_y = []
        self.graph2_y = []

        self.ani = FuncAnimation(
            self.fig, 
            self.updateGraph, 
            frames=21, 
            init_func=self.draw, 
            interval=500
        )
        plt.show()

    def setFigure(self) -> None:
        self.fig = plt.figure(figsize=(14,7))
        self.fig.suptitle("Wireless Communication in Disaster Region Simulation", size=16)
        self.fig.set(tight_layout=True)
        self.gs = gridspec.GridSpec(3, 2, width_ratios=[7,3])
        self.ax = [
            plt.subplot(self.gs[:,0]),
            plt.subplot(self.gs[0,1]),
            plt.subplot(self.gs[1,1]),
            plt.subplot(self.gs[2,1]),
        ]

    def updateGraph(self, current_time) -> None:
        # print(current_time)
        if current_time == 0:
            self.getClusteringAlgo()
            self.drawMap(update=True)
        self.drawGraph1()
        self.drawGraph2()
        self.drawGraph3()

    def draw(self) -> None:
        self.drawMap()
        self.drawGraph1()
        self.drawGraph2()
        self.drawGraph3()

    def drawMap(self, update: bool = False) -> None:
        self.ax[0].cla()
        pos = np.array(self.cluster_member_stats.getPosition())
        x,y,z = pos.T
        if update:
            self.ax[0].scatter(x,y,c=self.clustering.labels_)

            self.ax[0].scatter(
                self.center_x, 
                self.center_y, 
                s = 150, 
                c = "none", 
                marker = "X", 
                linewidths = 2.5, 
                edgecolors = "black")
            
            for head in self.cluster_head:
                circle = plt.Circle(
                    head.position[0:2], 
                    head.current_range,
                    fill=False, 
                    linewidth=2, 
                    alpha=0.9)
                # print(head.id)
                self.ax[0].add_artist(circle)
        else:
            self.ax[0].scatter(x,y)

        self.ax[0].set_xlim([0,self.config["length"]])
        self.ax[0].set_ylim([0,self.config["width"]])
        self.ax[0].set_title("Map")
    
    def drawGraph1(self) -> None:
        self.ax[1].cla()

        data = self.cluster_member_stats.getConnectivity(self.cluster_head)
        if len(self.graph1_y) > 20:
            self.graph1_y = self.graph1_y[1:]
        if data != []:
            self.graph1_y.append(data)
        self.graph1_x = [ind for ind in range(len(self.graph1_y))]

        self.ax[1].plot(self.graph1_x,self.graph1_y)
        self.ax[1].set_xlim([0,20])
        self.ax[1].set_ylim([0,100])
        self.ax[1].set_title("Connectivity Ratio")
        self.ax[1].legend([f"UAV {i+1}" for i in range(len(self.cluster_head))], loc="lower right")

    def drawGraph2(self) -> None:
        self.ax[2].cla()

        data = self.cluster_member_stats.getProbability(self.cluster_head)
        if len(self.graph2_y) > 20:
            self.graph2_y = self.graph2_y[1:]
        if data != []:
            self.graph2_y.append(data)
        self.graph2_x = [ind for ind in range(len(self.graph2_y))]
        
        self.ax[2].plot(self.graph2_x,self.graph2_y)
        self.ax[2].set_xlim([0,20])
        self.ax[2].set_ylim([80,100])
        self.ax[2].set_title("Uplink Probability")
        self.ax[2].legend([f"UAV {i+1}" for i in range(len(self.cluster_head))], loc="lower right")
    
    def drawGraph3(self) -> None:
        self.ax[3].cla()

        read_input = [
            "10"
        ]

        row_labels = [
            "Algorithm runtime", 
        ]

        cell_text = [
            [f"{read_input[0]} s"],
        ]

        self.ax[3].axis("off")
        self.ax[3].axis("tight")
        self.ax[3].tick_params(axis='x', which='major', pad=15)
        # self.ax[3].set_title("Data")
        self.ax[3].table(cellText=cell_text, rowLabels=row_labels, colWidths=[0.35, 0.2], loc='center right')
    
def initializeClusterMembers(
    length: float,
    width: float,
    random_nodes: list,
    cluster_nodes: list,
    energy: list,
    distribution: Distribution,
) -> list:
    all_cluster_members = []

    all_cluster_members_position = distribution.getDistribution(
        random_nodes[0]["distribution"],
        random_nodes[0]["param"])
    
    for nodes in cluster_nodes:
        current_node = distribution.getDistribution(
            nodes["distribution"],
            nodes["param"])
        all_cluster_members_position = np.append(
            all_cluster_members_position, 
            current_node, 
            axis=0)
    
    for position in all_cluster_members_position:
        cluster_member = ClusterMember(
            position = position,
            mobility = Mobility("stationary"),
            energy = distribution.getDistribution(
                energy[0]["distribution"],
                energy[0]["param"]),
        )
        all_cluster_members.append(cluster_member)

    return all_cluster_members