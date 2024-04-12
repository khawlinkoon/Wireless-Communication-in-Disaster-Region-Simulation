import yaml
import numpy as np
import logging
import time
import psutil
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

from src.utils import Mobility, Distribution
from src.ClusterAlgo import Kmeans, DensityBased, MiniKmeans, AffinityProp, Balanced, Spectral, Gaussian, LDA, Markov
from src.ClusterMember import ClusterMember, ClusterMemberStats
from src.ClusterHead import ClusterHead, ClusterHeadStats

class Simulation:
    def __init__(
        self,
    ) -> None:
        np.random.seed(1234)
        self.initialize()
    
    def getConfig(self) -> None:
        with open("config.yml") as f:
            self.config = yaml.safe_load(f)
    
    def getMapHeight(self) -> None:
        if not os.path.exists("map.csv"):
            os.system("python ./src/GenerateMapHeight.py")
        self.map_height = np.loadtxt("map.csv", delimiter=",", dtype=int)

    def initialize(self) -> None:
        self.getMapHeight()
        self.getConfig()
        center = [self.config['length']/2, self.config['width']/2, 0]
        self.uav_height = self.config['uav_height']
        self.disaster_zones = [] if "disaster_zones" not in self.config.keys() else self.config["disaster_zones"]
        self.evacuate_point = center if "evacuate_point" not in self.config.keys() else self.config["evacuate_point"]
        self.evacuate_point = np.append(self.evacuate_point, self.map_height[self.evacuate_point[0]][self.evacuate_point[1]])
        self.base_station_point = center if self.config["bs_total"] <=0 else self.config["bs_location"]
        self.base_station_point = [np.append(self.base_station_point[i], self.map_height[self.base_station_point[i][0]][self.base_station_point[i][1]]) for i in range(len(self.base_station_point))]
        self.evacuating = self.config['evacuating']
        self.dist_obj = Distribution()
        self.uav_total = self.config['uav_total']
        self.cluster_member = initializeClusterMembers(
            length = self.config["length"],
            width = self.config["width"],
            height = self.map_height,
            random_nodes = self.config["random_nodes"],
            cluster_nodes = self.config["cluster_nodes"],
            energy = self.config["energy"],
            distribution = self.dist_obj
        )
        self.uav = initializeClusterHeads(
            typ = "U",
            height = self.map_height,
            total = self.config['uav_total'],
            position = [np.array(self.evacuate_point) for i in range(self.config['uav_total'])],
            speed = self.config['uav_speed'],
            rnge = self.config['uav_range'],
            idle_energy = self.config['uav_idle_energy'],
            move_energy = self.config['uav_move_energy'],
        )
        self.base_station = initializeClusterHeads(
            typ = "B",
            height = self.map_height,
            total = self.config['bs_total'],
            position = self.base_station_point,
            speed = 0,
            rnge = self.config['bs_range'],
            idle_energy = 0,
            move_energy = 0,
        )
        self.current_cycle = -1
        self.start_cycle = self.config['start_cycle']

        # self.cluster_head = {}
        # self.cluster_head.update(self.base_station)
        # self.cluster_head.update(self.uav)
        # self.cluster_head_stats = ClusterHeadStats(self.cluster_head)
        self.uav_stats = ClusterHeadStats(self.uav)
        self.base_station_stats = ClusterHeadStats(self.base_station)
        self.cluster_head_stats = self.base_station_stats
        self.cluster_member_stats = ClusterMemberStats(self.cluster_member)
        self.initial_runtime = ""
        self.initial_memory = ""
        self.total_runtime = "0 ms"
        self.total_memory = "0"
        self.pathloss = []
    
    def getClusteringAlgo(self):
        def setClusterHeadWithCenters(cluster_center: np.array) -> None:
            # TODO: Check if cluster centers are sorted, if no need sort
            cluster_center = sorted(cluster_center,key=lambda x: [x[0],x[1],x[2]])
            self.center_x, self.center_y, self.center_z = np.array(cluster_center).T
            max_distance = self.cluster_member_stats.maxDistance(cluster_center)
            self.cluster_head_stats.updateEndPosition(len(cluster_center), cluster_center, self.uav_height)
            # self.cluster_head = []
            # for ind in range(len(np.unique(self.labels))):
            #     head = ClusterHead(
            #         id = ind,
            #         position = np.array([self.center_x[ind],self.center_y[ind],100]),
            #         speed = self.config["uav_speed"],
            #         max_range = self.config["uav_range"],
            #         current_range = min(max_distance[ind],self.config["uav_range"]),
            #         energy = self.config["uav_energy"]
            #     )
            #     self.cluster_head.append(head)
            
        def setClusterHeadWithoutCenters() -> None:
            temp_center_x = [[] for ind in range(len(np.unique(self.labels)))]
            temp_center_y = [[] for ind in range(len(np.unique(self.labels)))]
            temp_center_z = [[] for ind in range(len(np.unique(self.labels)))]
            
            for cluster in self.cluster_member:
                temp_center_x[cluster.base_station].append(cluster.position[0])
                temp_center_y[cluster.base_station].append(cluster.position[1])
                temp_center_z[cluster.base_station].append(cluster.position[2])
            
            self.center_x = [0 for ind in range(len(np.unique(self.labels)))]
            self.center_y = [0 for ind in range(len(np.unique(self.labels)))]
            self.center_z = [0 for ind in range(len(np.unique(self.labels)))]
            for ind in range(len(np.unique(self.labels))):
                self.center_x[ind] = sum(temp_center_x[ind])/len(temp_center_x[ind]) if len(temp_center_x[ind]) != 0 else 0
                self.center_y[ind] = sum(temp_center_y[ind])/len(temp_center_y[ind]) if len(temp_center_y[ind]) != 0 else 0
                self.center_z[ind] = sum(temp_center_z[ind])/len(temp_center_z[ind]) if len(temp_center_z[ind]) != 0 else 0
            
            cluster_center = np.array([self.center_x,self.center_y,self.center_z]).T
            cluster_center = sorted(cluster_center,key=lambda x: [x[0],x[1],x[2]])
            max_distance = self.cluster_member_stats.maxDistance(cluster_center)
            self.cluster_head_stats.updateEndPosition(len(cluster_center), cluster_center, self.uav_height)
            # self.cluster_head = []
            # for ind in range(len(np.unique(self.labels))):
            #     head = ClusterHead(
            #         id = ind,
            #         position = [self.center_x[ind],self.center_y[ind],100],
            #         speed = self.config["uav_speed"],
            #         max_range = self.config["uav_range"],
            #         current_range = min(max_distance[ind],self.config["uav_range"]),
            #         energy = self.config["uav_energy"]
            #     )
            #     self.cluster_head.append(head)

        current_algorithm = self.config["algorithm"].lower()
        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss
        if current_algorithm == "kmeans":
            algo = Kmeans()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel(n_clusters = self.cluster_head_stats.cluster_head_total)
            self.labels = algo.label
            self.cluster_member_stats.setBaseStation(self.labels)
            # setClusterHeadWithCenters(cluster_center=algo.cluster_center)
            setClusterHeadWithCenters(cluster_center=self.clustering.cluster_centers_)
        elif current_algorithm == "mini_kmeans":
            algo = MiniKmeans()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel(n_clusters = self.cluster_head_stats.cluster_head_total)
            self.labels = algo.getLabels()
            self.cluster_member_stats.setBaseStation(self.labels)
            setClusterHeadWithCenters(cluster_center=self.clustering.cluster_centers_)
        elif current_algorithm == "density_based":
            algo = DensityBased()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            self.cluster_member_stats.setBaseStation(self.labels)
            setClusterHeadWithoutCenters()
        elif current_algorithm == "affinity_propagation":
            algo = AffinityProp()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            self.cluster_member_stats.setBaseStation(self.labels)
            setClusterHeadWithoutCenters()
        elif current_algorithm == "birch":
            algo = Balanced()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            self.cluster_member_stats.setBaseStation(self.labels)
            setClusterHeadWithoutCenters()
        elif current_algorithm == "spectral":
            algo = Spectral()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            self.cluster_member_stats.setBaseStation(self.labels)
            setClusterHeadWithoutCenters()
        elif current_algorithm == "gaussian":
            algo = Gaussian()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            self.cluster_member_stats.setBaseStation(self.labels)
            setClusterHeadWithoutCenters()
        elif current_algorithm == "lda":
            return NotImplementedError
            algo = LDA()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            self.cluster_member_stats.setBaseStation(self.labels)
            setClusterHeadWithoutCenters()
        elif current_algorithm == "markov":
            algo = Markov()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            self.cluster_member_stats.setBaseStation(self.labels)
            setClusterHeadWithoutCenters()
        else:
            return NotImplementedError
        
        self.total_memory = str((psutil.Process(os.getpid()).memory_info().rss - start_mem)/1024) + " kB"
        self.total_runtime = f"{(time.time()-start_time)*1000:.2f} ms"
        if self.initial_runtime == "":
            self.initial_runtime = self.total_runtime
        if self.initial_memory == "":
            self.initial_memory = self.total_memory
    
    def run(self) -> None:
        self.setFigure()
        self.graph1_y = []
        self.graph2_y = []

        self.ani = FuncAnimation(
            self.fig, 
            self.updateGraph, 
            frames=self.config["cycle_frames"] + 1, 
            init_func=self.draw, 
            interval=self.config['time_per_frame']
        )
        plt.show()

    def setFigure(self) -> None:
        self.fig = plt.figure(figsize=(14,7))
        self.fig.suptitle("Wireless Communication in Disaster Region Simulation", size=16)
        self.fig.set(tight_layout=True)

        self.gs = gridspec.GridSpec(3, 2, width_ratios=[7,3])
        self.ax = [
            plt.subplot(self.gs[:,0]),
            plt.subplot(self.gs[1,1]),
            plt.subplot(self.gs[0,1]),
            plt.subplot(self.gs[2,1]),
        ]

    def updateGraph(self, current_time) -> None:
        # print(current_time)
        if current_time == 0:
            self.current_cycle += 1
            if self.current_cycle == self.start_cycle:
                self.cluster_head_stats = self.uav_stats
            self.getClusteringAlgo()
            self.drawGraph1()
            self.drawGraph2()
            self.drawGraph3()
            self.drawMap(update=True)
            
            if self.config['save']:
                self.fig.savefig("Results\\map.png",dpi=300)
                # self.saveGraph()
                print("Saved")
        elif current_time % 5 == 0:
            self.drawGraph1()
            self.drawGraph2()
            self.drawGraph3()
        else:
            pass
        if self.config["dynamic"]:
            self.cluster_member_stats.updatePosition(self.evacuate_point,self.evacuating)  
            self.cluster_head_stats.updatePosition()
        self.drawMap(update=True)

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
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            color_label = [colors[label%10] for label in self.labels]
            self.ax[0].scatter(x, y, c= color_label)
            
            patch = []
            for ind, group in enumerate(np.unique([f"Cluster {label+1}" for label in self.labels])):
                patch.append(mpatches.Patch(color=colors[ind%10], label=group))
            label = [p.get_label() for p in patch]
            label = sorted(label, key=lambda x: int("".join([i for i in x if i.isdigit()])))
            self.ax[0].legend(handles=patch, labels=label, loc="lower right")

            current_position = self.cluster_head_stats.getCurrentPosition()
            current_range = self.cluster_head_stats.getCurrentRange()

            # Cluster Centers
            self.ax[0].scatter(
                self.center_x, 
                self.center_y, 
                s = 150, 
                c = "none", 
                marker = "X", 
                linewidths = 2.5, 
                edgecolors = "black")

            uav_x, uav_y, uav_z = current_position.T
            # print(uav_x, uav_y)

            # UAV Locations
            self.ax[0].scatter(
                uav_x, 
                uav_y, 
                s = 150, 
                c = "none", 
                marker = "P", 
                linewidths = 2.5, 
                edgecolors = "black")
            
            bs_x, bs_y, bs_z = self.base_station_stats.getCurrentPosition().T

            # BS Location
            self.ax[0].scatter(
                bs_x,
                bs_y,
                s = 300,
                c = "none",
                marker = "^",
                linewidths = 8,
                edgecolors = "black"
            )
            
            for ind,head in enumerate(current_position):
                circle = plt.Circle(
                    head[0:2], 
                    current_range[ind],
                    fill=False, 
                    linewidth=2, 
                    alpha=0.9)
                # print(head.id)
                self.ax[0].add_artist(circle)
            
            for zone in self.disaster_zones:
                rect = plt.Rectangle(
                    zone[0],
                    zone[1],
                    zone[2],
                    color = "#d11111",
                    fill = False,
                    ls = "-",
                    lw = 3,
                    hatch = "x")
                self.ax[0].add_artist(rect)
            
            if self.config["dynamic"]:
                circle = plt.Circle(
                    self.evacuate_point[0:2], 
                    750,
                    fill = False, 
                    lw = 3, 
                    alpha = 0.5,
                    ls = "--",
                    color = "#000000")
                self.ax[0].add_artist(circle)

        else:
            self.ax[0].scatter(x,y)

        self.ax[0].set_xlim([0,self.config["length"]])
        self.ax[0].set_ylim([0,self.config["width"]])
        self.ax[0].set_title("Map")
    
    def drawGraph1(self, refresh = False) -> None:
        if refresh:
            self.ax[1].clear()
        else:
            self.ax[1].cla()
        
        data = self.cluster_member_stats.getConnectivity(self.cluster_head_stats.cluster_head_value)
        if len(self.graph1_y) > 0 and len(data) != len(self.graph1_y[0]):
            self.graph1_y = []
        if data != []:
            self.graph1_y.append(data)
        if len(self.graph1_y) > 12:
            self.graph1_y = self.graph1_y[1:]
        self.graph1_x = [ind for ind in range(len(self.graph1_y))]

        self.ax[1].plot(self.graph1_x,self.graph1_y)
        self.ax[1].set_xlim([0,self.config["cycle_frames"]/5])
        self.ax[1].set_ylim([0,100])
        self.ax[1].set_xlabel("Time (frame)")
        self.ax[1].set_ylabel("Coverage (%)")
        self.ax[1].set_title("Coverage probabilty")

        # self.ax[1].legend([f"UAV {i+1}" for i in range(len(self.cluster_head))], loc="lower right")

    def drawGraph2(self, refresh = False) -> None:
        if refresh:
            self.ax[2].clear()
        else:
            self.ax[2].cla()

        # data = self.cluster_member_stats.getProbability(self.cluster_head)
        self.pathloss = self.cluster_member_stats.getPathLoss(self.cluster_head_stats.cluster_head_value, self.config['terrain'])
        if len(self.graph2_y) > 0 and len(self.pathloss) != len(self.graph2_y[0]):
            self.graph2_y = []
        if self.pathloss != []:
            self.graph2_y.append(self.pathloss)
        if len(self.graph2_y) > 12:
            self.graph2_y = self.graph2_y[1:]
        self.graph2_x = [ind for ind in range(len(self.graph2_y))]
        
        self.ax[2].plot(self.graph2_x,self.graph2_y)
        self.ax[2].set_xlim([0,self.config["cycle_frames"]/5])
        self.ax[2].set_ylim([60,120])
        self.ax[2].set_xlabel("Time (frame)")
        self.ax[2].set_ylabel("Path Loss (dB)")
        self.ax[2].set_title("Path Loss")
        # self.ax[2].legend([f"UAV {i+1}" for i in range(len(self.cluster_head))], loc="lower right")
    
    def drawGraph3(self, refresh = False) -> None:
        if refresh:
            self.ax[3].clear()
        else:
            self.ax[3].cla()

        connectivity_mean = f"{np.mean(self.graph1_y)/100:.5f}" if len(self.graph1_y)>0 else "0"
        uplink_mean = f"{np.mean(self.graph2_y):.2f} dB" if len(self.graph2_y)>0 else "0 dB"

        read_input = [
            self.config["algorithm"].replace("_"," ").title(),
            self.initial_runtime.format(len(self.config["algorithm"])),
            # self.total_runtime.format(len(self.config["algorithm"])),
            self.initial_memory.format(len(self.config["algorithm"])),
            # self.total_memory.format(len(self.config["algorithm"])),
            connectivity_mean.format(len(self.config["algorithm"])),
            uplink_mean.format(len(self.config["algorithm"])),
        ]

        row_labels = [
            "Algorithm                ",
            "Algorithm initial runtime", 
            # "Algorithm runtime        ", 
            "Initial memory usage     ", 
            # "Algorithm memory usage   ", 
            "Average Coverage         ", 
            "Average Path Loss        ", 
        ]

        cell_text = [[text] for text in read_input]

        self.ax[3].axis("off")
        self.ax[3].axis("tight")
        self.ax[3].tick_params(axis='x', which='major', pad=15)
        self.ax[3].set_title("Data")
        table = self.ax[3].table(
            cellText = cell_text, 
            rowLabels = row_labels, 
            colWidths = [0.5, 0.4], 
            loc = 'best')
        table.auto_set_font_size(False)
        table.set_fontsize(11)

    def saveGraph(self) -> None:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(self.graph1_x,self.graph1_y)
        ax[0].set_xlim([0,self.config["cycle_frames"]/5])
        ax[0].set_ylim([0,100])
        ax[0].set_title("Connectivity Ratio")
        ax[0].legend([f"UAV {i+1}" for i in range(len(self.cluster_head))], loc="lower right")

        ax[1].plot(self.graph2_x,self.graph2_y)
        ax[1].set_xlim([0,self.config["cycle_frames"]/5])
        ax[1].set_ylim([60,120])
        ax[1].set_title("Path Loss")
        ax[1].legend([f"UAV {i+1}" for i in range(len(self.cluster_head))], loc="lower right")
        fig.savefig("Results\\results.png",dpi=300)
        plt.close()
    
def initializeClusterMembers(
    length: float,
    width: float,
    height: list,
    random_nodes: list,
    cluster_nodes: list,
    energy: list,
    distribution: Distribution,
) -> list:
    all_cluster_members = []

    all_cluster_members_position = distribution.getDistribution(
        random_nodes[0]["distribution"],
        random_nodes[0]["param"])
    # all_cluster_members_position = [[1000,3000,1]]
    
    for nodes in cluster_nodes:
        current_node = distribution.getDistribution(
            nodes["distribution"],
            nodes["param"])
        all_cluster_members_position = np.append(
            all_cluster_members_position, 
            current_node, 
            axis=0)
    
    for position in all_cluster_members_position:
        position[2] = height[position[0],position[1]]
        cluster_member = ClusterMember(
            position = position,
            mobility = Mobility("car"),
            energy = distribution.getDistribution(
                energy[0]["distribution"],
                energy[0]["param"]),
        )
        all_cluster_members.append(cluster_member)

    return all_cluster_members

def initializeClusterHeads(
    typ: str, # U = UAV, B = Base station
    height: list,
    total: int,
    position: list,
    speed: float,
    rnge: float,
    idle_energy: float,
    move_energy: float,
) -> list:
    all_cluster_heads = {}

    # print(position)
    if typ == "B":
        for i in range(len(position)):
            position[i][2] += 50  # Typical Base Station Height

    for i in range(total):
        cluster_head = ClusterHead(
            current_position = position[i],
            end_position = position[i],
            speed = speed,
            max_range = rnge,
            current_range = rnge,
            idle_energy = idle_energy,
            move_energy = move_energy,
        )
        all_cluster_heads[typ+str(i+1)] = (cluster_head)

    return all_cluster_heads