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
        if not os.path.exists("map.csv") or self.config["reload_map_height"]:
            os.system("python ./src/GenerateMapHeight.py")
        self.map_height = np.loadtxt("map.csv", delimiter=",", dtype=int)

    def initialize(self) -> None:
        self.getConfig()
        self.getMapHeight()
        self.center = [self.config["length"]/2, self.config["width"]/2, 0]
        self.uav_height = self.config["uav_height"]
        self.disaster_zones = [] if "disaster_zones" not in self.config.keys() else self.config["disaster_zones"]
        self.evacuate_point = self.center if "evacuate_point" not in self.config.keys() else self.config["evacuate_point"]
        self.evacuate_point = np.append(self.evacuate_point, self.map_height[self.evacuate_point[0]][self.evacuate_point[1]])
        self.evacuate_radius = self.config["evacuate_radius"]
        self.base_station_point = self.center if self.config["bs_total"] <=0 else self.config["bs_location"]
        self.base_station_point = [
            np.append(self.base_station_point[i], 
            self.map_height[self.base_station_point[i][0]][self.base_station_point[i][1]]) for i in range(len(self.base_station_point))
        ]
        self.base_station_point.append(self.evacuate_point)
        self.main_base_station = self.base_station_point[-1]
        self.evacuating = self.config["evacuating"]
        self.new_algo = self.config["new_algo"]
        self.dist_obj = Distribution()
        self.uav_total = self.config["uav_total"]
        self.communication_energy = self.config["communication_energy"]
        self.leach_probability = self.config["leach_probability"]
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
            total = self.config["uav_total"],
            position = [np.array(self.evacuate_point) for i in range(self.config["uav_total"])],
            speed = self.config["uav_speed"],
            rnge = self.config["uav_range"],
            idle_energy = self.config["uav_idle_energy"],
            move_energy = self.config["uav_move_energy"],
        )
        self.base_station = initializeClusterHeads(
            typ = "B",
            height = self.map_height,
            total = self.config["bs_total"]+1,
            position = self.base_station_point, 
            speed = 0,
            rnge = self.config["bs_range"],
            idle_energy = 0,
            move_energy = 0,
            tower_height = self.config["bs_height"],
        )
        self.current_cycle = -1
        self.start_cycle = self.config["start_cycle"]

        self.uav_stats = ClusterHeadStats(self.uav)
        self.base_station_stats = ClusterHeadStats(self.base_station)
        self.cluster_head_stats = self.base_station_stats
        self.cluster_member_stats = ClusterMemberStats(self.cluster_member)
        self.new_cluster_center = []
        self.initial_runtime = ""
        self.initial_memory = ""
        self.total_runtime = "0 ms"
        self.total_memory = "0"
        self.pathloss = []
        self.new_node_head_id = []
    
    def getClusteringAlgo(self):
        def setClusterHeadWithCenters(cluster_center: np.array) -> list:
            self.new_cluster_center = [[obj,ind] for ind,obj in enumerate(cluster_center)]
            self.new_cluster_center = sorted(
                self.new_cluster_center,
                key=lambda x: [x[0][0], x[0][0]+x[0][1], (self.center[0]-x[0][0])**2+(self.center[1]-x[0][1])**2+(self.center[2]-x[0][2])**2]
            )

            new_order = [obj[1] for obj in self.new_cluster_center]
            self.new_cluster_center = [obj[0] for obj in self.new_cluster_center]
            self.center_x, self.center_y, self.center_z = np.array(self.new_cluster_center).T
            self.cluster_head_stats.updateEndPosition(len(self.new_cluster_center), self.new_cluster_center, self.uav_height)
            return new_order
            
        def setClusterHeadWithoutCenters(total_cluster: int = 0) -> list:
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

            self.new_cluster_center = np.array([self.center_x,self.center_y,self.center_z]).T
            self.new_cluster_center = [[obj,ind] for ind,obj in enumerate(self.new_cluster_center)]
            self.new_cluster_center = sorted(
                self.new_cluster_center,
                key=lambda x: [x[0][0]+x[0][1], x[0][0], (self.center[0]-x[0][0])**2+(self.center[1]-x[0][1])**2+(self.center[2]-x[0][2])**2]
            )
            new_order = [obj[1] for obj in self.new_cluster_center]
            self.new_cluster_center = [obj[0] for obj in self.new_cluster_center]
            self.new_max_distance = self.cluster_member_stats.maxDistance(self.new_cluster_center)

            self.cluster_head_stats.updateEndPosition(len(self.new_cluster_center), self.new_cluster_center, self.uav_height)
            return new_order

        current_algorithm = self.config["algorithm"].lower()
        start_time = time.time()
        start_mem = psutil.Process(os.getpid()).memory_info().rss
        total_cluster = self.cluster_head_stats.cluster_head_total
        if self.current_cycle < self.start_cycle:
            total_cluster -= 1 

        if current_algorithm == "kmeans" or self.current_cycle < self.start_cycle:
            algo = Kmeans()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel(
                optimal = False if self.current_cycle < self.start_cycle else not self.config["use_all_uav"], 
                n_clusters = total_cluster
            )
            new_order = setClusterHeadWithCenters(cluster_center=self.clustering.cluster_centers_)
            self.labels = []
            for obj in algo.getLabels():
                for ind in range(len(new_order)):
                    if new_order[ind] == obj:
                        self.labels.append(ind)
                        break
            self.cluster_member_stats.setBaseStation(self.labels)
        elif current_algorithm == "mini_kmeans":
            algo = MiniKmeans()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel(
                optimal = False if self.current_cycle < self.start_cycle else not self.config["use_all_uav"], 
                n_clusters = total_cluster
            )
            new_order = setClusterHeadWithCenters(cluster_center=self.clustering.cluster_centers_)
            self.labels = []
            for obj in algo.getLabels():
                for ind in range(len(new_order)):
                    if new_order[ind] == obj:
                        self.labels.append(ind)
                        break
            self.cluster_member_stats.setBaseStation(self.labels)
        elif current_algorithm == "density_based":
            algo = DensityBased()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            new_order = setClusterHeadWithoutCenters()
            self.labels = []
            for obj in algo.getLabels():
                if obj == -1:
                    self.labels.append(0)
                    continue
                for ind in range(len(new_order)):
                    if new_order[ind] == obj:
                        self.labels.append(ind)
                        break
            self.cluster_member_stats.setBaseStation(self.labels)
        elif current_algorithm == "affinity_propagation":
            algo = AffinityProp()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            new_order = setClusterHeadWithoutCenters()
            self.labels = []
            for obj in algo.getLabels():
                for ind in range(len(new_order)):
                    if new_order[ind] == obj:
                        self.labels.append(ind)
                        break
            self.cluster_member_stats.setBaseStation(self.labels)
            setClusterHeadWithoutCenters()
        elif current_algorithm == "birch":
            algo = Balanced()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            new_order = setClusterHeadWithoutCenters()
            self.labels = []
            for obj in algo.getLabels():
                for ind in range(len(new_order)):
                    if new_order[ind] == obj:
                        self.labels.append(ind)
                        break
            self.cluster_member_stats.setBaseStation(self.labels)
            setClusterHeadWithoutCenters(total_cluster)
        elif current_algorithm == "spectral":
            algo = Spectral()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            new_order = setClusterHeadWithoutCenters()
            self.labels = []
            for obj in algo.getLabels():
                for ind in range(len(new_order)):
                    if new_order[ind] == obj:
                        self.labels.append(ind)
                        break
            self.cluster_member_stats.setBaseStation(self.labels)
            setClusterHeadWithoutCenters()
        elif current_algorithm == "gaussian":
            algo = Gaussian()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            new_order = setClusterHeadWithoutCenters()
            self.labels = []
            for obj in algo.getLabels():
                for ind in range(len(new_order)):
                    if new_order[ind] == obj:
                        self.labels.append(ind)
                        break
            self.cluster_member_stats.setBaseStation(self.labels)
            setClusterHeadWithoutCenters()
        elif current_algorithm == "lda":
            return NotImplementedError
            algo = LDA()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            new_order = setClusterHeadWithoutCenters()
            self.labels = []
            for obj in algo.getLabels():
                for ind in range(len(new_order)):
                    if new_order[ind] == obj:
                        self.labels.append(ind)
                        break
            self.cluster_member_stats.setBaseStation(self.labels)
        elif current_algorithm == "markov":
            algo = Markov()
            algo.setData(self.cluster_member_stats)
            self.clustering = algo.generateModel()
            self.labels = algo.getLabels()
            new_order = setClusterHeadWithoutCenters()
            self.labels = []
            for obj in algo.getLabels():
                for ind in range(len(new_order)):
                    if new_order[ind] == obj:
                        self.labels.append(ind)
                        break
            self.cluster_member_stats.setBaseStation(self.labels)
        else:
            return NotImplementedError
        
        if self.new_algo and self.current_cycle >= self.start_cycle:
            self.new_node_head_id = self.cluster_member_stats.getLeachCh(self.new_cluster_center, self.leach_probability, self.current_cycle)
        
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
        self.graph4_y = []

        self.ani = FuncAnimation(
            self.fig, 
            self.updateGraph, 
            frames=self.config["cycle_frames"] + 1, 
            init_func=self.draw, 
            interval=self.config["time_per_frame"]
        )
        plt.show()

    def setFigure(self) -> None:
        self.fig = plt.figure(figsize=(14,7))
        self.fig.suptitle("Wireless Communication in Disaster Region Simulation", size=16)
        self.fig.set(tight_layout=True)

        self.gs = gridspec.GridSpec(4, 2, width_ratios=[7,3])
        self.ax = [
            plt.subplot(self.gs[:,0]),
            plt.subplot(self.gs[1,1]),
            plt.subplot(self.gs[0,1]),
            plt.subplot(self.gs[3,1]),
            plt.subplot(self.gs[2,1]),
        ]

    def updateGraph(self, current_time) -> None:
        if current_time == 0:
            self.cluster_member_stats.setMobility(self.current_cycle)
            self.current_cycle += 1
            print("Cycle",self.current_cycle)
            if self.current_cycle == self.start_cycle:
                self.cluster_head_stats = self.uav_stats
            self.getClusteringAlgo()
            if self.config["save"]:
                self.saveGraph()
                print("Saved")
            self.drawMap(update=True)
            self.drawGraph1()
            self.drawGraph2()
            self.drawGraph3()
            self.drawGraph4()
            
        elif current_time % 5 == 0:
            self.drawGraph1()
            self.drawGraph2()
            self.drawGraph3()
            self.drawGraph4()
            if current_time == 50 and self.config["save"]:
                self.drawMap(update=True, save=True)
        if self.config["dynamic"]:
            self.cluster_member_stats.updatePosition(self.evacuate_point, self.evacuating, self.communication_energy/3)
        self.cluster_head_stats.updatePosition()
        self.drawMap(update=True)

    def draw(self) -> None:
        self.drawMap()
        self.drawGraph1()
        self.drawGraph2()
        self.drawGraph3()
        self.drawGraph4()

    def drawMap(self, update: bool = False, save: bool = False) -> None:
        self.ax[0].cla()
        pos = np.array(self.cluster_member_stats.getPosition())
        x,y,z = pos.T
        if update:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            color_label = [colors[label%10] for label in self.labels]
            self.ax[0].scatter(x, y, c= color_label)

            if self.new_algo and self.current_cycle >= self.start_cycle:
                for ind in self.new_node_head_id:
                    self.ax[0].scatter(x[ind],y[ind], s=100, c=color_label[ind], linewidth=3 ,edgecolors="black")
                    self.cluster_member_stats.updateEnergySingle(ind, self.communication_energy)
            
            patch = []
            for ind, group in enumerate(np.unique([f"Cluster {label+1}" for label in self.labels])):
                patch.append(mpatches.Patch(color=colors[ind%10], label=group))
            label = [p.get_label() for p in patch]
            label = sorted(label, key=lambda x: int("".join([i for i in x if i.isdigit()])))
            self.ax[0].legend(handles=patch, labels=label, loc="lower right")

            current_position = self.cluster_head_stats.getCurrentPosition()
            current_range = self.cluster_head_stats.getCurrentRange()

            # Cluster Centers
            if self.current_cycle >= self.start_cycle:
                self.ax[0].scatter(
                    self.center_x, 
                    self.center_y, 
                    s = 150, 
                    c = "none", 
                    marker = "X", 
                    linewidths = 2.5, 
                    edgecolors = "black")

            uav_x, uav_y, uav_z = current_position.T

            # UAV Locations
            self.ax[0].scatter(
                uav_x, 
                uav_y, 
                s = 150, 
                c = "none", 
                marker = "P", 
                linewidths = 2.5, 
                edgecolors = "black")
            
            base_station_position = self.base_station_stats.getCurrentPosition()
            bs_x, bs_y, bs_z = base_station_position.T

            # BS Location
            if self.current_cycle < self.start_cycle:
                self.ax[0].scatter(
                    bs_x,
                    bs_y,
                    s = 300,
                    c = "none",
                    marker = "^",
                    linewidths = 8,
                    edgecolors = "black"
                )
            else:
                self.ax[0].scatter(
                    bs_x[-1],
                    bs_y[-1],
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
                    self.evacuate_radius,
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

        if save:
            self.fig.savefig("Results\\map.png",dpi=300)
    
    def drawGraph1(self, refresh = False) -> None:
        if refresh:
            self.ax[1].clear()
        else:
            self.ax[1].cla()
        
        data = self.cluster_member_stats.getConnectivity(self.cluster_head_stats.cluster_head_value, self.main_base_station, self.new_node_head_id)
        if len(self.graph1_y) > 0 and len(data) != len(self.graph1_y[0]):
            self.graph1_y = []
        if data != []:
            self.graph1_y.append(data)
        if len(self.graph1_y) > 12:
            self.graph1_y = self.graph1_y[1:]
        self.graph1_x = [ind for ind in range(len(self.graph1_y))]

        self.ax[1].plot(self.graph1_x,self.graph1_y)
        self.ax[1].set_xlim([0,self.config["cycle_frames"]/5])
        self.ax[1].set_ylim([20,100])
        self.ax[1].set_xlabel("Time (frame)")
        self.ax[1].set_ylabel("Coverage (%)")
        self.ax[1].set_title("Coverage probabilty")

    def drawGraph2(self, refresh = False) -> None:
        if refresh:
            self.ax[2].clear()
        else:
            self.ax[2].cla()

        self.pathloss = self.cluster_member_stats.getPathLoss(self.cluster_head_stats.cluster_head_value, self.main_base_station, self.new_node_head_id, self.config["terrain"])
        if len(self.graph2_y) > 0 and len(self.pathloss) != len(self.graph2_y[0]):
            self.graph2_y = []
        if self.pathloss != []:
            self.graph2_y.append(self.pathloss)
        if len(self.graph2_y) > 12:
            self.graph2_y = self.graph2_y[1:]
        self.graph2_x = [ind for ind in range(len(self.graph2_y))]
        
        self.ax[2].plot(self.graph2_x,self.graph2_y)
        self.ax[2].set_xlim([0,self.config["cycle_frames"]/5])
        self.ax[2].set_ylim([60,110])
        self.ax[2].set_xlabel("Time (frame)")
        self.ax[2].set_ylabel("Path Loss (dB)")
        self.ax[2].set_title("Path Loss")
    
    def drawGraph3(self, refresh = False) -> None:
        if refresh:
            self.ax[3].clear()
        else:
            self.ax[3].cla()

        connectivity_mean = f"{np.mean(self.graph1_y)/100:.5f}" if len(self.graph1_y)>0 else "0"
        uplink_mean = f"{np.mean(self.graph2_y):.2f} dB" if len(self.graph2_y)>0 else "0 dB"
        energy_mean = f"{np.mean(self.graph4_y):.2f} kW" if len(self.graph4_y)>0 else "0 kW"

        algorithm = self.config['algorithm']
        read_input = [
            algorithm.replace("_"," ").title(),
            self.initial_runtime.format(len(algorithm)),
            # self.total_runtime.format(len(algorithm)),
            self.initial_memory.format(len(algorithm)),
            # self.total_memory.format(len(algorithm)),
            connectivity_mean.format(len(algorithm)),
            uplink_mean.format(len(algorithm)),
            energy_mean.format(len(algorithm))
        ]

        row_labels = [
            "Algorithm                ",
            "Algorithm initial runtime", 
            # "Algorithm runtime        ", 
            "Initial memory usage     ", 
            # "Algorithm memory usage   ", 
            "Average Coverage         ", 
            "Average Path Loss        ", 
            "Average Energy Used      ",
        ]

        cell_text = [[text] for text in read_input]

        self.ax[3].axis("off")
        self.ax[3].axis("tight")
        self.ax[3].tick_params(axis="x", which="major", pad=15)
        self.ax[3].set_title("Data")
        table = self.ax[3].table(
            cellText = cell_text, 
            rowLabels = row_labels, 
            colWidths = [0.5, 0.4], 
            loc = "best")
        table.auto_set_font_size(False)
        table.set_fontsize(11)

    def drawGraph4(self) -> None:
        self.ax[4].cla()
        energy = self.cluster_head_stats.getCurrentEnergy()
        
        if len(self.graph4_y) > 0 and len(energy) != len(self.graph4_y[0]):
            self.graph4_y = []
        if energy != []:
            self.graph4_y.append(energy)
        if len(self.graph4_y) > 12:
            self.graph4_y = self.graph4_y[1:]
        self.graph4_x = [ind for ind in range(len(self.graph4_y))]

        self.ax[4].plot(self.graph4_x,self.graph4_y)
        self.ax[4].set_xlim([0,self.config["cycle_frames"]/5])
        self.ax[4].set_xlabel("Time (frame)")
        self.ax[4].set_ylabel("Energy Used (kW)")
        self.ax[4].set_title("UAV energy content")


    def saveGraph(self) -> None:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(self.graph1_x,self.graph1_y)
        ax[0].set_xlim([0,self.config["cycle_frames"]/5])
        ax[0].set_ylim([0,100])
        ax[0].set_title("Connectivity Ratio")
        ax[0].legend(self.cluster_head_stats.id_num)

        ax[1].plot(self.graph2_x,self.graph2_y)
        ax[1].set_xlim([0,self.config["cycle_frames"]/5])
        ax[1].set_ylim([70,110])
        ax[1].set_title("Path Loss")
        ax[1].legend(self.cluster_head_stats.id_num)
        fig.savefig("Results\\results.png",dpi=300)
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(self.graph4_x,self.graph4_y)
        ax.set_xlim([0,self.config["cycle_frames"]/5])
        ax.set_title("UAV energy content")
        ax.set_xlabel("Time (frame)")
        ax.set_ylabel("Energy used (kW)")
        ax.legend(self.cluster_head_stats.id_num)
        fig.savefig("Results\\energy.png", dpi=300)
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
    
    for nodes in cluster_nodes:
        current_node = distribution.getDistribution(
            nodes["distribution"],
            nodes["param"])
        all_cluster_members_position = np.append(
            all_cluster_members_position, 
            current_node, 
            axis=0)
    
    for i,position in enumerate(all_cluster_members_position):
        position[0] = min(position[0], length-1)
        position[1] = min(position[1], width-1)
        position[2] = height[position[0]][position[1]]
        cluster_member = ClusterMember(
            position = position,
            mobility = np.random.choice([Mobility("stationary"),Mobility("walking"),Mobility("car")],p=[0.7,0.2,0.1]),
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
    tower_height: int = 0,
) -> list:
    all_cluster_heads = {}

    # print(position)
    if typ == "B":
        for i in range(len(position)):
            position[i][2] += tower_height

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