import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from src.ClusterMember import ClusterMember, ClusterMemberStats

class Kmeans:
    def setData(
        self,
        stats: ClusterMemberStats,
    ) -> None:
        self.stats = stats

        self.data = self.stats.getPosition()

    def generateModel(
        self,
        optimal : bool = False,
        n_clusters : int = 5,
        n_init : int = 10,
    ) -> KMeans:
        if n_clusters != 5:
            self.model = KMeans(n_clusters=n_clusters, n_init=n_init)
        elif optimal:
            self.model = KMeans(n_clusters=self.getOptimalCluster(), n_init=n_init)
        else:
            self.model = KMeans(n_clusters=n_clusters, n_init=n_init)
        self.model.fit(self.data)
        self.stats.setBaseStation(self.model.labels_)
        return self.model
    
    # TODO implement
    def getOptimalCluster(
        self,
    ) -> int:
        return 5


class DensityBased:
    def setData(
        self,
        stats: ClusterMemberStats,
    ) -> None:
        self.stats = stats

        self.data = self.stats.getPosition()

    def generateModel(
        self,
        eps : float = 200,
        min_samples: int = 15,
    ) -> DBSCAN:
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.model.fit(self.data)
        self.stats.setBaseStation(self.model.labels_)
        return self.model