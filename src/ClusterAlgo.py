import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans, AffinityPropagation, Birch, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import LatentDirichletAllocation
from hmmlearn.hmm import GaussianHMM

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
    
    def getLabels(
        self,
    ) -> list:
        return self.model.labels_
    
    # TODO implement
    def getOptimalCluster(
        self,
    ) -> int:
        return 5

# TODO: Test different parameters
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
        min_samples: int = 20,
    ) -> DBSCAN:
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.model.fit(self.data)
        self.stats.setBaseStation(self.model.labels_)
        return self.model
    
    def getLabels(
        self,
    ) -> list:
        return self.model.labels_

class MiniKmeans:
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
    ) -> MiniBatchKMeans:
        if n_clusters != 5:
            self.model = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init)
        elif optimal:
            self.model = MiniBatchKMeans(n_clusters=self.getOptimalCluster(), n_init=n_init)
        else:
            self.model = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init)
        self.model.fit(self.data)
        self.stats.setBaseStation(self.model.labels_)
        return self.model
    
    # TODO implement
    def getOptimalCluster(
        self,
    ) -> int:
        return 5
    
    def getLabels(
        self,
    ) -> list:
        return self.model.labels_

class AffinityProp:
    def setData(
        self,
        stats: ClusterMemberStats,
    ) -> None:
        self.stats = stats

        self.data = self.stats.getPosition()

    def generateModel(
        self,
        damping : float = 0.9,
    ) -> AffinityPropagation:
        self.model = AffinityPropagation(damping=damping)
        self.model.fit(self.data)
        self.stats.setBaseStation(self.model.labels_)
        return self.model
    
    def getLabels(
        self,
    ) -> list:
        return self.model.labels_

class Balanced:
    def setData(
        self,
        stats: ClusterMemberStats,
    ) -> None:
        self.stats = stats

        self.data = self.stats.getPosition()

    def generateModel(
        self,
        threshold: float = 0.01,
        n_clusters : int = 5,
    ) -> Birch:
        self.model = Birch(threshold=threshold, n_clusters=n_clusters)
        self.model.fit(self.data)
        self.stats.setBaseStation(self.model.labels_)
        return self.model
    
    def getLabels(
        self,
    ) -> list:
        return self.model.labels_

# TODO: check why spectral why cant work with random data?
class Spectral:
    def setData(
        self,
        stats: ClusterMemberStats,
    ) -> None:
        self.stats = stats

        self.data = self.stats.getPosition()

    def generateModel(
        self,
        n_clusters : int = 5,
    ) -> SpectralClustering:
        self.model = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors")
        self.model.fit(self.data)
        self.stats.setBaseStation(self.model.labels_)
        return self.model
    
    def getLabels(
        self,
    ) -> list:
        return self.model.labels_

class Gaussian:
    def setData(
        self,
        stats: ClusterMemberStats,
    ) -> None:
        self.stats = stats

        self.data = self.stats.getPosition()

    def generateModel(
        self,
        n_components : int = 5,
    ) -> GaussianMixture:
        self.model = GaussianMixture(n_components=n_components)
        self.model.fit(self.data)
        self.stats.setBaseStation(self.model.predict(self.data))
        return self.model

    def getLabels(
        self,
    ) -> list:
        return self.model.predict(self.data)

# TODO: How to make clsuter an interger not float
class LDA:
    def setData(
        self,
        stats: ClusterMemberStats,
    ) -> None:
        self.stats = stats

        self.data = self.stats.getPosition()

    def generateModel(
        self,
        n_components : int = 5,
        random_state: int = 0,
    ) -> LatentDirichletAllocation:
        self.model = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
        self.model.fit(self.data)
        self.stats.setBaseStation(self.model.transform(self.data))
        return self.model

    def getLabels(
        self,
    ) -> list:
        return self.model.transform(self.data)

class Markov:
    def setData(
        self,
        stats: ClusterMemberStats,
    ) -> None:
        self.stats = stats

        self.data = self.stats.getPosition()

    def generateModel(
        self,
        n_components : int = 5,
    ) -> GaussianHMM:
        self.model = GaussianHMM(n_components=n_components)
        self.model.fit(self.data)
        self.stats.setBaseStation(self.model.predict(self.data))
        return self.model

    def getLabels(
        self,
    ) -> list:
        return self.model.predict(self.data)

