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
        if optimal:
            n_clusters = self.optimalClusterSize(maxClusters = n_clusters)
        self.model = KMeans(n_clusters=n_clusters, n_init=n_init)
        self.model.fit(self.data)
        self.cluster_center = [list(x) for x in self.model.cluster_centers_]
        self.label = self.model.labels_
        self.stats.setBaseStation(self.label)

        return self.model
    
    def getLabels(self) -> list:
        return self.label

    def optimalClusterSize(
        self,
        maxClusters: int,
        nrefs: int = 5
    ) -> int:
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = []
        for gap_index, k in enumerate(range(1, maxClusters)):
            refDisps = np.zeros(nrefs)
            for i in range(nrefs):
                randomReference = np.random.random_sample(size=self.data.shape)
                km = KMeans(n_clusters=k, n_init=10)
                km.fit(randomReference)
                refDisp = km.inertia_
                refDisps[i] = refDisp

            km = KMeans(n_clusters=k, n_init=10)
            km.fit(self.data)
            origDisp = km.inertia_
            gap = np.log(np.mean(refDisps)) - np.log(origDisp)
            gaps[gap_index] = gap
            resultsdf.append([k,gap])
        return gaps.argmax() + 1
    

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
        eps : float = 300,
        min_samples: int = 100,
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
        if optimal:
            n_clusters = self.optimalClusterSize(maxClusters = n_clusters)
        self.model = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init)
        self.model.fit(self.data)
        self.stats.setBaseStation(self.model.labels_)
        return self.model
    
    def getLabels(
        self,
    ) -> list:
        return self.model.labels_
    
    def optimalClusterSize(
        self,
        maxClusters: int,
        nrefs: int = 5
    ) -> int:
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = []
        for gap_index, k in enumerate(range(1, maxClusters)):
            refDisps = np.zeros(nrefs)
            for i in range(nrefs):
                randomReference = np.random.random_sample(size=self.data.shape)
                km = KMeans(n_clusters=k, n_init=10)
                km.fit(randomReference)
                refDisp = km.inertia_
                refDisps[i] = refDisp

            km = KMeans(n_clusters=k, n_init=10)
            km.fit(self.data)
            origDisp = km.inertia_
            gap = np.log(np.mean(refDisps)) - np.log(origDisp)
            gaps[gap_index] = gap
            resultsdf.append([k,gap])
        return gaps.argmax() + 1

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
