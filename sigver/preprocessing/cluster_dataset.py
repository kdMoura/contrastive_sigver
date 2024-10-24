import numpy as np
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, 
    AffinityPropagation, MeanShift, 
    SpectralClustering, Birch, OPTICS
)
from sklearn.mixture import GaussianMixture
import skfuzzy as fuzz

PROTOTYPE_MODELS = ['kmeans','gmm','agglomerative','dbscan','affinity','meanshift','spectral','birch','optics','fuzzy_cmeans']

class PrototypeModel:
    
    def __init__(self, name, 
                 n_clusters=None, 
                 eps=35, 
                 min_samples=5, 
                 damping=0.9, 
                 bandwidth=None, 
                 m = 1.5, 
                 error = 0.005, 
                 maxiter = 1000
                 ):
        self.name = name
        self.n_clusters = n_clusters
        self.fuzzy_cmeans_parameters = {'m': m, 'error': error, 'maxiter': maxiter}
        

        if name == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42)

        elif name == 'gmm':
            self.model = GaussianMixture(n_components=n_clusters, covariance_type='diag', random_state=42)
            
        elif name == 'agglomerative':
            self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            #linkage='ward'
        
        elif name == 'dbscan':
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
        
        elif name == 'affinity':
            self.model = AffinityPropagation(damping=damping)
        
        elif name == 'meanshift':
            self.model = MeanShift(bandwidth=bandwidth)
        
        elif name == 'spectral':
            self.model = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
            # Could increase affinity parameter or assign eigen_solver for high-dimensional data
            #self.model = SpectralClustering(n_clusters=k, random_state=42, eigen_solver='arpack') 
        
        elif name == 'birch':
            self.model = Birch(n_clusters=n_clusters)
        
        elif name == 'optics':
            self.model = OPTICS(eps=eps, min_samples=min_samples)
        
    
    def __str__(self):
        return f"PrototypeModel(name={self.name}, k={self.n_clusters}, model={type(self.model).__name__})"
    
    def to_string(self):
        return self.__str__()
    
    def fit(self, data):
        self.data = data
        if self.name == 'fuzzy_cmeans':
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.T, 
                                                             c=self.n_clusters, 
                                                             **self.fuzzy_cmeans_parameters)
            self.model = {'cntr': cntr, 'u': u}

        else:
            self.model.fit(data)

    
    def get_k(self):
        return self.n_clusters
    
    def get_prototypes(self):
        if self.name == 'kmeans':
            prototypes = self.model.cluster_centers_
        
        elif self.name == 'gmm':
            prototypes = self.model.means_ 
        
        elif self.name == 'agglomerative':
            labels = self.model.labels_
            prototypes = np.array([self.data[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        
        elif self.name == 'dbscan':
            labels = self.model.labels_
            unique_labels = np.unique(labels)
            prototypes = []
            for label in unique_labels:
                if label != -1:
                    cluster_points = self.data[labels == label]
                    cluster_mean = cluster_points.mean(axis=0)
                    prototypes.append(cluster_mean)
            prototypes = np.array(prototypes)
        
        elif self.name == 'affinity':
            # Use exemplars as prototypes
            prototypes = self.data[self.model.cluster_centers_indices_]
        
        elif self.name == 'meanshift':
            prototypes = self.model.cluster_centers_
        
        elif self.name == 'spectral':
            labels = self.model.labels_
            prototypes = np.array([self.data[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        
        elif self.name == 'birch':
            if self.n_clusters:
                labels = self.model.labels_
                prototypes = np.array([self.data[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            else:
                prototypes = self.model.subcluster_centers_
        
        elif self.name == 'optics':
            labels = self.model.labels_
            unique_labels = np.unique(labels)
            prototypes = []
            for label in unique_labels:
                if label != -1:
                    cluster_points = self.data[labels == label]
                    cluster_mean = cluster_points.mean(axis=0)
                    prototypes.append(cluster_mean)
            prototypes = np.array(prototypes)
        
        elif self.name == 'fuzzy_cmeans':
            prototypes = self.model['cntr']

        return prototypes