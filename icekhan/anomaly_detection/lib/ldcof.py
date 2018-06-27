import math
import logging
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

class LDCOF(object):
    """
    Local density cluster-based outlier factor (LDCOF)
    @link https://www.google.ru/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwjGp_Ph0b3VAhVML8AKHUUDDEUQFggqMAA&url=https%3A%2F%2Fwww.dfki.de%2Fweb%2Fforschung%2Fpublikationen%2FrenameFileForDownload%3Ffilename%3Dslides.pdf%26file_id%3Duploads_1635&usg=AFQjCNEexJr1vakk96y5jgUN0IjPI8i-6w
    """

    def __init__(self, alpha = 0.8, n_clusters = 8):
        super(LDCOF, self).__init__()
        self.alpha = alpha # % of entries for large clusters (LC)
        self.n_clusters = n_clusters # number of clusters for KMeans
        self.data = []
        self.distances = {}
        self.cluster_centers = []
        self.clusterer = KMeans(n_clusters=n_clusters, n_jobs=-1)

    def __clusters_separation(self):
        """ split clusters into large clusters (LC) and small clusters (SC) """
        D = len(self.data)
        cluster_sizes = []
        for cluster in range(self.n_clusters):
            cluster_sizes.append((cluster, self.data_clusters.tolist().count(cluster)))
        self.cluster_sizes = sorted(cluster_sizes, key=lambda x: x[1], reverse=True)

        cumulative_size = 0
        threshold = None
        for i in range(self.n_clusters):
            cumulative_size += self.cluster_sizes[i][1]
            if cumulative_size > D * self.alpha:
                threshold = i
                break

        self.LC = [elem[0] for elem in self.cluster_sizes][:threshold]
        self.SC = [elem[0] for elem in self.cluster_sizes][threshold:]

    def __cluster_avg_distances(self):
        """ calculates mean distances within clusters """
        distances = {}
        for cluster in range(self.n_clusters):
            idx = np.where(self.data_clusters == cluster)[0]
            diff = self.data[idx] - self.cluster_centers[cluster]
            dists = [d for d in map(lambda t: math.sqrt(sum([pow(e, 2) for e in t])), diff)]
            if len(dists) > 0:
                distance = sum(dists) / len(dists)
            else:
                distance = 0.00000000001

            distances[cluster] = distance

        self.distances = distances

    def fit(self, data):
        """ fit model on data """
        self.data = data

        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(data)
        self.clusterer = kmeans
        logging.info('Fit has been completed')

        self.data_clusters = self.clusterer.predict(data)
        self.cluster_centers = self.clusterer.cluster_centers_
        logging.info('Cluster calculation has been completed')

        self.__clusters_separation()
        logging.info('Cluster separation has been completed')

        self.__cluster_avg_distances()
        logging.info('Cluster avg distances has been calculated')

    def __ldcof(self, data=[]):
        if len(data.shape) == 1: data = [data]

        clusters = self.clusterer.predict(data)
        lc_centers = self.cluster_centers[self.LC]

        res = []
        for i, cluster in zip(range(len(clusters)), clusters):
            if cluster in self.LC:
                _sum = [pow(elem[0] - elem[1], 2) for elem in zip(data[i], self.cluster_centers[cluster])]
                dist = math.sqrt(sum(_sum))
                if self.distances[cluster] == 0:
                    res.append(dist / 0.00001)
                else:
                    res.append(dist / self.distances[cluster])
            else:
                entry = data[i]
                min_dist_to_cluster = 99999999999
                for lc in self.LC:
                    center = self.cluster_centers[lc]
                    dist = math.sqrt(sum([e for e in map(lambda x: pow(x, 2), data[i] - center)]))
                    if dist < min_dist_to_cluster:
                        min_dist_to_cluster = dist

                if self.distances[cluster] == 0:
                    res.append(min_dist_to_cluster / 0.00001)
                else:
                    res.append(min_dist_to_cluster / self.distances[cluster])

        return res


    def transform(self, data):
        res = self.__ldcof(data)
        #logging.info("Result: " + str(res))
        return res
