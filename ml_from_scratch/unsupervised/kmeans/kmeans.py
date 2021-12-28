import matplotlib.pyplot as plt
import numpy as np 
import subprocess
from kmeans_viz import KMeansAnim
class KMeansClustering(KMeansAnim):
    def __init__(self, k):
        self.k = k
        self.centers = None
        self.clusters = None
        self.cluster_vals = []
        self.center_vals = []
        self.coords=[]

    def plot_state(self, X):
        plt.scatter(X[:, 0], X[:, 1], c=self.clusters)
        plt.scatter(self.centers[:, 0], self.centers[:, 1], c='r', s=70)
        plt.grid()
    def make_anim(self,class_name,file_name,quality='high'):
        # os.system()
        cmd="manim"
        if quality.lower()=="high":
            qual_cmd="qh"
        elif quality.lower()=="low":
            qual_cmd="ql"
        else:
            qual_cmd=""

        make_anim=f"manim -p{qual_cmd} {file_name} {class_name}"
    def fit(self, X, runs=10, plot=False, num_iter=100, plot_freq=0.1):
        # super().__init__()
        best_var = 10**9
        for _ in range(runs):
            fail = False
            centers = np.random.randn(self.k, X.shape[1]) * 3
            for iter_ in range(num_iter):
                arr = np.zeros((X.shape[0], self.k))
                for i, center in enumerate(centers, 0):
                    arr[:, i] = (((X - center) ** 2).sum(axis=1) ** 0.5)

                self.clusters = np.argmin(arr, axis=1)
                self.cluster_vals.append(self.clusters)

                if plot and iter_ % int(num_iter * plot_freq) == 0:
                    self.plot_state(X)
                    plt.title("Iteration " + str(iter_))
                    plt.show()

                for cno in range(self.k):
                    try:
                        centers[cno] = X[self.clusters == cno, :].mean(axis=0)
                    except:
                        fail = True
                        break
                self.center_vals.append(np.copy(centers))

            if fail:
                continue

            var = 0
            for center in centers:
                var += np.mean(((X - center) ** 2)) ** 0.5
            if var < best_var:
                best_var = var
                self.centers = centers




