import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter


class KMeans:
    def __init__(self, k=3, tol=1e-3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.tol = tol
        self.movement = None
    
    def generate_explanation(self, filename):
        with open(filename, 'w') as f:
            f.write('KMeans is a clustering algorithm that aims to partition n observations into k clusters. '
                    'The algorithm starts by randomly choosing k centroids, where k is the number of clusters. '
                    'It then iteratively assigns each data point to the nearest centroid and re-computes the '
                    'centroids based on the mean of all data points assigned to it. The algorithm continues '
                    'this process until the centroids stop moving or the maximum number of iterations is '
                    'reached.\n\n')
            
            f.write(f'The parameters used in this KMeans model are:\n')
            f.write(f'Number of clusters (k): {self.k}\n')
            f.write(f'Tolerance: {self.tol}\n')
            f.write(f'Maximum number of iterations: {self.max_iters}\n\n')
            
            f.write(f'The centroids for each iteration are:\n')
            for i, centroid in enumerate(self.centroids):
                f.write(f'Iteration {i+1}: {centroid}\n')



    def fit(self, X):
        self.centroids = X[np.random.choice(range(X.shape[0]), self.k, replace=False)]
        self.clusters = np.zeros(X.shape[0])
        for i in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.clusters = np.argmin(distances, axis=1)
            new_centroids = np.array([X[self.clusters == c].mean(axis=0) for c in range(self.k)])
            self.movement = np.max(np.abs(new_centroids - self.centroids))
            self.centroids = new_centroids
            print('Iteration {}, movement = {:.4f}'.format(i, self.movement))
            yield self.clusters, self.centroids
            if self.movement < self.tol:
                break


class KMeansAnimation:
    def __init__(self, kmeans_model, X):
        self.kmeans_model = kmeans_model
        self.writer = FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
        self.X = X
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim([np.min(X[:, 0]), np.max(X[:, 0])])
        self.ax.set_ylim([np.min(X[:, 1]), np.max(X[:, 1])])
        self.sc = self.ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5)
        self.centroid_sc = self.ax.scatter([], [], c='red')
        self.text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, va='top')
        self.iteration = 0

    def update(self, clusters_centroids):
        clusters, centroids = clusters_centroids
        self.sc.set_color([plt.cm.tab10(c) for c in clusters])
        self.centroid_sc.set_offsets(centroids)
        self.text.set_text('iteration: {}'.format(self.iteration))
        
        # Add text to explain the math happening in the animation
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        assigned_clusters = np.argmin(distances, axis=1)
        cluster_counts = [len(X[assigned_clusters == c]) for c in range(len(centroids))]
        text_objs = []
        text_str = 'Number of data points in each cluster:\n'
        for i, count in enumerate(cluster_counts):
            text_str += f'Cluster {i}: {count}\n'
        text_obj = self.ax.text(0.02, 0.92, text_str, transform=self.ax.transAxes, va='top')
        text_objs.append(text_obj)
        self.iteration += 1
        # Clear the text boxes at the start of each iteration
        for text_obj in text_objs:
            text_obj.remove()
        return self.sc, self.centroid_sc, self.text





    def animate(self, save_path=None):
        anim = FuncAnimation(self.fig, self.update, frames=kmeans.fit(X), repeat=False,interval=1000)
        if save_path:
            anim.save(save_path, writer=self.writer)
        else:
            plt.show()

if __name__ == '__main__':
    import pandas as pd 
    def test_kmeans_animation():
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 2), columns=['x', 'y'])
        kmeans = KMeans(k=3, max_iters=100)
        animation = KMeansAnimation(kmeans, X)
        animation.animate(save_path='kmeans_animation.mp4')    

    # test_kmeans_animation()
    kmeans = KMeans(k=3, max_iters=100)
    X = np.random.randn(100, 2)
    animation = KMeansAnimation(kmeans, X)
    animation.animate()
    # animation.animate(save_path='kmeans_animation.mp4')
    kmeans.generate_explanation("kmeans.txt")