from kmeans import KMeansClustering
# from kmeans_viz import KMeansAnim
import csv
import numpy as np 

if __name__=="__main__":
    # from 
    model=KMeansClustering(3)
    model.load_data("test_dat.csv")
    coords=model.coords
    num_iter=3
    model.fit(np.array(coords)[:, :2],
                       num_iter=num_iter, runs=1)