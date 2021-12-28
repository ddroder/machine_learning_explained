# from kmeans import KMeansClustering
from km import KMeansAnim
# from kmeans_viz import KMeansAnim
import csv
import numpy as np 

if __name__=="__main__":
    anim=KMeansAnim()
    anim.run_kmeans()
    # anim.runner(name_of_class="KMeansAnim",name_of_file="km.py")
    anim.runner()
