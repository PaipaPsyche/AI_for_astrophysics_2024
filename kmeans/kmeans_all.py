import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# This Function return a distance in ndim dimension
# between two points (arguments are table with all the dimensions) 
def dist_f(dat, cent):
	
	dist = np.sum((dat[:] - cent[:])**2)

	return dist
	
def do_kmeans_clustering(data_type,nb_k=4,stopping_threshold=1e-6):
    #data_type = "3d"

    # Usefull data, feel free to add the ones you may need
    # for your own implementation of the algorithm

    #nb_k = 4


    # Read data and skip the header
    input_data = np.loadtxt("kmeans_input_file_%s.dat"%(data_type), skiprows=1)

    print (np.shape(input_data))

    # Extract dimension and datalist size from the loaded table
    nb_dim, nb_dat = np.shape(input_data)

    # The origin of the centers are selected randomly
    # To the position of some points in the dataset
    init_pos = np.random.randint(low=0, high=nb_dat, size=nb_k)

    centers = input_data[:,init_pos]
    new_centers = np.zeros((nb_dim, nb_k))
    nb_points_per_center = np.zeros((nb_k))

    stopping_threshold = 1e-6
    ############################### ################################
    #      Main loop, until the new centers do not move anymore
    ############################### ################################

    while dist_f(new_centers, centers) > stopping_threshold: 
        lbls = np.zeros(nb_dat)
        for  i in range(nb_dat):
            i_distances = np.array([ dist_f(input_data[:,i], centers[:,j]) for j in range(nb_k) ])
            ind_min = np.argmin(i_distances)
            nb_points_per_center[ind_min] += 1
            lbls[i] = ind_min
        for cnt in range(nb_k):
            new_centers[:,cnt] = np.sum(input_data[:,lbls == cnt], axis=1) / nb_points_per_center[cnt]
        centers = new_centers
    ################################ ################################
    #      Save the ending centroid position for visualisation
    ################################ ################################

    np.savetxt("kmeans_output_%s.dat"%(data_type), centers.T, header="%d %d"%(nb_dim,nb_k), comments="")
    plot_kmeans(data_type)
	
	
# VISUAL ++++++++++++++++++

def plot_kmeans(data_type):
    centers = np.loadtxt("kmeans_output_%s.dat"%(data_type), skiprows=1).T
    nb_dim, nb_k = np.shape(centers)

    data = np.loadtxt("kmeans_input_file_%s.dat"%(data_type), skiprows=1)
    nb_data = np.shape(data)[1]

    distances = np.zeros((nb_k, nb_data))

    for i in range(0,nb_data):
        for j in range(0,nb_k):
            for k in range(0, nb_dim):
                distances[j,i] += (centers[k, j] - data[k,i])**2

    arg_min = np.zeros(nb_data)
    for i in range(0,nb_data):
        arg_min[i] = np.argmin(distances[:,i])

    if (nb_dim == 2 ):

        for i in range(0, nb_k):
            ind = np.where(arg_min[:] == i)
            plt.plot(data[0,ind], data[1,ind], '.', color = plt.cm.rainbow(float(nb_k-i)/nb_k))
        plt.plot(centers[0,:], centers[1,:], '.', markersize=14)
    else :
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for i in range(0, nb_k):
            ind = np.where(arg_min[:] == i)
            ax.scatter(data[0,ind], data[1,ind], data[2,ind], color = plt.cm.rainbow(float(nb_k-i)/nb_k))
        ax.scatter(centers[0,:], centers[1,:], centers[2,:], marker = "x", c = "k", s = 64, depthshade="False", linewidth=3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    plt.show()
    plt.savefig(f"kmeans_{data_type}_res.png", dpi=200)
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

