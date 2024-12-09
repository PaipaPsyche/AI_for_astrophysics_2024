############################### ################################
# K-means exercise for the M2-OSAE Machine Learning lessons
# contact : David Cornu - david.cornu@observatoiredeparis.psl.eu
################################ ################################
import numpy as np


# This Function return a distance in ndim dimension
# between two points (arguments are table with all the dimensions) 
def dist_f(dat, cent):
	
	dist = np.sum((dat[:] - cent[:])**2)

	return dist
	

data_type = "3d"

# Usefull data, feel free to add the ones you may need
# for your own implementation of the algorithm

nb_k = 4


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

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
