####################################### Code K-means (machine learning) #########################################
########################################### Ibsen P. S. Gomes ###################################################
############################ Observatório Nacional - Universidade Federal Fluminense ############################

# Libraries:
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import sys
import random
from math import sqrt


# Normalization:

def norm_data(data):
    
    '''
    Data normalization using the "minmax" method
    
    input: 
    data = column in a database 
    
    operation:
    new_data = [data - min(data)]/[max(data) - min(data)]
    
    output:
    data_norm = normalized data
    '''
    
    data_norm = np.zeros((len(data)))
    for i in range(len(data)):
        data_norm[i] = (data[i] - min(data))/(max(data) - min(data))
        
    return data_norm


# Metrics:

def euclidian(v1, v2, check_input=True):
    
    '''
    Given two vectors and calculate the Euclidean distance between vectors
    
    input:
    v1 = 1D vector
    v2 = 1D vector
    
    operation:
    distance = sqrt(sum(v1 - v2)**2)
    
    output:
    euclidian_distance
    '''
    
    # "Assert" to ensure the entries are 1D:
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if check_input is True:
        assert v1.ndim == 1, 'a must be a 1D' 
        assert v2.ndim == 1, 'x must be a 1D'
        
    dist = 0.0
    for i in range(len(v1)):
        dist += (v1[i] - v2[i])**2
 
    euclidian_distance = sqrt(dist)
    
    return euclidian_distance


def manhattan(v1, v2, check_input=True):
    
    '''
    Given two vectors and calculate the Manhattan distance between vectors
    
    input:
    v1 = 1D vector
    v2 = 1D vector
    
    operation:
    distance = |sum(v1 - v2)|
    
    output:
    euclidian_distance
    '''
    
    # "Assert" to ensure the entries are 1D:
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if check_input is True:
        assert v1.ndim == 1, 'a must be a 1D' 
        assert v2.ndim == 1, 'x must be a 1D'
    
    dist = 0.0
    for i in range(len(v1)):
        dist += abs(v1[i] - v2[i])
 
    manhattan_distance = dist
    return manhattan_distance


### Kmeans++ inicialization:

def kmeans_plus_plus(data, k):
    
    '''
    Initialization using kmeans++:
     inputs:
     data = numpy array where an initial point will be drawn and, from it, select other centroids
     k = number of clusters
    
     output:
     centroid coordinates, after using kmeans++
    
     Note: the code does not select the next centroid through a probability distribution,
     it just selects the next centroid as being the farthest from the already selected centroids!
    '''
    
    ## initialize a list of centroids and add a randomly selected data point to the list:
    centroids = []
    centroids.append(data[np.random.randint(
            data.shape[0]), :])
  
    ## iterations to generate the other centroids (k - 1):
    for c_id in range(k - 1):
         
        ## initialize a list to store distances from data to the nearest centroid:
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize
             
            ## calculate the distance from a given to the selected centroids and store the minimum distance:
            for j in range(len(centroids)):
                temp_dist = euclidian(point, centroids[j])
                d = min(d, temp_dist)
            dist.append(d)
             
        ## select a die with maximum distance as our next centroid
        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        centroids.append(next_centroid)
        dist = []
        
    return centroids


# K-means:

def K_means(data, k, it, tol, random='range'):
    
    '''
    This code applies the k-means method to an input dataset
    
     Prohibited:
    
     data = data to be grouped;
     k = number of centroids (how many groups you want to divide the data)
     tol = tolerance
     random = way to initialize the centroids
    
     method:
    
     -> calculation of the distance between each data with the centroids
     -> update of centroid position to group mean
     -> if position [i] - position [i-1] of the centroids is less than tolerance (tol) the code stops!
     -> if the tolerance criterion is not met, the code continues to perform iterations (maybe do all iterations!)
    
     exit:

     centroids = coordinates of the centroids after the end of iterations
     count = how many iterations were performed
     index = label of each sample after convergence
     inertia = sum of squared intra-cluster distance
    '''
     
    nprop = data.shape[1]
    centroids = np.zeros((k, nprop))
    
    # Conditional -> if "range", the centroids will be drawn within a range:  
    if random == 'range':
        
        for npro in range(nprop):
            for i in range(k):
                centroids[i,npro]  = np.array(np.random.uniform(min(data[:,npro]),max(data[:,npro])))
            
    # Conditional -> if "input_data", the centroids will be chosen from the input data:     
    if random == 'input_data':
        
        for i in range(k):
            index = np.random.randint(data.shape[0])
            centroids[i] = data[index,:]
            
    # Conditional -> for k-means++ initialization:
    if random == 'kmeans++':
        
        centroids = kmeans_plus_plus(data, k) # chamando a função kmeans++
        
        
    conta = 0

    index = [0.0]*np.size(data,0) # to store the labels of each sample
    
    for t in range(it):
        bestmatches = [[] for i in range(k)]
     
        #Check which centroid is closer to each instance
        for j in range(len(data)):
            row=data[j]
            bestmatche = 0 ## Here, store the index of the shortest distance for comparison
            for i in range(k):
                d = euclidian(centroids[i],row) 
                if d < euclidian(centroids[bestmatche],row): #Storing the smallest distance between the sample and the centroid
                    bestmatche = i 
            index[j] = bestmatche # add an index that represents the nearest centroid
            bestmatches[bestmatche].append(j)
     
        #Update the centroids to the new cluster mean
        for i in range(k):
            avgs=[0.0]*len(data[0])
            if len(bestmatches[i])>0:
                for rowid in bestmatches[i]:
                    for m in range(len(data[rowid])):
                        avgs[m] += data[rowid][m]
                for j in range(len(avgs)):
                    avgs[j] /= len(bestmatches[i])
                centroids[i]=avgs
                
        # Conditional for convergence: if position [i] - previous position[i-1]...
         # ...if less than a tolerance the code stops and converges!
        if np.allclose(centroids[i],centroids[i-1], rtol=tol) == True:
            break 
            
        conta += 1
    
    # Calculate Inertia:
    inercia = 0
    for centro_index in range (k):
        for amostra in range (len(index)):
            # If the sample index has the same index as the centroid, the operation for Inertia will be performed:
            if index[amostra] == centro_index: 
                inercia += sum(abs((data[amostra] - centroids[centro_index])**2))
                
    centroids = np.asarray(centroids)
    
    return centroids, conta, index, inercia


#################################################### END CODE ###################################################