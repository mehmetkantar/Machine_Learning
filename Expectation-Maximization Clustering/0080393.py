import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-5.0, +0.0],
                        [+0.0, +5.0],
                        [+0.0, -5.0],
                        [+0.0, +0.0],
                        [+5.0, -2.5]])
group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+0.1, +0.0],
                               [+0.0, +1.5]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 5

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    


    # Read initial centroids from file
    centroids = np.genfromtxt("hw05_initial_centroids.csv", delimiter=",")
    

    # Get number of data points
    N = X.shape[0]
    

    # Calculate distances from each point to each centroid
    
    distances = np.zeros((N, K))
    for k in range(K):
        distances[:, k] = np.sum((X - centroids[k])**2, axis=1)
    



    
    # Assign each point to the nearest centroid
    
    assignments = np.argmin(distances, axis=1)
    

    # Initialize means as the centroids
    
    means = centroids.copy()
    
    # Initialize covariances by calculating sample covariance for each cluster
    
    covariances = np.zeros((K, 2, 2))
    
    
    for k in range(K):
    
        cluster_points = X[assignments == k]
    
        if len(cluster_points) > 0:
            diff = cluster_points - means[k]
            covariances[k] = np.dot(diff.T, diff) / len(cluster_points)
    


    
    # Initialize priors as the proportion of points in each cluster
    
    priors = np.zeros(K)
    for k in range(K):
        priors[k] = np.sum(assignments == k) / N
    


    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below
    


    N = X.shape[0]
    

    
    # Run EM for 100 iterations
    
    
    for iteration in range(100):
    
        # E-step: Calculate responsibilities (posterior probabilities)
    
        responsibilities = np.zeros((N, K))
        
        for k in range(K):

            
            # Calculate multivariate Gaussian PDF for each point
            
            diff = X - means[k]
            inv_cov = np.linalg.inv(covariances[k])
            det_cov = np.linalg.det(covariances[k])
            
            
            # Mahalanobis distance
            
            mahal_dist = np.sum(diff @ inv_cov * diff, axis=1)
            
            # PDF calculation
            
            normalization = 1.0 / np.sqrt((2 * np.pi)**2 * det_cov)
            responsibilities[:, k] = priors[k] * normalization * np.exp(-0.5 * mahal_dist)
        


        
        # Normalize responsibilities so they sum to 1 for each data point
        
        responsibilities = responsibilities / np.sum(responsibilities, axis=1, keepdims=True)
        
        # M-step: Update parameters
        # Effective number of points assigned to each cluster
        
        
        N_k = np.sum(responsibilities, axis=0)
        


        # Update priors
        
        priors = N_k / N
        



        # Update means
        for k in range(K):
            means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / N_k[k]
        



        # Update covariances
        for k in range(K):
        
            diff = X - means[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            covariances[k] = np.dot(weighted_diff.T, diff) / N_k[k]
    


    # Final assignments based on maximum responsibility
    
    assignments = np.argmax(responsibilities, axis=1)
    





    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    


    colors = ['red', 'blue', 'green', 'purple', 'orange']
    


    plt.figure(figsize=(8, 8))
    
    for k in range(K):
        cluster_points = X[assignments == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=colors[k], s=10, alpha=0.6)
    


    # Function to draw ellipse contour at a specific probability level
    def draw_ellipse(mean, cov, color, linestyle):
        # Create grid for contour
        x_range = np.linspace(-8, 8, 200)
        y_range = np.linspace(-8, 8, 200)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        pos = np.dstack((X_grid, Y_grid))
        
        # Calculate PDF values
        rv = stats.multivariate_normal(mean, cov)
        Z = rv.pdf(pos)
        
        # Draw contour at level 0.01
        plt.contour(X_grid, Y_grid, Z, levels=[0.01], colors=color, linestyles=linestyle)
    


    # Draw original Gaussian densities (dashed lines)
    for k in range(K):
        draw_ellipse(group_means[k], group_covariances[k], 'k', 'dashed')
    


    # Draw fitted Gaussian densities (solid lines)
    for k in range(K):
        draw_ellipse(means[k], covariances[k], colors[k], 'solid')
    



    # Set labels and limits
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.grid(False)
    
    
    # Show plot
    plt.show()
    
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)