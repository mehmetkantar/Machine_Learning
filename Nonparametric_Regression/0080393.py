import math
import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt("yellowstone_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("yellowstone_test.csv", delimiter = ",", skip_header = 1)

# get x and y values
x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1]
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 1.6
maximum_value = 5.1
x_interval = np.arange(start = minimum_value, stop = maximum_value, step = 0.001)

def plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(x_train, y_train, "b.", markersize = 10)
    plt.plot(x_test, y_test, "r.", markersize = 10)
    plt.plot(x_interval, y_interval_hat, "k-")
    plt.xlim([1.55, 5.15])
    plt.xlabel("Eruption time (min)")
    plt.ylabel("Waiting time to next eruption (min)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)



# STEP 3
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def regressogram(x_query, x_train, y_train, left_borders, right_borders):
    # your implementation starts below
    
    N = x_query.shape[0]
    y_hat = np.zeros(N)


    # Check if the x_query is a dense grid
    is_dense_grid = (N > 1000) and (np.median(np.diff(x_query)) < 0.01)



    for i in range(N):

        assigned = False



        for j in range(len(left_borders)):  # iterate through each bin
            


            if left_borders[j] < x_query[i] <= right_borders[j]:
                
                
                indices = np.where((x_train > left_borders[j]) & (x_train <= right_borders[j]))[0]
                
                if len(indices) > 0:
                    y_hat[i] = np.mean(y_train[indices])
                
                assigned = True
                break



        # Iniatilly starts with 0, but if no bin is assigned and is dense grid, set to NaN
        if (not assigned) and is_dense_grid:
            y_hat[i] = np.nan

            


    # your implementation ends above
    return(y_hat)
    
bin_width = 0.35
left_borders = np.arange(start = minimum_value, stop = maximum_value, step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width, stop = maximum_value + bin_width, step = bin_width)

y_interval_hat = regressogram(x_interval, x_train, y_train, left_borders, right_borders)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("regressogram.pdf", bbox_inches = "tight")

y_test_hat = regressogram(x_test, x_train, y_train, left_borders, right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Regressogram => RMSE is {} when h is {}".format(rmse, bin_width))



# STEP 4
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def running_mean_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below
    
    
    N = x_query.shape[0]
    y_hat = np.zeros(N)
    


    for i in range(N):
        
        
        # Create a window centered at x_query[i] with width 2*bin_width
        left = x_query[i] - bin_width / 2
        right = x_query[i] + bin_width / 2
        


        # Find all training points within this window
        indices = np.where((x_train >= left) & (x_train < right))[0]
        

        if len(indices) > 0:
            y_hat[i] = np.mean(y_train[indices])
    
    
    # your implementation ends above
    return(y_hat)

bin_width = 0.35

y_interval_hat = running_mean_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("running_mean_smoother.pdf", bbox_inches = "tight")

y_test_hat = running_mean_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Running Mean Smoother => RMSE is {} when h is {}".format(rmse, bin_width))



# STEP 5
# assuming that there are N query data points
# should return a numpy array with shape (N,)
def kernel_smoother(x_query, x_train, y_train, bin_width):
    # your implementation starts below
    
    
    N = x_query.shape[0]
    y_hat = np.zeros(N)
    



    for i in range(N):
        
        
        
        # Calculate internal = (x - xi) / h
        internal = (x_query[i] - x_train) / bin_width
        


        # Gaussian kernel K(internal) = (1/sqrt(2*pi)) * exp(-0.5 * internal^2)
        K = (1 / np.sqrt(2 * math.pi)) * np.exp(-0.5 * internal**2)
        


        # Weighted average sum(K * y) / sum(K)
        if np.sum(K) > 0:
            y_hat[i] = np.sum(K * y_train) / np.sum(K)
    


    
    # your implementation ends above
    return(y_hat)

bin_width = 0.35

y_interval_hat = kernel_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)
fig.savefig("kernel_smoother.pdf", bbox_inches = "tight")

y_test_hat = kernel_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Kernel Smoother => RMSE is {} when h is {}".format(rmse, bin_width))
