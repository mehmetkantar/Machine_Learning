import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



X_train = np.genfromtxt("bird_sounds_features_train.csv", delimiter = ",")
y_train = np.genfromtxt("bird_sounds_labels_train.csv", delimiter = ",", dtype = int)
X_test = np.genfromtxt("bird_sounds_features_test.csv", delimiter = ",")
y_test = np.genfromtxt("bird_sounds_labels_test.csv", delimiter = ",", dtype = int)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

train_mean = X_train.mean(axis = 0)
train_std = X_train.std(axis = 0)

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std



# STEP 3
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def sigmoid(X, W, w0):
    # your implementation starts below

    scores= 1/(1+np.exp(-(np.matmul(X,W)+w0)) )
    
    # your implementation ends above
    return(scores)



# STEP 4
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def one_hot_encoding(y):
    # your implementation starts below

    N = y.shape[0]
    K = int(np.max(y))
    Y = np.zeros((N, K))


    for n in range(N):
        Y[n, int(y[n]) - 1] = 1.0 # one-hot encoding
    
    # your implementation ends above
    return(Y)



np.random.seed(421)
D = X_train.shape[1]
K = np.max(y_train)
Y_train = one_hot_encoding(y_train)
W_initial = np.random.uniform(low = -0.001, high = 0.001, size = (D, K))
w0_initial = np.random.uniform(low = -0.001, high = 0.001, size = (1, K))



# STEP 5
# assuming that there are D features and K classes
# should return a numpy array with shape (D, K)
def gradient_W(X, Y_truth, Y_predicted):
    # your implementation starts below

    gradient =np.matmul( X.T , ((Y_predicted - Y_truth) * (Y_predicted * (1.0 - Y_predicted))))  
    
    
    
    # your implementation ends above
    return(gradient)

# assuming that there are K classes
# should return a numpy array with shape (1, K)
def gradient_w0(Y_truth, Y_predicted):
    # your implementation starts below


    gradient = ((Y_predicted - Y_truth) * (Y_predicted * (1.0 - Y_predicted)))  
    gradient = np.sum(gradient, axis=0, keepdims=True)  


    # your implementation ends above
    return(gradient)



# STEP 6
# assuming that there are N data points and K classes
# should return three numpy arrays with shapes (D, K), (1, K), and (1000,)
def discrimination_by_regression(X_train, Y_train,
                                 W_initial, w0_initial):
    eta = 0.05
    iteration_count = 1000

    W = W_initial
    w0 = w0_initial

    iteration = 1
    objective_values = []
    # your implementation starts below

    while iteration <= iteration_count:

        iteration += 1
        Y_predicted = sigmoid(X_train, W, w0)  # (N, K)


        error = 0.5 * np.sum((Y_train - Y_predicted) ** 2)
        objective_values.append(error)


        gradien_W = gradient_W(X_train, Y_train, Y_predicted)  
        gradien_w0 = gradient_w0(Y_train, Y_predicted)  


        W += -eta * gradien_W
        w0 += -eta * gradien_w0


    objective_values = np.array(objective_values, dtype=float)




    # your implementation ends above
    return(W, w0, objective_values)

W, w0, objective_values = discrimination_by_regression(X_train, Y_train,
                                                       W_initial, w0_initial)

print(W)
print(w0)
print(objective_values[0:10])



fig = plt.figure(figsize = (10, 6))
plt.plot(range(1, len(objective_values) + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()
fig.savefig("hw02_iterations.pdf", bbox_inches = "tight")



# STEP 7
# assuming that there are N data points
# should return a numpy array with shape (N,)
def calculate_predicted_class_labels(X, W, w0):
    # your implementation starts below

    Y_predicted = sigmoid(X, W, w0)  
    y_predicted = np.argmax(Y_predicted, axis=1) + 1  
    
    # your implementation ends above
    return(y_predicted)

y_hat_train = calculate_predicted_class_labels(X_train, W, w0)
print(y_hat_train[0:10])

y_hat_test = calculate_predicted_class_labels(X_test, W, w0)
print(y_hat_test[0:10])



# STEP 8
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, y_predicted):
    # your implementation starts below

    K = int(np.max(y_truth))
    confusion_matrix = np.zeros((K, K), dtype=int)  

    for true_label, pred_label in zip(y_truth, y_predicted):
        confusion_matrix[int(true_label) - 1, int(pred_label) - 1] += 1  # increment count

    
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, y_hat_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, y_hat_test)
print(confusion_test)

print("Training accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_train)) / np.sum(confusion_train)))

print("Test accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_test)) / np.sum(confusion_test)))

