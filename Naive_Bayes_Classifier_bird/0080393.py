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



# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    K=np.max(y)
    class_priors=np.zeros(K)
    for i in range(K):
        class_priors[i]=np.sum(y==i+1)/len(y)

    
    
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 4
# assuming that there are K classes and D features
# should return two numpy arrays with shape (K, D)
def estimate_means_and_deviations(X, y):
    # your implementation starts below
    K=np.max(y)
    D=X.shape[1]
    means=np.zeros((K,D))
    deviations=np.zeros((K,D))


    for i in range(K):
        means[i]=np.sum(X[y==i+1],axis=0)/np.sum(y==i+1)
        deviations[i]=np.sqrt(np.sum((X[y==i+1]-means[i])**2,axis=0)/np.sum(y==i+1))
        


        
    # your implementation ends above
    return(means, deviations)

means, deviations = estimate_means_and_deviations(X_train, y_train)
print(means)
print(deviations)



# STEP 5
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, means, deviations, class_priors):
    # your implementation starts below
    
    N=X.shape[0] 
    K=means.shape[0]
    score_values=np.zeros((N,K))
    score_values1=np.zeros((N,K))

    for i in range(N):
        for j in range(K):
            score_values[i,j]=np.log(class_priors[j])+np.sum(-0.5*np.log(2*np.pi)-np.log(deviations[j])-((X[i]-means[j])**2)/(2*deviations[j]**2))

    
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, means, deviations, class_priors)
print(scores_train[0:10, :])

scores_test = calculate_score_values(X_test, means, deviations, class_priors)
print(scores_test[0:10, :])



# STEP 6
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    K=np.max(y_truth)
    confusion_matrix=np.zeros((K,K), dtype=int)
    for i in range(len(y_truth)):
        confusion_matrix[y_truth[i]-1,np.argmax(scores[i])]+=1 #
    
   
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)
print("Training accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_train)) / np.sum(confusion_train)))

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
print("Test accuracy is {:.2f}%.".format(100 * np.sum(np.diag(confusion_test)) / np.sum(confusion_test)))
