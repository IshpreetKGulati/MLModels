import numpy as np
import random
import matplotlib.pyplot as plt

def genrate_data(theta, n, m):
    x = np.random.randn(n , m+1)
    for i in range(0, len(x)):
        x[i][0] = 1
    beta = np.random.randn(m+1)
    x_beta =  np.dot(x , beta) #matrix product
    sigmoid = 1 / (1 + np.exp(-x_beta)) #calculate probability i.e. sigmoid function
    y = np.where(sigmoid >=0.5, 1, 0)
    random_number = random.randint(1, 5000)
    
    # add noise 
    # change 1 to 0 and vice versa for theta elements
    for i in range(0, int(theta*n)):
        if y[i] == 0:
            y[i] = 1

        elif y[i] == 1:
            y[i] = 0

    return (x, y, beta)

def logistic_regression(X, Y, epoch, learning_rate, tou):
    n = X.shape[0]
    m = X.shape[1]

    beta = np.random.randn(m) #initialize m betas, randomly
    prev_cost = float('inf')
    X_beta = np.dot(X, beta)
    sig_value = 1 / (1 + np.exp(-X_beta))
    y = np.where(sig_value > 0.5, 1, 0)
    cost = -(np.sum(Y * np.log(sig_value) + (1 - Y) * np.log(1 - sig_value))) / n  # cost function
    print("initial cost", cost)
    
    for i in range(0, epoch):
        X_beta = np.dot(X , beta)
        sig_value = 1 / (1 + np.exp(-X_beta))#calculete probability for X_beta (initial value)

        y = np.where(sig_value > 0.5, 1, 0)#if probability is greater that 0.5 assign 1 else 0


        #cross entropy, the cost function
        #took negative so that it can be minimized
        cost = -(np.sum(Y * np.log(sig_value) + (1 - Y) * np.log(1 - sig_value)))/n # cost function
        
        # differentition of cost
        def_cost = 1/n * np.dot(X.T, sig_value - Y) 
        
        #reinitialize the betas
        beta = beta - learning_rate * def_cost
        
        if abs(prev_cost-cost) < tou:# if the difference is less than the threshold(tou) then break
            break
        prev_cost = cost
    return (beta, cost)


# calculate accuracy on test data
def accuracy(X_test_data, Y_test_data, beta):
    x_b = np.dot(X_test_data, beta)
    sigmoid = 1 / (1 + np.exp(-x_b))
    y = np.where(sigmoid >= 0.5, 1, 0)
    result = np.equal(y, Y_test_data) #assigns True if predicted y is same as original y
    return np.sum(result)/len(y) * 100


 #used to calculate the similarity between the original betas and the predicted betas
def cosine_similarity(beta, beta_predicted):
    cos_sim = np.dot(beta, beta_predicted) / (np.linalg.norm(beta) * np.linalg.norm(beta_predicted))
    return cos_sim


X, Y, beta = genrate_data(0.03, 5000, 2)
X_train_data, X_test_data = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
Y_train_data, Y_test_data = Y[:int(0.8 * len(X))], Y[int(0.8 * len(X)):]
print("original beta",beta)

print("##############################################################################")

beta_predicted_logistic, cost_logistic = logistic_regression(X_train_data, Y_train_data, 1000, 0.01, 0.0000000001)
print("Accuracy Logistic Regression run2",accuracy(X_train_data, Y_train_data, beta_predicted_logistic))
print("Cost for Logistic Regression run2",cost_logistic)
print("predicted beta in logistic regression run2",beta_predicted_logistic)
print("cosine similarity run2",cosine_similarity(beta, beta_predicted_logistic))
print("##############################################################################")


