"""
Linear Regression
"""
import random
import numpy as np
import matplotlib.pyplot as plt

# this function generates random data
#sigma -> varience
#n -> total number of instances
#m -> number of columns
def genrate_data(sigma, n, m):

	# e is unexplained error, it is normally distributed with mean = 0 and varience = 1
    e = random.gauss(0, sigma)
    x = np.random.rand(n , m+1) #randomly generates x
    for i in range(0, len(x)): #makes the first column of x as 1
        x[i][0] = 1
    beta = np.random.rand(m+1) #randomly generates beta
    x_beta =  x @ beta #matrix product of x with beta
    y = x_beta + e #adding the matrix product and e we get are original y, which can be comapared with the predicted y to calculate accuracy

    return(x, y, beta)

def Linear_Regression(X, Y, k, tou, alpha):

    m = len(X[0])
    # Step 1 -> Initialisation
    beta = np.random.rand(m) # randomly selected beta value

    E = Y - X @ beta #error between the original y and predicted y
    curr_cost = np.dot(E, E) #the initial cost

    differentiation_of_cost = - (2 * E @ X)

    # Step 2 -> iterate until the desired epochs has been reached or until the threshold. Upgrade the betas and minimise the cost
    for i in range (0, k):
        E = Y - X @ beta
        cost = np.dot(E,E)
        differentiation_of_cost = - (2 * E @ X)

        if abs(curr_cost- cost) < tou: #tou is the threshold on change in cost function from previous to current iteration
        #    print(1)
            break
        else:
            beta = beta - alpha * differentiation_of_cost #update the beta value

        curr_cost = cost #updating the current cost

    return ( beta, curr_cost)

X, Y, beta = genrate_data(10, 300, 2)
x = np.array(X)
#print(x.shape)
y = np.array([Y])
x = np.transpose(x)
#print(x.shape)
#y = Y.reshape(-1,1)

# scatter plot of X and Y
for i in range(0, len(x)):
    plt.scatter(x[i], y)

predicted_beta, cost = Linear_Regression(X, Y, 500, 0.0001, 0.001)
#print("original beta ",beta)
#print("predicted beta",predicted_beta)
#print("cost",cost)

Y_predicted = X @ predicted_beta
y_predicted = np.array([Y_predicted])

# scatter plot for X and predicted Y
#for i in range(0, len(x)):
 #   plt.scatter(x[i], y_predicted)
plt.show()
#print(Y)

print(Y[12], Y_predicted[12])
#print(Y[220], Y_predicted[220])
print(Y[37], Y_predicted[37])
print(Y[112], Y_predicted[112])



