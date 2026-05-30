import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_sin = pd.read_csv("data/sin_data_train.csv", index_col=0)

# Plot the example dataset to examine the relationship between x and y.
plt.figure(figsize=(14, 6))
sns.scatterplot(df_sin, x="x", y="y")
plt.show()

def sin_model(X, theta):
    """
    Outputs predictions (Y-hat) given input X, and Model parameters theta.

    Keyword arguments:
    x -- The vector of values x.
    theta -- A vector of length 2, where theta[0] = theta_1 and theta[1] = theta_2.
    """
    theta_1 = theta[0]
    theta_2 = theta[1]
    return theta_1 * X + np.sin(theta_2 * X)


def sin_MSE(theta, X, Y):
    """
    Compute the numerical value of the l2 loss of our sinusoidal model given theta.

    Keyword arguments:
    theta -- The vector of values theta.
    X     -- The vector of x values - note that sin_model only needs a vector of X values and handles the transformations.
    Y     -- The vector of y values.
    """
    Y_hat = sin_model(X, theta)
    return np.mean((Y - Y_hat) ** 2)

def sin_MSE_dt0(theta, X, Y):
    """
    Compute the numerical value of the partial derivative of l2 loss with respect to theta_0.

    Keyword arguments:
    theta -- The vector of values theta.
    X     -- The vector of x values.
    Y     -- The vector of y values.
    """
    Y_hat = sin_model(X, theta)
    residual = Y - Y_hat
    return -2 * np.mean(residual * X)

def sin_MSE_dt1(theta, X, Y):
    """
    Compute the numerical value of the partial of l2 loss with respect to theta_1.

    Keyword arguments:
    theta -- The vector of values theta.
    X     -- The vector of x values.
    Y     -- The vector of y values.
    """
    Y_hat = sin_model(X, theta)
    residual = Y - Y_hat
    return -2 * np.mean(residual * np.cos(theta[1] * X) * X)

# This function calls dt1 and dt2 and returns the gradient dt.
# It is already implemented for you.
def sin_MSE_gradient(theta, X, Y):
    """
    Returns the gradient of l2 loss with respect to vector theta.

    Keyword arguments:
    theta -- The vector of values theta.
    X     -- The vector of x values.
    Y     -- The vector of y values.
    """
    return np.array([sin_MSE_dt0(theta, X, Y), sin_MSE_dt1(theta, X, Y)])

theta = [1, 1]
sin_MSE_gradient(theta, df_sin["x"], df_sin["y"])

def init_theta():
    """Creates an initial theta [0, 0] of shape (2,) as a starting point for gradient descent."""
    return np.array([0, 0])

def grad_desc(loss_f, gradient_loss_f, theta, data, num_iter=20, alpha=0.1):
    """
    Run gradient descent update for a finite number of iterations and static learning rate.

    Keyword arguments:
    loss_f -- The loss function to be minimized (used for computing loss_history).
    gradient_loss_f -- The gradient of the loss function to be minimized.
    theta -- The vector of values theta to use at the first iteration.
    data -- The data used in the model.
    num_iter -- The max number of iterations.
    alpha -- The learning rate (also called the step size).

    Return:
    theta -- The optimal value of theta after num_iter of gradient descent.
    theta_history -- The list of theta values over each iteration of gradient descent.
    loss_history -- The list of loss values over each iteration of gradient descent.
    """
    theta_history = []
    loss_history = []

    X = data["x"].values
    Y = data["y"].values

    for _ in range(num_iter):
        theta_history.append(theta.copy())
        loss = loss_f(theta, X, Y)
        loss_history.append(loss)
        grad = gradient_loss_f(theta, X, Y)
        theta = theta - alpha * grad

    return theta, theta_history, loss_history

theta_start = init_theta()
theta_hat, thetas_used, losses_calculated = grad_desc(
    sin_MSE, sin_MSE_gradient, theta_start, df_sin, num_iter=20, alpha=0.1
)

""" for b, l in zip(thetas_used, losses_calculated):
    print(f"theta: {b}, Loss: {l}") """

theta_init = init_theta()
theta_est, thetas, loss = grad_desc(sin_MSE, sin_MSE_gradient, theta_init, df_sin)

X, Y = df_sin['x'], df_sin['y']
Y_pred = sin_model(X, theta_est)

plt.plot(X, Y_pred, label='Model ($\hat{y}$)')
plt.scatter(X, Y, alpha=0.5, label='Observation ($y$)', color='gold')
plt.legend()
plt.show()

""" 
# Bonus Plot: Step-by-Step Progression of the Estimation
X, Y = df_sin['x'], df_sin['y']
theta_est = init_theta()

for step in range(5):
    theta_est, thetas, loss = grad_desc(sin_MSE, sin_MSE_gradient, theta_est, df_sin, num_iter=1)
    Y_pred = sin_model(X, theta_est)

    plt.plot(X, Y_pred, label=f'Model estimation after {step+1} steps')
    plt.scatter(X, Y, alpha=0.5, label='Observation ($y$)', color='gold')
    plt.legend()
    plt.show()
 """

