import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_lin = pd.read_csv("data/linear_data_train.csv")

# Plot the example dataset to examine the relationship between x and y.
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_lin, x="x", y="y")
plt.title("Linear Data Example")
plt.show()

def linear_model(X, theta):
    """
    Outputs predictions (Y-hat) given input X, and Model parameters theta.

    Keyword arguments:
    x -- The vector of values x.
    theta -- A vector of length 2, where theta[0] = theta_1 and theta[1] = theta_2.
    """
    y = theta[0] + (theta[1] * X)
    return y


def linear_MSE(theta, X, Y):
    """
    Compute the numerical value of the l2 loss of our linear model given theta.

    Keyword arguments:
    theta -- The vector of values theta.
    X     -- The vector of x values.
    Y     -- The vector of y values.
    """
    y_pred = linear_model(X, theta)
    loss = np.mean((Y - y_pred)**2)
    return loss

def linear_MSE_dt0(theta, X, Y):
    """
    Compute the numerical value of the partial derivative of l2 loss with respect to theta_0.

    Keyword arguments:
    theta -- The vector of values theta.
    X     -- The vector of x values.
    Y     -- The vector of y values.
    """
    y_pred = linear_model(X, theta)
    dt0 = 2 * np.mean(y_pred - Y)
    return dt0
    ...

def linear_MSE_dt1(theta, X, Y):
    """
    Compute the numerical value of the partial derivative of l2 loss with respect to theta_1.

    Keyword arguments:
    theta -- The vector of values theta.
    X     -- The vector of x values.
    Y     -- The vector of y values.
    """
    y_pred = linear_model(X, theta)

    dt1 = 2 * np.mean((y_pred - Y) * X)
    return dt1
    ...

def linear_MSE_gradient(theta, X, Y):
    """
    Returns the gradient of l2 loss with respect to vector theta.

    Keyword arguments:
    theta -- The vector of values theta.
    X     -- The vector of x values.
    Y     -- The vector of y values.
    """
    return np.array([
        linear_MSE_dt0(theta, X, Y),
        linear_MSE_dt1(theta, X, Y)
    ])


def init_linear_theta():
    """Creates an initial theta [0, 0] of shape (2,) as a starting point for gradient descent."""
    return np.array([0.0, 0.0])

def grad_desc(theta, data, num_iter=20, alpha=0.1):
    """
    Run gradient descent update for a finite number of iterations and static learning rate.

    Keyword arguments:
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

    for _ in range(num_iter):
        loss = linear_MSE(theta, data['x'], data['y'])
        theta_history.append(theta)
        loss_history.append(loss)
        if np.isnan(loss):
            break
        grads = linear_MSE_gradient(theta, data['x'], data['y'])
        theta = theta - (alpha * grads)

    ...

    return theta, theta_history, loss_history

theta_init = init_linear_theta()
theta_est, theta_hist, loss_hist = grad_desc(theta_init, df_lin, num_iter=100, alpha=0.1)

X = df_lin["x"]
Y = df_lin["y"]
Y_pred = linear_model(X, theta_est)

plt.plot(X, Y_pred, label='Model ($\hat{y}$)', color='blue')
plt.scatter(X, Y, alpha=0.5, label='Observation ($y$)', color='orange')
plt.title("Final Linear Model Fit via Gradient Descent")
plt.legend()
plt.show()
