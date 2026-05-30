import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_poly = pd.read_csv("data/poly_data_train.csv")

# Plot the example dataset to examine the relationship between x and y.
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df_poly, x='x', y='y')
plt.title("Polynomial Data")
plt.show()

def poly_model(X, theta):
    return sum(theta[i] * X**i for i in range(len(theta)))

def poly_MSE(theta, X, Y):
    Y_hat = poly_model(X, theta)
    return np.mean((Y - Y_hat) ** 2)

def poly_MSE_gradient(theta, X, Y):
    Y_hat = poly_model(X, theta)
    residual = Y - Y_hat
    grad = [-2 * np.mean(residual * (X ** i)) for i in range(len(theta))]
    return np.array(grad)

def init_poly_theta(degree):
    return np.zeros(degree + 1)

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

# Train polynomial model
degree = 4
theta_init = init_poly_theta(degree)
theta_est, theta_hist, loss_hist = grad_desc(poly_MSE, poly_MSE_gradient, theta_init, df_poly, alpha=0.01, num_iter=50)

# Visualize prediction
X = df_poly['x']
Y = df_poly['y']
Y_pred = poly_model(X, theta_est)

plt.plot(X, Y_pred, label='Model ($\\hat{y}$)', color='blue')
plt.scatter(X, Y, alpha=0.5, label='Observation ($y$)', color='orange')
plt.title("Final Polynomial Model Fit via Gradient Descent")
plt.legend()
plt.show()
