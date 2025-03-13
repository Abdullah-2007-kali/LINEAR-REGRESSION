# LINEAR-REGRESSION
the linear regression progeckt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv("f.csv")
print(data)

# Extract x and y values
x = data["x"].values
y = data["y"].values

plt.figure(figsize=(8, 6))
plt.title("Linear Regression")  # Fixed spelling
plt.scatter(x, y)
plt.xlabel("x:")
plt.ylabel("y")
plt.show()
 #Linear Regression Function
def linear_regression(x, y):
    learning_rate = 0.001
    iterations = 100
    theta_0 = 0
    theta_1 = 0
    n = x.shape[0]
    losses = []

    for i in range(iterations):
        h_x = theta_0 + theta_1 * x  # Hypothesis function
        mse = (1 / n) * np.sum((h_x - y) ** 2)  # Correct MSE formula
        losses.append(mse)

        # Compute gradients
        gradient_theta0 = (2 / n) * np.sum(h_x - y)
        gradient_theta1 = (2 / n) * np.sum(x * (h_x - y))

        # Update parameters using gradient descent
        theta_0 -= learning_rate * gradient_theta0
        theta_1 -= learning_rate * gradient_theta1

    print("Theta0: ", theta_0)
    print("Theta1: ", theta_1)
    print("MSE: ", mse)

    # Predicting a new value
    new_x = 9
    prediction = theta_0 + theta_1 * new_x
    print("Prediction for x=9: ", prediction)

    # Plot the results
    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = theta_0 + theta_1 * x_line

    plt.figure(figsize=(8, 6))
    plt.title("Linear Regression")
    plt.scatter(x, y, label="Data points")
    plt.plot(x_line, y_line, c='r', label="Regression line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# Run regression
linear_regression(x, y)
