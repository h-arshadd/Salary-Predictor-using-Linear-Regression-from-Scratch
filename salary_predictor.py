import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
ds = pd.read_csv("Salary_Data.csv")
x = ds["YearsExperience"].values
y = ds["Salary"].values

# Initialize parameters
w = 0.0
b = 0.0
lr = 0.01
iterations = 1500

# Cost function (not used in training but can be used to monitor training progress)
# Calculates the mean squared error for the current weights
def cost_function(x, y, w, b):
    m = len(y)
    total_error = np.sum((w*x + b - y)**2)
    return total_error / (2*m)

# Gradient descent function
def gradientDescent_function(x, y, w, b, lr, iterations):
    m = len(y)
    for i in range(iterations):
        dw = np.sum((w*x + b - y) * x) / m
        db = np.sum((w*x + b - y)) / m
        w -= lr * dw
        b -= lr * db
    return w, b

# Train the model
w, b = gradientDescent_function(x, y, w, b, lr, iterations)

# Predict salaries
y_pred = w*x + b

# Plot actual vs predicted
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x, y_pred, color='red', label='Predicted')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()

# Function to predict salary for given years of experience
def predict_salary(years):
    return w * years + b

# Interactive prediction
while True:
    user_input = input("Enter years of experience (or 'q' to quit): ")
    if user_input.lower() == 'q':
        print("Exiting...")
        break
    try:
        years = float(user_input)
        salary = predict_salary(years)
        print(f"Predicted Salary: ${salary:.2f}")
    except ValueError:
        print("Please enter a valid number.")
