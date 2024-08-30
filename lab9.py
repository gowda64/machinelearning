#Implement the non-parametric Locally Weighted Regression algorithm in order to fit data
#points. Select appropriate data set for your experiment and draw graphs 

import numpy as np
import matplotlib.pyplot as plt

# Function to perform Locally Weighted Regression
def locally_weighted_regression(query_point, data_points, target_values, bandwidth):
    # Add a bias term (1) to the query point
    query_point_with_bias = [1, query_point]
    
    # Add a bias term (1) to each data point
    data_points_with_bias = [[1, point] for point in data_points]
    data_points_with_bias = np.asarray(data_points_with_bias)
    
    # Calculate the weights for each data point using a Gaussian kernel
    weights = np.exp(np.sum((data_points_with_bias - query_point_with_bias) ** 2, axis=1) / (-2 * bandwidth))
    
    # Apply the weights to the data points
    weighted_data_points = (data_points_with_bias.T) * weights
    
    # Calculate the model parameters (beta) using weighted least squares
    beta = np.linalg.pinv(weighted_data_points @ data_points_with_bias) @ weighted_data_points @ target_values @ query_point_with_bias
    
    # Return the prediction for the query point
    return beta

# Function to draw the regression curve
def plot_regression_curve(bandwidth):
    # Predict the target values for each point in the domain using the locally weighted regression
    predictions = [locally_weighted_regression(point, data_points, target_values, bandwidth) for point in domain]
    
    # Plot the original data points
    plt.plot(data_points, target_values, 'o', color='black')
    
    # Plot the regression curve
    plt.plot(domain, predictions, color='red')
    plt.show()

# Generate 1000 evenly spaced data points between -3 and 3
data_points = np.linspace(-3, 3, num=1000)


# Define the domain for plotting
domain = data_points

# Calculate the target values using a logarithmic function
target_values = np.log(np.abs(data_points ** 2 - 1) + 0.5)

# Plot the regression curve with a small bandwidth (tau)
plot_regression_curve(0.001)
