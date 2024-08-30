#Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same data set for
#clustering using k-Means algorithm. Compare the results of these two algorithms and
#comment on the quality of clustering


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as sm
import pandas as pd
import numpy as np

# Load the Iris dataset from sklearn
iris = datasets.load_iris()

# Create a DataFrame 'X' with the features (Sepal_Length, Sepal_Width, Petal_Length, Petal_Width)
X = pd.DataFrame(iris.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])

# Create a DataFrame 'Y' with the target labels (which species each sample belongs to)
Y = pd.DataFrame(iris.target, columns=['Targets'])

# Display the feature and target data
print("Feature Data (X):")
print(X)
print("\nTarget Labels (Y):")
print(Y)

# Define a colormap for visualization purposes
colormap = np.array(['red', 'lime', 'black'])

# Plot the actual clusters based on the true target labels
plt.figure(figsize=(10, 4))  # Adjust figure size for better readability

plt.subplot(1, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[Y.Targets], s=40)
plt.title('Real Clustering')

# Apply K-Means clustering with 3 clusters
kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(X)

# Plot the clusters formed by K-Means
plt.subplot(1, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[kmeans_model.labels_], s=40)
plt.title('K-Means Clustering')

# Display the K-Means clustering results
plt.show()

# Apply Gaussian Mixture Model (GMM) with 3 components (clusters)
gmm_model = GaussianMixture(n_components=3)
gmm_model.fit(X)

# Plot the clusters formed by GMM
plt.figure(figsize=(5, 4))  # Adjust figure size for better readability
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[gmm_model.predict(X)], s=40)
plt.title('EM Clustering')

# Display the GMM clustering results
plt.show()

# Print the actual target labels for comparison
print("Actual Target Labels:\n", iris.target)

# Print the labels predicted by K-Means
print("K-Means Predicted Labels:\n", kmeans_model.labels_)

# Print the labels predicted by GMM (EM Algorithm)
print("EM Predicted Labels:\n", gmm_model.predict(X))

# Calculate and print the accuracy of K-Means clustering
kmeans_accuracy = sm.accuracy_score(Y, kmeans_model.labels_)
print(f"Accuracy of K-Means: {kmeans_accuracy:.2f}")

# Calculate and print the accuracy of EM clustering
gmm_accuracy = sm.accuracy_score(Y, gmm_model.predict(X))
print(f"Accuracy of EM: {gmm_accuracy:.2f}")
