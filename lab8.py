#Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set.
#Print both correct and wrong predictions

# Import necessary libraries
from sklearn.model_selection import train_test_split  # For splitting the dataset into training and testing sets
from sklearn.neighbors import KNeighborsClassifier  # For the k-Nearest Neighbour algorithm
from sklearn import datasets  # For loading datasets like the Iris dataset

# Load the Iris dataset
iris = datasets.load_iris()
print("Iris dataset loaded...")

# Split the dataset into training and testing sets
# test_size=0.1 means 10% of the data will be used for testing, 90% for training
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1)

# Display the label names (0: setosa, 1: versicolor, 2: virginica)
for i in range(len(iris.target_names)):
    print("Label", i, "-", str(iris.target_names[i]))

# Create a K-Nearest Neighbour classifier with K=2 (number of neighbors)
classifier = KNeighborsClassifier(n_neighbors=2)

# Train the classifier using the training data
classifier.fit(x_train, y_train)

# Predict the labels for the test data
y_pred = classifier.predict(x_test)

# Display the results of the classification
print("Results of Classification using K-NN with K=2")
for r in range(len(x_test)):
    print("Sample:", str(x_test[r]), 
          "Actual label:", str(y_test[r]), 
          "Predicted label:", str(y_pred[r]))

# Calculate and print the classification accuracy
print("Classification Accuracy:", classifier.score(x_test, y_test))