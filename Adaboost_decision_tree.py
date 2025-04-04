import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y

def experiment(scale,depth):
    X, labels = make_moons_3d(n_samples=1000, noise=0.2)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

    # Initialize a decision tree as the base estimator for AdaBoost
    base_estimator = DecisionTreeClassifier(max_depth=depth)

    # Set up the AdaBoost classifier with the decision tree
    adaboost = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=scale,
        algorithm='SAMME',
        random_state=42
    )

    # Train the AdaBoost classifier
    adaboost.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = adaboost.predict(X_test)

    # Evaluate and print the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

scale=[10,50,100]
depth=[2,3,5,10]
for i in scale:
    for j in depth:
        print(f"Scale: {i}, Depth: {j}")
        experiment(i,j)
        print("\n")