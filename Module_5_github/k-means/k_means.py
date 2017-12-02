import matplotlib.pyplot as plt
import numpy as np


def initialize_centroids(points, k):
    """
        Selects k random points as initial
        points from dataset
    """
    centr = points.copy()
    np.random.shuffle(centr)
    return centr[:k]


def closest_centroid(points, centroids):
    """
        Returns an array containing the index to the nearest centroid for each point
    """
    distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))  # newaxis gives new dimention
    return np.argmin(distances, axis=0)


def move_centroids(points, closest, centroids):
    """
        Returns the new centroids assigned from the points closest to them
    """
    return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])


def show_centroids(points, centr, plot_title):
    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(centr[:, 0], centr[:, 1], c='r', s=100)
    ax = plt.gca()
    plt.title(plot_title)
    plt.show()


def main(points):
    num_iterations = 100
    k = 3

    # Initialize centroids
    centr = initialize_centroids(points, k)

    show_centroids(points, centr, 'Initial Centroids')

    # Run iterative process
    for i in range(num_iterations):
        closest = closest_centroid(points, centr)
        centr = move_centroids(points, closest, centr)

    show_centroids(points, centr, 'Final Centroids')


if __name__ == '__main__':
    # centroids = initialize_centroids(points, 3)

    # Generate Data
    points = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
                        (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
                        (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))

    main(points)
