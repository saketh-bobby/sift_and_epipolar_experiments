import numpy as np
import cv2
import matplotlib.pyplot as plt

def euclidean_distance(p1, p2):
    # sum = 0
    # for i in range(p1.shape[0]):
    #     sum += (p1[i] - p2[i])**2 

    # return np.sqrt(sum)

    # faster alternative
    return np.linalg.norm(p1 - p2)

def update_closest_centroids(X, centroids):
    print("Calculating closest centroid")
    closest_centroids = np.empty((X.shape[0], 1))
    for i, row in enumerate(X):
        min_val = np.finfo(np.float).max
        min_idx = 0
        for idx, center in enumerate(centroids):
            d = euclidean_distance(row, center)
            if d < min_val:
                min_val = d
                min_idx = idx
        closest_centroids[i] = min_idx
    return closest_centroids


def update_centroids(X, closest_centroids, K):
    for i in range(K):
        group = np.array([X[idx] for idx, j in enumerate(closest_centroids) if j == i])
        centroids[i] = np.mean(group, axis=0)

    return centroids

def plot_centroids(centroids, K):
    colors = ["red", "green", "blue"]
    plt.figure()
    for i in range(K):
        plt.scatter(*centroids[i], c=colors[i], marker="o")
        plt.text(*centroids[i], s=(centroids[i][0], centroids[i][1]))


def plot_classifier(centroids, closest_centroids, K):
    colors = ["red", "green", "blue"]
    plt.figure()
    
    for i, color in zip(range(K), colors):
        group = np.array([X[idx] for idx, j in enumerate(closest_centroids) if j == i])

        x, y = group[:, 0], group[:, 1]

        plt.scatter(x, y, edgecolors=color, marker="^", facecolors="None")
        
        for a, b in zip(x, y):
            plt.text(a, b, s=(a, b))

def run_kmeans(image, centroids, num_iterations, K):
    for i in range(num_iterations):
        print("Iteration", i+1, "of", num_iterations)
        closest_centroids = update_closest_centroids(image, centroids)
        centroids = update_centroids(image, closest_centroids, K)
    return centroids, closest_centroids


if __name__ == "__main__":

    UBIT = "sakethva"
    np.random.seed(sum([ord(c) for c in UBIT]))

    print("Task 3:")

    X = np.array([[5.9, 3.2],
    [4.6, 2.9],
    [6.2, 2.8],
    [4.7, 3.2],
    [5.5, 4.2],
    [5.0, 3.0],
    [4.9, 3.1],
    [6.7, 3.1],
    [5.1, 3.8],
    [6.0, 3.0]])

    K = 3

    closest_centroids = np.empty((X.shape[0], 1))

    centroids = np.array([[6.2, 3.2], [6.6, 3.7], [6.5, 3.0]])

    closest_centroids = update_closest_centroids(X, centroids)
    print("Classification vector for the first iteration:", closest_centroids)

    plot_centroids(centroids, K)
    plt.savefig("task3_iter1_a.jpg")

    plot_classifier(centroids, closest_centroids, K)
    plt.savefig("task3_iter1_b.jpg")

    centroids = update_centroids(X, closest_centroids, K)
    print("Updated mean values are:", centroids)

    closest_centroids = update_closest_centroids(X, centroids)
    print("Classification vector for the second iteration:", closest_centroids)

    plot_centroids(centroids, K)
    plt.savefig("task3_iter2_a.jpg")

    plot_classifier(centroids, closest_centroids, K)
    plt.savefig("task3_iter2_b.jpg")

    centroids = update_centroids(X, closest_centroids, K)
    print("Updated mean values are:", centroids)

    print("3.4:")

    image = cv2.imread("./data/baboon.jpg")
    
    h, w, _ = image.shape

    K_values = [3, 5, 10, 20]

    NUM_ITERATIONS = 40
    # for each K run color quantization 
    for K in K_values:
        print("Applying color quantization with", K, "clusters")
        image = image.reshape((h * w, 3))

        # pick K random centroids from the image
        random_centroids = np.random.permutation(image)[:K]

        centroids = random_centroids

        centroids, closest_centroids = run_kmeans(image, centroids, NUM_ITERATIONS, K)

        color_q = np.empty((h*w, 3))
            

        closest_centroids = np.array(closest_centroids, dtype=np.int)
        for idx, val in enumerate(closest_centroids):
            color_q[idx] = random_centroids[val]

        color_q = color_q.reshape((h, w, 3))
        image = image.reshape((h, w, 3))

        
        color_q = np.array(color_q, dtype=np.uint8)
        
        cv2.imwrite("task3_baboon_%d.jpg" % K, np.hstack([image, color_q]))