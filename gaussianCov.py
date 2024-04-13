import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

import matplotlib.transforms as transforms


def compute_gaussian2d(a_flat, b_flat):
    # Compute covariance matrix
    a_mean = np.mean(a_flat)
    b_mean = np.mean(b_flat)
    covariance_matrix = np.cov(a_flat, b_flat)

    a_std = np.sqrt(np.var(a_flat))
    b_std = np.sqrt(np.var(b_flat))

    return a_mean, b_mean, covariance_matrix, a_std, b_std, a_flat,b_flat






def confidence_ellipse(x, y, ax, n_std=2.0, plot_axes=False, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    lambda_, v = np.linalg.eig(cov)

    lambda_ = np.sqrt(lambda_)
    ellipse = Ellipse(xy=(np.mean(x), np.mean(y)),
                      width=lambda_[0] * n_std * 2, height=lambda_[1] * n_std * 2,
                      angle=np.degrees(np.arctan2(*v[:, 0][::-1])),
                      edgecolor='red', facecolor='none', **kwargs)

    if plot_axes:
        # Add major and minor axis lines
        major = v[:, 0] * lambda_[0] * n_std
        minor = v[:, 1] * lambda_[1] * n_std
        center = np.array([np.mean(x), np.mean(y)])
        ax.plot([center[0], center[0] + major[0]], [center[1], center[1] + major[1]], 'k-')
        ax.plot([center[0], center[0] + minor[0]], [center[1], center[1] + minor[1]], 'k-')

        print(center, major)

        if lambda_[0] < lambda_[1]:
            v[:, 0], v[:, 1] = v[:, 1], v[:, 0]
            lambda_[0], lambda_[1] = lambda_[1], lambda_[0]


        major2 = v[:, 0] * n_std * (0.9*lambda_[0]- lambda_[1])
        p1 = center + major2
        p2 = center - major2
        max_dist = n_std * lambda_[1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--')

        print("Points for detector: ", p1, p2, "Distance=", max_dist)
        # Add the circle to the axes
        circle = Circle(p1,  max_dist, color='blue', fill=False)  # Center at (0.5, 0.5), radius 0.1
        ax.add_patch(circle)
        circle = Circle(p2, max_dist, color='blue', fill=False)  # Center at (0.5, 0.5), radius 0.1
        ax.add_patch(circle)

    return ax.add_patch(ellipse)


def combine_gaussians(means, covariances, weights):
    result = 0
    for i in range(len(means)):
        mu = means[i]  # mean vector
        Sigma = covariances[i]  # covariance matrix
        weight = weights[i]  # weight of the Gaussian

        # Compute the exponent term
        exponent = -0.5 * np.dot(np.dot(mu.T, np.linalg.inv(Sigma)), mu)

        # Compute the Gaussian PDF
        pdf = (1 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))) * np.exp(exponent)

        # Accumulate the PDF with the corresponding weight
        result += weight * pdf

    return result


def plot_stats(stats, ax, plot_data=False, plot_cov=False, plot_var=False, plot_summary=False):


    for a_mean, b_mean, covariance_matrix, a_std, b_std, a_flat, b_flat in stats:
        if plot_data:
            ax.plot(a_flat, b_flat, '.')

        ax.plot(a_mean, b_mean, 'o')

        if plot_cov:
            confidence_ellipse(a_flat, b_flat, ax, n_std=2)

        if plot_var:
            # Non rotated ellipse
            ellipse = Ellipse((a_mean, b_mean), width=2*a_std * 2, height=2*b_std * 2,
                              edgecolor='blue', facecolor='none')
            ax.add_patch(ellipse)


    # covariance of all means
    if plot_summary:
        a_means = [a_mean for a_mean, b_mean, covariance_matrix, a_std, b_std, a_flat, b_flat in stats]
        b_means = [b_mean for a_mean, b_mean, covariance_matrix, a_std, b_std, a_flat, b_flat in stats]


        confidence_ellipse(np.array(a_means), np.array(b_means), ax, n_std=2, plot_axes=True)


        # Point
        x = np.array([.5, -4])
        mu = np.array([np.average(a_means), np.average(b_means)])
        Sigma = np.cov(a_means, b_means)

        # Mahalanobis Distance
        distance, mu = mahalanobis_distance(mu, Sigma, x)
        print("Mahalanobis Distance:", distance, " point=", x, "from mu=", mu)
    # means =


def mahalanobis_distance(mu, Sigma, x):
    Sigma_inv = np.linalg.inv(Sigma)
    distance = np.sqrt((x - mu).T @ Sigma_inv @ (x - mu))
    return distance, mu


import os

def extract_all(folder_path, extend_data=False):
    # all hist
    all_hist = []
    # List all files in the specified directory
    try:
        # os.listdir() returns a list of all files and directories in 'directory'
        files = os.listdir(folder_path)
        # print("Files in directory:", files)

        for file in files[:]:
            # Read the image
            image = cv2.imread(folder_path + "/" + file)

            if image is None:
                print("Error: Image could not be read.")
                return

            # Convert from BGR to LAB color space
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

            # Split the LAB image into its channels
            l_channel, a_channel, b_channel = cv2.split(lab_image)

            # Flatten the channels to 1D arrays for covariance calculation
            a_flat = np.array(a_channel.flatten(), dtype=np.float32)
            b_flat = np.array(b_channel.flatten(), dtype=np.float32)

            a_flat = 100 * (a_flat - 128) / 256
            b_flat = 100 * (b_flat - 128) / 256

            #

            # Compute stats
            if extend_data:
                n = len(a_flat)
                n2 = n // 2
                stats = compute_gaussian2d(a_flat[:n2], b_flat[:n2])
                all_hist.append(stats)

                # stats = compute_gaussian2d(-a_flat[n2:], b_flat[n2:])
                # all_hist.append(stats)

                # stats = compute_gaussian2d(a_flat[-n2:], b_flat[:n2])
                # all_hist.append(stats)
                #
                stats = compute_gaussian2d(a_flat[:n2], b_flat[-n2:])
                all_hist.append(stats)
            else:
                stats = compute_gaussian2d(a_flat, b_flat)
                all_hist.append(stats)



    except FileNotFoundError:
        print("The directory does not exist.")


    return all_hist

# Specify the path to your folder
folder_path = "purple"
# folder_path = "images/lab-500pm-646569/"


if __name__ == "__main__":
    stats = extract_all(folder_path, extend_data=False)

    # Create figure and axis
    fig, ax = plt.subplots()
    plot_stats(stats, ax, plot_cov=False, plot_summary=True)

    ax.grid()
    plt.tight_layout()
    plt.show()
    # plot_summary(stats)


