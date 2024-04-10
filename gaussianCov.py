import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import matplotlib.transforms as transforms


def compute_gaussian2d(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image could not be read.")
        return

    # Convert from BGR to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    a_channel = a_channel / 256
    b_channel = b_channel / 256

    # Flatten the channels to 1D arrays for covariance calculation
    a_flat = a_channel.flatten()
    b_flat = b_channel.flatten()



    # Compute covariance matrix
    a_mean = np.mean(a_flat)
    b_mean = np.mean(b_flat)
    covariance_matrix = np.cov(a_flat, b_flat)

    a_std = np.sqrt(np.var(a_flat))
    b_std = np.sqrt(np.var(b_flat))
    return (a_mean, b_mean, covariance_matrix, a_std, b_std, a_flat,b_flat)






def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='red', **kwargs):
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
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      edgecolor='red',facecolor='none', **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_stats(stats):
    # Create figure and axis
    fig, ax = plt.subplots()

    for a_mean, b_mean, covariance_matrix, a_std, b_std, a_flat, b_flat in stats:
        if PLOT_DATA:
            ax.plot(a_flat, b_flat, '.')

        ax.plot(a_mean, b_mean, 'o')

        if PLOT_COVARIANCE:
            confidence_ellipse(a_flat, b_flat, ax, n_std=2)

        if PLOT_VARIANCE:
            # Non rotated ellipse
            ellipse = Ellipse((a_mean, b_mean), width=2*a_std * 2, height=2*b_std * 2,
                              edgecolor='blue', facecolor='none')
            ax.add_patch(ellipse)


    ax.grid()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.show()




def plot_summary(stats):
    # Create figure and axis
    fig, ax = plt.subplots()


    a_stds = [a_std for a_mean, b_mean, covariance_matrix, a_std, b_std, a_flat, b_flat in stats]
    b_stds = [b_std for a_mean, b_mean, covariance_matrix, a_std, b_std, a_flat, b_flat in stats]
    ax.plot(a_stds)
    ax.plot(b_stds)

    # Average
    a_av = np.average(a_stds) * np.ones_like(a_stds)
    b_av = np.average(b_stds) * np.ones_like(b_stds)
    ax.plot(a_av, '--')
    ax.plot(b_av, '--')

    # ax.set_xlim(100,150)
    # ax.set_ylim(100,150)
    # ax.set_xlim(0,255)
    # ax.set_ylim(0,255)
    ax.grid()
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.show()


import os

def extract_all(folder_path):
    # all hist
    all_hist = []
    # List all files in the specified directory
    try:
        # os.listdir() returns a list of all files and directories in 'directory'
        files = os.listdir(folder_path)
        # print("Files in directory:", files)

        for file in files:
            stats = compute_gaussian2d(folder_path + "/" + file)

            all_hist.append(stats)
    except FileNotFoundError:
        print("The directory does not exist.")


    return all_hist

# Specify the path to your folder
folder_path = "green"
PLOT_DATA = False
PLOT_COVARIANCE = True
PLOT_VARIANCE = True
stats = extract_all(folder_path)
plot_stats(stats)
plot_summary(stats)


