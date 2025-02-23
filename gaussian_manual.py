import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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
                       facecolor='none', **kwargs)

    if plot_axes:
        # Add major and minor axis lines
        major = v[:, 0] * lambda_[0] * n_std
        minor = v[:, 1] * lambda_[1] * n_std
        center = np.array([np.mean(x), np.mean(y)])
        ax.plot([center[0], center[0] + major[0]], [center[1], center[1] + major[1]], 'k-')
        ax.plot([center[0], center[0] + minor[0]], [center[1], center[1] + minor[1]], 'k-')

        # print(center, major)

        if lambda_[0] < lambda_[1]:
            v[:, 0], v[:, 1] = v[:, 1], v[:, 0]
            lambda_[0], lambda_[1] = lambda_[1], lambda_[0]


        major2 = v[:, 0] * n_std * (0.8*lambda_[0]- lambda_[1])
        p1 = center + major2
        p2 = center - major2
        max_dist = n_std * lambda_[1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--')

    return ax.add_patch(ellipse)

def mahalanobis_distance(mu, Sigma, x):
    Sigma_inv = np.linalg.inv(Sigma)
    print("mean and Inverted Covariance: ", mu.tolist(),", ", Sigma_inv.tolist())
    distance = np.sqrt((x - mu).T @ Sigma_inv @ (x - mu))
    return distance, mu

def compute_gaussian2d(a_flat, b_flat):
    # Compute covariance matrix
    a_mean = np.mean(a_flat)
    b_mean = np.mean(b_flat)
    covariance_matrix = np.cov(a_flat, b_flat)

    a_std = np.sqrt(np.var(a_flat))
    b_std = np.sqrt(np.var(b_flat))

    return a_mean, b_mean, covariance_matrix, a_std, b_std, a_flat,b_flat

def plot_stats(points, ax, plot_summary=False,edgecolor='k'):
    for a_mean, b_mean in points:
        ax.plot(a_mean, b_mean, '.', color=edgecolor)

    # covariance of all means
    if plot_summary:
        a_means = [a_mean for a_mean, b_mean in points]
        b_means = [b_mean for a_mean, b_mean in points]

        confidence_ellipse(np.array(a_means), np.array(b_means), ax, n_std=2, plot_axes=True, edgecolor=edgecolor)

        # Point
        x = np.array([.5, -4])
        mu = np.array([np.average(a_means), np.average(b_means)])
        Sigma = np.cov(a_means, b_means)

        # Mahalanobis Distance
        distance, mu = mahalanobis_distance(mu, Sigma, x)
    
    return distance, mu



if __name__ == "__main__":

    green_mean_ab = [(-27, 2),(-31, 18), (-35, 21), (-24, 18), (-18,9),(-32,9),(-27, 13),(-23, 14), (-25,9), (-31, 10), (-23,11), (-30,8), (-29,27),(-19,22),(-30,16),(-32,34),(-21,26),(-18,27),(-27,17),(-29,30),(-23,20),(-24,29),(-24,23),(-23,30),(-16,28),(-22,16),(-23,29),(-33,18),(-26,4),(-28,21),(-41,27)]
    green_mean_ab = [(1, -14), (-5, -8), (17, -22),(2, -19), (-5, -12), (13, -16),(25, -43), (22, -39), (10, -15)]
    blue_mean_ab = [(10, -42), (9, -40), (8, -38), (7, -40), (8, -42), (6, -43), (5, -26), (8, -31), (6, -23), (10, -24), (12, -34), (0, -17), (3, -10), (8, -25), (5, -19), (-1, -2), (1, -5), (0, -4), ]

    points = blue_mean_ab
    # Create figure and axis
    fig, ax = plt.subplots()
    plot_stats(points, ax, plot_summary=True)

    ax.grid()
    plt.tight_layout()
    plt.show()



