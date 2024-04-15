from matplotlib import pyplot as plt

from gaussianCov import extract_all, plot_stats


# folder_path = "images/lab-500pm-646569/"
PLOT_DATA = False
PLOT_COVARIANCE = True
PLOT_VARIANCE = False
ALL_MEANS = True

if __name__ == "__main__":
    colors = ["purple", "green_w_filter"]
    # Create figure and axis
    fig, ax = plt.subplots()

    for c in colors:
        print(c)
        stats = extract_all(c)
        plot_stats(stats, ax, plot_cov=False, plot_summary=True)

    ax.grid()
    plt.tight_layout()
    plt.show()

    # plot_summary(stats)