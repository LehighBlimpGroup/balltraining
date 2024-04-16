from matplotlib import pyplot as plt

from gaussianCov import extract_all, plot_stats


# folder_path = "images/lab-500pm-646569/"
PLOT_DATA = False
PLOT_COVARIANCE = True
PLOT_VARIANCE = False
ALL_MEANS = True
EXTEND_DATA = True

if __name__ == "__main__":
    colors = [("purple", 'b'), ("green_w_filter", 'g')]

    # Create figure and axis
    fig, ax = plt.subplots()

    for (folder, c) in colors:
        print(folder)
        stats = extract_all(folder, extend_data=EXTEND_DATA)
        plot_stats(stats, ax, plot_cov=False, plot_summary=True, color=c)

    ax.grid()
    plt.tight_layout()
    plt.show()

    # plot_summary(stats)