from matplotlib import pyplot as plt

from gaussian_manual import plot_stats

if __name__ == "__main__":

    green_mean_ab = [(-31, 39), (-28, 42), (-19, 19), (-29, 31),(-30, 36), (-30, 42),(-18, 17), (-26, 30)]
    purple = [(1, -2), (5, -4), (-1, 1), (2, -2),(1, -2), (5, -3)]
    blue_mean_ab = [(10, -42), (9, -40), (8, -38), (7, -40), (8, -42), (6, -43), (5, -26), (8, -31), (6, -23), (10, -24), (12, -34), (0, -17), (3, -10), (8, -25), (5, -19), (-1, -2), (1, -5), (0, -4), ]

    colors = [("green", green_mean_ab, 'g'), ("blue", blue_mean_ab, 'b'), ("Purple", purple, 'c')]

    # Create figure and axis
    fig, ax = plt.subplots()

    for name, points, c in colors:
        print (name)
        plot_stats(points, ax, plot_summary=True, edgecolor=c)

    ax.grid()
    plt.tight_layout()
    plt.show()


