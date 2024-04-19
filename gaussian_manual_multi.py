from matplotlib import pyplot as plt

from gaussian_manual import plot_stats

import re

def read_tuples_from_file(filename):
    tuples_list = []

    try:
        with open(filename, 'r') as file:
            for line in file:
                # Use regular expression to extract tuples
                tuples = re.findall(r'\(([^)]+)\)', line)
                for tpl in tuples:
                    # Split each tuple by comma and convert values to integers
                    a, b = map(int, tpl.split(','))
                    # Append the tuple to the list
                    if abs(a) > 5 or abs(b) > 5:
                        tuples_list.append((a, b))
    except FileNotFoundError:
        print("File not found. Please provide a valid filename.")

    return tuples_list

if __name__ == "__main__":

    green = read_tuples_from_file("green.txt")
    purple = read_tuples_from_file("purple.txt")
    blue = read_tuples_from_file("blue.txt")
    red = read_tuples_from_file("red.txt")

    #colors = [("green", green_mean_ab, 'g'), ("blue", blue_mean_ab, 'b'), ("Purple", purple, 'c')]
    colors = [("purple", purple, 'm'), ("green", green, 'g'), ("blue", blue, 'b'), ("red", red, 'r')]

    # Create figure and axis
    fig, ax = plt.subplots()

    for name, points, c in colors:
        print ("-------", name)
        plot_stats(points, ax, plot_summary=True, edgecolor=c)
    
    plt.xlabel("B")
    plt.ylabel("A")

    ax.grid()
    plt.tight_layout()
    plt.show()


