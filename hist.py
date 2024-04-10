import cv2
import numpy as np
import matplotlib.pyplot as plt




def compute_lab_histogram(image_path, numbins):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image could not be read.")
        return

    # Convert from BGR to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Define the number of bins and range for each channel
    channels = [l_channel, a_channel, b_channel]
      # L ranges from 0 to 255, A and B from -128 to 127

    histograms = []
    for i, channel in enumerate(channels):
        # Calculate histogram
        hist = cv2.calcHist([channel], [0], None, [numbins], histRange[i])
        hist /= hist.sum()
        #hist = cv2.calcHist([image], [0], None, [num_bins], [0, 256])


        #plt.xlim(histRange[i])

        histograms.append(hist)

    return histograms





# Provide the path to your image
# image_path = 'example912a.jpg'
# compute_lab_histogram(image_path, numbins=10)

import os

def extract_histograms(folder_path, histRange, numbins):
    # all hist
    all_hist = []
    # List all files in the specified directory
    try:
        # os.listdir() returns a list of all files and directories in 'directory'
        files = os.listdir(folder_path)
        print("Files in directory:", files)

        for file in files:
            histogram = compute_lab_histogram(folder_path + "/" + file, numbins=numbins)
            all_hist.append(histogram)
    except FileNotFoundError:
        print("The directory does not exist.")


    return all_hist



def plot_histograms(histograms, histRange, numbins):
    colors = ("black", "green", "blue")
    channel_names = ['L', 'A', 'B']


    # Plot histograms for each channel
    plt.figure(figsize=(10, 5))
    for lab_hist in histograms:
        for i, (hist,color, name) in enumerate(zip(lab_hist, colors, channel_names)):
            # Plot histogram
            plt.subplot(1, 3, i + 1)
            plt.plot([-128 + histRange[i][0] + (histRange[i][1] - histRange[i][0]) * j / numbins
                      for j in range(numbins)], hist, '-.', color=color)
            # plt.bar(range(numbins), hist, width=1.0, color='gray')
            plt.title( name+' channel Histogram')
            plt.xlabel('Intensity Value')
            plt.ylabel('Frequency')


    plt.tight_layout()
    plt.show()



histRange = [[0, 255], [100, 165], [80, 155]]
NUM_BINS = 15
# Specify the path to your folder
folder_path = "purple"
histograms = extract_histograms(folder_path, histRange, NUM_BINS)
plot_histograms(histograms, histRange, NUM_BINS)


