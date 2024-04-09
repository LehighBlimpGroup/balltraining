import cv2
import numpy as np
import matplotlib.pyplot as plt

NUMBINS = 20


def compute_lab_histogram(image_path, numbins=16):
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
    colors = ("black", "green", "blue")
    channel_names = ['L', 'A', 'B']
    histRange = [[0, 256], [100,155], [100, 155]]  # L ranges from 0 to 255, A and B from -128 to 127


    for i, (channel, color, name) in enumerate(zip(channels, colors, channel_names)):
        # Calculate histogram
        hist = cv2.calcHist([channel], [0], None, [numbins], histRange[i])
        hist /=  hist.sum()
        #hist = cv2.calcHist([image], [0], None, [num_bins], [0, 256])

        # Plot histogram
        plt.subplot(1, 3, i + 1)
        plt.plot([-128+histRange[i][0] + (histRange[i][1]-histRange[i][0] ) * j / numbins
                  for j in range(numbins)],hist, '-.',color=color)
        # plt.bar(range(numbins), hist, width=1.0, color='gray')
        plt.title(f'{name} channel Histogram')
        plt.xlabel('Intensity Value')
        plt.ylabel('Frequency')
        #plt.xlim(histRange[i])

    plt.tight_layout()



# Provide the path to your image
# image_path = 'example912a.jpg'
# compute_lab_histogram(image_path, numbins=10)

import os

def read_files_in_directory(folder_path):
    # Plot histograms for each channel
    plt.figure(figsize=(10, 5))

    # List all files in the specified directory
    try:
        # os.listdir() returns a list of all files and directories in 'directory'
        files = os.listdir(folder_path)
        print("Files in directory:", files)

        for file in files:
            compute_lab_histogram(folder_path + "/" + file, numbins=NUMBINS)
    except FileNotFoundError:
        print("The directory does not exist.")

    plt.show()


# Specify the path to your folder
folder_path = "purple"
read_files_in_directory(folder_path)


