
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_2d_histogram(image_path, numbins, range_a, range_b):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image could not be read.")
        return

    # Convert from BGR to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Calculate the 2D histogram for the a and b channels
    # Ensure that ranges are specified as a list of tuples/lists
    hist = cv2.calcHist([lab_image], [1, 2], None, [numbins, numbins], [120, 165, 80, 155])#[120, 165, 80, 155]

    # Normalize the histogram
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)




    return hist

def plot_2d_histogram(hist, numbins, range_a, range_b):
    print(hist)
    plt.imshow(hist, interpolation='nearest', origin='lower',
               extent=[range_a[0], range_a[1], range_b[0], range_b[1]])
    plt.colorbar()
    plt.title('2D Color Histogram for a and b channels')
    plt.xlabel('a channel')
    plt.ylabel('b channel')

def compute_statistics(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image could not be read.")
        return

    # Convert from BGR to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Compute mean and variance for each channel
    l_mean, l_var = np.mean(l_channel), np.var(l_channel)
    a_mean, a_var = np.mean(a_channel), np.var(a_channel)
    b_mean, b_var = np.mean(b_channel), np.var(b_channel)

    print(f"L channel: Mean = {l_mean:.2f}, Variance = {l_var:.2f}")
    print(f"A channel: Mean = {a_mean:.2f}, Variance = {a_var:.2f}")
    print(f"B channel: Mean = {b_mean:.2f}, Variance = {b_var:.2f}")

    return a_mean,a_var, b_mean, b_var

# Parameters
image_path = 'purple/purple10.jpg'  # Ensure this path is correctly specified
# image_path = 'green/green1.jpg'  # Ensure this path is correctly specified
numbins = 20  # You can adjust the number of bins
range_a = (0, 127)  # Typical range for a channel in LAB
range_b = (0, 127)  # Typical range for b channel in LAB

# Compute and plot the histogram
hist_ab = compute_2d_histogram(image_path, numbins, range_a, range_b)
plot_2d_histogram(hist_ab, numbins, range_a, range_b)
a_mean,a_var, b_mean, b_var = compute_statistics(image_path)


# plt.scatter([a_mean//2], [b_mean//2], label=' Mean', color='red', zorder=5)
# circle = plt.Circle((means[i], 0), np.sqrt(variances[i]), color='blue', fill=False, linewidth=2, alpha=0.5)

plt.show()
