"""
date: 2024-02-29
author: Gesine Mueller
aim:    input:      3D tif stacks of the autofluorescence, the MBP channel,
        output:     3D tif stacks of the Gradient transform on AF, TXT of the dominant direction for every patch,
                    TXT of the corrected dominant directions along the cortex surface
"""

import numpy as np
from skimage.io import imread, imsave
from skimage import filters, feature
from skimage.util import view_as_windows
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from scipy.optimize import curve_fit


def normalize_image(image):
    lower, upper = np.percentile(image, (1, 99))
    image_clipped = np.clip(image, lower, upper)
    return (image_clipped - image_clipped.min()) / (image_clipped.max() - image_clipped.min())

def calculate_dominant_direction(window, sigma, coherence_threshold=0.1):
    """
    # Orientation analysis: https://forum.image.sc/t/orientationj-or-similar-for-python/51767/2
    Calculate the dominant direction of a 2D window
    :param window: 2D array
    :param sigma: sigma used in the structure tensor calculation
    :param coherence_threshold: minimum coherence value for the dominant direction to be considered
    :return: dominant direction in degrees (+/- 90 degrees, 0 degrees is vertical, +/- 90 degrees is horizontal)
    """
    eps = 1e-20
    axx, axy, ayy = feature.structure_tensor(window.astype(np.float32), sigma=sigma, mode="reflect")
    coh = ((ayy - axx) / (ayy + axx + eps)) ** 2
    energy = np.sqrt(axx.mean() ** 2 + ayy.mean() ** 2)
    measure = coh.mean() * energy
    if measure < coherence_threshold:
        return np.nan
    else:
        return np.rad2deg(np.arctan2(2 * axy.mean(), (ayy.mean() - axx.mean())) / 2)

def rolling_window_orientation_analysis2D(image, window_size, sigma, overlap, coherence_threshold=0.2):
    """
    Apply the dominant direction calculation rolling window analysis to a 2D image
    :param image: 2D image array
    :param window_size: size of the window
    :param sigma: sigma used in the structure tensor calculation
    :param overlap: overlap between windows
    :param coherence_threshold: coherence_threshold: minimum coherence value for the dominant direction to be considered
    :return: array of dominant directions in 2D (y, x, direction)
    """
    step_size = window_size - overlap
    windows = view_as_windows(image, (window_size, window_size), step=step_size)
    dominant_directions = np.array([(i * step_size + window_size // 2, j * step_size + window_size // 2,
                      calculate_dominant_direction(windows[i, j], sigma, coherence_threshold=coherence_threshold))
                     for i in range(windows.shape[0]) for j in range(windows.shape[1])])
    return dominant_directions

def plot_dominant_directions(image, dominant_directions, z):
    """
    Plot the dominant directions on top of the image
    :param image: original image 2D
    :param dominant_directions: array of dominant directions in 2D (y, x, direction)
    :return: plot of the image with the dominant directions
    """
    plt.imshow(image, cmap='gray')
    for y, x, direction in dominant_directions:
        if not np.isnan(direction):
            scale_factor = 5  # Adjust this value to change the length of the arrow
            dx = scale_factor * np.sin(np.deg2rad(direction))
            dy = scale_factor * np.cos(np.deg2rad(direction))
            plt.gca().add_patch(FancyArrow(x, y, dx, dy, color='r', width=1))
    plt.savefig('dominant_directions' + str(z) + '.png', dpi=300)
    plt.show()


def curve_func(y, a, b, c):
    return a * y**2 + b * y + c


def fit_curve_AF(img, threshold=0.25, step_size=3):
    # threshold = filters.threshold_otsu(img)
    # Initialize an empty list to store the y-coordinates
    coordinates = []
    for i in range(0, img.shape[0], step_size):
        row = img[i]
        # Find the y-coordinate where the pixel value first exceeds the threshold
        x = np.argmax(row > threshold)
        coordinates.append((i, x))
    coordinates = np.array(coordinates)

    # Fit the curve to every nth leftmost point
    popt, pcov = curve_fit(curve_func, coordinates[:, 0], coordinates[:, 1])
    y_approx = []
    for i in range(img.shape[0]):
        j = int(curve_func(i, *popt))
        y_approx.append(j)

    return y_approx

def correct_dominant_directions(dominant_directions, sobel, window_size=(24, 24)):
    # Iterate over the z-coordinates
    dominant_directions_corrected = []
    for z in range(sobel.shape[0]):
        dd_z = dominant_directions[dominant_directions[:, 0] == z]
        for i in range(dd_z.shape[0]):
            _, y, x, direction = dd_z[i]
            y, x = int(y), int(x)
            # Get the corresponding Sobel window
            window = sobel[int(z), max(0, y - window_size[0] // 2):min(sobel.shape[1], y + window_size[0] // 2),
                               max(0, x - window_size[1] // 2):min(sobel.shape[2], x + window_size[1] // 2)]
            # Calculate the mean value of the Sobel window
            masked_window = np.ma.masked_equal(window, 0)
            mean_sobel = np.mean(masked_window)
            # Correct the dominant direction based on the mean value
            if mean_sobel < 90:
                dd_z[i, 3] += abs(90 - mean_sobel)
            elif mean_sobel > 90:
                dd_z[i, 3] -= abs(90 - mean_sobel)
        # Append the corrected dominant direction to the list
        dominant_directions_corrected.append(dd_z)

    # Concatenate the corrected dominant directions into a single array
    return np.concatenate(dominant_directions_corrected, axis=0)


if __name__ == "__main__":
    # Calculate the dominant directions on the MBP channel
    path = "..."
    mbp = imread(path + 'MBP.tif')
    dominant_directions3D = []
    Sato = []
    for z in range(mbp.shape[0]):
        mbp_median = median_filter(mbp[z], size=5)
        # Apply the Sato filter
        filtered_image = filters.sato(mbp_median, sigmas=range(3, 4), black_ridges=False)
        # Normalize the filtered image
        filtered_image_normalized = normalize_image(filtered_image)
        Sato.append(filtered_image_normalized)
        # Calculate the dominant directions
        dominant_directions2D = rolling_window_orientation_analysis2D(filtered_image_normalized, 24, 5, 6, coherence_threshold=0.05)

        # Recalculate orientations to be between 0 and 180 degrees (180 degrees is north, 0 deg is south, 90 degrees is east)
        not_nan_indices = ~np.isnan(dominant_directions2D[:, 2])
        dominant_directions2D[:, 2][not_nan_indices] = [abs(value) if value < 0 else 180 - value for value in dominant_directions2D[:, 2][not_nan_indices]]
        dominant_directions2D = dominant_directions2D[not_nan_indices]

        # Create an array with the z-index
        z_index = np.full(len(dominant_directions2D), z)
        # Create a 3D array with the z-index and the dominant directions
        dominant_directions_z = np.column_stack((z_index, dominant_directions2D)) # (z,y,x,direction)
        dominant_directions3D.append(dominant_directions_z)
        # Plot the dominant directions
        # plot_dominant_directions(filtered_image_normalized, dominant_directions2D, z) #takes ages
    # Concatenate all arrays in the list into one final array
    dominant_directions_final = np.concatenate(dominant_directions3D, axis=0)
    # Save the dominant directions as txt
    np.savetxt(path + 'dominant_directions.txt', dominant_directions_final)
    imsave(path + 'sato.tif', (np.stack(Sato, axis=0)* 255).astype(np.uint8))



    # Calculate the gradient filter over distance transform on AF channel
    af = imread(path + 'AF.tif')
    Binary_approximated = []
    Sobel_smooth = []
    for z in range(af.shape[0]):
        af_median = median_filter(af[z], size=5)
        image_normalized = normalize_image(af_median)

        # Perform the curve fitting
        # split_images = np.array_split(image_normalized, 5, axis=0)
        # Extract the y-coordinates of the surface from each part
        y = fit_curve_AF(image_normalized)
        #Y = [fit_curve_MBP(img) for img in split_images]
        # Concatenate the y-coordinates into a single array
        #y = np.concatenate(Y, axis=0)
        # Create an array of x-coordinates
        x = np.arange(0, image_normalized.shape[0], 1)
        # Create a spline that fits the x and y coordinates
        spl = UnivariateSpline(x, y, s=10000)
        # Generate a new set of y-coordinates using the spline
        y_processed = spl(x)
        # Create a new binary image
        binary_smooth = np.zeros_like(image_normalized)
        for i in range(image_normalized.shape[0]):
            j = int(y_processed[i])
            if j < image_normalized.shape[1]:
                binary_smooth[i, :j] = 1
        binary_approximated = 1 - binary_smooth
        Binary_approximated.append(binary_approximated)

        # Calculate the distance transform of the image
        distance = ndimage.distance_transform_edt(binary_approximated, return_distances=True)
        # Apply the Sobel filter to the distance transform
        sx = ndimage.sobel(distance, axis=0, mode='nearest')
        sy = ndimage.sobel(distance, axis=1, mode='nearest')
        # Apply the np.arctan2 function to the result of the Sobel filter
        sobel = np.arctan2(sy, sx) * 180 / np.pi
        # Apply the Gaussian filter to the result of the np.arctan2 function
        sobel_smooth = ndimage.gaussian_filter(sobel, sigma=2)
        # set all values below 45 to zero since that is the min value before Gaussian filter
        sobel_smooth = np.where(sobel_smooth < 45, 0, sobel_smooth)
        Sobel_smooth.append(sobel_smooth)



    # Save the results as 3D tif stacks
    imsave(path + 'binary_approximated.tif', (np.stack(Binary_approximated, axis=0) * 255).astype(np.uint8))
    imsave(path + 'sobel_smooth.tif', np.stack(Sobel_smooth, axis=0).astype(np.uint8))


    # Correct the dominant directions
    dominant_directions_corrected = correct_dominant_directions(dominant_directions_final,
                                                                np.stack(Sobel_smooth, axis=0))
    # Correct for 0-180 since we don't have directed orientations
    dominant_directions_corrected[:, 3] = np.where(dominant_directions_corrected[:, 3] > 180,
                                                   dominant_directions_corrected[:, 3] - 180,
                                                   dominant_directions_corrected[:, 3])
    dominant_directions_corrected[:, 3] = np.where(dominant_directions_corrected[:, 3] < 0,
                                                   180 + dominant_directions_corrected[:, 3],
                                                   dominant_directions_corrected[:, 3])
    # Save the corrected dominant directions as txt
    np.savetxt(path + 'dominant_directions_corrected.txt', dominant_directions_corrected)