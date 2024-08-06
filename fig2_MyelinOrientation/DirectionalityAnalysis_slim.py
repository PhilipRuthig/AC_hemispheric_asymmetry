"""
date: 2024-03-07
author: Gesine Mueller
aim:    input:      3D tif stacks the MBP channel,
        output:     TXT of the corrected dominant directions along the cortex surface for every patch/window
"""

import numpy as np
from skimage.io import imread, imsave
from skimage import filters, feature
from skimage.util import view_as_windows
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
import time

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

def curve_func(y, a, b, c):
    return a * y**2 + b * y + c

def fit_curve_AF(img, threshold=0.25, step_size=3):
    """
    Fit a curve to the leftmost points of the image
    :param img: normalized image (0,1)
    :param threshold: must be exceeded to be considered as the left-most point
    :param step_size: take only every nth row of the image into consideration
    :return: y-coordinates of the fitted curve
    """
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

def correct_dominant_directions(dominant_directions, sobel, window_size=24):
    """
    Correct the dominant directions by comparing the mean sobel value of the window with 90 degrees
    Assumption is that neurons are oriented perpendicular to the cortex surface as the default state - we have to correct for that assumption
    :param dominant_directions: dominant directions from direactionality analysis (180 degrees: north, 90 degrees: east, 0 degrees: south)(array of z, y, x, direction)
    :param sobel: gradient filter over distance transformof the cortex surface, estimated on the MBP channel
    :param window_size: sliding window
    :return: corrected dominant directions in degrees (array of z, y, x, direction)
    """
    for i in range(dominant_directions.shape[0]):
        z, y, x, direction = dominant_directions[i]
        z, y, x = int(z), int(y), int(x)
        window = sobel[z, max(0, y - window_size // 2):min(sobel.shape[1], y + window_size // 2),
                       max(0, x - window_size // 2):min(sobel.shape[2], x + window_size // 2)]
        masked_window = np.ma.masked_equal(window, 0)
        mean_sobel = np.mean(masked_window)
        if mean_sobel < 90:
            dominant_directions[i, 3] += abs(90 - mean_sobel)
        elif mean_sobel > 90:
            dominant_directions[i, 3] -= abs(90 - mean_sobel)
    return dominant_directions


if __name__ == "__main__":
    start_time = time.time()
    window_size = 24
    path = "..."
    mbp = imread(path + 'MBP.tif')
    af = imread(path + 'AF.tif')

    dominant_directions_final = np.empty((0, 4))
    for z in range(mbp.shape[0]):
        mbp_median = median_filter(mbp[z], size=5)
        filtered_image = filters.sato(mbp_median, sigmas=range(3, 4), black_ridges=False)
        filtered_image_normalized = normalize_image(filtered_image)
        dominant_directions2D = rolling_window_orientation_analysis2D(filtered_image_normalized, window_size, 5, 6, coherence_threshold=0.05)
        not_nan_indices = ~np.isnan(dominant_directions2D[:, 2])
        dominant_directions2D[:, 2][not_nan_indices] = [abs(value) if value < 0 else 180 - value for value in dominant_directions2D[:, 2][not_nan_indices]]
        dominant_directions2D = dominant_directions2D[not_nan_indices]
        z_index = np.full(len(dominant_directions2D), z)
        dominant_directions_z = np.column_stack((z_index, dominant_directions2D)) # (z,y,x,direction)
        dominant_directions_final = np.concatenate((dominant_directions_final, dominant_directions_z), axis=0)

    Sobel_smooth = []
    x = np.arange(0, af.shape[1], 1)
    for z in range(af.shape[0]):
        image_normalized = normalize_image(median_filter(af[z], size=5))
        y = fit_curve_AF(image_normalized)
        y_processed = UnivariateSpline(x, y, s=10000)(x)
        binary_smooth = np.zeros_like(image_normalized)
        for i in range(image_normalized.shape[0]):
            j = int(y_processed[i])
            if j < image_normalized.shape[1]:
                binary_smooth[i, :j] = 1
        binary_approximated = 1 - binary_smooth

        distance = ndimage.distance_transform_edt(binary_approximated, return_distances=True)
        sobel = np.arctan2(ndimage.sobel(distance, axis=1, mode='nearest'),
                           ndimage.sobel(distance, axis=0, mode='nearest')) * 180 / np.pi
        sobel_smooth = ndimage.gaussian_filter(sobel, sigma=2)
        sobel_smooth = np.where(sobel_smooth < 45, 0, sobel_smooth)
        Sobel_smooth.append(sobel_smooth)
    sobel_smooth = np.stack(Sobel_smooth, axis=0)

    dominant_directions_corrected = correct_dominant_directions(dominant_directions_final, sobel_smooth)
    dominant_directions_corrected[:, 3] = np.where(dominant_directions_corrected[:, 3] > 180,
                                                   dominant_directions_corrected[:, 3] - 180,
                                                   dominant_directions_corrected[:, 3])
    dominant_directions_corrected[:, 3] = np.where(dominant_directions_corrected[:, 3] < 0,
                                                   180 + dominant_directions_corrected[:, 3],
                                                   dominant_directions_corrected[:, 3])

    np.savetxt(path + 'dominant_directions_corrected-slim.txt', dominant_directions_corrected)
    imsave(path + 'sobel_smooth-slim.tif', np.stack(sobel_smooth, axis=0).astype(np.uint8))


    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
