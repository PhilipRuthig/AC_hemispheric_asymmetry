"""
Input: corrected dominant_directions with dimensions (z, y, x, direction)
Aim: add layer identifer, create basic statistics, plt stats, create hierachical models as an explanatory for L,R, tonotopy and layer, maybe later sex
"""

import numpy as np
from scipy.stats import circmean, circvar, circstd
import matplotlib.pyplot as plt
from skimage.io import imread
import pandas as pd
import seaborn as sns
import re
import os

def prepare_dominantDirections_for_stats(path, side, sampleID):
    data = np.loadtxt(path + sampleID + side + "/dominant_directions_corrected-slim.txt")
    layers = np.array([0, 130, 410, 650, 1150, 1350]) #start point in pixel
    sobel_smooth = imread(path + sampleID + side + "/sobel_smooth-slim.tif")
    binary_sobel_smooth = np.where(sobel_smooth > 0, 1, 0)
    distances = np.array([[np.argmax(row) for row in slice] for slice in binary_sobel_smooth])
    # Calculate the distance for every row in dominant_directions
    distances_to_cortex_surface = []
    for i in range(data.shape[0]):
        z, y, x, direction = data[i]
        z, y, x = int(z), int(y), int(x)
        dist = x - distances[z][y]
        distances_to_cortex_surface.append(dist)
    distances_to_cortex_surface = np.array(distances_to_cortex_surface)
    distances_to_cortex_surface = np.where(distances_to_cortex_surface < 0, 0,distances_to_cortex_surface)
    # Add layer identifier
    # Calculate the layer for every entry in dominant_directions
    layers_id = []
    for distance in distances_to_cortex_surface:
        layer = np.maximum(0, np.searchsorted(layers, distance, side='left') - 1)
        layers_id.append(layer)
    layers_id = np.array(layers_id)
    if side == "L":
        side_id = np.full(len(data), 0)
    else:
        side_id = np.full(len(data), 1)

    sample_id = np.full(len(data), int(re.findall(r'\d+', sampleID)[0]))
    # Add layers_id as the first column to dominant_directions
    dd = np.column_stack((sample_id, side_id, layers_id, data))
    np.savetxt(path + sampleID + side + "_VC" +"/dominant_directions_final.txt", dd) # VC: + "_VC"

'''path = "/Volumes/SSD16_4TB/ACx data/"
sampleID = "PR006_"
side = "L"
prepare_dominantDirections_for_stats(path, side, sampleID)'''

# loop over the directories
path = "/Volumes/SSD16_4TB/ACx data/"
folders = sorted([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])[1:]
for folder in folders:
    sampleID = folder[0:-1]
    side = folder[-1]
    prepare_dominantDirections_for_stats(path, side, sampleID)

##############################################################################################################
# merge all final files to one
path = "/Volumes/SSD16_4TB/ACx data/"
folders = sorted([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])[1:]
final_array = np.empty((0, 7))
for folder in folders:
    arr = np.loadtxt(path + folder + "/dominant_directions_final.txt")
    final_array = np.concatenate((final_array, arr))
np.savetxt(path + "/dominant_directions_fused_vc.txt", final_array)


##############################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean, circvar, circstd
# Basic statistics
# Calculate the mean and standard deviation of the dominant directions for every layer and plot
path = "..."
data = np.loadtxt(path + "/dominant_directions_fused.txt")

# Create a rough stats from the data: L, R histograms and stats, Levy plots for all layers and both sides,
# Filter the data
data_filtered_0 = data[(data[:, 1] == 0) & (data[:, 2] != 0) & (data[:, 2] != 4) & (data[:, 2] != 5)]
data_filtered_1 = data[(data[:, 1] == 1) & (data[:, 2] != 0) & (data[:, 2] != 4) & (data[:, 2] != 5)]
# Calculate the circular means
circmean_0 = circmean(data_filtered_0[:, -1], high=180, low=0)
circmean_1 = circmean(data_filtered_1[:, -1], high=180, low=0)


# Plot the histograms
palette = sns.color_palette()
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(data_filtered_0[:, -1], bins=180, density=True, color=palette[0])
plt.axvline(circmean_0, color='k', linestyle='dashed', linewidth=1)
plt.text(circmean_0, plt.gca().get_ylim()[1]*0.9, f'Mean: {circmean_0:.2f}', color='k', ha='center')
plt.title('Left ACx')
plt.xlabel('Dominant Direction (°)')
plt.ylabel('Normalized Frequency')
plt.xlim(0, 180)  # Set y-axis limits
plt.xticks(range(0, 181, 20))  # Set y-axis labels

plt.subplot(1, 2, 2)
plt.hist(data_filtered_1[:, -1], bins=180, density=True, color=palette[1])
plt.axvline(circmean_1, color='k', linestyle='dashed', linewidth=1)
plt.text(circmean_1, plt.gca().get_ylim()[1]*0.9, f'Mean: {circmean_1:.2f}', color='k', ha='center')
plt.title('Right ACx')
plt.xlabel('Dominant Direction (°)')
plt.ylabel('Normalized Frequency')
plt.xlim(0, 180)  # Set y-axis limits
plt.xticks(range(0, 181, 20))  # Set y-axis labels
plt.suptitle('Histogram of the Dominant Directions')
plt.savefig(path + 'LR_comparison-VC.png', dpi=300)
plt.tight_layout()
plt.show()

#### same plot by layer
# Unique layer values
layers = (1.,2.,3.)
layerid = ["23", "4", "5"]

# Loop over the layers
for i, layer in enumerate(layers):

    # Filter the data for the current layer
    data_filtered_0 = data[(data[:, 1] == 0) & (data[:, 2] == layer)]
    data_filtered_1 = data[(data[:, 1] == 1) & (data[:, 2] == layer)]

    # Calculate the circular means
    circmean_0 = circmean(data_filtered_0[:, -1], high=180, low=0)
    circmean_1 = circmean(data_filtered_1[:, -1], high=180, low=0)

    # Plot the histograms
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(data_filtered_0[:, -1], bins=180, density=True, alpha=0.6, color=palette[0])
    plt.axvline(circmean_0, color='k', linestyle='dashed', linewidth=1)
    plt.text(circmean_0, plt.gca().get_ylim()[1]*0.9, f'Mean: {circmean_0:.2f}', color='k', ha='center')
    plt.title(f'Left ACx')
    plt.xlabel('Dominant Direction (°)')
    plt.ylabel('Normalized Frequency')
    plt.xlim(0, 180)  # Set y-axis limits
    plt.xticks(range(0, 181, 20))  # Set y-axis labels

    plt.subplot(1, 2, 2)
    plt.hist(data_filtered_1[:, -1], bins=180, density=True, alpha=0.6, color=palette[1])
    plt.axvline(circmean_1, color='k', linestyle='dashed', linewidth=1)
    plt.text(circmean_1, plt.gca().get_ylim()[1]*0.9, f'Mean: {circmean_1:.2f}', color='k', ha='center')
    plt.title(f'Right ACx')
    plt.xlabel('Dominant Direction (°)')
    plt.ylabel('Normalized Frequency')
    plt.xlim(0, 180)  # Set y-axis limits
    plt.xticks(range(0, 181, 20))  # Set y-axis labels

    plt.suptitle(f'Histogram of the Dominant Directions - Layer ' + layerid[i])
    plt.savefig(path + f'LR_comparison_layer_{layerid[i]}.png', dpi=300)
    plt.tight_layout()
    plt.show()


