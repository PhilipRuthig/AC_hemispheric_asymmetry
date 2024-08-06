import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage.io import imread
import os

# ###############################################################
# prepare the data
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
    np.savetxt(path + sampleID + side + "_VC" +"/dominant_directions_final.txt", dd)

# loop over the directories
path = ""
folders = sorted([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])[1:]
for folder in folders:
    sampleID = folder[0:-1]
    side = folder[-1]
    prepare_dominantDirections_for_stats(path, side, sampleID)

# merge all final files to one
path = ""
folders = sorted([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])[1:]
final_array = np.empty((0, 7))
for folder in folders:
    arr = np.loadtxt(path + folder + "/dominant_directions_final.txt")
    final_array = np.concatenate((final_array, arr))
np.savetxt(path + "/dominant_directions_fused_vc.txt", final_array)


# ##############################################################################################################
# plot dominant directons along tonotopy and layers, mean over all z-slices
def fig_layer_tonotopy(path, dominant_directions, max_z, max_y, max_x, patch_size = 18, cmap = "twilight"):
    filled_array = np.full((max_z, max_y, max_x), np.nan)
    for row in dominant_directions:
        z, y, x, value = row
        x += (2 * (260-z))  # cortex surface moves in x direction during z stack (ACx): (2 * (260-z)); VCx R: (1.5 * (z))
        if int(x / 18) >= max_x:
            continue
        else:
            filled_array[int(z), int(y / 18), int(x / 18)] = value
    averaged_array = np.nanmean(filled_array, axis=0)

    fig, (ax) = plt.subplots(1, 1, figsize=(10, 15))
    fig.subplots_adjust(bottom=0.2)
    sns.heatmap(averaged_array, ax=ax, cmap=cmap, square=True, xticklabels=False, yticklabels=False,
                vmin=0, vmax=180, center=90, cbar_kws={"shrink": .35})
    ax.set_ylabel('Tonotopic axis', fontsize=14)
    #ax.invert_xaxis()
    ax.xaxis.set_label_position('bottom')
    ax.set_xlabel('Layer', fontsize=14)
    layer_mid = np.array([0, 65, 130, 270, 410, 530, 650, 900, 1150, 1250, 1350])
    layer_mid = (layer_mid + dominant_directions[:, 2].min() + patch_size)/ patch_size
    new_tick_locations = layer_mid.astype(int)
    ax.set_xticks(new_tick_locations)
    labels = ['', 'I', '', 'II/III', '', 'IV', '', 'V', '', 'VI', '']
    ax.set_xticklabels(labels, fontsize=14, horizontalalignment='center')
    plt.savefig(path + '/Layers_tonotopy.png', dpi=300)
    plt.show()

path = ""
directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith('PR') and d.endswith('R')]
for directory in directories:
    file_path = os.path.join(directory, 'dominant_directions_corrected-slim.txt')
    # Check if the file exists
    if os.path.isfile(file_path):
        # Load the data
        dominant_directions = np.loadtxt(file_path)
        # Run the function
        fig_layer_tonotopy(directory, dominant_directions, int(260), int(2379/18), int(2007/18))

# ##############################################################################################################
# polar plots along tonotopy
path = ""
data = np.loadtxt(path + "dominant_directions_fused.txt")
y_min = np.min(data[:, 4])
y_max = np.max(data[:, 4])

# Divide this range into three equal parts
anterior = (int(y_min), int(y_min + (y_max - y_min) / 3))
middle = (int(y_min + (y_max - y_min) / 3), int(y_min + 2 * (y_max - y_min) / 3))
posterior = (int(y_min + 2 * (y_max - y_min) / 3), int(y_max))


def plot_directionalityPolarAlongAP(data_l, data_r, ax1, ax2):
    '''
    Plot 1D directionality per layer; comparison between left and right cortex
    patch_size:         size of patch on which the directionality distributions were computed
    data_l, data_r:     sum of all distributions for each layer left/right
    nbr_l, nbr_r:       nbr od patches per layer for normalization left/right
    path_output:        path to where to save the resulting image
    return:             polar plot of both sides per layer
    '''
    color = ['#252525', '#a1dab4', '#41b6c4', '#225ea8', '#252525']
    labels = ['L1', 'L2/3', 'L4', 'L5']
    for i in (np.unique(data_l[:, 2])[:-1]):
        i = int(i)
        hist_l, bins_l = np.histogram(data_l[data_l[:, 2] == i][:, 6], bins=180, density=True)
        hist_r, bins_r = np.histogram(data_r[data_r[:, 2] == i][:, 6], bins=180, density=True)
        ax1.plot(np.deg2rad(np.round(bins_l, 0)[:-1]), hist_l, color=color[i], label=labels[i])
        ax1.fill(np.deg2rad(bins_l[:-1]), hist_l, color=color[i], alpha=0.3)
        ax2.plot(np.deg2rad(np.round(bins_r, 0)[:-1]), hist_r, color=color[i], label=labels[i])
        ax2.fill(np.deg2rad(bins_r[:-1]), hist_r, color=color[i], alpha=0.3)
    ax1.set_thetamin(0)
    ax1.set_thetamax(180)
    ax1.set_theta_zero_location("N")
    ax1.invert_xaxis()
    ax1.grid(b=True, which='major', color='#bdbdbd', linestyle='-')
    ax1.minorticks_on()
    ax1.grid(b=True, which='minor', color='#d9d9d9', linestyle='-', alpha=0.2)
    ax2.set_thetamin(0)
    ax2.set_thetamax(180)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.grid(b=True, which='major', color='#bdbdbd', linestyle='-')
    ax2.minorticks_on()
    ax2.grid(b=True, which='minor', color='#d9d9d9', linestyle='-', alpha=0.2)
    ax2.set_yticklabels([])
    #ax2.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1, 1))

# Create a figure with three subplots
import pylustrator
pylustrator.start()
fig = plt.figure(figsize=(8, 10), dpi=300)
ax = fig.subplot_mosaic(
    """
    AAABBBGGGHHH
    AAABBBGGGHHH
    CCCDDDIIIJJJ
    CCCDDDIIIJJJ
    EEEFFFKKKLLL
    EEEFFFKKKLLL
    """
)
for key in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]:
    fig.delaxes(ax[key])
    ax[key] = fig.add_subplot(ax[key].get_subplotspec(), polar=True)

column_labels = ['L', 'R']
row_labels = ['Anterior', 'Middle', 'Posterior']
axs = [ax["A"], ax["B"], ax["C"], ax["D"], ax["E"], ax["F"]]
for i, part_range in enumerate([anterior, middle, posterior]):
    data_l_part = data[np.isin(data[:, 0], [3, 5, 6, 7, 8, 9]) &
                     (data[:, 1] == 0) &
                     (data[:, 2] != 0) &
                     (data[:, 2] != 5) &
                     (data[:, 4] >= part_range[0]) &
                     (data[:, 4] < part_range[1])]
    data_r_part = data[np.isin(data[:, 0], [3, 5, 6, 7, 8, 9]) &
                     (data[:, 1] == 1) &
                     (data[:, 2] != 0) &
                     (data[:, 2] != 5) &
                     (data[:, 4] >= part_range[0]) &
                     (data[:, 4] < part_range[1])]
    plot_directionalityPolarAlongAP(data_l_part, data_r_part, axs[2 * i], axs[2 * i + 1])

axs = [ax["G"], ax["H"], ax["I"], ax["J"], ax["K"], ax["L"]]
for i, part_range in enumerate([anterior, middle, posterior]):
    data_l_part = data[np.isin(data[:, 0], [1, 10, 12, 13, 14]) &
                       (data[:, 1] == 0) &
                       (data[:, 2] != 0) &
                       (data[:, 2] != 5) &
                       (data[:, 4] >= part_range[0]) &
                       (data[:, 4] < part_range[1])]
    data_r_part = data[np.isin(data[:, 0], [1, 10, 12, 13, 14]) &
                       (data[:, 1] == 1) &
                       (data[:, 2] != 0) &
                       (data[:, 2] != 5) &
                       (data[:, 4] >= part_range[0]) &
                       (data[:, 4] < part_range[1])]
    plot_directionalityPolarAlongAP(data_l_part, data_r_part, axs[2 * i], axs[2 * i + 1])
    ''' # Set the column labels
    if i == 0:
        for j in range(2):
            axs[i, j].set_title(column_labels[j])

    # Set the row labels
    axs[i, 0].set_ylabel(row_labels[i], rotation=90, size='large')'''
plt.tight_layout()
save_name = 'PolarLayer_Female_AP.png'
#plt.savefig(path+save_name, dpi = 300)

plt.figure(1).axes[0].set(position=[0.2202, 0.7229, 0.4549, 0.2269], ylim=(-0.005, 0.04011))
plt.figure(1).axes[0].spines[['start', 'end']].set_visible(True)
plt.figure(1).axes[1].set_yticks([0.002, 0.004, 0.006, 0.008, 0.012, 0.014, 0.016, 0.018, 0.022, 0.024, 0.026, 0.028, 0.032, 0.034, 0.036, 0.038], ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], minor=True)
plt.figure(1).axes[1].set(position=[0.5782, 0.7229, 0.4549, 0.2269], yticks=[0., 0.01, 0.02, 0.03, 0.04], yticklabels=['', '', '', '', ''], ylim=(-0.005, 0.04011))
plt.figure(1).axes[1].spines[['start', 'end']].set_visible(True)
plt.figure(1).axes[2].set_yticks([0.002, 0.004, 0.006, 0.008, 0.012, 0.014, 0.016, 0.018, 0.022, 0.024, 0.026, 0.028, 0.032, 0.034], ['', '', '', '', '', '', '', '', '', '', '', '', '', ''], minor=True)
plt.figure(1).axes[2].set(position=[0.2202, 0.4262, 0.4549, 0.2269], yticks=[0., 0.01, 0.02, 0.03], yticklabels=['0.0', '0.01', '0.02', '0.03'], ylim=(-0.005, 0.036))
plt.figure(1).axes[2].spines[['start', 'end', 'inner']].set_visible(True)
plt.figure(1).axes[3].set_yticks([0.002, 0.004, 0.006, 0.008, 0.012, 0.014, 0.016, 0.018, 0.022, 0.024, 0.026, 0.028, 0.032, 0.034], ['', '', '', '', '', '', '', '', '', '', '', '', '', ''], minor=True)
plt.figure(1).axes[3].set(position=[0.5781, 0.4262, 0.4549, 0.2269], yticks=[0., 0.01, 0.02, 0.03], yticklabels=['', '', '', ''], ylim=(-0.005, 0.036))
plt.figure(1).axes[3].spines[['start', 'end']].set_visible(True)
plt.figure(1).axes[4].set_yticks([0.002, 0.004, 0.006, 0.008, 0.012, 0.014, 0.016, 0.018, 0.022, 0.024, 0.026, 0.028, 0.032, 0.034], ['', '', '', '', '', '', '', '', '', '', '', '', '', ''], minor=True)
plt.figure(1).axes[4].set(position=[0.2202, 0.1217, 0.4549, 0.2269], yticks=[0., 0.01, 0.02, 0.03], yticklabels=['0.0', '0.01', '0.02', '0.03'], ylim=(-0.005, 0.034))
plt.figure(1).axes[4].spines[['start', 'end']].set_visible(True)
plt.figure(1).axes[5].set_yticks([0.002, 0.004, 0.006, 0.008, 0.012, 0.014, 0.016, 0.018, 0.022, 0.024, 0.026, 0.028, 0.032], ['', '', '', '', '', '', '', '', '', '', '', '', ''], minor=True)
plt.figure(1).axes[5].legend(loc=(-1.108, 3.416), frameon=False, fontsize=12.)
plt.figure(1).axes[5].set(position=[0.5781, 0.1217, 0.4549, 0.2269], yticks=[0., 0.01, 0.02, 0.03], yticklabels=['', '', '', ''], ylim=(-0.005, 0.034))
plt.figure(1).axes[5].spines[['start', 'end']].set_visible(True)

plt.figure(1).axes[0].set(position=[0.1481, 0.7521, 0.2468, 0.1975], xticklabels=['0°', '45°', '90°', '135°', '180°'], yticklabels=['−0.01', '0.00', '0.01', '0.02', '0.03', '0.04', '0.05'])
plt.figure(1).axes[1].set(position=[0.3406, 0.7521, 0.2468, 0.1975], xticklabels=['0°', '45°', '90°', '135°', '180°'])
plt.figure(1).axes[2].set(position=[0.1481, 0.4591, 0.2468, 0.1975], xticklabels=['0°', '45°', '90°', '135°', '180°'])
plt.figure(1).axes[2].spines[['inner']].set_visible(False)
plt.figure(1).axes[3].set(position=[0.3406, 0.4591, 0.2468, 0.1975], xticklabels=['0°', '45°', '90°', '135°', '180°'])
plt.figure(1).axes[4].set(position=[0.1481, 0.1509, 0.2468, 0.1975], xticklabels=['0°', '45°', '90°', '135°', '180°'])
plt.figure(1).axes[5].legend(loc=(0.765, 1.083), frameon=False, labelspacing=0.2, fontsize=12.)
plt.figure(1).axes[5].set(position=[0.3406, 0.1509, 0.2468, 0.1975], xticklabels=['0°', '45°', '90°', '135°', '180°'])
plt.figure(1).axes[6].set_yticks([0.002, 0.004, 0.006, 0.008, 0.012, 0.014, 0.016, 0.018, 0.022, 0.024, 0.026, 0.028, 0.032, 0.034, 0.036, 0.038], ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], minor=True)
plt.figure(1).axes[6].set(position=[0.5627, 0.7521, 0.2468, 0.1975], xticklabels=['0°', '45°', '90°', '135°', '180°'], yticks=[0., 0.01, 0.02, 0.03], yticklabels=['0.0', '0.01', '0.02', '0.03'], ylim=(-0.005, 0.038))
plt.figure(1).axes[6].spines[['start', 'end']].set_visible(True)
plt.figure(1).axes[7].set(position=[0.765, 0.7521, 0.2468, 0.1975], xticklabels=['0°', '45°', '90°', '135°', '180°'], ylim=(-0.005, 0.038))
plt.figure(1).axes[7].spines[['start', 'end']].set_visible(True)
plt.figure(1).axes[8].set(position=[0.5627, 0.4591, 0.2468, 0.1975], xticklabels=['0°', '45°', '90°', '135°', '180°'], ylim=(-0.005, 0.038))
plt.figure(1).axes[8].spines[['start', 'end']].set_visible(True)
plt.figure(1).axes[9].set_yticks([0.002, 0.004, 0.006, 0.008, 0.012, 0.014, 0.016, 0.018, 0.022, 0.024, 0.026, 0.028, 0.032, 0.034, 0.036], ['', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], minor=True)
plt.figure(1).axes[9].set(position=[0.7633, 0.4591, 0.2468, 0.1975], xticklabels=['0°', '45°', '90°', '135°', '180°'], yticks=[0., 0.01, 0.02, 0.03], yticklabels=['', '', '', ''], ylim=(0., 0.038))
plt.figure(1).axes[9].spines[['start', 'end']].set_visible(True)
plt.figure(1).axes[10].set_yticks([0.002, 0.004, 0.006, 0.008, 0.012, 0.014, 0.016, 0.018, 0.022, 0.024, 0.026, 0.028, 0.032, 0.034, 0.036], ['', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], minor=True)
plt.figure(1).axes[10].set(position=[0.5627, 0.1509, 0.2468, 0.1975], xticklabels=['0°', '45°', '90°', '135°', '180°'], yticks=[0., 0.01, 0.02, 0.03], yticklabels=['0.0 ', '0.01 ', '0.02 ', '0.03'], ylim=(-0.005, 0.036))
plt.figure(1).axes[10].spines[['start', 'end']].set_visible(True)
plt.figure(1).axes[11].set_yticks([0.002, 0.004, 0.006, 0.008, 0.012, 0.014, 0.016, 0.018, 0.022, 0.024, 0.026, 0.028, 0.032, 0.034, 0.036], ['', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], minor=True)
plt.figure(1).axes[11].legend(loc=(-0.9583, 2.593), frameon=False, labelspacing=0.2, fontsize=12.)
plt.figure(1).axes[11].set(position=[0.7633, 0.1509, 0.2468, 0.1975], xticklabels=['0°', '45°', '90°', '135°', '180°'], yticks=[0., 0.01, 0.02, 0.03], yticklabels=['', '', '', ''], ylim=(-0.005, 0.036))
plt.figure(1).axes[11].spines[['start', 'end']].set_visible(True)
plt.figure(1).text(0.6762, 0.0948, 'L', transform=plt.figure(1).transFigure, fontsize=18.)  # id=plt.figure(1).texts[6].new
plt.figure(1).text(0.8736, 0.0948, 'R', transform=plt.figure(1).transFigure, fontsize=18.)  # id=plt.figure(1).texts[7].new
plt.figure(1).text(0.7794, 0.0320, '♂', transform=plt.figure(1).transFigure, fontsize=24.)  # id=plt.figure(1).texts[8].new

plt.figure(1).text(0.0246, 0.8450, 'Anterior', transform=plt.figure(1).transFigure, fontsize=16.)  # id=plt.figure(1).texts[9].new
plt.figure(1).text(0.3474, 0.0320, '♀', transform=plt.figure(1).transFigure, fontsize=24.)  # id=plt.figure(1).texts[10].new
plt.figure(1).text(0.0246, 0.5519, 'Center', transform=plt.figure(1).transFigure, fontsize=16.)  # id=plt.figure(1).texts[11].new
plt.figure(1).text(0.0246, 0.2433, 'Posterior', transform=plt.figure(1).transFigure, fontsize=16.)  # id=plt.figure(1).texts[12].new
plt.figure(1).text(0.2632, 0.0948, 'L', transform=plt.figure(1).transFigure, fontsize=18.)  # id=plt.figure(1).texts[13].new
plt.figure(1).text(0.4546, 0.0948, 'R', transform=plt.figure(1).transFigure, fontsize=18.)  # id=plt.figure(1).texts[14].new

plt.show()


# ##############################################################################################################
# plot the fit
path_3p = "bpnr3p_allsamples_25.10000.0.05/"
fit_3p = np.array(pd.read_csv(path_3p + 'bpnr3p_fit.csv', header=None, delimiter=r"\s+"))
fit_means_3p = np.mean(fit_3p, axis=0)
fit_std_3p = np.std(fit_3p, axis=0)

path_2p = "bpnr2p_allsamples_25.10000.0.05/"
fit_2p = np.array(pd.read_csv(path_2p + 'bpnr1p_fit.csv', header=None, delimiter=r"\s+"))
fit_means_2p = np.mean(fit_2p, axis=0)
fit_std_2p = np.std(fit_2p, axis=0)

path_1p = "bpnr1p_allsamples_25.10000.0.05/"
fit_1p = np.array(pd.read_csv(path_1p + 'bpnr1p_fit.csv', header=None, delimiter=r"\s+"))
fit_means_1p = np.mean(fit_1p, axis=0)
fit_std_1p = np.std(fit_1p, axis=0)

x = range(3)  # or whatever x-coordinates you want
plt.plot((0,1,2), (fit_means_1p[1],fit_means_2p[1], fit_means_3p[1]),color ='k')
plt.errorbar((0,1,2), (fit_means_1p[1],fit_means_2p[1], fit_means_3p[1]), yerr=(fit_std_1p[1], fit_std_2p[1], fit_std_3p[1]), fmt='o', color ='k')
plt.xticks([0, 1, 2], ['domDir ~ side', 'domDir ~ side + layer', 'domDir ~ side + layer + tonotopy'])
plt.xlabel('Regression model complexity')
plt.ylabel('DIC')
plt.title('Fit Means with Standard Deviations')
plt.tight_layout()
plt.savefig("/Volumes/SSD16_4TB/ACx data/stats/" + "fit_1p-2p-3p.png", dpi=200)
plt.show()