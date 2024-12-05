import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data for the different plots
# for the polar plot: data_l, data_r
path = "/Volumes/Gesine1/ACx data/"
data = np.loadtxt(path + "dominant_directions_fused.txt")
unique_values = np.unique(data[:, 0])
data_l = data[(data[:, 1] == 0)  & (data[:, 2] != 5)]
data_r = data[(data[:, 1] == 1)  & (data[:, 2] != 5)]

# for the Violin plot 2p L23: df_melted2p_L23
means = []
lB = []
uB = []
path = "/Volumes/Gesine1/ACx data/4Revisions/stats/2p_ACx_runs_randomMC_25.10000.0.05/"
for name in sorted([name for name in os.listdir(path) if not name.startswith('._')]):
    L_23 = np.mean(np.array(pd.read_csv(path + name + '/bpnr2p_Intercept.csv', header=None, delimiter=r"\s+")), axis=0)
    R_23 = np.mean(np.array(pd.read_csv(path + name + '/bpnr2p_sideR.csv', header=None, delimiter=r"\s+")), axis=0)
    L_4 = np.mean(np.array(pd.read_csv(path + name + '/bpnr2p_layerL4.csv', header=None, delimiter=r"\s+")), axis=0)
    R_4 = np.mean(np.array(pd.read_csv(path + name + '/bpnr2p_siderlayerL4.csv', header=None, delimiter=r"\s+")), axis=0)
    L_5 = np.mean(np.array(pd.read_csv(path + name + '/bpnr2p_layerL5.csv', header=None, delimiter=r"\s+")), axis=0)
    R_5 = np.mean(np.array(pd.read_csv(path + name + '/bpnr2p_siderlayerL5.csv', header=None, delimiter=r"\s+")), axis=0)
    means.append((L_23[0], L_4[0], L_5[0], R_23[0], R_4[0], R_5[0]))
    lB.append((L_23[3], L_4[3], L_5[3], R_23[3], R_4[3], R_5[3]))
    uB.append((L_23[4], L_4[4], L_5[4], R_23[4], R_4[4], R_5[4]))
means = np.array(means)
df = pd.DataFrame({
    'Group 1': np.concatenate((means[:,0], means[:,3])),
    'Side': np.repeat(['left ACx', 'right ACx'], len(means))
})
df_melted2p_L23 = df.melt(id_vars='Side', var_name='Group', value_name='Values')
df = pd.DataFrame({
    'Group 1': np.concatenate((means[:,1], means[:,4])),
    'Side': np.repeat(['left ACx', 'right ACx'], len(means))
})
df_melted2p_L4 = df.melt(id_vars='Side', var_name='Group', value_name='Values')
df = pd.DataFrame({
    'Group 1': np.concatenate((means[:,2], means[:,5])),
    'Side': np.repeat(['left ACx', 'right ACx'], len(means))
})
df_melted2p_L5 = df.melt(id_vars='Side', var_name='Group', value_name='Values')

# for the posteriors 2p: data_distr2, data_params2
path = '/Volumes/Gesine1/ACx data/4Revisions/stats/2p_ACx_randomMC_25.10000.0.05/'
L_23 = np.mean(np.array(pd.read_csv(path + 'bpnr2p_Intercept.csv', header=None, delimiter=r"\s+")), axis=0)
R_23 = np.mean(np.array(pd.read_csv(path + 'bpnr2p_sideR.csv', header=None, delimiter=r"\s+")), axis=0)
L_4 = np.mean(np.array(pd.read_csv(path + 'bpnr2p_layerL4.csv', header=None, delimiter=r"\s+")), axis=0)
R_4 = np.mean(np.array(pd.read_csv(path + 'bpnr2p_siderlayerL4.csv', header=None, delimiter=r"\s+")), axis=0)
L_5 = np.mean(np.array(pd.read_csv(path + 'bpnr2p_layerL5.csv', header=None, delimiter=r"\s+")), axis=0)
R_5 = np.mean(np.array(pd.read_csv(path + 'bpnr2p_siderlayerL5.csv', header=None, delimiter=r"\s+")), axis=0)
beta1_1 = np.array(pd.read_csv(path + 'bpnr2p_beta1_1.csv', header=None, delimiter=r"\s+"))
beta1_2 = np.array(pd.read_csv(path + 'bpnr2p_beta1_2.csv', header=None, delimiter=r"\s+"))
beta1_3 = np.array(pd.read_csv(path + 'bpnr2p_beta1_3.csv', header=None, delimiter=r"\s+"))
beta1_4 = np.array(pd.read_csv(path + 'bpnr2p_beta1_4.csv', header=None, delimiter=r"\s+"))
beta2_1 = np.array(pd.read_csv(path + 'bpnr2p_beta2_1.csv', header=None, delimiter=r"\s+"))
beta2_2 = np.array(pd.read_csv(path + 'bpnr2p_beta2_2.csv', header=None, delimiter=r"\s+"))
beta2_3 = np.array(pd.read_csv(path + 'bpnr2p_beta2_3.csv', header=None, delimiter=r"\s+"))
beta2_4 = np.array(pd.read_csv(path + 'bpnr2p_beta2_4.csv', header=None, delimiter=r"\s+"))
fit = np.array(pd.read_csv(path + 'bpnr2p_fit.csv', header=None, delimiter=r"\s+"))
# plot the distribution and the lb, ub and median of the coefficients
L_23distr = np.arctan2(beta2_1, beta1_1).reshape(-1)
L_4distr = np.arctan2(beta2_1+beta2_3, beta1_1+beta1_3).reshape(-1)
L_5distr = np.arctan2(beta2_1+beta2_4, beta1_1+beta1_4).reshape(-1)
R_23distr = np.arctan2(beta2_1+beta2_2, beta1_1+beta1_2).reshape(-1)
R_4distr = np.arctan2(beta2_1+beta2_2+beta2_3, beta1_1+beta1_2+beta1_3).reshape(-1)
R_5distr = np.arctan2(beta2_1+beta2_2+beta2_4, beta1_1+beta1_2+beta1_4).reshape(-1)
data_distr2 = np.column_stack((L_23distr, L_4distr, L_5distr, R_23distr, R_4distr, R_5distr))
data_params2 = np.column_stack((L_23, L_4, L_5, R_23, R_4, R_5))

# Load data for posterior 1p: data1, data_params1
# 1p - mean distributions
path = '/Volumes/Gesine1/ACx data/4Revisions/stats/1p_ACx_randomMC_25.10000.0.05/'
L_means = np.mean(np.array(pd.read_csv(path + 'bpnr1p_Intercept.csv', header=None, delimiter=r"\s+")), axis=0)
R_means = np.mean(np.array(pd.read_csv(path + 'bpnr1p_sideR.csv', header=None, delimiter=r"\s+")), axis=0)
beta1_1 = np.array(pd.read_csv(path + 'bpnr1p_beta1_1.csv', header=None, delimiter=r"\s+"))
beta1_2 = np.array(pd.read_csv(path + 'bpnr1p_beta1_2.csv', header=None, delimiter=r"\s+"))
beta2_1 = np.array(pd.read_csv(path + 'bpnr1p_beta2_1.csv', header=None, delimiter=r"\s+"))
beta2_2 = np.array(pd.read_csv(path + 'bpnr1p_beta2_2.csv', header=None, delimiter=r"\s+"))
fit = np.array(pd.read_csv(path + 'bpnr1p_fit.csv', header=None, delimiter=r"\s+"))
# plot the distribution and the lb, ub and median of the coefficients
L_distr = np.arctan2(beta2_1, beta1_1).reshape(-1)
R_distr = np.arctan2(beta2_1+beta2_2, beta1_1+beta1_2).reshape(-1)
data1 = np.column_stack((L_distr, R_distr))
data_params1 = np.column_stack((L_means, R_means))

# load data for violin plot 1p:
means = []
lB = []
uB = []
path = '/Volumes/Gesine1/ACx data/4Revisions/stats/1p_ACx_runs_randomMC_25.10000.0.05/'
for name in sorted([name for name in os.listdir(path) if not name.startswith('._')]):
    L = np.mean(np.array(pd.read_csv(path + name + '/bpnr1p_Intercept.csv', header=None, delimiter=r"\s+")), axis=0)
    R = np.mean(np.array(pd.read_csv(path + name + '/bpnr1p_sideR.csv', header=None, delimiter=r"\s+")), axis=0)
    means.append((L[0], R[0]))
    lB.append((L[3], R[3]))
    uB.append((L[4], R[4]))
means = np.array(means)
df = pd.DataFrame({
    'Group 1': np.concatenate((means[:,0], means[:,1])),
    'Side': np.repeat(['left ACx', 'right ACx'], len(means))
})
df_melted1p = df.melt(id_vars='Side', var_name='Group', value_name='Values')

# load data violin plot male/female 1p: df_melted1pMale, df_melted1pFemale
base_path = '/Volumes/Gesine1/ACx data/4Revisions/stats/1p_ACx_runs_randomMC_25.10000.0.05/'
Male = ['1', '10', '12', '13', '14']
Female = ['3', '5', '6', '7', '8', '9']

# Function to filter directories by values in their names
def filter_directories_by_values(base_path, values):
    matching_dirs = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and any(value in item for value in values):
            matching_dirs.append(item_path)
    return matching_dirs

# Filter directories
filtered_male = filter_directories_by_values(base_path, Male)
filtered_female = filter_directories_by_values(base_path, Female)[:-1]
# Initialize lists to store means and bounds
means = []
lB = []
uB = []
# Process each filtered directory
for dir_path in sorted(filtered_male):
    L = np.mean(np.array(pd.read_csv(os.path.join(dir_path, 'bpnr1p_Intercept.csv'), header=None, delimiter=r"\s+")), axis=0)
    R = np.mean(np.array(pd.read_csv(os.path.join(dir_path, 'bpnr1p_sideR.csv'), header=None, delimiter=r"\s+")), axis=0)
    means.append((L[0], R[0]))
    lB.append((L[3], R[3]))
    uB.append((L[4], R[4]))
means = np.array(means)
lB = np.array(lB)
uB = np.array(uB)
labels = ["left ACx", "right ACx"]
# split violin plot
df = pd.DataFrame({
    'Group 1': np.concatenate((means[:,0], means[:,1])),
    'Side': np.repeat(['left ACx', 'right ACx'], len(means))
})
df_melted1pMale = df.melt(id_vars='Side', var_name='Group', value_name='Values')
means = []
lB = []
uB = []
for dir_path in sorted(filtered_female):
    L = np.mean(np.array(pd.read_csv(os.path.join(dir_path, 'bpnr1p_Intercept.csv'), header=None, delimiter=r"\s+")), axis=0)
    R = np.mean(np.array(pd.read_csv(os.path.join(dir_path, 'bpnr1p_sideR.csv'), header=None, delimiter=r"\s+")), axis=0)
    means.append((L[0], R[0]))
    lB.append((L[3], R[3]))
    uB.append((L[4], R[4]))
means = np.array(means)
lB = np.array(lB)
uB = np.array(uB)
labels = ["left ACx", "right ACx"]
# split violin plot
df = pd.DataFrame({
    'Group 1': np.concatenate((means[:,0], means[:,1])),
    'Side': np.repeat(['left ACx', 'right ACx'], len(means))
})
df_melted1pFemale = df.melt(id_vars='Side', var_name='Group', value_name='Values')

# load data for VCx: dataVC, data_paramsVC
path = '/Volumes/Gesine1/ACx data/4Revisions/stats/1p_VCx_randomMC_25.10000.0.1/'
L_means = np.mean(np.array(pd.read_csv(path + 'bpnr1p_Intercept.csv', header=None, delimiter=r"\s+")), axis=0)
R_means = np.mean(np.array(pd.read_csv(path + 'bpnr1p_sideR.csv', header=None, delimiter=r"\s+")), axis=0)
beta1_1 = np.array(pd.read_csv(path + 'bpnr1p_beta1_1.csv', header=None, delimiter=r"\s+"))
beta1_2 = np.array(pd.read_csv(path + 'bpnr1p_beta1_2.csv', header=None, delimiter=r"\s+"))
beta2_1 = np.array(pd.read_csv(path + 'bpnr1p_beta2_1.csv', header=None, delimiter=r"\s+"))
beta2_2 = np.array(pd.read_csv(path + 'bpnr1p_beta2_2.csv', header=None, delimiter=r"\s+"))
fit = np.array(pd.read_csv(path + 'bpnr1p_fit.csv', header=None, delimiter=r"\s+"))
# plot the distribution and the lb, ub and median of the coefficients
L_distr = np.arctan2(beta2_1, beta1_1).reshape(-1)
R_distr = np.arctan2(beta2_1+beta2_2, beta1_1+beta1_2).reshape(-1)
dataVC = np.column_stack((L_distr, R_distr))
data_paramsVC = np.column_stack((L_means, R_means))

# Load data for posterior plot male: data1_male
# 1p - mean distributions
path = '/Volumes/Gesine1/ACx data/4Revisions/stats/1p_ACx_Male_randomMC_25.10000.0.05/'
L_means = np.mean(np.array(pd.read_csv(path + 'bpnr1p_Intercept.csv', header=None, delimiter=r"\s+")), axis=0)
R_means = np.mean(np.array(pd.read_csv(path + 'bpnr1p_sideR.csv', header=None, delimiter=r"\s+")), axis=0)
beta1_1 = np.array(pd.read_csv(path + 'bpnr1p_beta1_1.csv', header=None, delimiter=r"\s+"))
beta1_2 = np.array(pd.read_csv(path + 'bpnr1p_beta1_2.csv', header=None, delimiter=r"\s+"))
beta2_1 = np.array(pd.read_csv(path + 'bpnr1p_beta2_1.csv', header=None, delimiter=r"\s+"))
beta2_2 = np.array(pd.read_csv(path + 'bpnr1p_beta2_2.csv', header=None, delimiter=r"\s+"))
fit = np.array(pd.read_csv(path + 'bpnr1p_fit.csv', header=None, delimiter=r"\s+"))
# plot the distribution and the lb, ub and median of the coefficients
L_distr = np.arctan2(beta2_1, beta1_1).reshape(-1)
R_distr = np.arctan2(beta2_1+beta2_2, beta1_1+beta1_2).reshape(-1)
data1_male = np.column_stack((L_distr, R_distr))
data_params1_male = np.column_stack((L_means, R_means))

# Load data for posterior plot femlae: data1_female
# 1p - mean distributions
path = '/Volumes/Gesine1/ACx data/4Revisions/stats/1p_ACx_Female_randomMC_25.10000.0.1/'
L_means = np.mean(np.array(pd.read_csv(path + 'bpnr1p_Intercept.csv', header=None, delimiter=r"\s+")), axis=0)
R_means = np.mean(np.array(pd.read_csv(path + 'bpnr1p_sideR.csv', header=None, delimiter=r"\s+")), axis=0)
beta1_1 = np.array(pd.read_csv(path + 'bpnr1p_beta1_1.csv', header=None, delimiter=r"\s+"))
beta1_2 = np.array(pd.read_csv(path + 'bpnr1p_beta1_2.csv', header=None, delimiter=r"\s+"))
beta2_1 = np.array(pd.read_csv(path + 'bpnr1p_beta2_1.csv', header=None, delimiter=r"\s+"))
beta2_2 = np.array(pd.read_csv(path + 'bpnr1p_beta2_2.csv', header=None, delimiter=r"\s+"))
fit = np.array(pd.read_csv(path + 'bpnr1p_fit.csv', header=None, delimiter=r"\s+"))
# plot the distribution and the lb, ub and median of the coefficients
L_distr = np.arctan2(beta2_1, beta1_1).reshape(-1)
R_distr = np.arctan2(beta2_1+beta2_2, beta1_1+beta1_2).reshape(-1)
data1_female = np.column_stack((L_distr, R_distr))
data_params1_female = np.column_stack((L_means, R_means))




# pylustrator
import pylustrator
pylustrator.start()

fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.subplot_mosaic(
    """
    AAABBBCCCDDDD
    AAABBBCCCDDDD
    AAABBBEEEFFFF
    AAABBBEEEFFFF
    IIIJJJEEEFFFF
    IIIJJJGGGHHHH
    IIIJJJGGGHHHH
    LLLLMMMMNNNNN
    LLLLMMMMNNNNN
    """
)
for key in ["A", "B"]:
    fig.delaxes(ax[key])
    ax[key] = fig.add_subplot(ax[key].get_subplotspec(), polar=True)

for key in ["C", "D", "E", "F", "G", "H"]:
    fig.delaxes(ax[key])
    ax[key] = fig.add_subplot(ax[key].get_subplotspec(), polar=False)

# Polar plot
color = ["#d9d9d9", "#969696", "#525252", "#252525"]
labels = ['L1', 'L2/3', 'L4', 'L5', 'L6']
for i in np.unique(data_l[:, 2])[:-1]:
    i = int(i)
    hist_l, bins_l = np.histogram(data_l[data_l[:, 2] == i][:, 6], bins=180, density=True)
    hist_r, bins_r = np.histogram(data_r[data_r[:, 2] == i][:, 6], bins=180, density=True)
    # Filling the histogram for the left data
    ax["A"].fill(np.deg2rad(bins_l[:-1]), hist_l, color=color[i], alpha=0.3)
    # Adding the outline for the left data
    ax["A"].plot(np.deg2rad(bins_l[:-1]), hist_l, color=color[i], linewidth=1.5, label=labels[i])
    # Filling the histogram for the right data
    ax["B"].fill(np.deg2rad(bins_r[:-1]), hist_r, color=color[i], alpha=0.3)
    # Adding the outline for the right data
    ax["B"].plot(np.deg2rad(bins_r[:-1]), hist_r, color=color[i], linewidth=1.5, label=labels[i])
ax["A"].set_thetamin(0)
ax["A"].set_thetamax(180)
ax["A"].set_theta_zero_location("N")
ax["A"].invert_xaxis()
ax["A"].grid(b=True, which='major', color='#bdbdbd', linestyle='-')
ax["A"].minorticks_on()
ax["A"].grid(b=True, which='minor', color='#d9d9d9', linestyle='-', alpha=0.2)
ax["B"].set_thetamin(0)
ax["B"].set_thetamax(180)
ax["B"].set_theta_zero_location("N")
ax["B"].set_theta_direction(-1)
ax["B"].grid(b=True, which='major', color='#bdbdbd', linestyle='-')
ax["B"].minorticks_on()
ax["B"].grid(b=True, which='minor', color='#d9d9d9', linestyle='-', alpha=0.2)
ax["B"].set_yticklabels([])
ax["B"].legend(fontsize=10, loc='upper right', bbox_to_anchor=(1, 1))

# Violin plot 2p L23
sns.violinplot(x='Group', y='Values', hue='Side', data=df_melted2p_L23, split=True, gap=0.3, palette=['blue', 'goldenrod'], inner="point", alpha=0.7, ax=ax["C"])
groups = df_melted2p_L23['Group'].unique()
for group in groups:
    left_side = df_melted2p_L23[(df_melted2p_L23['Group'] == group) & (df_melted2p_L23['Side'] == 'left ACx')]['Values'].values
    right_side = df_melted2p_L23[(df_melted2p_L23['Group'] == group) & (df_melted2p_L23['Side'] == 'right ACx')]['Values'].values
    x_position = np.where(groups == group)[0][0]
    for i, (left, right) in enumerate(zip(left_side, right_side)):
        ax["C"].plot([x_position - 0.06, x_position + 0.06], [left, right], color=plt.cm.tab10(i), linestyle='-', linewidth=1)
ax["C"].set_xlabel('')
ax["C"].set_xticklabels('')
ax["C"].set_ylabel('Mean orientation [°]', rotation=90)
ax["C"].set_ylim(80, 100)
ax["C"].legend().remove()

# Violin plot 2p L4
sns.violinplot(x='Group', y='Values', hue='Side', data=df_melted2p_L4, split=True, gap=0.3, palette=['blue', 'goldenrod'], inner="point", alpha=0.7, ax=ax["E"])
groups = df_melted2p_L4['Group'].unique()
for group in groups:
    left_side = df_melted2p_L4[(df_melted2p_L4['Group'] == group) & (df_melted2p_L4['Side'] == 'left ACx')]['Values'].values
    right_side = df_melted2p_L4[(df_melted2p_L4['Group'] == group) & (df_melted2p_L4['Side'] == 'right ACx')]['Values'].values
    x_position = np.where(groups == group)[0][0]
    for i, (left, right) in enumerate(zip(left_side, right_side)):
        ax["E"].plot([x_position - 0.06, x_position + 0.06], [left, right], color=plt.cm.tab10(i), linestyle='-', linewidth=1)
ax["E"].set_xlabel('')
ax["E"].set_xticklabels('')
ax["E"].set_ylabel('Mean orientation [°]', rotation=90)
ax["E"].set_ylim(80, 100)
ax["E"].legend().remove()

# Violin plot 2p L5
sns.violinplot(x='Group', y='Values', hue='Side', data=df_melted2p_L5, split=True, gap=0.3, palette=['blue', 'goldenrod'], inner="point", alpha=0.7, ax=ax["G"])
groups = df_melted2p_L5['Group'].unique()
for group in groups:
    left_side = df_melted2p_L5[(df_melted2p_L5['Group'] == group) & (df_melted2p_L5['Side'] == 'left ACx')]['Values'].values
    right_side = df_melted2p_L5[(df_melted2p_L5['Group'] == group) & (df_melted2p_L5['Side'] == 'right ACx')]['Values'].values
    x_position = np.where(groups == group)[0][0]
    for i, (left, right) in enumerate(zip(left_side, right_side)):
        ax["G"].plot([x_position - 0.06, x_position + 0.06], [left, right], color=plt.cm.tab10(i), linestyle='-', linewidth=1)
ax["G"].set_xlabel('')
ax["G"].set_xticklabels('')
ax["G"].set_ylabel('Mean orientation [°]', rotation=90)
ax["G"].set_ylim(80, 100)
ax["G"].legend().remove()

# Posterior 2p
colors = ['blue', 'goldenrod']
d = ['L', 'R']
axs = [ax["D"], ax["F"], ax["H"]]
for j, a in enumerate(axs):
    data = data_distr2[:, [j + 0, j + 3]]
    data_param = data_params2[:, [j + 0, j + 3]]
    for i in range(len(d)):
        dat = np.rad2deg(data[:, i])
        sns.set(style="ticks")
        a.hist(dat, bins=90, alpha=0.7, label=d[i], color=colors[i], edgecolor=colors[i], histtype='stepfilled')
        a.axvline(data_param[0, i], c="black", ls="-", lw=1)
        a.axvline(data_param[3, i], c="grey", ls="--", lw=1)
        a.axvline(data_param[4, i], c="grey", ls="--", lw=1)
        a.set_ylabel("")
        a.set_yticks([])
        a.set_yticklabels([])
        a.set_xlim(87, 94)
        a.set_xticklabels([87, 88, 89, 90, 91, 92, 93, 94])

# Posterior 1p
d = ['Left', 'Right']
colors = ['blue', 'goldenrod']
for i in range(len(d)):
    dat = np.rad2deg(data1[:, i])
    sns.set(style="ticks")
    ax["J"].hist(dat, bins=90, alpha=0.7, label=d[i], color=colors[i], edgecolor=colors[i], histtype='stepfilled')
    ax["J"].axvline(data_params1[0, i], c="black", ls="-", lw=1)
    ax["J"].axvline(data_params1[3, i], c="grey", ls="--", lw=1)
    ax["J"].axvline(data_params1[4, i], c="grey", ls="--", lw=1)
    ax["J"].set_ylabel("")
    ax["J"].set_yticklabels([])
ax["J"].set_xlim(89, 92)
ax["J"].set_xticklabels([89., 89.5, 90., 90.5, 91., 91.5, 92.])

# Violin plot 1p
sns.violinplot(x='Group', y='Values', hue='Side', data=df_melted1p, split=True, gap=0.3, palette=['blue', 'goldenrod'], inner="point", alpha=0.7, ax=ax["I"])
groups = df_melted1p['Group'].unique()
for group in groups:
    left_side = df_melted1p[(df_melted1p['Group'] == group) & (df_melted1p['Side'] == 'left ACx')]['Values'].values
    right_side = df_melted1p[(df_melted1p['Group'] == group) & (df_melted1p['Side'] == 'right ACx')]['Values'].values
    x_position = np.where(groups == group)[0][0]
    for i, (left, right) in enumerate(zip(left_side, right_side)):
        ax["I"].plot([x_position - 0.06, x_position + 0.06], [left, right], color=plt.cm.tab10(i), linestyle='-', linewidth=1)
ax["I"].set_xlabel('')
ax["I"].set_xticklabels('')
ax["I"].set_ylabel('Mean orientation [°]', rotation=90)
ax["I"].set_ylim(80, 100)
ax["I"].legend().remove()

"""# Violin plot Male
sns.violinplot(x='Group', y='Values', hue='Side', data=df_melted1pMale, split=True, gap=0.3, palette=['blue', 'goldenrod'], inner="point", alpha=0.7, ax=ax["L"])
groups = df_melted1pMale['Group'].unique()
for group in groups:
    left_side = df_melted1pMale[(df_melted1pMale['Group'] == group) & (df_melted1pMale['Side'] == 'left ACx')]['Values'].values
    right_side = df_melted1pMale[(df_melted1pMale['Group'] == group) & (df_melted1pMale['Side'] == 'right ACx')]['Values'].values
    x_position = np.where(groups == group)[0][0]
    for i, (left, right) in enumerate(zip(left_side, right_side)):
        ax["L"].plot([x_position - 0.06, x_position + 0.06], [left, right], color='black', linestyle='--', linewidth=1)
ax["L"].set_xlabel('')
ax["L"].set_xticklabels('')
ax["L"].set_ylabel('Mean orientation [°]', rotation=90)
ax["L"].set_ylim(80, 100)
ax["L"].legend().remove()

# Violin plot Female
sns.violinplot(x='Group', y='Values', hue='Side', data=df_melted1pFemale, split=True, gap=0.3, palette=['blue', 'goldenrod'], inner="point", alpha=0.7, ax=ax["M"])
groups = df_melted1pFemale['Group'].unique()
for group in groups:
    left_side = df_melted1pFemale[(df_melted1pFemale['Group'] == group) & (df_melted1pFemale['Side'] == 'left ACx')]['Values'].values
    right_side = df_melted1pFemale[(df_melted1pFemale['Group'] == group) & (df_melted1pFemale['Side'] == 'right ACx')]['Values'].values
    x_position = np.where(groups == group)[0][0]
    for i, (left, right) in enumerate(zip(left_side, right_side)):
        ax["M"].plot([x_position - 0.06, x_position + 0.06], [left, right], color='black', linestyle='--', linewidth=1)
ax["M"].set_xlabel('')
ax["M"].set_xticklabels('')
ax["M"].set_ylabel('Mean orientation [°]', rotation=90)
ax["M"].set_ylim(80, 100)
ax["M"].legend().remove()"""

# posterior plot Male
d = ['Left', 'Right']
colors = ['blue', 'goldenrod']
for i in range(len(d)):
    dat = np.rad2deg(data1_male[:, i])
    sns.set(style="ticks")
    ax["L"].hist(dat, bins=90, alpha=0.7, label=d[i], color=colors[i], edgecolor=colors[i], histtype='stepfilled')
    ax["L"].axvline(data_params1_male[0, i], c="black", ls="-", lw=1)
    ax["L"].axvline(data_params1_male[3, i], c="grey", ls="--", lw=1)
    ax["L"].axvline(data_params1_male[4, i], c="grey", ls="--", lw=1)
    ax["L"].set_ylabel("")
    ax["L"].set_yticklabels([])
ax["L"].set_xlim(87, 93)
ax["L"].set_xticklabels([87., 88., 89., 90., 91., 92., 93.])

# posterior plot Female
d = ['L', 'R']
colors = ['blue', 'goldenrod']
for i in range(len(d)):
    dat = np.rad2deg(data1_female[:, i])
    sns.set(style="ticks")
    ax["M"].hist(dat, bins=90, alpha=0.7, label=d[i], color=colors[i], edgecolor=colors[i], histtype='stepfilled')
    ax["M"].axvline(data_params1_female[0, i], c="black", ls="-", lw=1)
    ax["M"].axvline(data_params1_female[3, i], c="grey", ls="--", lw=1)
    ax["M"].axvline(data_params1_female[4, i], c="grey", ls="--", lw=1)
    ax["M"].set_ylabel("")
    ax["M"].set_yticklabels([])
ax["M"].set_xlim(87, 93)
ax["M"].set_xticklabels([87., 88., 89., 90., 91., 92., 93.])


# Posterior 1p VCx
d = ['Left', 'Right']
colors = ['blue', 'goldenrod']
for i in range(len(d)):
    dat = np.rad2deg(dataVC[:, i])
    sns.set(style="ticks")
    ax["N"].hist(dat, bins=90, alpha=0.7, label=d[i], color=colors[i], edgecolor=colors[i], histtype='stepfilled')
    ax["N"].axvline(data_paramsVC[0, i], c="black", ls="-", lw=1)
    ax["N"].axvline(data_paramsVC[3, i], c="grey", ls="--", lw=1)
    ax["N"].axvline(data_paramsVC[4, i], c="grey", ls="--", lw=1)
    ax["N"].set_ylabel("")
    ax["N"].set_yticklabels([])

plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
plt.figure(1).ax_dict["I"].set(position=[0.07488, 0.3597, 0.197, 0.2464], xticks=[], xticklabels=[], ylim=(78., 102.))
plt.figure(1).ax_dict["J"].set(position=[0.2956, 0.3893, 0.197, 0.2168], xlabel='Posterior distribution [°]', yticks=[], yticklabels=[], ylim=(0., 11500.))
plt.figure(1).ax_dict["L"].set(position=[0.07488, 0.04435, 0.197, 0.2464], xlabel='Male', ylim=(78., 102.))
plt.figure(1).ax_dict["M"].set(position=[0.2956, 0.04435, 0.197, 0.2464], xlabel='Female', ylabel='', yticks=[], yticklabels=[], ylim=(78., 102.))
plt.figure(1).ax_dict["M"].get_yaxis().get_label().set(text='')
plt.figure(1).ax_dict["N"].set(position=[0.603, 0.06406, 0.3153, 0.2267], xlabel='Posterior distribution [°]', ylabel='', yticks=[], yticklabels=[], ylim=(0., 10300.))
plt.figure(1).axes[5].set_xticks([0.1963, 0.3927, 0.589, 0.9817, 1.178, 1.374, 1.767, 1.963, 2.16, 2.553, 2.749, 2.945], ['', '', '', '', '', '', '', '', '', '', '', ''], minor=True)
plt.figure(1).axes[5].set(position=[0.03547, 0.68, 0.2049, 0.2562], xticks=[0., 0.7854, 1.571, 2.356, 3.142], xticklabels=['0°', '45°', '90°', '135°', '180°'], yticklabels=['0.00', '0.01', '0.02', '0.03', '0.04'])
plt.figure(1).axes[5].spines[['start', 'end']].set_visible(True)
plt.figure(1).axes[6].set_xticks([0.1963, 0.3927, 0.589, 0.9817, 1.178, 1.374, 1.767, 1.963, 2.16, 2.553, 2.749, 2.945], ['', '', '', '', '', '', '', '', '', '', '', ''], minor=True)
plt.figure(1).axes[6].set(position=[0.197, 0.675, 0.2049, 0.2562], xticks=[0., 0.7854, 1.571, 2.356, 3.142], xticklabels=['0°', '45°', '90°', '135°', '180°'])
plt.figure(1).axes[6].spines[['start', 'end']].set_visible(True)
plt.figure(1).axes[7].set(position=[0.5872, 0.7982, 0.1695, 0.1774], xlabel='L2/3', xticks=[], xticklabels=[], ylim=(76., 102.))
plt.figure(1).axes[8].set(position=[0.7685, 0.7982, 0.2167, 0.1774], xlabel='L2/3', xticks=[], xticklabels=[])
plt.figure(1).axes[8].xaxis.labelpad = 3.387389
plt.figure(1).axes[8].get_xaxis().get_label().set(position=(0.5, 1233.), text='L2/3')
plt.figure(1).axes[9].set(position=[0.5872, 0.5913, 0.1695, 0.1774], xlabel='L4', xticks=[], xticklabels=[], ylim=(76., 102.))
plt.figure(1).axes[9].get_xaxis().get_label().set(position=(0.5, 916.6), text='L4')
plt.figure(1).axes[10].set(position=[0.7685, 0.5913, 0.2167, 0.1774], xlabel='L4', xticks=[], xticklabels=[])
plt.figure(1).axes[10].xaxis.labelpad = 4.144380
plt.figure(1).axes[10].get_xaxis().get_label().set(position=(0.4715, 1227.), text='L4')
plt.figure(1).axes[11].set(position=[0.5872, 0.3843, 0.1695, 0.1774], xlabel='L5', xticks=[], xticklabels=[], ylim=(76., 102.))
plt.figure(1).axes[12].set(position=[0.7685, 0.3843, 0.2167, 0.1774], xlabel='Posterior distribution [°]', xticks=[87., 88., 89., 90., 91., 92., 93., 94.], xticklabels=['87', '88', '89', '90', '91', '92', '93', '94'])
plt.figure(1).axes[5].set(position=[0.03547, 0.6798, 0.2049, 0.2561], yticklabels=['', '', '', '', ''], ylim=(-0.01, 0.038))
plt.figure(1).axes[6].set_yticks([-0.001, 0.002, 0.004, 0.006, 0.008, 0.012, 0.014, 0.016, 0.018, 0.022, 0.024, 0.026, 0.028, 0.032, 0.034, 0.036, 0.038], ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], minor=True)
plt.figure(1).axes[6].set(position=[0.1931, 0.6798, 0.2049, 0.2561], yticks=[-0.01, 0., 0.01, 0.02, 0.03], yticklabels=['-0.01', '0.00', '0.01', '0.02', '0.03'], ylim=(-0.01, 0.038))
plt.figure(1).ax_dict["I"].set(xlabel='Left            Right')
plt.figure(1).axes[6].legend(loc=(0.5895, 0.5694), frameon=False, borderpad=3.5, labelspacing=0.2, handlelength=1.8, handletextpad=0.6, fontsize=10.)
plt.figure(1).text(0.0197, 0.9608, '(a)', transform=plt.figure(1).transFigure, fontsize=16.)  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.0197, 0.6159, '(b)', transform=plt.figure(1).transFigure, fontsize=16.)  # id=plt.figure(1).texts[1].new
plt.figure(1).text(0.4926, 0.9608, '(c)', transform=plt.figure(1).transFigure, fontsize=16.)  # id=plt.figure(1).texts[2].new
plt.figure(1).text(0.0197, 0.3055, '(d)', transform=plt.figure(1).transFigure, fontsize=16.)  # id=plt.figure(1).texts[3].new
plt.figure(1).text(0.5517, 0.3055, '(e)', transform=plt.figure(1).transFigure, fontsize=16.)  # id=plt.figure(1).texts[4].new
plt.figure(1).text(0.8789, 0.2562, 'VC', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[5].new
plt.figure(1).axes[6].legend(loc=(0.5895, 0.5694), frameon=False, borderpad=3.5, labelspacing=0.2, handlelength=1.8, handletextpad=0.6, fontsize=10.)

plt.figure(1).ax_dict["I"].set(position=[0.07488, 0.3646, 0.197, 0.2464])
plt.figure(1).ax_dict["J"].set(position=[0.2956, 0.3646, 0.197, 0.2464])
plt.figure(1).ax_dict["L"].set(position=[0.07488, 0.0542, 0.197, 0.2464])
plt.figure(1).ax_dict["M"].set(position=[0.2956, 0.0542, 0.197, 0.2464])
plt.figure(1).ax_dict["N"].set(position=[0.603, 0.0542, 0.3783, 0.2464])
plt.figure(1).texts[5].set(position=(0.9497, 0.2735))

#plt.figure(1).ax_dict["J"].legend(loc=(0.5312, 1.027), frameon=False, borderpad=0.5, labelspacing=0.3)
plt.figure(1).ax_dict["N"].set(position=[0.5862, 0.0542, 0.398, 0.2464], xticks=[89., 90., 91., 92., 93., 94., 95., 96., 97., 98., 99.], xticklabels=['89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99'])
plt.figure(1).axes[10].xaxis.labelpad = 4.387050
plt.figure(1).axes[11].set(position=[0.5862, 0.3843, 0.1695, 0.1774])
plt.figure(1).texts[4].set(position=(0.536, 0.3055))
plt.figure(1).texts[5].set(position=(0.9458, 0.2685))
plt.figure(1).text(0.2246, 0.6800, 'R', transform=plt.figure(1).transFigure, fontsize=14.)  # id=plt.figure(1).texts[6].new
plt.figure(1).text(0.1951, 0.6800, 'L', transform=plt.figure(1).transFigure, fontsize=14.)  # id=plt.figure(1).texts[7].new

plt.figure(1).ax_dict["I"].set(xlabel='', ylim=(75., 105.))
#plt.figure(1).ax_dict["J"].legend(loc=(0.5937, 1.063), frameon=False, borderpad=0.5, labelspacing=0.3)
plt.figure(1).ax_dict["J"].set(xlim=(89.5, 91.5), ylim=(0., 10660.))
plt.figure(1).ax_dict["L"].set(xticks=[87., 88., 89., 90., 91., 92., 93.], xticklabels=['87', '88', '89', '90', '91', '92', '93'], yticks=[], yticklabels=[], ylim=(0., 10660.))
plt.figure(1).ax_dict["M"].set(xticks=[87., 88., 89., 90., 91., 92., 93.], xticklabels=['87', '88', '89', '90', '91', '92', '93'], ylim=(0., 10660.))
plt.figure(1).ax_dict["N"].set(xticks=[83., 85., 87., 89., 91., 93., 95., 97., 99., 101., 103., 105., 107.], xticklabels=['83', '85', '87', '89', '91', '93', '95', '97', '99', '101', '103', '105', '107'], xlim=(83., 107.), ylim=(0., 10660.))
plt.figure(1).axes[7].set(ylim=(75., 105.))
plt.figure(1).axes[8].set(xlim=(87.5, 92.5), ylim=(0., 10660.))
plt.figure(1).axes[9].set(ylim=(75., 105.))
plt.figure(1).axes[10].set(xlim=(87.5, 92.5), ylim=(0., 10660.))
plt.figure(1).axes[11].set(ylim=(75., 105.))
plt.figure(1).axes[12].set(position=[0.7675, 0.3843, 0.2167, 0.1774], xlim=(87.5, 92.5), ylim=(0., 10660.))
plt.figure(1).text(0.2365, 0.2587, '♂', transform=plt.figure(1).transFigure, fontsize=16.)  # id=plt.figure(1).texts[8].new
plt.figure(1).text(0.3153, 0.2587, '♀', transform=plt.figure(1).transFigure, fontsize=16.)  # id=plt.figure(1).texts[9].new

plt.figure(1).ax_dict["M"].legend(loc=(0.02839, 0.03717), frameon=False, labelspacing=0.2)
plt.figure(1).ax_dict["M"].set(xlabel='Posterior distribution [°]')
plt.figure(1).texts[6].set(position=(0.2936, 0.6553))
plt.figure(1).texts[7].set(position=(0.132, 0.6553))

plt.show()

####### calculates means for fit
path = '/Volumes/Gesine1/ACx data/4Revisions/stats/2p_ACx_randomMC_25.10000.0.05/bpnr2p_fit.csv'
df = pd.read_csv(path, delimiter=r'\s+')
# Calculate the mean and standard deviation of the second column
mean_value = df.iloc[:, 1].mean()
std_value = df.iloc[:, 1].std()
print(f"Mean: {mean_value}")
print(f"Standard Deviation: {std_value}")

# Calculate means and std for the models
path = '/Volumes/Gesine1/ACx data/4Revisions/stats/2p_ACx_randomMC_25.10000.0.05/'
L_means = np.mean(np.array(pd.read_csv(path + 'bpnr2p_layerL5.csv', header=None, delimiter=r"\s+")), axis=0)
R_means = np.mean(np.array(pd.read_csv(path + 'bpnr2p_siderlayerL5.csv', header=None, delimiter=r"\s+")), axis=0)
L_std = np.std(np.array(pd.read_csv(path + 'bpnr2p_layerL5.csv', header=None, delimiter=r"\s+")), axis=0)
R_std = np.std(np.array(pd.read_csv(path + 'bpnr2p_siderlayerL5.csv', header=None, delimiter=r"\s+")), axis=0)