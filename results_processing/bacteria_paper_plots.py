# %%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcls
# plt.style.use("ggplot")
# plt.style.use("default")
from glob import glob
from colony_comparisons import *
nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif" : "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)

def set_size(width, fraction=1, square=False, ratio=None):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    if ratio == None:
        golden_ratio = (5 ** 0.5 - 1) / 2
    else:
        golden_ratio = ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if square:
        fig_height_in = fig_width_in
    else:
        fig_height_in = fig_width_in * golden_ratio

    return fig_width_in, fig_height_in

# =============================================================================
# Plots for the nufeb ABM paper:
# - Histogram of the median normed colony areas for the test runs in each case
#       compared against histograms from simulation
# - Total population of species in simulation vs predicted
# =============================================================================

# %%
# =========================
# Load data
DATA_DIR = "../../data/prnn-results/"
SAVE_DIR = "../../data/prnn-videos/"
# LOC = "wells-1_microns-30_interp-20-64-64_layers-64_lr-0.0003"
# LOC = "wells-1_microns-30_interp-20-32-32_hsv_layers-32_lr-0.0003"
LOC = "wells-1_microns-30_interp-20-32-32_hsv_layers-32_lr-0.0003-test"
num_frames = 20
# =========================

batches = []
for batch_dir in glob(DATA_DIR + LOC + "/*"):
    if "test" in batch_dir:
        batches.append(batch_dir.split("/")[-1])
    else:
        batches.append(int(batch_dir.split("/")[-1]))

batches.sort()
tests = []
for test_dir in glob(DATA_DIR + LOC + "/{}/*".format(batches[0])):
    tests.append(int(test_dir.split("/")[-1]))

tests.sort()

dir_num = batches[-1]

# %%
# Collect images
im_arrs = []
p_im_arrs = []
for test_num in tests:
    batch_dir = DATA_DIR + LOC + "/{}/{}".format(dir_num, test_num)
    img_array = []
    p_img_array = []
    gt_img = []
    pd_img = []

    for file_num in range(1,num_frames):
        gt_img.append(batch_dir + "/gt" + str(file_num) + ".png")

    for file_num in range(num_frames//2 + 1, 3*num_frames//2):
        pd_img.append(batch_dir + "/pd" + str(file_num) + ".png")

    for gt_file, pd_file in zip(gt_img, pd_img):
        if len(plt.imread(gt_file).shape) == 2:
            img_array.append(plt.imread(gt_file))
            p_img_array.append(plt.imread(pd_file))
        else:
            if "hsv" in LOC:
                img_array.append(mcls.hsv_to_rgb(plt.imread(gt_file)[:, :, ::-1]))
                p_img_array.append(mcls.hsv_to_rgb(plt.imread(pd_file)[:, :, ::-1]))
            else:
                img_array.append(plt.imread(gt_file)[:, :, ::-1])
                p_img_array.append(plt.imread(pd_file)[:, :, ::-1])
    img_array = np.array(img_array)
    p_img_array = np.array(p_img_array)

    # img_array = img_array[num_frames//2:]
    # p_img_array = p_img_array[num_frames//2:]

    im_arrs.append(img_array)
    p_im_arrs.append(p_img_array)

# %%
# Collect population information
r_sizes = []; p_r_sizes = []; g_sizes = []; p_g_sizes = [];
r_areas = []; g_areas = []; p_r_areas = []; p_g_areas = [];
r_t_sizes = []; p_r_t_sizes = []; g_t_sizes = []; p_g_t_sizes = []
for i in range(len(im_arrs)):
    # Populations
    r_sizes.append(extract_population(im_arrs[i], "red"))
    g_sizes.append(extract_population(im_arrs[i], "green"))
    r_areas.append([np.sum(extract_colony_areas(im_arrs[i][j], "red")) for j in range(0, len(im_arrs[i]))])
    g_areas.append([np.sum(extract_colony_areas(im_arrs[i][j], "green")) for j in range(0, len(im_arrs[i]))])
    r_t_sizes.append(extract_thresh_population(im_arrs[i], "red"))
    g_t_sizes.append(extract_thresh_population(im_arrs[i], "green"))
    p_r_sizes.append(extract_population(p_im_arrs[i], "red"))
    p_g_sizes.append(extract_population(p_im_arrs[i], "green"))
    p_r_areas.append([np.sum(extract_colony_areas(p_im_arrs[i][j], "red")) for j in range(0, len(p_im_arrs[i]))])
    p_g_areas.append([np.sum(extract_colony_areas(p_im_arrs[i][j], "green")) for j in range(0, len(p_im_arrs[i]))])
    p_r_t_sizes.append(extract_thresh_population(p_im_arrs[i], "red"))
    p_g_t_sizes.append(extract_thresh_population(p_im_arrs[i], "green"))

r_sizes = np.array(r_sizes)
g_sizes = np.array(g_sizes)
r_areas = np.array(r_areas)
g_areas = np.array(g_areas)
r_t_sizes = np.array(r_t_sizes)
g_t_sizes = np.array(g_t_sizes)
p_r_sizes = np.array(p_r_sizes)
p_g_sizes = np.array(p_g_sizes)
p_r_areas = np.array(p_r_areas)
p_g_areas = np.array(p_g_areas)
p_r_t_sizes = np.array(p_r_t_sizes)
p_g_t_sizes = np.array(p_g_t_sizes)

# %%
# Plot total population growth
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
if "test" in LOC:
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=set_size(470, square=False, ratio=.8))
else:
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=set_size(470, square=True))

p_mask = slice(num_frames//2, num_frames+1)
dom = range(1, num_frames)
dom2 = range(num_frames//2+1, num_frames)
lines = []
labels = ["Red", "Predicted red", "Green", "Predicted green"]
for ti in range(len(r_sizes)):
    if "test" in LOC:
        pcol = ti % 2
        prow = ti // 2
    else:
        pcol = ti % 3
        prow = ti // 3
    ax[prow, pcol].grid(True, zorder=-1)
    lines.append(ax[prow, pcol].plot(dom,r_sizes[ti], color="red", ls="--", alpha=.8)[0])# label="Red"))
    lines.append(ax[prow, pcol].plot(dom2,p_r_sizes[ti][p_mask], color="red")[0])# label="Predicted red"))
    lines.append(ax[prow, pcol].plot(dom,g_sizes[ti],color="green", ls="--", alpha=.8)[0])# label="Green"))
    lines.append(ax[prow, pcol].plot(dom2,p_g_sizes[ti][p_mask], color="green")[0])# label="Predicted green"))
    # ax[prow, pcol].scatter(dom,r_sizes[ti], color="red", zorder=10)
    ax[prow, pcol].scatter(dom2,p_r_sizes[ti][p_mask], color="red", marker="^", zorder=10)
    # ax[prow, pcol].scatter(dom,g_sizes[ti], color="green", zorder=10)
    ax[prow, pcol].scatter(dom2,p_g_sizes[ti][p_mask], color="green", marker="^", zorder=10)
    ax[prow, pcol].set_title("Test well: {}".format(ti+1), fontsize=12)
    ax[prow, pcol].xaxis.set_tick_params(labelsize=10)
    ax[prow, pcol].yaxis.set_tick_params(labelsize=10)
    # ax[prow, pcol].legend(fontsize=8)
# fig.text(0.5, -0.1, 'Prediction time', ha='center', fontsize=18)
# fig.text(-0.1, 0.5, 'Total population area', va='center', rotation='vertical', fontsize=18)
fig.legend(
        lines[:4],
        labels,
        fontsize=10,
        loc=(.085, .76)
        # loc="center right"
)
plt.tight_layout(pad=1.5)

# Share ticks and labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Prediction time", fontsize=12)
# plt.ylabel("Population", fontsize=12)
plt.ylabel("Intensity values", fontsize=12)
# fig.suptitle("Global Population comparison")
# plt.savefig("/home/connor/GDrive/SCGSR/docs/presentations/21-8-30_cfsm-group/figures/global_pop.png", transparent=True, bbox_inches="tight")
# plt.savefig("/home/connor/GDrive/SCGSR/data/papers/2021_bacteria/global_pop.png", transparent=True, bbox_inches="tight")
plt.savefig("../../data/prnn-videos/" + LOC + "/global_population.pdf")
plt.show()

# %%
# Plot total population area
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
if "test" in LOC:
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=set_size(470, square=False, ratio=.8))
else:
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=set_size(470, square=True))

p_mask = slice(num_frames//2, num_frames+1)
dom = range(1, num_frames)
dom2 = range(num_frames//2+1, num_frames)
lines = []
labels = ["Red", "Predicted red", "Green", "Predicted green"]
for ti in range(len(r_sizes)):
    if "test" in LOC:
        pcol = ti % 2
        prow = ti // 2
    else:
        pcol = ti % 3
        prow = ti // 3
    ax[prow, pcol].grid(True, zorder=-1)
    lines.append(ax[prow, pcol].plot(dom,r_areas[ti], color="red", ls="--", alpha=.8)[0])# label="Red"))
    lines.append(ax[prow, pcol].plot(dom2,p_r_areas[ti][p_mask], color="red")[0])# label="Predicted red"))
    lines.append(ax[prow, pcol].plot(dom,g_areas[ti],color="green", ls="--", alpha=.8)[0])# label="Green"))
    lines.append(ax[prow, pcol].plot(dom2,p_g_areas[ti][p_mask], color="green")[0])# label="Predicted green"))
    # ax[prow, pcol].scatter(dom,r_areas[ti], color="red", zorder=10)
    ax[prow, pcol].scatter(dom2,p_r_areas[ti][p_mask], color="red", marker="^", zorder=10)
    # ax[prow, pcol].scatter(dom,g_areas[ti], color="green", zorder=10)
    ax[prow, pcol].scatter(dom2,p_g_areas[ti][p_mask], color="green", marker="^", zorder=10)
    ax[prow, pcol].set_title("Test well: {}".format(ti+1), fontsize=12)
    ax[prow, pcol].xaxis.set_tick_params(labelsize=10)
    ax[prow, pcol].yaxis.set_tick_params(labelsize=10)
    # ax[prow, pcol].legend(fontsize=8)
# fig.text(0.5, -0.1, 'Prediction time', ha='center', fontsize=18)
# fig.text(-0.1, 0.5, 'Total population area', va='center', rotation='vertical', fontsize=18)
fig.legend(
        lines[:4],
        labels,
        fontsize=10,
        # loc=(.1, .8)
        loc=(.085, .76)
        # loc="center right"
)
plt.tight_layout(pad=1.5)

# Share ticks and labels
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Prediction time", fontsize=12)
# plt.ylabel("Total population area (pixels)", fontsize=12)
plt.ylabel("Total number of pixels", fontsize=12)
# fig.suptitle("Global Population comparison")
# plt.savefig("/home/connor/GDrive/SCGSR/docs/presentations/21-8-30_cfsm-group/figures/global_pop.png", transparent=True, bbox_inches="tight")
# plt.savefig("/home/connor/GDrive/SCGSR/data/papers/2021_bacteria/global_pop.png", transparent=True, bbox_inches="tight")
plt.savefig("../../data/prnn-videos/" + LOC + "/global_population_area.pdf")
plt.show()

# %%
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15,12))
for ti in range(len(r_sizes)):
    pcol = ti % 3
    prow = ti // 3
    ax[prow, pcol].plot(r_t_sizes[ti], color="red", label="Red", ls="--", alpha=.8)
    ax[prow, pcol].plot(p_r_t_sizes[ti], color="red", label="Predicted red")
    ax[prow, pcol].plot(g_t_sizes[ti], color="green", label="Green", ls="--", alpha=.8)
    ax[prow, pcol].plot(p_g_t_sizes[ti], color="green", label="Predicted green")
    ax[prow, pcol].set_title("Test well: {}".format(ti+1))
    ax[prow, pcol].legend(fontsize=12)
fig.text(0.5, 0.00, 'Prediction time', ha='center', fontsize=14)
fig.text(0.00, 0.5, 'Total population area', va='center', rotation='vertical', fontsize=14)
fig.suptitle("Global Population comparison (thresholded)")
plt.show()

# %%
# Track colonies and measure their size
r_labels = []; g_labels = []; p_r_labels = []; p_g_labels = []
r_areas = []; g_areas = []; p_r_areas = []; p_g_areas = []
for i in range(len(tests)):
# for i in [2]:
    # Colony labels
    r_labels.append(track_colonies(im_arrs[i], "red"))
    g_labels.append(track_colonies(im_arrs[i], "green"))
    p_r_labels.append(track_colonies(p_im_arrs[i], "red"))
    p_g_labels.append(track_colonies(p_im_arrs[i], "green"))

    # Number of initial colonies
    num_r_cols = len(np.unique(r_labels[i])) - 1
    num_g_cols = len(np.unique(g_labels[i])) - 1
    num_p_r_cols = len(np.unique(p_r_labels[i])) - 1
    num_p_g_cols = len(np.unique(p_g_labels[i])) - 1

    # Areas
    temp_r_areas = np.zeros((num_r_cols, r_labels[i].shape[0]))
    temp_g_areas = np.zeros((num_g_cols, r_labels[i].shape[0]))
    temp_p_r_areas = np.zeros((num_p_r_cols, r_labels[i].shape[0]))
    temp_p_g_areas = np.zeros((num_p_g_cols, r_labels[i].shape[0]))
    for ti in range(r_labels[i].shape[0]):
        for ci, val in enumerate(np.unique(r_labels[i])[1:]):
            temp_r_areas[ci, ti] = np.sum(r_labels[i][ti] == val)
        for ci, val in enumerate(np.unique(g_labels[i])[1:]):
            temp_g_areas[ci, ti] = np.sum(g_labels[i][ti] == val)
        for ci, val in enumerate(np.unique(p_r_labels[i])[1:]):
            temp_p_r_areas[ci, ti] = np.sum(p_r_labels[i][ti] == val)
        for ci, val in enumerate(np.unique(p_g_labels[i])[1:]):
            temp_p_g_areas[ci, ti] = np.sum(p_g_labels[i][ti] == val)
    # Remove any negligible colonies
    r_areas.append(temp_r_areas)
    g_areas.append(temp_g_areas)
    p_r_areas.append(temp_p_r_areas)
    p_g_areas.append(temp_p_g_areas)


# %%
# Plot average colony areas
if "test" in LOC:
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=set_size(470, square=False, ratio=.8))
else:
    fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=set_size(470, square=True))

p_mask = slice(num_frames//2, num_frames+1)
lines = []
labels = ["Red", "Predicted red", "Green", "Predicted green"]
for ti in range(len(r_areas)):
    if "test" in LOC:
        pcol = ti % 2
        prow = ti // 2
    else:
        pcol = ti % 3
        prow = ti // 3
    lines.append(ax[prow, pcol].plot(dom,np.mean(r_areas[ti], axis=0), color="red",ls="--", alpha=.8)[0])
    lines.append(ax[prow, pcol].plot(dom2,np.mean(p_r_areas[ti], axis=0)[p_mask], color="red")[0])
    lines.append(ax[prow, pcol].plot(dom,np.mean(g_areas[ti], axis=0), color="green", ls="--", alpha=.8)[0])
    lines.append(ax[prow, pcol].plot(dom2,np.mean(p_g_areas[ti], axis=0)[p_mask], color="green")[0])
    # ax[prow, pcol].scatter(dom,np.mean(r_areas[ti], axis=0), color="red", zorder=10)
    ax[prow, pcol].scatter(dom2,np.mean(p_r_areas[ti], axis=0)[p_mask], color="red", marker="^", zorder=10)
    # ax[prow, pcol].scatter(dom,np.mean(g_areas[ti], axis=0), color="green", zorder=10)
    ax[prow, pcol].scatter(dom2,np.mean(p_g_areas[ti], axis=0)[p_mask], color="green", marker="^", zorder=10)
    ax[prow, pcol].set_title("Test well: {}".format(ti+1), fontsize=12)
    ax[prow, pcol].xaxis.set_tick_params(labelsize=10)
    ax[prow, pcol].yaxis.set_tick_params(labelsize=10)
    ax[prow, pcol].grid(True)
    # ax[prow, pcol].legend(fontsize=14)
# fig.text(0.5, 0.00, 'Prediction time', ha='center', fontsize=18)
# fig.text(0.00, 0.5, 'Average colony area', va='center', rotation='vertical', fontsize=18)
fig.legend(
        lines[:4],
        labels,
        fontsize=10,
        # loc=(.72, .8)
        loc=(.74, .76)
        # loc="center right"
)
plt.tight_layout(pad=1.5)
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Prediction time", fontsize=12)
plt.ylabel("Average colony area (pixels)", fontsize=12)
# fig.suptitle("Colony Population comparison")
# plt.savefig("/home/connor/GDrive/SCGSR/docs/presentations/21-8-30_cfsm-group/figures/ave_colony_area.png", transparent=True, bbox_inches="tight")
# plt.savefig("/home/connor/GDrive/SCGSR/data/papers/2021_bacteria/ave_colony_area.png", transparent=True, bbox_inches="tight")
plt.savefig("../../data/prnn-videos/" + LOC + "/colony_population.pdf")
plt.show()

# %%
# Plot colony number vs area in scatter plot
for ti in range(len(r_areas)):
    if "test" in LOC:
        fig, ax = plt.subplots(1, 3, sharex=False, sharey=True, figsize=set_size(470, square=False, ratio=.33))
    else:
        fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=set_size(470, square=True))

    lines = []
    labels = ["True red", "Predicted red", "True green", "Predicted green"]
    for i, frame in enumerate([10, 14, 18]):
        lines.append(ax[i].scatter([r_areas[ti].shape[0]], np.mean(r_areas[ti][:, frame]), color="red", zorder=10))
        lines.append(ax[i].scatter([p_r_areas[ti].shape[0]], np.mean(p_r_areas[ti][:, frame]), marker="^", color="red", zorder=10))
        lines.append(ax[i].scatter([g_areas[ti].shape[0]], np.mean(g_areas[ti][:, frame]), color="green", zorder=10))
        lines.append(ax[i].scatter([p_g_areas[ti].shape[0]], np.mean(p_g_areas[ti][:, frame]), marker="^", color="green", zorder=10))
        ax[i].set_title("Time: {}".format(frame+1), fontsize=12)
        ax[i].xaxis.set_tick_params(labelsize=10)
        ax[i].yaxis.set_tick_params(labelsize=10)
        ax[i].set_xticks(range(1, 7), range(1,7))
        ax[i].grid(True)

    fig.legend(
            lines[:4],
            labels,
            fontsize=10,
            # loc=(.72, .8)
            loc=(.72, .42)
            # loc="center right"
    )
    plt.tight_layout(pad=1.5)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("Number of colonies", fontsize=12)
    plt.ylabel("Average pixel area", fontsize=12)
    plt.suptitle("Test well {}".format(ti+1)) 
    plt.savefig("../../data/prnn-videos/" + LOC + "/colony_scatter" + str(ti) + ".pdf")

# plt.xlabel("Prediction time", fontsize=12)
# plt.ylabel("Average colony area (pixels)", fontsize=12)
# plt.savefig("../../data/prnn-videos/" + LOC + "/colony_scatter.pdf")
# plt.show()


# %%
# Plot each colony area
fig, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15,12))

for batch in range(len(r_areas)):
    pcol = batch % 3
    prow = batch // 3
    for ci in range(r_areas[batch].shape[0]):
        ax[prow, pcol].plot(r_areas[batch][ci], color="red", label="Red {}".format(ci+1), ls="--", alpha=.8)
    for ci in range(p_r_areas[batch].shape[0]):
        ax[prow, pcol].plot(p_r_areas[batch][ci], color="red", label="Predicted Red {}".format(ci+1))
    for ci in range(g_areas[batch].shape[0]):
        ax[prow, pcol].plot(g_areas[batch][ci], color="green", label="Green {}".format(ci+1), ls="--", alpha=.8)
    for ci in range(p_g_areas[batch].shape[0]):
        ax[prow, pcol].plot(p_g_areas[batch][ci], color="green", label="Predicted Green {}".format(ci+1))
    ax[prow, pcol].set_title("Batch: {}".format(batch+1))
    ax[prow, pcol].legend(fontsize=6)
fig.text(0.5, 0.00, 'Prediction time', ha='center', fontsize=14)
fig.text(0.00, 0.5, 'Average colony area', va='center', rotation='vertical', fontsize=14)
fig.suptitle("Colony Population comparison")
# plt.tight_layout()
# plt.savefig("/home/connor/GDrive/SCGSR/docs/presentations/21-8-30_cfsm-group/figures/ave_colony_area.png", transparent=True, bbox_inches="tight")
plt.show()


# %%
mse = []
lpips = []

