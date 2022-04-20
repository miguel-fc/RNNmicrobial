# The goal of this file is to be able to run all the data processing elements
# - Converting the right .tiffs to .npy arrays (or .pngs, .mp4s, etc)
# - Transforming the raw data with rotations, blurs, etc.
# - Generate a predRNN usable datasets

# %%
# =============================================================================
# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from glob import glob
import sys
sys.path.append("data_processing")
import transforms
import sampling
import formatting
import subprocess

# %%
# =============================================================================
# =============================================================================
# =============================================================================

# Parameters

# =============================================================================
# Dataset generation
# =============================================================================
DATA_DIR = "data/"
DATASET_DIR = "datasets/"
MICRONS = 30
show_samples = 10 # Show randomly sampled images

# Cut the later frames of the data to stay away from a full lawn at prediction
# cut_size is the number of frames to keep
data_cut = True
cut_size = 7

# Interpolate to add frames and/or add pixels
interpolation = True
interp_size = (20, 32, 32)

# Other transforms
threshold = False
flips = True
rotates = True
gauss_blur = True
gauss_noise = True
poisson_noise = False

# Split images into subimages?
subimages = False
sub_step = 10
sub_dim = 32

# Spit out HSV values?
hsv = True

# Train, valid, test split
train_split = .8
valid_split = .2
test_split = 0.0
# Test runs
test_runs = [42, 26, 3, 15, 21]

# Save string
cut_str = ""
if data_cut:
    cut_str = "_cut-{}".format(cut_size)
interp_str = ""
if interpolation:
    interp_str = "_interp-{}-{}-{}".format(*interp_size)
threshold_str = ""
if threshold:
    threshold_str = "_threshold"
    # threshold_str = "_red"
    # threshold_str = "_green"
sub_str = ""
if subimages:
    sub_str = "_subims-step-{}-dim-{}".format(sub_step, sub_dim)
hsv_str = ""
if hsv:
    hsv_str = "_hsv"
DESC_STR = "wells-1_microns-{}".format(MICRONS) + interp_str + threshold_str + sub_str + hsv_str
# =============================================================================
# =============================================================================
# =============================================================================

# %%
# Load the data
ims = []
test_ims = []
print("Loading data from: ", DATA_DIR, " with wells of size ", MICRONS, " microns")
print("-------------------------")
for file in glob(DATA_DIR + "wells-1_microns-{}*.npy".format(MICRONS)):
    temp_ims = np.load(file)
    # Brighten images
    temp_ims *= 1/temp_ims.max()
    # for ti in temp_ims:
    #     plt.imshow(ti)
    #     plt.show()
    run_num = int(file.split(".")[-2].split("w")[-1])
    if run_num in test_runs:
        test_ims.append(temp_ims)
    else:
        ims.append(temp_ims)
ims = np.stack(ims)
test_ims = np.stack(test_ims)
# Data shape is now: (num_wells, num_frames, pixel height, pixel width, num channels)

#-----------------------------------------------------------------------------
# Transforming the data
#-----------------------------------------------------------------------------
if data_cut:
    ims = ims[:, :cut_size, ...]
    test_ims = test_ims[:, :cut_size, ...]

if interpolation:
    print("Interpolating images to make stacks of size: ", interp_size)
    new_ims = np.zeros((ims.shape[0], *interp_size, *ims.shape[len(interp_size)+1:]))
    for i in range(ims.shape[0]):
        new_ims[i] = transforms.interpolate_stack(ims[i], interp_size)
    ims = new_ims
    new_ims = "" # Hack to free the memory of new_ims
    if len(test_ims) > 0:
        new_ims = np.zeros((test_ims.shape[0], *interp_size, *test_ims.shape[len(interp_size)+1:]))
        for i in range(test_ims.shape[0]):
            new_ims[i] = transforms.interpolate_stack(test_ims[i], interp_size)
        test_ims = new_ims
    new_ims = "" # Hack to free the memory of new_ims

if threshold:
    print("Thresholding images")
    for i in range(ims.shape[0]):
        ims[i] = transforms.threshold_stack(ims[i])
    # ims = ims[:, :, :, :, 0] # For red only
    # ims = ims[:, :, :, :, 1] # For green only
    for i in range(test_ims.shape[0]):
        test_ims[i] = transforms.threshold_stack(test_ims[i])

if hsv:
    print("Transform to HSV")
    for i in range(ims.shape[0]):
        ims[i] = transforms.hsv_stack(ims[i])
    for i in range(test_ims.shape[0]):
        test_ims[i] = transforms.hsv_stack(test_ims[i])

# Flips, blurs, rotates, etc.
final_ims = [np.concatenate(ims)]
if flips:
    print("Flipping images")
    for i in range(ims.shape[0]):
        for flip in ["x", "y"]:
            final_ims.append(transforms.flip_stack(ims[i], direction=flip))
if rotates:
    print("Rotating images")
    for i in range(ims.shape[0]):
        for theta in [90, 180, 270]:
            final_ims.append(transforms.rotate_stack(ims[i], theta=theta))
if gauss_blur:
    print("Gaussian blurring images")
    for i in range(ims.shape[0]):
        final_ims.append(transforms.gaussian_blur_stack(ims[i]))
if gauss_noise:
    print("Gaussian noising images")
    for i in range(ims.shape[0]):
        final_ims.append(transforms.gaussian_noise_stack(ims[i]))
if poisson_noise:
    print("Poisson noising images")
    for i in range(ims.shape[0]):
        final_ims.append(transforms.poisson_noise_stack(ims[i]))


batch_size = ims.shape[1]
ims = np.concatenate(final_ims)
test_ims = np.concatenate(test_ims)

# Subimage split
if subimages:
    print("-------------------------")
    print("Splitting images into subimages of size {}x{} with steps of size {} between their centers".format(sub_dim, sub_dim, sub_step))
    ims = sampling.get_subimages(ims, sub_step, sub_dim, batch_size)
else:
    ims = sampling.get_batches(ims, batch_size)
    test_ims = sampling.get_batches(test_ims, batch_size)
# Data shape is now: (num_batches, num_frames, pixel height, pixel width, num channels)

# Shuffle dataset
rng = np.random.default_rng(seed=1)
rng.shuffle(ims)

# Show samples
for _ in range(show_samples):
    batch = rng.integers(0, ims.shape[0])
    frame = rng.integers(0, ims.shape[1])
    if hsv:
        plt.imshow(hsv_to_rgb(ims[batch, frame]))
    else:
        plt.imshow(ims[batch, frame])
    plt.title("Batch: {}, frame: {}".format(batch, frame))
    plt.show()

#-----------------------------------------------------------------------------
# Generating a predRNN dataset
if len(ims.shape) == 4:
    ims = ims[..., np.newaxis]
train_ims, valid_ims, _, train_clips, valid_clips, _, dims = \
    formatting.predrnn_format(ims, train_split, valid_split, test_split, batch_size)
test_ims, _, _, test_clips, _, _, dims = \
    formatting.predrnn_format(test_ims, 1, 0, 0, batch_size)
print("-------------------------")
print("Train size: ", train_ims.shape)
print("Valid size: ", valid_ims.shape)
print("Test size: ", test_ims.shape)

# Saving the results
print("-------------------------")
print("Files saved: ")
train_str = DATASET_DIR + DESC_STR + "-train.npz"
valid_str = DATASET_DIR + DESC_STR + "-valid.npz"
test_str = DATASET_DIR + DESC_STR + "-test.npz"
print(train_str)
print(valid_str)
print(test_str)

np.savez(train_str, clips=train_clips, dims=dims, input_raw_data=train_ims)
np.savez(valid_str, clips=valid_clips, dims=dims, input_raw_data=valid_ims)
np.savez(test_str, clips=test_clips, dims=dims, input_raw_data=test_ims)

