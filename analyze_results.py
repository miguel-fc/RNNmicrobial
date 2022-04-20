import os
import shutil
import sys
sys.path.append("results_processing/")
from gt_vs_pd_panels import gt_vs_pd_panels
from gt_vs_pd_video import gt_vs_pd_videos
from predrnn_accuracy import predrnn_accuracy
from bacteria_populations import bacteria_populations

# --------------------------------------------------------------------------
# Parameters
RES_DIR = "results/validation_images/"
LOG_DIR = "results/logs/"
PLOTS_DIR = "results/"
POP_DIR = "results/metrics/"
LOC = "wells-1_microns-30"

show_plots = True
save_plots = True
show_videos = False
save_videos = False
num_frames = 20
jump_size = 2

layers = 32
lr = 0.0003
threshold = False
interpolation = True
interp_sizes = [20, 32, 32]
subimages = False
sub_step = 10
sub_dim = 32
hsv = True
test = False

# layer_str = "_layers-{}_lr-{:.5f}".format(layers, lr)
layer_str = "_layers-{}_lr-{}".format(layers, lr)
threshold_str = ""
if threshold:
    threshold_str = "_threshold"
    # threshold_str = "_red"
    # threshold_str = "_green"
interp_str = ""
if interpolation:
    interp_str = "_interp-{}-{}-{}".format(*interp_sizes)
sub_str = ""
if subimages:
    sub_str = "_subims-step-{}-dim-{}".format(sub_step, sub_dim)
hsv_str = ""
if hsv:
    hsv_str = "_hsv"
test_str = ""
if test:
    test_str = "-test"
LOC += interp_str + hsv_str + threshold_str + sub_str + layer_str + test_str
print(LOC)

# Create output folder
if os.path.exists(PLOTS_DIR + LOC) == True:
    shutil.rmtree(PLOTS_DIR + LOC)
os.mkdir(PLOTS_DIR + LOC)
os.mkdir(PLOTS_DIR + LOC + "/panels")
os.mkdir(PLOTS_DIR + LOC + "/videos")
# --------------------------------------------------------------------------

# Accuracy plots from log
if test == False:
    predrnn_accuracy(LOG_DIR, PLOTS_DIR, LOC, show_plots, save_plots, layer_str)
# Make panels
gt_vs_pd_panels(RES_DIR, PLOTS_DIR, LOC, num_frames, show_plots, save_plots, jump_size)
# Make videos
gt_vs_pd_videos(RES_DIR, PLOTS_DIR, LOC, num_frames, show_videos, save_videos, jump_size)
# Make population curves
# bacteria_populations(RES_DIR, POP_DIR, LOC, num_frames, show_plots, save_plots)
