import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from glob import glob
nice_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif" : "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)

def set_size(width, fraction=1):
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
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    return fig_width_in, fig_height_in

def predrnn_accuracy(DATA_DIR, SAVE_DIR, LOC, show_plots, save_plots, layer_str):
    LOCS = [LOC] # This is legacy to be able to compare various layers and lrs

    all_mse = []
    all_ssim = []
    all_psnr = []
    all_lpips = []
    all_tr = []
    all_tr_itr = []
    all_te_itr = []

    all_frame_mse = []
    all_frame_ssim = []
    all_frame_psnr = []
    all_frame_lpips = []

    print("Prediction accuracy (mse, etc.): ")
    print("------------------------------")
    for LOC in LOCS:
        BASE = DATA_DIR + LOC + ".log"
        print("Read: ", BASE)
        mse = []
        ssim = []
        psnr = []
        lpips = []

        training_loss = []
        training_itr = []
        test_itr = []
        params = ""
        active = 0
        frame_mse = []
        frame_ssim = []
        frame_psnr = []
        frame_lpips = []

        with open(BASE, "r") as file:
            for line in file.readlines()[12:]:
                if "Namespace" in line:
                    params += line
                if "training loss" in line:
                    training_loss.append(float(line.split(" ")[-1].strip()))
                if "itr" in line:
                    training_itr.append(int(line.split(" ")[-1].strip()))
                if "test" in line:
                    test_itr.append(training_itr[-1])
                if "mse per seq" in line:
                    mse.append(float(line.split(" ")[-1].strip()))
                    active = 1
                if "ssim per frame" in line:
                    ssim.append(float(line.split(" ")[-1].strip()))
                    active = 2
                if "psnr per frame" in line:
                    psnr.append(float(line.split(" ")[-1].strip()))
                    active = 3
                if "lpips per frame" in line:
                    lpips.append(float(line.split(" ")[-1].strip()))
                    active = 4
                if "itr" in line:
                    active = 0
                    frame_mse = []
                    frame_ssim = []
                    frame_psnr = []
                    frame_lpips = []
                if line[:3] not in "mse"+"ssim"+"psnr"+"lpips":
                    collect_frames = True
                else:
                    collect_frames = False
                if collect_frames & (active == 1):
                    frame_mse.append(float(line.strip()))
                elif collect_frames & (active == 2):
                    frame_ssim.append(float(line.strip()))
                elif collect_frames & (active == 3):
                    frame_psnr.append(float(line.strip()))
                elif collect_frames & (active == 4):
                    frame_lpips.append(float(line.strip()))

        mse = np.array(mse).T
        ssim = np.array(ssim).T
        psnr = np.array(psnr).T
        lpips = np.array(lpips).T
        tr_loss = np.array(training_loss).T
        tr_itr = np.array(training_itr).T
        test_itr = np.array(test_itr).T

        frame_mse = np.array(frame_mse).T
        frame_ssim = np.array(frame_ssim).T
        frame_psnr = np.array(frame_psnr).T
        frame_lpips = np.array(frame_lpips).T

        all_mse.append(mse)
        all_ssim.append(ssim)
        all_psnr.append(psnr)
        all_lpips.append(lpips)
        if "test" in LOC:
            all_tr.append(np.zeros(1))
            all_tr_itr.append(np.zeros(1))
            all_te_itr.append(np.zeros(1))
        else:
            all_tr.append(tr_loss)
            all_tr_itr.append(tr_itr)
            all_te_itr.append(test_itr)
        all_frame_mse.append(frame_mse)
        all_frame_ssim.append(frame_ssim)
        all_frame_psnr.append(frame_psnr)
        all_frame_lpips.append(frame_lpips)

    # Save parameters
    params = [p.strip() for p in params.split(",")]
    max_iterations = 0
    test_interval = 0
    for p in params:
        if "max_iterations" in p:
            max_iterations = int(p.split("=")[-1])
        if "test_interval" in p:
            test_interval = int(p.split("=")[-1])
    # Plots
    def make_plots(all_items, all_xticks, ylabel, xlabel, save_name, dense_ticks=False, grid=True, scatter=False, nsubplots=3):
        # plt.figure(figsize=(8, 6))
        plt.figure(figsize=set_size(470/nsubplots))
        colors = cm.get_cmap("Set1").colors
        if len(all_items) == 1:
            # plt.scatter(all_xticks[0], all_items[0])
            if dense_ticks:
                plt.xticks(np.array(all_xticks[0])+11, np.array(all_xticks[0])+11)
            plt.plot(np.array(all_xticks[0])+11, all_items[0], color=colors[0], lw=3, label=layer_str)
            if grid:
                plt.grid()
            if scatter:
                plt.scatter(np.array(all_xticks[0])+11, all_items[0], color=colors[0])
        else:
            for i, sub_item in enumerate(all_items):
                plt.plot(np.array(all_xticks[i])+11, sub_item, label=layer_str)
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        # plt.legend()
        # plt.title(LOC[:-len(layer_str)])
        if save_plots:
            save_loc = SAVE_DIR + LOC + "/{}.pdf".format(save_name)
            print("Save: {}".format(save_loc))
            plt.tight_layout(pad=0.2)
            plt.savefig(save_loc)
        if show_plots:
            plt.show()
        else:
            plt.close()

    make_plots(all_mse, all_te_itr, "MSE", "Epoch", "mse")
    make_plots(all_ssim, all_te_itr, "SSIM", "Epoch", "ssim")
    make_plots(all_psnr, all_te_itr, "PSNR", "Epoch", "psnr")
    make_plots(all_lpips, all_te_itr, "LPIPS", "Epoch", "lpips")
    make_plots(all_tr, all_tr_itr, "Training Loss", "Epoch", "training_loss")

    make_plots(all_frame_mse, [range(len(frame_mse))], "MSE", "Time", "frame_mse", True, True, True, 2)
    make_plots(all_frame_ssim, [range(len(frame_ssim))], "SSIM", "Time", "frame_ssim", True, True, True, 2)
    make_plots(all_frame_psnr, [range(len(frame_psnr))], "PSNR", "Time", "frame_psnr", True, True, True, 2)
    make_plots(all_frame_lpips, [range(len(frame_lpips))], "LPIPS", "Time", "frame_lpips", True, True, True, 2)
