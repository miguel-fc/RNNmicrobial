export CUDA_VISIBLE_DEVICES=0

# Neural net parameters
export LAYERS="32"
export LR="0.0003"

# Directories
export LOGDIR=$PWD"/results/logs/"
export DATADIR=$PWD"/datasets/"
export CHECKDIR=$PWD"/results/checkpoints/"
export RESDIR=$PWD"/results/validation_images/"

# Dataset
export LOC="wells-1_microns-30_interp-20-32-32_hsv"

echo Data from $DATADIR$LOC
echo Logs at $LOGDIR$LOC.log
echo Results at $DATRESDIR$LOC"_layers-"$LAYERS"_lr-"$LR
echo Model checkpoints at $CHECKDIR$LOC"_layers-"$LAYERS"_lr-"$LR
cd ..
(python -u /home/connor/GDrive/SCGSR/connor_to_miguel/predrnn-pytorch/run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths $DATADIR$LOC-train.npz \
    --valid_data_paths $DATADIR$LOC-valid.npz \
    --save_dir $CHECKDIR$LOC"_layers-"$LAYERS"_lr-"$LR \
    --gen_frm_dir $RESDIR$LOC"_layers-"$LAYERS"_lr-"$LR \
    --model_name predrnn \
    --reverse_input 1 \
    --img_width 32 \
    --img_channel 3 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden $LAYERS,$LAYERS,$LAYERS,$LAYERS \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --lr $LR \
    --batch_size 8 \
    --max_iterations 500 \
    --display_interval 100 \
    --test_interval 100 \
    --snapshot_interval 100 \
) 2>&1 | tee $LOGDIR$LOC"_layers-"$LAYERS"_lr-"$LR.log
