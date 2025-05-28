#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate funcog

# Set base parameters
DATA_ROOT="/mnt/sdc/samy/dataset/Ines_Abdullah/script_mc"
N_JOBS=40
TIMECOURSE_FOLDER="Timecourses_updated_03052024"
LAG=1
N_RUNS=1000
GAMMA=100

# Window sizes to test
# WINDOW_SIZES=(5 7 10 11)
WINDOW_SIZES=(9) # Uncomment this line to test only window size 9

# Make logs directory
mkdir -p logs

# Loop over window sizes
for WIN in "${WINDOW_SIZES[@]}"; do
    LOG_FILE="logs/allegiance_ws${WIN}.log"
    echo "Launching allegiance job with window_size=${WIN}, logging to $LOG_FILE"
    nohup python run_all_allegiance_local.py \
        --n_jobs $N_JOBS \
        --data_root $DATA_ROOT \
        --timecourse_folder $TIMECOURSE_FOLDER \
        --lag $LAG \
        --window_size $WIN \
        --n_runs $N_RUNS \
        --gamma $GAMMA \
        > $LOG_FILE 2>&1 &
done

# Make the script executable
# chmod +x launch_allegiance_jobs.sh

# Run the script
# ./launch_allegiance_jobs.sh
