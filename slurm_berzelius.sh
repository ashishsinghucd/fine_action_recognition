#!/bin/bash

# --- Slurm Job Configuration ---
#
# This section configures the resources requested from the Slurm scheduler.

# Job name for easy identification in the queue
#SBATCH --job-name=posec3dfinegym

# File to write standard output to. %j is the job ID.
#SBATCH --output=/proj/tinyml_htg_ltu/users/x_ashsi/Research/Logs/finegym/%j.out

# File to write standard error to.
#SBATCH --error=/proj/tinyml_htg_ltu/users/x_ashsi/Research/Logs/finegym/%j.err

# Request a single node
#SBATCH --nodes=1

# Request 3 GPUs on the node
#SBATCH --gres=gpu:3

# Request CPUs for data loading. 4 CPUs per GPU is a good starting point.
#SBATCH --cpus-per-task=12

# Request memory. 16GB per GPU is a safe bet.
#SBATCH --mem=48G

# Set a time limit for the job (e.g., 48 hours).
#SBATCH --time=48:00:00

# IMPORTANT: Specify the partition (queue) to submit to.
# This is cluster-specific. Common names are 'gpu', 'main', 'a100'.
# Check your cluster's documentation and change 'gpu_partition' accordingly.
# SBATCH --partition=gpu_partition

# --- Job Execution ---
#
# This section contains the commands that will be run when the job starts.

echo "----------------------------------------------------"
echo "Job started on: $(hostname)"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "----------------------------------------------------"

# Load necessary modules (if required by your cluster environment)
# module load anaconda3
# module load cuda/11.8

nvidia-smi

source /home/x_ashsi/miniconda3/etc/profile.d/conda.sh
conda activate openmmlab
module load buildenv-gcccuda/12.1.1-gcc12.3.0
# Activate your Conda environment
# source activate mmaction-env # Replace with your actual environment name

export PROJECT_PATH=/home/x_ashsi/Research/Projects/mmaction2

# ------------------------------------------------------------
echo "Create directory for log"
CURRENTDATE=`date + "%Y-%m-%d"`
echo $CURRENTDATE
PATHLOG="/proj/tinyml_htg_ltu/users/x_ashsi/Research/Logs/finegym/"
echo $PATHLOG
output_file="${PATHLOG}/${SLURM_JOB_ID}_logs.txt"

# The main command to start distributed training
# It uses the provided script to launch the job on 3 GPUs.
# The config file has auto_scale_lr enabled, so the learning rate
# will be automatically adjusted for the new total batch size (16 * 3 = 48).
echo "Starting MMAction2 distributed training..."

cd $PROJECT_PATH
./tools/dist_train.sh \
    configs/fine_action_recognition/posec3d/posec3d_finegym99.py \
    3 \
    --work-dir /proj/tinyml_htg_ltu/users/x_ashsi/Research/Logs/finegym/posec3d_finegym99

echo "----------------------------------------------------"
echo "Job finished at: $(date)"
echo "----------------------------------------------------"
