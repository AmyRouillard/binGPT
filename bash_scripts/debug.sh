#!/bin/bash
#PBS -N EOT_debug
#PBS -q gpu_1
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -l walltime=00:10:00
#PBS -o /mnt/lustre/users/arouillard/debug_out.log
#PBS -e /mnt/lustre/users/arouillard/debug_error.log
#PBS -P PHYS1216
#PBS -M "rouillardamy@gmail.com"
#PBS -m be

module add chpc/BIOMODULES
module load python/0.3.12.2
export LD_LIBRARY_PATH=/mnt/lustre/users/arouillard/venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

wdir=/mnt/lustre/users/arouillard
source $wdir/venv/bin/activate
python $wdir/binGPT/scripts/2_transformer.py