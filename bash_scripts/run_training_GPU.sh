#!/bin/bash
#PBS -N EOT
#PBS -q serial
#PBS -l select=1:ncpus=24
#PBS -l walltime=10:00:00
#PBS -o /mnt/lustre/users/arouillard/results/qft_24/reports/output_qft_n24_j0.log
#PBS -e /mnt/lustre/users/arouillard/results/qft_24/reports/error_qft_n24_j0.log
#PBS -P PHYS1216
#PBS -M "rouillardamy@gmail.com"
#PBS -m be

module add chpc/BIOMODULES
module load python/0.3.12.2
export LD_LIBRARY_PATH=/mnt/lustre/users/arouillard/venv/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
wdir=/mnt/lustre/users/arouillard
source $wdir/venv/bin/activate
python $wdir/binGPT/scripts/2_transformer.py