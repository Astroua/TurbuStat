#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --mem=512000MB
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --job-name=turbustat_scaling
#SBATCH --output=turbustat_scaling-%J.out
#SBATCH --error=turbustat_scaling-%J.err
#SBATCH --account=def-eros-ab

export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE

module restore my_default

# Use version loaded while building pyfftw
module load fftw-mpi/3.3.6

source /home/ekoch/.bashrc
source /home/ekoch/preload.bash

export scratch_path=/home/ekoch/scratch/turbustat_scaling
export project_path=/home/ekoch/projects/rrg-eros-ab/ekoch/

# Ensure the most recent version of TurbuStat is installed
cd $HOME/code/TurbuStat_ewk_fork
$HOME/anaconda3/bin/python setup.py install

cd $scratch_path

# Call script with number of cores
$HOME/anaconda3/bin/python $HOME/code/TurbuStat_ewk_fork/Examples/scaling_tests.py

# Copy the output files to the project path
cp $scratch_path/*.txt $project_path/turbustat_scaling/
