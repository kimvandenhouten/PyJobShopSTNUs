#!/bin/bash

#SBATCH --job-name="evaluation_pipeline"  # Job name
#SBATCH --time=05:00:00            # Set an estimated time for your job (5 hours)
#SBATCH --ntasks=1                 # Using 1 task since this is not an MPI-parallel job
#SBATCH --cpus-per-task=2         # Number of CPUs per task (adjust based on your job requirements)
#SBATCH --partition=compute        # Default partition to use
#SBATCH --mem-per-cpu=3968MB                 # Adjust memory based on the dataset and Hugging Face model requirements
#SBATCH --account=Education-EEMCS-BSc-TI       # Set your account here

# Load necessary modules
module load 2024r1
module load python/3.10.12          # Loads a compatible Python version
module load py-numpy               # Load numpy for array operations
module load cuda/11.6

# Optionally, create a virtual environment (if not done yet)
# You can skip this if your environment is already set up
# python3 -m venv venv
source /scratch/ashiamishis/rp/PyJobShopSTNUs/venv/bin/activate
export PYTHONPATH=/scratch/ashiamishis/rp/PyJobShopSTNUs:$PYTHONPATH
export JAVA_HOME=$HOME/jdk/jdk-21
export PATH=$JAVA_HOME/bin:$PATH

# Upgrade pip and install necessary Python libraries
# pip install --upgrade pip
# pip install -r requirements.txt

# Run the Python script
srun python PyJobShopIntegration/pyjobshop_pipeline.py mmrcpspd > output.log
