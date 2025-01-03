#!/bin/bash
#SBATCH --job-name=ibb_face             # Job name
#SBATCH --partition=gpu                   # Partition (queue) to submit to
#SBATCH --gres=gpu:1                      # Request 1 GPU (full GPU)
#SBATCH --ntasks=1                        # Run a single task (1 instance of your program)
#SBATCH --cpus-per-task=16                 # Number of CPU cores per task (adjust based on your needs)
#SBATCH --mem=64G                         # Total memory (RAM) for the job (adjust based on your dataset)
#SBATCH --time=32:00:00                    # Time limit (24 hours)
#SBATCH --output=train_camilenet%j.log               # Standard output and error log (%j is replaced by job ID)

export WANDB_API_KEY=APIDATAKEY

# Run your Python training script
python maml_anil/train_camilenet.py --ways 10 --shots 5 --run_str camilenet