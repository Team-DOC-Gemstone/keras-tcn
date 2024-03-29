#!/bin/bash

#SBATCH --job-name=keras-tcn-gems497
#SBATCH --output=output-keras-tcn-gems497.txt
#SBATCH --error=err-keras-tcn-gems497.err
#SBATCH --time=10:00:00
#SBATCH --mem=31gb
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --account=class
#SBATCH --partition=class

cd /fs/classhomes/spring2024/gems497/ge497000/keras-tcn/tasks/mnist_pixel
source /fs/class-projects/spring2024/gems497/ge497g00/team-doc-env/bin/activate
module add cuda/11.8.0 cudnn/v8.8.0
srun bash -c "python3 main.py" &
wait
