#!/bin/bash -l

# Request 4 cores. This will set NSLOTS=4
#$ -pe omp 4
# Request 1 GPU
#$ -l gpus=0.25
# Request at least compute capability 3.5
#$ -l gpu_c=3.5
#$ -o output_resnet101
#$ -l h_rt=48:00:00
#$ -m ea
#$ -j y
# Give the job a name
#$ -N resnet101

scc-singularity exec /share/singularity/images/scc-centos7.img /projectnb/cs585/kai/food_detection/tf_experiments/resnet101_sub_singularity0.sh 


