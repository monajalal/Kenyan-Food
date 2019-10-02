#!/bin/bash -l
/usr/local/etc/distro

module load python3/3.6.5
module load gcc/5.5.0
module load cuda/9.2
module load pytorch/1.0
python food_detection_fold0_Resnet101.py
