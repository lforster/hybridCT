#!/bin/bash


for t in n705 n60 n456 n261 nadir 261 456 60 705 multiangle; do
  python pyct/train_cloud.py -y /scratch/3D-RT_LES-clouds/nlahaye/111226/config_${t}_mimo_sweep.yaml --prepare-data
done

