#!/bin/bash
# chmod +x run_final.sh
echo '**' `date +%H:%M:%S` 'start **'

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O ./data/voc2012.tar
mkdir -p ./data/voc2012
tar -xf ./data/voc2012.tar -C ./data/voc2012
ls ./data/voc2012/VOCdevkit/VOC2012 # Explore the dataset
