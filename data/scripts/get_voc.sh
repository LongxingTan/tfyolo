#!/bin/bash
# chmod +x get_voc.sh
echo '**' `date +%H:%M:%S` 'start **'

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O ./data/voc2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -O ./data/voc2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar -O ./data/voc2007_test.tar
echo 'Downloaded'

mkdir -p ./data/voc/train ./data/voc/test
tar -xf ./data/voc2012.tar -C ./data/voc/train
tar -xf ./data/voc2007.tar -C ./data/voc/train
tar -xf ./data/voc2007_test.tar -C ./data/voc/test
echo 'Done'
