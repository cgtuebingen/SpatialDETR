#!/bin/bash

# this script should run inside the docker container
echo "installing opencv with cuda support "
MMCV_WITH_OPS=1 pip install -e /deps/mmcv/

echo "installing mmdetection3d"
pip install -e /workspace/mmdetection3d

echo "installing DETR3d"
pip install -e /workspace/detr3d/

echo "installing SpatialDETR"
pip install -e /workspace

echo "DONE"