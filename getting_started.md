
## Setup
## Repository
1. Clone the repository `git clone https://github.com/cgtuebingen/SpatialDETR`
2. Clone mmdetection3d@1.0.0rc1 `git clone --branch v1.0.0rc1 https://github.com/open-mmlab/mmdetection3d.git mmdetection3d`
3. Clone fork of DETR3D `git clone https://github.com/SimonDoll/detr3d.git detr3d`

## Environment
This repository was tested with
- `Python>=3.7`
- `Cuda>=10.2 <=11.3`
- `Pytorch>=1.9`
- `Docker` with GPU-support (for the Docker version only)  

but should work with other versions as well.

### Docker Setup
To ease the setup of this project we provide a docker container and some convenience scripts (see `docker`). To setup use:
- (if not done alread) setup [Docker](https://docs.docker.com/engine/install/ubuntu/) and [Nvidia-Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
- `./docker/build.sh` to build the container
- Adapt and run `./docker/run_command.sh` by changing the TODO's to start and link data and repo to the container. This will result in a shell inside the container

### In Container Setup
Inside the docker container install the packages in dev mode to allow for changes.  
You can use the script `./docker/in_docker_setup.sh` or run the commands below.  
1. `MMCV_WITH_OPS=1 pip install -e /deps/mmcv/`
2. `pip install -e /workspace/mmdetection3d` the mmdetection3d base on the `v1.0.0rc1` tag (see the mmdet3d rc1.0.0 changelog for important information on coordinate system refactorings).
3. `pip install -e detr3d/` to install DETR3D. 
4. `pip install -e .` install our plugin (SpatialDETR) as extension to DETR3D / mmdetection3d

### Alternative: Custom Setup
Simply setup mmdetection3d on tag `v1.0.0rc1` and all its deps such as:
1. mmcv (https://github.com/open-mmlab/mmcv)
2. mmdet (https://github.com/open-mmlab/mmdetection)
3. mmseg (https://github.com/open-mmlab/mmsegmentation)
4. Fork of DETR3D (https://github.com/SimonDoll/detr3d.git)
5. `pip install -e ,` install our plugin (SpatialDETR) as extension to DETR3D / mmdetection3d

### Data
1. Follow the [mmdetection3d instructions](https://mmdetection3d.readthedocs.io/en/v1.0.0rc1/datasets/nuscenes_det.html) to preprocess the data of the nuScenes dataset.

### Train
1. As for DETR3D download the [pretrained weights](https://drive.google.com/drive/folders/1h5bDg7Oh9hKvkFL-dRhu5-ahrEp2lRNN?usp=sharing) of FCOS3D to `pretrained/` or retrain FCOS3D with the new coordinate conventions using mmdetection3d.
2. Use the configs in the `configs/`folder to train SpatialDETR.  
For a basline on a single gpu use:

`python ./mmdetection3d/tools/train.py configs/submission/frozen_4/query_proj_value_proj.py`  
  or for multi-gpu e.g. 4 gpus:  
`./mmdetection3d/tools/dist_train.sh configs/submission/frozen_4/query_proj_value_proj.py 4`

3. To test, use  
`./mmdetection3d/tools/dist_test.sh configs/submission/frozen_4/query_proj_value_proj.py /path/to/.pth 4 --eval=bbox`

 