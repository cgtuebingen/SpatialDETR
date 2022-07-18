
# path to directory where nusenes data is stored
nusc_data_dir="TODO"
# path to this repository root
repo_dir="TODO"
# path to directory where models / logs shall be stored in
exp_dir="TODO"

docker run \
--gpus all --shm-size=16g \
--mount source=$repo_dir,target=/workspace,type=bind,consistency=cached \
--mount source=$nusc_data_dir,target=/workspace/data/nuscenes,type=bind,consistency=cached \
--mount source=$exp_dir,target=/workspace/work_dirs,type=bind,consistency=cached \
-it \
--name=spatial_detr_release \
spatial_detr
