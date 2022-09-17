# SpatialDETR: Robust Scalable Transformer-Based 3D Object Detection from Multi-View Camera Images with Global Cross-Sensor Attention (ECCV 2022)

![](img/overview.png "Overview")  

This is the official repository for [SpatialDETR](https://uni-tuebingen.de/securedl/sdl-eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE2NjMwODI0NjQsImV4cCI6MTY2MzE3MjQ2MSwidXNlciI6MCwiZ3JvdXBzIjpbMCwtMV0sImZpbGUiOiJmaWxlYWRtaW5cL1VuaV9UdWViaW5nZW5cL0Zha3VsdGFldGVuXC9JbmZvS29nbmlcL1dTSVwvQ29tR3JhcGhcL3B1YmxpY2F0aW9uc1wvMjAyMlwvRUNDVjIwMjItU3BhdGlhbERldHJcL0VDQ1YyMDIyLXNwYXRpYWxfZGV0ci5wZGYiLCJwYWdlIjoyMzczOTl9.iLxLzTD_7W1HiUOYb1RdNqR0o1FPIPXW-JYvC-__st8/ECCV2022-spatial_detr.pdf) which will be published at ECCV 2022.   


https://user-images.githubusercontent.com/46648831/190854092-9f9c7fc5-f890-4ec1-809a-9a3807b5993a.mp4  

**Authors**: Simon Doll, Richard Schulz, Lukas Schneider, Viviane Benzin, Markus Enzweiler, Hendrik P.A. Lensch
## Abstract
Based on the key idea of DETR this paper introduces an object-centric 3D object detection framework that operates on a limited number of 3D object queries instead of dense bounding box proposals followed by non-maximum suppression. After image feature extraction a decoder-only transformer architecture is trained on a set-based loss. SpatialDETR infers the classification and bounding box estimates based on attention both spatially within each image and across the different views. To fuse the multi-view information in the attention block we introduce a novel geometric positional encoding that incorporates the view ray geometry to explicitly consider the extrinsic and intrinsic camera setup. This way, the  spatially-aware cross-view attention exploits arbitrary receptive fields to integrate cross-sensor data and therefore global context. Extensive experiments on the nuScenes benchmark demonstrate the potential of global attention and result in state-of-the-art performance.

If you find this repository useful, please cite
```bibtex
@inproceedings{Doll2022ECCV,
  author = {Doll, Simon and Schulz, Richard and Schneider, Lukas and Benzin, Viviane and Enzweiler Markus and Lensch, Hendrik P.A.},
  title = {SpatialDETR: Robust Scalable Transformer-Based 3D Object Detection from Multi-View Camera Images with Global Cross-Sensor Attention},
  booktitle = {European Conference on Computer Vision(ECCV)},
  year = {2022}
}
```

[You can find the Paper here](https://uni-tuebingen.de/securedl/sdl-eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE2NjMwODI0NjQsImV4cCI6MTY2MzE3MjQ2MSwidXNlciI6MCwiZ3JvdXBzIjpbMCwtMV0sImZpbGUiOiJmaWxlYWRtaW5cL1VuaV9UdWViaW5nZW5cL0Zha3VsdGFldGVuXC9JbmZvS29nbmlcL1dTSVwvQ29tR3JhcGhcL3B1YmxpY2F0aW9uc1wvMjAyMlwvRUNDVjIwMjItU3BhdGlhbERldHJcL0VDQ1YyMDIyLXNwYXRpYWxfZGV0ci5wZGYiLCJwYWdlIjoyMzczOTl9.iLxLzTD_7W1HiUOYb1RdNqR0o1FPIPXW-JYvC-__st8/ECCV2022-spatial_detr.pdf)
## Setup
To setup the repository and run trainings we refer to [getting_started.md](getting_started.md)

## Changelog
### 06/22
- We moved the codebase to the new coordinate conventions of `mmdetection3d rc1.0ff`
- The performancy might slightly vary compared to the original runs on `mmdetection3d 0.17` reported in the paper


## Experimental results
The baseline models have been trained on `4xV100 GPUs`, the submission models on `8xA100 GPUs`. For more details we refer to the corresponding configuration / log files. Keep in mind that the performance can vary between runs and that the current codebase uses `mmdetection3d@rc1.0`
| Config  | Logfile | Set | #GPUs| mmdet3d | mAP | ATE | ASE | AOE | AVE | AAE | NDS|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---| --- |
| [query_proj_value_proj.py (baseline)](configs/submission/frozen_4/query_proj_value_proj.py)  | [log](training_logs/frozen_4/query_proj_value_proj.log) / [model](https://drive.google.com/file/d/1Tm6M0e-8QYBUeqwBwYpJk1EwxQbyZl6M/view?usp=sharing) | val | 4 | rc1.0 | 0.315 | 0.839 | 0.274 | 0.484 | 0.810 | 0.217 | 0.395 |
| [query_proj_value_proj.py](configs/submission/frozen_4/query_proj_value_proj.py)  | [log](training_logs/frozen_4/query_proj_value_proj_0_17.log) | val | 4 | 0.17 | 0.313 | 0.850 | 0.274 | 0.494 | 0.814 | 0.213 | 0.392 |
| [query_center_proj_no_value_proj_shared.py](configs/submission/frozen_1/query_center_proj_no_value_proj_shared.py)  | [log](training_logs/frozen_1/query_center_proj_no_value_proj_shared.log) | val | 8 | 0.17 | 0.351 | 0.772 | 0.274 | 0.395 | 0.847 | 0.217 | 0.425
| [query_center_proj_no_value_proj_shared_cbgs_vovnet_trainval.py](configs/submission/frozen_1/query_center_proj_no_value_proj_shared_cbgs_vovnet_trainval.py)  |[log](training_logs/frozen_1/query_center_proj_no_value_proj_shared_cbgs_vovnet_trainval.log)| test | 8 | 0.17| 0.425 | 0.614 | 0.253 | 0.402 | 0.857 | 0.131 | 0.487

## Qualitative results
![](img/qualitative_results.png "Visualization of predictions")
![](img/attention.png "Attention visualization for multiple queries")

## License
See [license_infos.md](license_infos.md) for details.

## Acknowledgement
This repo contains the implementations of SpatialDETR. Our implementation is a plugin to [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) and also uses a [fork](https://github.com/SimonDoll/fusion_detr3d) of [DETR3D](https://github.com/WangYueFt/detr3d). Full credits belong to the contributors of those frameworks and we truly thank them for enabling our research!
