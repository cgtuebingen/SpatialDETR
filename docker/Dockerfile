ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# fix for non signed nvidia pub keys (taken from https://jdhao.github.io/2022/05/05/nvidia-apt-repo-public-key-error-fix/)
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# install some useful libs
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 git nano sudo python3-dev build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV  PATH="/root/.local/bin:${PATH}"


# Install MMCV, MMDetection and MMSegmentation
# we do this by cloning all repos to /deps/ -> can install in editable mode
WORKDIR /deps

RUN git clone https://github.com/open-mmlab/mmcv.git mmcv \
    && cd mmcv \
    && git checkout tags/v1.5.0 -b v1.5.0 

RUN git clone https://github.com/open-mmlab/mmdetection.git mmdetection \
    && cd mmdetection \
    && git checkout tags/v2.23.0 -b v2.23.0 \
    && pip install -e .

RUN pip install mmsegmentation==0.20.0 \
    && pip uninstall -y pycocotools \
    && pip install mmpycocotools
RUN conda clean --all
ENV FORCE_CUDA="1"

WORKDIR /workspace