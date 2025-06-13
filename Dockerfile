# FROM nvcr.io/nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
FROM ubuntu:20.04
LABEL maintainer="Sarah Gillet"

ENV DEBIAN_FRONTEND noninteractive

# This sets the shell to fail if any individual command
# fails in the build pipeline, rather than just the last command
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install the base toolchain
RUN apt-get update && \
  apt-get install --yes --no-install-recommends \
  python3.8 \
  python3-pip \
  python3.8-dev \
  # bzip2 \
  # ca-certificates \
  # sudo \
  # wget \
  build-essential 
  

RUN pip3 install \
  pytorch-lightning==1.7.7 \
  numpy==1.19.5 \
  scipy==1.8.1 \
  wandb==0.13.2 \
  pygsheets==2.0.6 \
  pandas==1.4.2 \
  ml_collections==0.1.1 \
  xgboost \
  optuna


RUN pip3 install --no-cache-dir \
  torch==1.9.0+cu111 \
  torchmetrics==0.11.4 \
  torchvision==0.10.0+cu111 \
  torchaudio==0.9.0 \
  torch_geometric==2.1.0 \
  # torch-cluster==1.6.0 \
  -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install --no-index \
  torch-scatter==2.0.9 \
  torch-sparse==0.6.12 \
  torch-spline-conv==1.2.1 \
  torch-cluster \
  -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html

COPY offline_training /usr/local/offline_training
COPY training_data /usr/local/training_data
COPY dicts /usr/local/dicts

WORKDIR /usr/local/offline_training
#ENTRYPOINT [ "python3.8"]

