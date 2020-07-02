FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

COPY ./docker/bash.bashrc /etc/bash.bashrc
WORKDIR /src

