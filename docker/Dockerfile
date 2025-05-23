ARG BASE_IMAGE
FROM $BASE_IMAGE

ENV CUDA_HOME="/usr/local/cuda"

# Set APT configuration and disable installation of recommended packages
RUN echo 'APT::Install-Recommends "0";\nAPT::Install-Suggests "0";' > /etc/apt/apt.conf.d/01norecommend && \
    echo 'APT::Get::Assume-Yes "true";' > /etc/apt/apt.conf.d/02assumeyes && \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/03keep-cache && \
    echo 'APT::Clean-Installed "false";' > /etc/apt/apt.conf.d/04clean-installed-off && \
    rm -f /etc/apt/apt.conf.d/docker-clean

RUN echo ${PACKAGE_NAME} > /etc/debian_chroot

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
      apt-get -y install sudo

# Install pip packages
RUN pip install tqdm plotly kaleido pandas nbformat ipykernel triton onnxruntime cuda-python onnx_graphsurgeon
RUN pip install -U jetson-stats
RUN pip install torch==2.4.0 torchvision==0.19.0a0+48b1edf

# this doesn't work for now, so we need to install torch2trt manually inside docker
# Install torch2trt 
# Adapted from https://github.com/dusty-nv/jetson-containers/blob/master/packages/pytorch/torch2trt/Dockerfile
# RUN cd /opt && \
#     git clone --depth=1 https://github.com/NVIDIA-AI-IOT/torch2trt && \
#     cd torch2trt && \
#     #sed 's|collections.Sequence|collections.abc.Sequence|g' -i torch2trt/converters/interpolate.py && \
#     #cat torch2trt/converters/interpolate.py | grep Sequence && \
#     pip3 install . && \
#     sed 's|^set(CUDA_ARCHITECTURES.*|#|g' -i CMakeLists.txt && \
#     sed 's|Catch2_FOUND|False|g' -i CMakeLists.txt && \
#     cmake -B build -DCUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} . && \
#     cmake --build build --target install && \
#     ldconfig && \
#     pip3 install nvidia-pyindex && \
#     pip3 install onnx-graphsurgeon

#######
# USER and permission setup
#######
# Set locale
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
locale-gen
ENV LANG=en_US.UTF-8  
ENV LANGUAGE=en_US:en  
ENV LC_ALL=en_US.UTF-8

# Set up user
ARG _UID
ARG _GID
ARG USER
RUN if ! id -u ${USER} >/dev/null 2>&1; then useradd -u ${_UID} -s /bin/bash -m ${USER}; fi && \
if ! getent group ${_GID} >/dev/null 2>&1; then groupmod -g ${_GID} ${USER}; fi && \
usermod -a -G ${USER} ${USER} && \
usermod -a -G video ${USER} && \
echo "${USER} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USER} && \
chown -R ${USER}:${USER} /root

USER ${USER}
WORKDIR /home/${USER}

CMD ["/bin/bash"]
