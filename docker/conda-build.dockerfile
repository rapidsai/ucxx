ARG CUDA_VERSION=11.5.2
ARG DISTRIBUTION_VERSION=ubuntu20.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-${DISTRIBUTION_VERSION}

# Make available to later build stages
ARG DISTRIBUTION_VERSION
# Where to install conda, and what to name the created environment
ARG CONDA_HOME=/opt/conda
ENV CONDA_HOME="${CONDA_HOME}"

# Where cuda is installed
ENV CUDA_HOME="/usr/local/cuda"

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y \
    && apt-get --fix-missing upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && apt-get install -y \
        automake \
        dh-make \
        git \
        libcap2 \
        libnuma-dev \
        libtool \
        make \
        pkg-config \
        udev \
        curl \
    && apt-get autoremove -y \
    && apt-get clean

RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
    -o /minimamba.sh \
    && bash /minimamba.sh -b -p ${CONDA_HOME} \
    && rm /minimamba.sh

ENV PATH="${CONDA_HOME}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${CUDA_HOME}/bin"

RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate base && \
    conda config --prepend channels rapidsai-nightly && \
    conda config --set conda_build.root-dir /ucxx/.conda-bld && \
    mamba install boa

WORKDIR /ucxx

CMD ["/bin/bash", "./docker/conda-build.sh"]
