FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS rdagentbase
# For GPU support, please choose the proper tag from https://hub.docker.com/r/pytorch/pytorch/tags

RUN apt-get clean && apt-get update && apt-get install -y \  
    curl \  
    vim \  
    git \
    wget \
    unzip \
    openssh-server \
    build-essential \
    && rm -rf /var/lib/apt/lists/* 

RUN conda install conda=25.3.1

COPY kaggle_environment.yaml /workspace

RUN conda env create -f kaggle_environment.yaml

WORKDIR /workspace

###### rdagentbase end ######

FROM rdagentbase AS final
# FROM rdagentbase:latest AS final

RUN mkdir -p /root/.config/kaggle
COPY kaggle.json /root/.config/kaggle

SHELL ["/bin/bash", "-c"]

ENV RDAGENT_BRANCH=develop
ENV RDAGENT_GH_REPO=https://github.com/jayantjha/RD-Agent.git
ENV DS_CODER_COSTEER_ENV_TYPE=conda

RUN echo "source activate kaggle" >> ~/.bashrc && \
    source activate kaggle && \
    echo "Cloning repository from ${RDAGENT_GH_REPO} | branch ${RDAGENT_BRANCH}" && \
    git clone --depth 1 --branch "${RDAGENT_BRANCH}" "${RDAGENT_GH_REPO}" rdagent_repo && \
    pip install -r ./rdagent_repo/requirements.txt && \
    pip install ./rdagent_repo
