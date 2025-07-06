FROM apache/airflow:2.10.3 AS builder
USER root
# RUN set -ex apt-get autoremove -y \
#     && apt-get update && apt-get -y install build-essential \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get -y install build-essential libgl1-mesa-glx libglib2.0-0 openslide-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER airflow

FROM builder AS build1
COPY requirements.txt .
RUN pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt
run pip cache purge