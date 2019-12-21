FROM ubuntu:latest
MAINTAINER PEI-i1

RUN apt-get update && apt-get install -y --force-yes \
    bc \
    curl \
    git \
    python3.7 \
    python3-pip \
    sudo \
    unzip \
    wget \
    vim

RUN useradd --create-home --shell /bin/bash solver
RUN echo 'root:root' | chpasswd
RUN echo 'solver:solver' | chpasswd

USER solver

WORKDIR /home/solver
RUN python3.7 -m pip install pip

COPY . Tech_Problems
WORKDIR /home/solver/Tech_Problems

RUN echo 'PATH=$PATH:/home/solver/.local/bin' >> ../.bashrc

RUN python3.7 -m pip install -r requirements.txt --user

WORKDIR /home/solver
