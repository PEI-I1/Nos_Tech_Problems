FROM archlinux:latest
MAINTAINER PEI-i1

RUN pacman -Syu --noconfirm && pacman -S --noconfirm \
    bc \
    curl \
    git \
    python \
    python-pip \
    sudo \
    unzip \
    wget \
    vim

RUN useradd -m -G wheel -s /bin/bash solver
RUN echo 'root:root' | chpasswd
RUN echo 'solver:solver' | chpasswd

RUN sed -irs 's/# (%wheel ALL=\(ALL\) ALL)/\1/g' /etc/sudoers

USER solver

WORKDIR /home/solver

RUN git clone https://github.com/PEI-I1/Nos_Tech_Problems.git
WORKDIR /home/solver/Nos_Tech_Problems

RUN echo 'PATH=$PATH:/home/solver/.local/bin' >> ../.bashrc

RUN git checkout Technical_NLP

RUN pip install -r requirements.txt --user --no-warn-script-location

WORKDIR /home/solver
