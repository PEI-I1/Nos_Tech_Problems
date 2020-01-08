FROM python:3.7
MAINTAINER PEI-i1

RUN apt-get update && apt-get install -y --force-yes \
    bc \
    curl \
    unzip \
    wget \
    vim

RUN useradd --create-home --shell /bin/bash solver
RUN echo 'root:root' | chpasswd
RUN echo 'solver:solver' | chpasswd

USER solver

WORKDIR /home/solver

COPY --chown=solver . NTP_Bot
WORKDIR /home/solver/NTP_Bot

ENV PATH="/home/solver/.local/bin:${PATH}"

RUN python3.7 -m pip install -r requirements.txt --user

CMD ./app.py
