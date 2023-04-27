# syntax=docker/dockerfile:1.0.0-experimental

ARG AWSCLI=2.1.25
ARG POETRY=1.2.2
ARG TENSORRT=8.2.3.0
ARG PYTHON=3.7.3
ARG CUDA=11.1.1
ARG CUDNN=8
ARG UBUNTU=18.04
FROM acesdev/algo-base:awscli${AWSCLI}-poetry${POETRY}-tensorrt${TENSORRT}-python${PYTHON}-cuda${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU}
COPY pyproject.toml poetry.lock poetry.toml $WORKDIR/

RUN mkdir -m 700 $HOME/.ssh && ssh-keyscan github.com > $HOME/.ssh/known_hosts
RUN pip install --upgrade pip
RUN --mount=type=ssh poetry install --no-root

RUN pip install git+https://github.com/elliottzheng/face-detection.git@master
RUN echo "alias flake8='pflake8'" >> $HOME/.bash_aliases
