ARG DEBIAN_FRONTEND=noninteractive

FROM python:3.7.7-buster

WORKDIR ws
RUN apt update -y

# Install the 100m map
RUN wget https://downloads.teqnohaxor.com/quake/maps/100m.zip && \
    echo 'bbc92aa728aaaac2335f2a67ddd2c1b2b09a33403da604d7aa7e99e23b69c42b  100m.zip' | sha256sum --check && \
    mkdir -p quake-base/id1/maps && \
    unzip 100m.zip -d quake-base/id1/maps

# Install modified quakespasm
RUN apt install -y libsdl2-dev && \
    git clone 'https://github.com/matthewearl/quakespasm.git' quakespasm-hacks -b me/hacks && \
    cd quakespasm-hacks/quakespasm/Quake/ && \
    make

# Install pyquake
RUN git clone 'https://github.com/matthewearl/pyquake.git' pyquake && \
    pip install -e pyquake

# Install q1physrl (and dependencies)
COPY archive.tar.gz .
RUN tar zxvf archive.tar.gz && \
    pip install -r q1physrl/requirements_train.txt && \
    pip install -e q1physrl/q1physrl_env && \
    pip install -e q1physrl && \
    rm -r /root/.cache

# Expose tensorboard port.
EXPOSE 6006

# Run this to start tensorboard:   tensorboard --logdir q1physrl --bind_all
