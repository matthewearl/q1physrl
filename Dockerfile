ARG DEBIAN_FRONTEND=noninteractive

FROM python:3.7.7-buster

WORKDIR ws
RUN apt update -y

# Install quake shareware
RUN wget https://ftp.gwdg.de/pub/misc/ftp.idsoftware.com/idstuff/quake/quake106.zip && \
    echo 'ec6c9d34b1ae0252ac0066045b6611a7919c2a0d78a3a66d9387a8f597553239  quake106.zip' | sha256sum --check && \
    apt install -y lhasa && \
    unzip quake106.zip -d quake-sw-installer && \
    lha -xw=quake-sw quake-sw-installer/resource.1

# Install the 100m map
RUN wget https://downloads.teqnohaxor.com/quake/maps/100m.zip && \
    echo 'bbc92aa728aaaac2335f2a67ddd2c1b2b09a33403da604d7aa7e99e23b69c42b  100m.zip' | sha256sum --check && \
    unzip 100m.zip -d quake-sw/id1

# Install modified quakespasm
RUN apt install -y libsdl2-dev && \
    git clone 'https://github.com/matthewearl/quakespasm.git' quakespasm-hacks -b me/hacks && \
    cd quakespasm-hacks/quakespasm/Quake/ && \
    make

# Install q1physrl (and dependencies)
COPY archive.tar.gz .
RUN tar zxvf archive.tar.gz && \
    pip install -r q1physrl/requirements_train.txt && \
    pip install -e q1physrl && rm -r /root/.cache

# Expose tensorboard port.
EXPOSE 6006

# Run this to start tensorboard:   tensorboard --logdir q1physrl --bind_all
