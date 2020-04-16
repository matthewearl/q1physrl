FROM python:3.7.7-buster

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR ws
RUN apt update -y

# Install ray with gaussian squashed gaussian support
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" > \
        /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt update -y && \
    apt install -y cmake bazel=1.1.0 && \
    git clone 'https://github.com/matthewearl/ray.git' ray-gsg -b me/gsg && \
    cd ray-gsg/python && \
    python setup.py bdist_wheel && \
    pip install dist/*.whl && \
    cd ../.. && \
    rm -rf ray-gsg && \
    rm -r /root/.cache

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
COPY . q1physrl
RUN pip install -e q1physrl

# TODO: Install and expose tensorboard?
