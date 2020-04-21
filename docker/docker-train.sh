#!/usr/bin/env bash

set -e

DOCKER="$(which docker)"
"${DOCKER}" run -it --rm --mount type=volume,source=ray-results,destination=/root/ray_results q1physrl q1physrl_train
