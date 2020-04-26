#!/usr/bin/env bash

set -e

PARAMS_FNAME="$1"; shift

[ -z "$PARAMS_FNAME" ] && echo "Usage: $0 [params.yml file]" 2>&1 && exit 1

PARAMS_FNAME="$(realpath "$PARAMS_FNAME")"

cd "$(dirname "$0")/.."

HOST_CMD="
pip install -r binds/q1physrl-host/requirements_q1physrl.txt &&
pip install -v binds/q1physrl-host/q1physrl_env &&
pip install -v binds/q1physrl-host &&
q1physrl_train /ws/binds/params.yml
"
DOCKER="$(which docker)"
"${DOCKER}" run -it --rm \
    --mount type=bind,source="$(pwd)",destination=/ws/binds/q1physrl-host,readonly \
    --mount type=bind,source="$PARAMS_FNAME",destination=/ws/binds/params.yml,readonly \
    --mount type=bind,source="$HOME/ray_results",destination=/root/ray_results \
    matthewearl/q1physrl:q1physrl \
    bash -e -c "${HOST_CMD}"
