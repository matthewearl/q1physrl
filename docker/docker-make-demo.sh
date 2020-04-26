#!/usr/bin/env bash

set -e

GIT="$(which git)"
DOCKER="$(which docker)"
TOUCH="$(which touch)"
RM="$(which rm)"

QUAKE_BASE_FNAME="$1"; shift
CHECKPOINT_FNAME="$1"; shift
PARAMS_FNAME="$1"; shift
DEMO_FNAME="$1"; shift

[ -z "$DEMO_FNAME" ] && echo "Usage: $0 [pak1 file] [checkpoint file] [params file] [demo file]" 2>&1 && exit 1

QUAKE_BASE_FNAME="$(realpath "$QUAKE_BASE_FNAME")"
CHECKPOINT_FNAME="$(realpath "$CHECKPOINT_FNAME")"
METADATA_FNAME="$CHECKPOINT_FNAME.tune_metadata"
PARAMS_FNAME="$(realpath "$PARAMS_FNAME")"
DEMO_FNAME="$(realpath "$DEMO_FNAME")"

cd "$(dirname "$0")/.."

if [ -f "$QUAKE_BASE_FNAME/PAK0.PAK" ]; then
    PAK0_FNAME="$QUAKE_BASE_FNAME/PAK0.PAK"
else
    PAK0_FNAME="$QUAKE_BASE_FNAME/pak0.pak"
fi

if [ -f "$QUAKE_BASE_FNAME/PAK1.PAK" ]; then
    PAK1_FNAME="$QUAKE_BASE_FNAME/PAK1.PAK"
else
    PAK1_FNAME="$QUAKE_BASE_FNAME/pak1.pak"
fi

HOST_CMD="
pip install -r binds/q1physrl-host/requirements_q1physrl.txt &&
pip install -v binds/q1physrl-host/q1physrl_env &&
pip install -v binds/q1physrl-host &&
q1physrl_make_demo \\
    /ws/binds/checkpoint \\
    /ws/binds/params.json \\
    /ws/quakespasm-hacks/quakespasm/Quake/quakespasm \\
    /ws/quake-base/ \\
    /ws/binds/temp.dem
"

"${TOUCH}" "$DEMO_FNAME"  # Docker requires that the file exists to be able to mount it?
"${DOCKER}" run -it --rm \
    --mount type=bind,source="$(pwd)",destination=/ws/binds/q1physrl-host,readonly \
    --mount type=bind,source="$PAK0_FNAME",destination=/ws/quake-base/id1/pak0.pak,readonly \
    --mount type=bind,source="$PAK1_FNAME",destination=/ws/quake-base/id1/pak1.pak,readonly \
    --mount type=bind,source="$CHECKPOINT_FNAME",destination=/ws/binds/checkpoint,readonly \
    --mount type=bind,source="$METADATA_FNAME",destination=/ws/binds/checkpoint.tune_metadata,readonly \
    --mount type=bind,source="$PARAMS_FNAME",destination=/ws/binds/params.json,readonly \
    --mount type=bind,source="$DEMO_FNAME",destination=/ws/binds/temp.dem \
    matthewearl/q1physrl:q1physrl \
    bash -e -c "${HOST_CMD}"

