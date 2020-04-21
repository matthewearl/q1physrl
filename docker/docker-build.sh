#!/usr/bin/env bash

set -e

cd "$(dirname "$0")/.."

ARCHIVE_NAME="archive.tar.gz"
GIT="$(which git)"
DOCKER="$(which docker)"
RM="$(which rm)"

[ -e "$ARCHIVE_NAME" ] && \
    echo "File ${ARCHIVE_NAME} already exists. Please remove it first." 2>&1 && \
    exit 1

"${GIT}" archive --format=tar.gz --prefix=q1physrl/ $(git stash create) > archive.tar.gz
"${DOCKER}" build . -t q1physrl

"${RM}" archive.tar.gz
