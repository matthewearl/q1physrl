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

TREE="$(git stash create)"
if [ -z "$TREE" ]; then
    TREE="HEAD"
fi
"${GIT}" archive --format=tar.gz --prefix=q1physrl/ "$TREE" > archive.tar.gz
"${DOCKER}" build . -t q1physrl

"${RM}" archive.tar.gz
