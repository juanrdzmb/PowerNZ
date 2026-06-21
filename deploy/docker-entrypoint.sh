#!/bin/sh
set -eu

# A named Docker volume is mounted after the image is built and starts owned by
# root. Correct it here, then run the Python process without root privileges.
mkdir -p /data
chown -R powernz:powernz /data
exec gosu powernz "$@"
