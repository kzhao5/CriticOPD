#!/usr/bin/env bash
# Fetch Relay-OPD at the pinned commit and add the CriticOPD files on top of it.
#   bash setup.sh [target directory, default ./Relay-OPD]
# The training code lives in <target>/relay-opd afterwards; `git -C <target> diff` shows every change.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST=${1:-Relay-OPD}
BASE_REPO=https://github.com/zju-real/Relay-OPD.git
BASE_COMMIT=eab21451f99e1a40fbb244f556de766d153c88f5

if [[ ! -d "${DEST}/.git" ]]; then
  git clone "${BASE_REPO}" "${DEST}"
fi
git -C "${DEST}" checkout --quiet "${BASE_COMMIT}"
cp -r "${HERE}/overlay/." "${DEST}/relay-opd/"
chmod +x "${DEST}"/relay-opd/opd/scripts/criticopd/*.sh
echo "CriticOPD added to ${DEST}/relay-opd (base commit ${BASE_COMMIT:0:7})."
git -C "${DEST}" status --short
