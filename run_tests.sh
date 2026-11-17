#!/bin/sh
# Every test of the released reference model, scorer and generator check, one command:  sh code/run_tests.sh
# Needs python with numpy and pytest. Tests that need data not in the release (the simulator's player, earlier
# study folders, run folders) are skipped when it is absent; environment variables name it: OBS_PLAY, OBS_DATA, and
# OBS_SET (the folder that holds the set's problems/, when they are not beside the code).
cd "$(dirname "$0")" || exit 1
export PYTHONDONTWRITEBYTECODE=1
exec "${PYTHON:-python}" -m pytest -p no:cacheprovider -q tests generator/tests "$@"
