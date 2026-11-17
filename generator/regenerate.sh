#!/bin/sh
# Regenerate the Frames problems of the Open Benchmark Set from nothing, in one sequence.
#
#   sh regenerate.sh             # the study's twenty structures (seeds 0 to 19, failing ones replaced from 32 up)
#                                # and the tuning seeds 30, 31
#
# Environment: OBS_DATA (default ./obs-data), where the exports and the problems are written; PYTHON (default python),
# with numpy, scipy and cvxpy. Every step can be rerun: the member choices are cached under the exports.
#
# The option --audit of the study's copy of this script is removed in this release. As the study had it, it asked
# for twenty seeds from 20 (20 to 39, which takes in the tuning seeds and the study's replacement seeds) and wrote
# into the study's own exports and problems folders. The held-back structures were generated instead by the audit's
# own scripts, with ten seeds from 20 and every write in separate folders: OBS_GEN_BULK set to a separate exports
# folder, then `obs_gen2.py seeds --first 20 --count 10 --audit`, `obs_gen2.py all --seeds LIST --audit`, and
# stage2.py with --out a separate problems folder.
set -e
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
PY="${PYTHON:-python}"
export OBS_DATA="${OBS_DATA:-$(pwd)/obs-data}"
export OBS_GEN_BULK="${OBS_GEN_BULK:-$OBS_DATA/exports/obs_gen2}"
SF="$OBS_GEN_BULK/study_seeds.json"
if [ -n "$1" ]; then echo "regenerate.sh takes no option (the study's --audit is removed; see the comment above)"; exit 2; fi
# 1. the member's place and tuning per seed, and the seed list with replacements
$PY obs_gen2.py seeds --first 0 --count 20
SEEDS=$($PY -c "import json; d=json.load(open('$SF')); print(','.join(str(s) for s in d['seeds'] + d['tuning']))")
echo "seeds: $SEEDS"
# 2. the exports (one process per case); a structure a mode of which can reach 4,096 Hz is refused here
$PY obs_gen2.py all --seeds "$SEEDS" --jobs 4
# 3. staging, with the stager's checks and the reference model's
$PY stage2.py --seeds "$SEEDS" --jobs 6 --out "$OBS_DATA/problems"
