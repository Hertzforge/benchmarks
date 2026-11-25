#!/bin/sh
# Start drive.py on a job list as a launchd agent at normal priority (ProcessType Interactive), for batches whose
# clock time is reported: on macOS a process started from an interactive tool's shell may run at priority 20 and get
# about half a core on a loaded machine, where a launchd agent of this kind gets a whole one.
#
#   sh sine/agree/start_agent.sh <label> <jobs.json> <problems folder> [workers]
#   launchctl bootout gui/$(id -u)/<label>        stops it and every process it started
#
# The plist and the agent's output go beside the jobs file. Nothing here reads the synced project folder.
set -u
[ $# -ge 3 ] || { echo "usage: sh start_agent.sh <label> <jobs.json> <problems> [workers]"; exit 2; }
LABEL="$1"; JOBS="$2"; PROBLEMS="$3"; WORKERS="${4:-8}"
case "$LABEL" in ''|*[!A-Za-z0-9._]*) echo "label: letters, digits, . and _ only"; exit 2;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="$(cd "$(dirname "$JOBS")" && pwd)"
PL="$DIR/$LABEL.plist"; OUT="$DIR/$LABEL.out"; DOM="gui/$(id -u)"
if launchctl print "$DOM/$LABEL" 2>/dev/null | grep -q 'state = running'; then echo "$LABEL is running: refused"; exit 2; fi
launchctl bootout "$DOM/$LABEL" 2>/dev/null
cat > "$PL" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>$LABEL</string>
<key>ProgramArguments</key><array>
  <string>/usr/bin/python3</string><string>$HERE/drive.py</string><string>$DIR/$(basename "$JOBS")</string>
  <string>--problems</string><string>$PROBLEMS</string><string>--workers</string><string>$WORKERS</string>
</array>
<key>EnvironmentVariables</key><dict>
  <key>PATH</key><string>/usr/bin:/bin:/usr/sbin:/sbin</string>
  <key>PYTHONDONTWRITEBYTECODE</key><string>1</string>
</dict>
<key>ProcessType</key><string>Interactive</string>
<key>RunAtLoad</key><true/>
<key>StandardOutPath</key><string>$OUT</string>
<key>StandardErrorPath</key><string>$OUT</string>
</dict></plist>
EOF
plutil -lint "$PL" > /dev/null || { echo "the plist does not parse: $PL"; exit 3; }
launchctl bootstrap "$DOM" "$PL"; echo "launchctl bootstrap $LABEL: exit $?"
