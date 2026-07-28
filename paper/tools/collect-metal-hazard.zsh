#!/bin/zsh
# Collect the Metal hazard-tracking pilot with retained raw observations.
#
# One invocation compiles the harness and runs every session into a fresh
# timestamped collection directory, with power state and revisions recorded:
#
#     zsh paper/tools/collect-metal-hazard.zsh [sessions]   # default 10
#
# Copy the printed directory into paper/data/raw/ on the analysis machine.
set -euo pipefail

sessions=${1:-10}
root=${0:A:h:h:h}
cd $root

id=$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s)-hazard
dir=paper/data/raw/$id
mkdir -p $dir

bench=$(mktemp -d)/metal-hazard-bench
echo "compiling harness..." >&2
xcrun swiftc -O paper/tools/metal-hazard-bench.swift -o $bench

{
  echo "created_utc: $(date -u +%FT%TZ)"
  echo "host: $(hostname -s)"
  echo "os: macOS $(sw_vers -productVersion)"
  echo "hardware: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
  echo "blade_revision: $(git rev-parse HEAD)"
  echo "blade_status: $(git status --porcelain -- ':(exclude)paper/data' | tr '\n' ';')"
  echo "sessions: $sessions"
} > $dir/metadata.txt
pmset -g batt > $dir/power.txt
pmset -g custom >> $dir/power.txt 2>/dev/null || true

if grep -q 'lowpowermode *1' $dir/power.txt; then
  echo "note: Low Power Mode appears enabled; it is recorded in power.txt" >&2
fi

for i in $(seq -w 1 $sessions); do
  echo "=== session r$i of $sessions" >&2
  $bench --raw-output $dir/metal-hazard-r$i-raw.csv \
    | tee $dir/metal-hazard-r$i-summary.csv > /dev/null
done

echo "done: $dir"
echo "copy this directory into paper/data/raw/ on the analysis machine"
