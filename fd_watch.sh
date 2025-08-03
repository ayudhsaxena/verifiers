#!/usr/bin/env bash
# fd_watch.sh – watch total open file-descriptors system-wide
# Usage:  ./fd_watch.sh [interval_seconds]
#
# If the "TOTAL" column keeps climbing (and your workload is stable),
# you likely have a descriptor leak somewhere.

INTERVAL=${1:-1}               # default: 1-second sampling
prev=0

printf "%-20s %-10s %-10s\n" "TIMESTAMP" "TOTAL_FDs" "ΔsincePrev"
while :; do
    # Count all file descriptors across all processes
    total=$(find /proc/[0-9]*/fd -type l 2>/dev/null | wc -l)
    
    delta=$(( total - prev ))
    printf "%-20s %-10s %-10s\n" "$(date +%H:%M:%S)" "$total" "$delta"
    
    prev=$total
    sleep "$INTERVAL"
done