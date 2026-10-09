#!/bin/sh
# Install the garage clock's home screen (time + battery gauge) and make it the
# only app in the rotation. Tracker notifications still show on top of it.
#   clock/install.sh [CLOCK_IP]
set -eu
HOST=${1:-192.168.4.37}
API="http://$HOST/api/v1"
DIR=$(dirname "$0")
curl -fsS -X PUT "$API/apps/script/TimeBat" -H 'Content-Type: text/plain' --data-binary @"$DIR/timebat.be"; echo
curl -fsS -X PUT "$API/apps/order" -H 'Content-Type: application/json' \
  -d '{"order":["TimeBat"],"disabled":["Time","Date","Battery"]}'; echo
