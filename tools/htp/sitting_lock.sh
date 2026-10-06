#!/usr/bin/env bash
# One sitting at a time on a shared phone.
#   tools/htp/sitting_lock.sh take    <serial> "<purpose>" [expect_min]
#   tools/htp/sitting_lock.sh release <serial>
#   tools/htp/sitting_lock.sh status  <serial>
# The lock is /data/local/tmp/nntrainer/.sitting.lock on the device. `take`
# exits 1 and prints the holder when another sitting owns it; `release`
# removes it only if the owner matches (SITTING_OWNER, default user@host).
set -u
LOCK=/data/local/tmp/nntrainer/.sitting.lock
cmd=${1:-}; serial=${2:-}
[ -n "$cmd" ] && [ -n "$serial" ] || { sed -n 2,7p "$0"; exit 2; }
owner=${SITTING_OWNER:-$(id -un)@$(hostname)}
sh() { adb -s "$serial" shell "$@"; }
case "$cmd" in
take)
  purpose=${3:?purpose}; mins=${4:-30}
  held=$(sh "cat $LOCK 2>/dev/null")
  if [ -n "$held" ]; then
    echo "LOCKED by: $held" >&2; exit 1
  fi
  sh "mkdir -p $(dirname $LOCK) && printf 'owner=%s purpose=%s start=%s expect_min=%s\n' '$owner' '$purpose' '$(date -u +%FT%TZ)' '$mins' > $LOCK"
  sh "cat $LOCK" ;;
release)
  held=$(sh "cat $LOCK 2>/dev/null")
  case "$held" in
    "") echo "not locked" ;;
    "owner=$owner "*) sh "rm -f $LOCK" && echo "released" ;;
    *) echo "LOCKED by someone else, not released: $held" >&2; exit 1 ;;
  esac ;;
status)
  held=$(sh "cat $LOCK 2>/dev/null"); echo "${held:-not locked}" ;;
*) sed -n 2,7p "$0"; exit 2 ;;
esac
