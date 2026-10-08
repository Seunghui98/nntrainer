#!/system/bin/sh
##
# @file    216-sampler.sh
# @brief   Device-side system sampler for #216 (runs on the phone, fork-light)
#
# Every $2 s (default 0.5) appends to $1: uptime, per-core /proc/stat lines,
# per-core scaling_cur_freq, DDR / LLCC bus_dcvs cur_freq, kswapd / swap /
# pgpgin / file-refault counters, MemFree / MemAvailable / Cached, PSI
# totals, loadavg, and every nntrainer_causallm task's stat line with its
# Cpus_allowed_list. Stops when the file $1.stop appears.
# Usage (on device): sh 216-sampler.sh <out> [interval]
O=$1; I=${2:-0.5}; BD=/sys/devices/system/cpu/bus_dcvs
rm -f $O.stop
while [ ! -e $O.stop ]; do
  { read u x < /proc/uptime; echo "S $u"
    while read l; do case $l in cpu[0-7]*) echo "C $l";; intr*) break;; esac; done < /proc/stat
    f=""; for c in 0 1 2 3 4 5 6 7; do read v < /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq; f="$f $v"; done
    echo "F$f"; read d < $BD/DDR/cur_freq; read c < $BD/LLCC/cur_freq; echo "B $d $c"
    v=""; while read k n; do case $k in pgscan_kswapd|pgscan_direct|pswpin|pswpout|pgmajfault|pgpgin|workingset_refault_file|pgsteal_kswapd) v="$v $k=$n";; esac; done < /proc/vmstat; echo "V$v"
    m=""; while read k n x; do case $k in MemFree:|MemAvailable:|Cached:|SwapFree:) m="$m ${k%:}=$n";; Shmem:) break;; esac; done < /proc/meminfo; echo "M$m"
    for r in cpu memory io; do while read k a b c t; do echo "P $r $k ${t#total=}"; done < /proc/pressure/$r; done
    read la < /proc/loadavg; echo "L $la"
    p=$(pidof nntrainer_causallm)
    if [ -n "$p" ]; then
      for t in /proc/$p/task/*; do read l < $t/stat && echo "A $l"; done 2>/dev/null
      grep -H Cpus_allowed_list /proc/$p/task/*/status 2>/dev/null | sed 's,^/proc/[0-9]*/task/\([0-9]*\)/status:Cpus_allowed_list:[[:space:]]*,G \1 ,'
    fi; } >> $O
  sleep $I
done
