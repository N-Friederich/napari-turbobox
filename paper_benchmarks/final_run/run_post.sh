#!/bin/bash
# Panel (c) memory and panel (f) image size, in the same environment as run_final.sh.
# Called by run_final.sh (stage post); needs TB_THUMBS, PACING and EDIT_VIEW. The repository root is
# taken from the location of this script; BB (Python of the benchmark environment, required) and OUT
# (results directory) as in run_final.sh.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd)
cd $R
: "${BB:?set BB to the Python of the benchmark environment}"
LOCK="$BB $R/paper_benchmarks/final_run/with_lock.py"
LOG=$R/benchmark_logs/final
V="${OUT:-$R/benchmark_results}"
mkdir -p $LOG
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
mkdir -p $V/resolution $V/memory
: "${TB_THUMBS:?}" "${PACING:?}" "${EDIT_VIEW:?}"
export EDIT_VIEW
$LOCK --tag POST-RES -- /bin/bash -c "cd $R && PYTHONPATH=src:. $BB paper_benchmarks/final_run/resolution_final.py $V/resolution > $LOG/POST-RES.log 2>&1"; rc_res=$?
echo "$(date +%H:%M) POST-RES rc=$rc_res"
$LOCK --tag POST-MEM -- /bin/bash -c "cd $R && PYTHONPATH=src:. $BB paper_benchmarks/final_run/memory_final.py $V/memory > $LOG/POST-MEM.log 2>&1"; rc_mem=$?
echo "$(date +%H:%M) POST-MEM rc=$rc_mem"
echo "$(date +%H:%M) POST DONE"
[ $rc_res -eq 0 ] && [ $rc_mem -eq 0 ]
