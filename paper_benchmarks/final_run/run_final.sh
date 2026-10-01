#!/bin/bash
# Driver of the final benchmark run (25 Sep 2026). One environment for both tools (napari 0.7.0,
# napari-bbox 0.1.1, PyQt6, bermuda). Protocol: xy_visible drags, XZ/YZ views follow the dragged box
# (--ortho-follow), one untimed warm-up drag per configuration, pacing $PACING, thumbnails as set.
# Every process waits for AC power, runs under caffeinate and is checked against the pmset log;
# a run with a sleep/battery event is moved aside and repeated once. Usage:
#   TB_THUMBS=on|off BB_SYNC=per_box|per_box_nothumb|per_box_quiet PACING=sleep|spin EDIT_VIEW=api|mouse [STAGES="large main sleep post tut"] run_final.sh
# EDIT_VIEW=mouse: both tools drag the box in the XY view through napari's mouse dispatch and their own
# drag handlers (napari-bbox: needs BB_SYNC=per_box_nothumb or per_box_quiet for the other views).
# Final configuration: TB_THUMBS=off BB_SYNC=per_box_quiet PACING=spin EDIT_VIEW=mouse.
# Results go to $OUT (default: benchmark_results/; a relative OUT is taken from the repository root; the
# reported run is in paper_results/). Runs whose result file already exists in $OUT are skipped, so a
# stopped run resumes. $OUT/environment/ gets the pip freeze of BB and $OUT/provenance/ the code state.
# Logs go to benchmark_logs/. Stop cleanly between processes: touch benchmark_logs/STOP (the running
# process finishes first).
# macOS only (pmset, caffeinate, /usr/bin/python3 for sleep_check.py); the checkout path must not contain
# spaces (paths are not quoted). The repository root is taken from the location of this script. BB must
# be set to the Python of the benchmark environment (napari 0.7.0, napari-bbox 0.1.1). Example:
#   BB=/path/to/python TB_THUMBS=off BB_SYNC=per_box_quiet PACING=spin EDIT_VIEW=mouse paper_benchmarks/final_run/run_final.sh
# The lock script with_lock.py (next to this script) keeps two timed processes from overlapping.
set -u
: "${TB_THUMBS:?}" "${BB_SYNC:?}" "${PACING:?}" "${EDIT_VIEW:?}"
export EDIT_VIEW
STAGES="${STAGES:-large main sleep post tut}"
MAIN_DRAGS="${MAIN_DRAGS:-50}"   # preliminary run: MAIN_DRAGS=10 MAIN_DIR=main_prelim
MAIN_DIR="${MAIN_DIR:-main}"
LARGE_DRAGS="${LARGE_DRAGS:-50}"  # preliminary run: LARGE_DRAGS=10 LARGE_DIR=large_n_prelim
LARGE_DIR="${LARGE_DIR:-large_n}"
# refuse to run with reduced priority (e.g. started with & in zsh, which nices background jobs)
[ "$(ps -o nice= -p $$ | tr -d ' ')" = 0 ] || { echo "refusing to run: nice=$(ps -o nice= -p $$)"; exit 1; }
R=$(cd "$(dirname "$0")/../.." && pwd)
cd $R
[ -e $R/benchmark_logs/STOP ] && { echo "refusing to run: benchmark_logs/STOP exists (remove it first)"; exit 1; }
: "${BB:?set BB to the Python of the benchmark environment}"
export BB
PYTHONPATH=src "$BB" -c "import napari, napari_bbox, napari_turbobox" \
  || { echo "refusing to run: $BB cannot import napari, napari_bbox and napari_turbobox (from src/)"; exit 1; }
LOCK="$BB $R/paper_benchmarks/final_run/with_lock.py"
LOG=$R/benchmark_logs/final
V="${OUT:-$R/benchmark_results}"
mkdir -p $LOG $V/environment $V/provenance $V/$MAIN_DIR $V/$LARGE_DIR $V/sleep_pacing $V/sleep_pacing_spin $V/tutorial_full
caffeinate -dims -w $$ &  # no idle/display/system sleep while the driver runs (not even between processes)
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
# code state of this run set (the sidecars also record it per process)
(git rev-parse HEAD > $V/provenance/code_state_head.txt; git diff HEAD -- src paper_benchmarks > $V/provenance/code_state_vs_head.patch; git ls-files --others --exclude-standard src paper_benchmarks > $V/provenance/code_state_untracked.txt)
# the environment freeze, taken as the harness takes it (without PYTHONPATH; its sidecars record the
# sha256 and report_final.py compares)
PIP_DISABLE_PIP_VERSION_CHECK=1 $BB -m pip freeze 2>/dev/null > $V/environment/pip_freeze_benchmark_env.txt
HB="-m paper_benchmarks.drag_session_benchmark --drag-protocol xy_visible --ortho-follow --warmup 1"
H="$HB --pacing $PACING"
EV="--edit-view $EDIT_VIEW"
MS=$([ $EDIT_VIEW = mouse ] && echo -mouse)
TB="--tool modern --thumbnails $TB_THUMBS $EV"
LG="--tool legacy --legacy-sync $BB_SYNC $EV"
tagfile() {  # expected file tag of a run tag (for skipping finished runs on a restart)
  case $1 in
    *-SLEEP-TB-*) echo "modern-xy_visible$([ $TB_THUMBS = on ] && echo -thumbs_on)-follow-warm1$MS";;
    *-SLEEP-BB-*) echo "legacy-${BB_SYNC}-xy_visible-follow-warm1$MS";;
    *-TB-*) echo "modern-xy_visible$([ $TB_THUMBS = on ] && echo -thumbs_on)-follow-warm1$([ $PACING = spin ] && echo -spin)$MS";;
    *-TUT-*) echo "legacy-xy_visible-follow-warm1$([ $PACING = spin ] && echo -spin)";;
    *) echo "legacy-${BB_SYNC}-xy_visible-follow-warm1$([ $PACING = spin ] && echo -spin)$MS";;
  esac
}
wait_ac() { until pmset -g batt | grep -q "AC Power"; do echo "$(date +%H:%M) waiting for AC power"; sleep 30; done; }
stop_requested() { [ -e $R/benchmark_logs/STOP ]; }
run() {  # tag outdir args...  (RUN_H overrides the harness prefix for one stage)
  tag=$1; out=$2; shift 2
  local HH="${RUN_H:-$H}"
  if stop_requested; then echo "$(date +%H:%M) $tag skipped (STOP)"; return; fi
  if [ -n "$(ls $out/drag_session_*_rep*.csv 2>/dev/null | grep -F "_rep${tag##*-r}.csv" | grep -F "$(tagfile $tag)")" ]; then
    echo "$(date +%H:%M) $tag skipped (result exists)"; return
  fi
  for attempt in 1 2; do
    wait_ac
    t0=$(date +%s)
    before=$(ls $out 2>/dev/null)
    { echo "load at start: $(sysctl -n vm.loadavg)"; } > $LOG/$tag.power
    caffeinate -dims $LOCK --tag $tag -- /bin/bash -c "cd $R && PYTHONPATH=src $BB $HH $* --out $out > $LOG/$tag.log 2>&1"
    rc=$?
    { echo "load at end: $(sysctl -n vm.loadavg)"; } >> $LOG/$tag.power
    if [ $rc -eq 0 ] && /usr/bin/python3 $R/paper_benchmarks/final_run/sleep_check.py $t0 $(date +%s) >> $LOG/$tag.power 2>&1; then
      echo "$(date +%H:%M) $tag rc=$rc clean"; return
    fi
    echo "$(date +%H:%M) $tag rc=$rc INVALID (crash, sleep or battery), attempt $attempt"
    mkdir -p $V/_invalid/$tag-a$attempt
    for f in $(ls $out); do echo "$before" | grep -qxF "$f" || mv -n $out/$f $V/_invalid/$tag-a$attempt/; done
    mv $LOG/$tag.log $LOG/INVALID-$tag-a$attempt.log; mv $LOG/$tag.power $LOG/INVALID-$tag-a$attempt.power
  done
  echo "$(date +%H:%M) $tag INVALID twice - aborting the run for a person to decide"; touch $V/ABORTED; exit 1
}
# 1) large N (first, so N = 10^4 is in hand early) (Figure 3a/d): N = 2000/5000/10000, k = 5 x $LARGE_DRAGS drags
[[ " $STAGES " == *" large "* ]] && for r in 0 1 2 3 4; do
  run FIN2-LARGE-TB-r$r $V/$LARGE_DIR $TB --panels a --ns 2000,5000,10000 --drags $LARGE_DRAGS --replicate $r
  run FIN2-LARGE-BB-r$r $V/$LARGE_DIR $LG --panels a --ns 2000,5000,10000 --drags $LARGE_DRAGS --replicate $r
done
echo "$(date +%H:%M) LARGE DONE"
# 2) main: panels a (N = 10..1000) + b, k = 5 x $MAIN_DRAGS drags, tools interleaved per replicate
[[ " $STAGES " == *" main "* ]] && for r in 0 1 2 3 4; do
  run FIN2-MAIN-TB-r$r $V/$MAIN_DIR $TB --drags $MAIN_DRAGS --replicate $r
  run FIN2-MAIN-BB-r$r $V/$MAIN_DIR $LG --drags $MAIN_DRAGS --replicate $r
done
echo "$(date +%H:%M) MAIN DONE"
# 3) pacing sensitivity: the same protocol with idle waiting (sleep) between moves, k = 3 x 10 drags,
#    panel a N = 10/100/1000/10000 and panel b, each replicate followed by a matched busy-waiting control
[[ " $STAGES " == *" sleep "* ]] && for r in 0 1 2; do
  RUN_H="$HB --pacing sleep" run FIN2-SLEEP-TB-r$r $V/sleep_pacing $TB --ns 10,100,1000,10000 --drags 10 --replicate $r
  RUN_H="$HB --pacing sleep" run FIN2-SLEEP-BB-r$r $V/sleep_pacing $LG --ns 10,100,1000,10000 --drags 10 --replicate $r
  # matched busy-waiting control (same drags, same order in time), for a clean per-tool sleep-vs-spin comparison
  RUN_H="$HB --pacing spin" run FIN2-SPINCTL-TB-r$r $V/sleep_pacing_spin $TB --ns 10,100,1000,10000 --drags 10 --replicate $r
  RUN_H="$HB --pacing spin" run FIN2-SPINCTL-BB-r$r $V/sleep_pacing_spin $LG --ns 10,100,1000,10000 --drags 10 --replicate $r
done
unset RUN_H
echo "$(date +%H:%M) SLEEP DONE"
# 4) panels c and f (memory; image size with the same pacing/thumbnail settings)
if [[ " $STAGES " == *" post "* ]] && ! stop_requested && [ ! -s $V/memory/benchmark_memory.csv -o ! -s $V/resolution/benchmark_resolution.csv ]; then
  for attempt in 1 2; do
    wait_ac; t0=$(date +%s)
    PACING=$PACING TB_THUMBS=$TB_THUMBS EDIT_VIEW=$EDIT_VIEW OUT=$V caffeinate -dims $R/paper_benchmarks/final_run/run_post.sh > $LOG/post.out 2>&1; rc=$?
    if [ $rc -eq 0 ] && /usr/bin/python3 $R/paper_benchmarks/final_run/sleep_check.py $t0 $(date +%s) > $LOG/POST.power 2>&1; then
      echo "$(date +%H:%M) POST rc=$rc clean"; break
    fi
    echo "$(date +%H:%M) POST rc=$rc INVALID (crash, sleep or battery), attempt $attempt"
    mkdir -p $V/_invalid/POST-a$attempt
    mv -n $V/memory $V/resolution $LOG/post.out $LOG/POST.power $V/_invalid/POST-a$attempt/ 2>/dev/null
    mkdir -p $V/memory $V/resolution
    if [ $attempt -eq 2 ]; then echo "$(date +%H:%M) POST INVALID twice - aborting"; touch $V/ABORTED; exit 1; fi
  done
fi
# 5) full-reassignment baseline (napari's multiple-viewer example pattern), panel a N <= 1000, k = 3 x 10 drags
[[ " $STAGES " == *" tut "* ]] && for r in 0 1 2; do
  run FIN2-TUT-BB-r$r $V/tutorial_full --tool legacy --legacy-sync full --edit-view api --panels a --drags 10 --replicate $r
done
echo "$(date +%H:%M) FINAL DONE"
