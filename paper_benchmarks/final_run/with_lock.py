"""Run a command under an exclusive machine-wide lock so timing-sensitive or
CPU-heavy runs (anything that builds napari viewers, pytest, profiles) never
overlap. Usage: python with_lock.py [--tag TAG] -- cmd args...
The lock file and the journal (start/end/duration/rc lines, lock_journal.log) are in
benchmark_logs/ in the repository root. Works on Python 3.9+ (fcntl.flock); POSIX only."""
import fcntl, os, subprocess, sys, time, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "benchmark_logs")
os.makedirs(ROOT, exist_ok=True)
LOCK = os.path.join(ROOT, ".timing.lock")
JOURNAL = os.path.join(ROOT, "lock_journal.log")
args = sys.argv[1:]
tag = "untagged"
if args and args[0] == "--tag":
    tag, args = args[1], args[2:]
if args and args[0] == "--":
    args = args[1:]
if not args:
    sys.exit("usage: with_lock.py [--tag TAG] -- cmd ...")
def jot(msg):
    with open(JOURNAL, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} [{tag}] {msg}\n")
with open(LOCK, "w") as lk:
    t_wait = time.time()
    fcntl.flock(lk, fcntl.LOCK_EX)
    waited = time.time() - t_wait
    jot(f"START (waited {waited:.0f}s): {' '.join(args)}")
    t0 = time.time()
    rc = subprocess.call(args)
    jot(f"END rc={rc} dur={time.time()-t0:.1f}s")
sys.exit(rc)
