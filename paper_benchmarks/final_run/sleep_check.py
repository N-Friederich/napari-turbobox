"""Exit 1 if macOS slept (pmset 'Entering Sleep') or ran on battery between two epoch times."""
import subprocess
import sys
from datetime import datetime

start, end = float(sys.argv[1]), float(sys.argv[2])
proc = subprocess.run(["pmset", "-g", "log"], capture_output=True, text=True)
if proc.returncode != 0 or not proc.stdout.strip():
    print(f"power history unknown (pmset rc={proc.returncode}); treating the run as invalid")
    sys.exit(2)
log = proc.stdout.splitlines()
bad = []
for line in log:
    if len(line) < 25 or not line[:4].isdigit():
        continue
    try:
        t = datetime.strptime(line[:25], "%Y-%m-%d %H:%M:%S %z").timestamp()
    except ValueError:
        continue
    bad_event = any(k in line for k in ("Entering Sleep", "Using Batt", "Using BATT", "DarkWake", "Clamshell",
                                         "Display is turned off"))
    if start <= t <= end + 30 and bad_event:
        bad.append(line[:120])
print("\n".join(bad[:5]) if bad else "no sleep, AC power throughout")
sys.exit(1 if bad else 0)
