"""Run a command; on failure, repeat the tail of its output as a GitHub annotation."""
import collections
import subprocess
import sys

tail = collections.deque(maxlen=60)
p = subprocess.Popen(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, encoding="utf-8", errors="replace")
for line in p.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
    tail.append(line.rstrip())
code = p.wait()
if code != 0:
    msg = "%0A".join(l.replace("%", "%25").replace("\r", "") for l in tail)
    print(f"::error title=step failed ({code})::{msg}")
sys.exit(code)
