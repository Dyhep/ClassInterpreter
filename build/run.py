"""Run a command; on failure, repeat the tail of its output as a GitHub annotation."""
import collections
import subprocess
import sys

notice = sys.argv[1] == "--notice"
args = sys.argv[2:] if notice else sys.argv[1:]
tail = collections.deque(maxlen=80)
p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, encoding="utf-8", errors="replace")
for line in p.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
    tail.append(line.rstrip())
code = p.wait()
if code != 0 or notice:
    msg = "%0A".join(l.replace("%", "%25").replace("\r", "") for l in tail)
    kind = "error" if code else "notice"
    print(f"::{kind} title=output ({code})::{msg}")
sys.exit(code)
