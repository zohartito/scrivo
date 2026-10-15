# Stand-in: Scrivo was called Sotto, and launchers, login items and the updater built before the rename run this file.
# Delete after installs from before the scrivo.py rename have updated (their next update rebuilds them for scrivo.py).
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    # The same process (pid, launcher, signals, exit code), interpreter options and arguments, now running scrivo.py.
    options = sys.orig_argv[1:len(sys.orig_argv) - len(sys.argv)]
    os.execv(sys.executable, [sys.orig_argv[0], *options, str(Path(__file__).resolve().with_name("scrivo.py")),
                              *sys.argv[1:]])
