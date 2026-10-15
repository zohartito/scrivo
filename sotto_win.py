# Stand-in: Scrivo was called Sotto, and a console run or restart started before the rename runs this file.
# Delete after installs from before the scrivo_win.py rename have updated.
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    # In this process (Windows has no exec): scrivo_win.py as __main__, with its own path as argv[0].
    sys.argv[0] = str(Path(__file__).resolve().with_name("scrivo_win.py"))
    runpy.run_path(sys.argv[0], run_name="__main__")
