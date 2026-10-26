"""Entry point for the packaged Mac app (PyInstaller)."""
import multiprocessing
import sys

from towerdesigner.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
