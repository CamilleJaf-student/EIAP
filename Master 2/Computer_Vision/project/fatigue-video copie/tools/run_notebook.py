"""Convert project.py (jupytext percent format) to project.ipynb and execute it in place.

Usage: python tools/run_notebook.py [--no-exec]
"""
import sys
import time

import jupytext
import nbformat
from nbclient import NotebookClient

nb = jupytext.read("project.py")
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
if "--no-exec" not in sys.argv:
    t0 = time.time()
    NotebookClient(nb, timeout=-1, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
    print(f"executed in {time.time() - t0:.0f} s")
nbformat.write(nb, "project.ipynb")
print("written project.ipynb")
