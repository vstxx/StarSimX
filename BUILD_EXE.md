# StarSimX build instructions

# 1) Create a venv and install deps
# py -3.11 -m venv .venv
# .venv\Scripts\pip install -r requirements.txt

# 2) Build an EXE with PyInstaller (GUI app)
# .venv\Scripts\pip install pyinstaller
# GUI, windowed (no console):
# .venv\Scripts\pyinstaller -w -F -n StarSimX --hidden-import matplotlib.backends.backend_tkagg --hidden-import mpl_toolkits.mplot3d --paths . StarSimX/StarSimX.py
# The EXE will be in dist/StarSimX.exe

# If you prefer console + GUI:
# .venv\Scripts\pyinstaller -F -n StarSimX --hidden-import matplotlib.backends.backend_tkagg --hidden-import mpl_toolkits.mplot3d --paths . StarSimX/StarSimX.py

# Note: Tkinter ships with most CPython installers. Ensure Python was installed with tcl/tk.

