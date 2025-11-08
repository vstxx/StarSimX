# StarSimX

Educational N-body star formation simulator.

Quick start (GUI):
- py -3 -m pip install -r requirements.txt
- py StarSimX/StarSimX.py     # GUI launches by default

If Visual Studio shows only "Press any key to continue...":
- Set the startup file to StarSimX/StarSimX.py or run: py run_gui.py
- Ensure the working directory is the repo root.
- Make sure Python environment is Python 3.9 where numpy/matplotlib are installed.

CLI examples:
- py StarSimX/StarSimX.py --cli --N 800 --vrms 0.1 --omega 0.3 --t_end 5.0 --dt 0.002 --gif out.gif
- py StarSimX/StarSimX.py --cli --config example.config.json

Build EXE (Windows with PyInstaller):
- See BUILD_EXE.md

