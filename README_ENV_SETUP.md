# Environment Setup for human_detection_tracking_drone.py

This document explains how to create a Python virtual environment and install the required dependencies to run the script `human_detection_tracking_drone.py`.

## Steps

1. Create a virtual environment named `venv`:

```bash
python -m venv venv
```

2. Activate the virtual environment:

- On Windows (cmd):

```bash
venv\Scripts\activate
```

- On Windows (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Run the script:

```bash
python human_detection_tracking_drone.py
```

## Notes

- Ensure you have Python 3.6 or higher installed.
- The dependencies include:
  - opencv-python
  - numpy
  - pillow
  - transformers
