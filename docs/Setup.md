# Setup notes

## Unity
This repo does **not** include a full Unity project to keep it lightweight.
Copy scripts from `unity/Assets/Scripts/` into your Unity project.

### Assigning joints
In the Unity Inspector, set `Joints[0..5]` to the UR3e joint transforms in order.

## Python
Install dependencies:
```bash
cd python
pip install -r requirements.txt
```

### Head-motion mode
Run:
```bash
python stream_accel_to_unity.py --device-name "BA MINI 003"
```

### EEG mode
1) Put your weights in:
- `python/models/theta.pth`
- `python/models/lambda.pth`

2) Start Unity, then run:
```bash
python eeg_to_unity_bridge.py --eeg-listen-port 12346 --unity-port 12345
```

3) Run your EEG chunk sender (the process that pickles and sends (T,C) windows to port 12346).

## Media
- `media/demo.gif` (your recorded demo)
- `media/pipeline.png` (included)
