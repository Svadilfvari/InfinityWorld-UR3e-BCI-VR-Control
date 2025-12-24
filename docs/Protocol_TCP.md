# TCP Protocol (Python → Unity)

Unity runs a TCP server on **port 12345** (see `InverseKinematicsKeyboardControl.cs`) and expects short ASCII commands.

## Message format

Messages are plain text; examples:

### Discrete translation commands
- `left`
- `right`
- `up`
- `down`
- `forward`
- `backward`

The script updates the target Cartesian position `(px, py, pz)` by a fixed step (`positionStep`).

### Orientation commands (degrees)
- `rx <deg>`
- `ry <deg>`
- `rz <deg>`

Examples:
- `rx 70`
- `ry -15`
- `rz 90`

## Notes / recommended improvements

- Add a delimiter (e.g., newline) and parse multiple commands per packet to avoid partial reads.
- Consider sending JSON like `{"rx":70.0,"ry":0.0,"rz":0.0}` to reduce ambiguity and enable batching.
