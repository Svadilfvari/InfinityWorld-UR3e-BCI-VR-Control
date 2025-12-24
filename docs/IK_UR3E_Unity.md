# Inverse kinematics used in Unity (UR3e-like arm)

This project uses a lightweight analytical IK to move a UR3e-like 6-DOF chain in Unity.

> The script uses a simplified geometric model with link lengths `L1`, `L2`, `L3` for the first three joints and then solves
> the wrist from orientation vectors.

---

## 1) Inputs

- Target position: `(px, py, pz)`
- Target tool orientation: `(rx, ry, rz)` (degrees, treated like Euler angles)
- Link lengths (meters): `L1`, `L2`, `L3`

---

## 2) Solve base + shoulder + elbow (θ0, θ1, θ2)

### θ0 (base yaw)
`θ0 = atan2(p5y, p5x)`

### θ2 (elbow) — cosine rule

Let:

- `r² = p5x² + p5y² + (p5z - L1)²`

Then:

- `C3 = (r² - L2² - L3²) / (2 L2 L3)`  
- Clamp `C3` to `[-1, 1]`
- Elbow-up solution:
  - `θ2 = atan2(+sqrt(1 - C3²), C3)`
- Elbow-down alternative:
  - `θ2 = atan2(-sqrt(1 - C3²), C3)`

### θ1 (shoulder)

- `A = sqrt(p5x² + p5y²)`
- `B = p5z - L1`
- `M = L2 + L3*C3`
- `θ1 = atan2(M*A - L3*sin(θ2)*B, L3*sin(θ2)*A + M*B)`

---

## 3) Solve wrist orientation (θ3, θ4, θ5)

The script computes two direction vectors from `(rx, ry, rz)`:

- `a = (ax, ay, az)` tool direction  
- `b = (bx, by, bz)` secondary direction

Then rotates them into the wrist frame and solves:

- `θ3 = atan2(asy, asx)`
- `θ4 = atan2(cos(θ3)*asx + sin(θ3)*asy, asz)`
- `θ5 = atan2(cos(θ3)*bsy - sin(θ3)*bsx, -bsz / sin(θ4))`

A guard is used when `sin(θ4)` is near 0.

---

## Common pitfalls

- Units mismatch: link lengths are meters but Unity coordinates may be scaled.
- Multiple solutions (elbow up/down, wrist flip) and singularities.
- Euler conventions: if wrist motion looks wrong, switch to quaternions and compute vectors from a rotation matrix.
