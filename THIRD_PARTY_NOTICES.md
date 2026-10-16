# EIEM attribution

The stance calibration, Tda Miku reference constants, semi-standard MMD control
folds, swing-twist projection, staged FK/IK design, and confidence-weighted knee
hint in `core/mmd_animation.py` and `core/mmd_reference.py` are adapted from
[Sasye/EIEM](https://github.com/Sasye/EIEM), revision
`1bc9baa15df9eef712e53c9ee57efc7c14cf8436`.

Reference files: `src/direct_vmd_pose.h`, `src/direct_vmd_source.h`, and
`src/direct_vmd_knee.h`. Original author: Sasye and EIEM contributors.
The Python/NumPy implementation and Blender action baking are modifications
for Mod Tool. The upstream Unity runtime, hooks, and FinalIK binaries are not
included.

Upstream is licensed under GNU Affero General Public License version 3.
The adapted modules are distributed under AGPL-3.0-only; the license text
is in [LICENSES/EIEM-AGPL-3.0.txt](LICENSES/EIEM-AGPL-3.0.txt).
Redistribution of the combined addon must comply with the upstream license,
including its corresponding-source requirements. This notice does not assert
that unrelated pre-existing files were originally authored by EIEM.
