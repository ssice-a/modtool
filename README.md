# Mod Tool

This Blender add-on focuses on a few practical mesh workflows that are common when cleaning up game or mod assets.

## Features

- Target-preserving welding of two closed boundary loops with different vertex counts
- Joint editing of selected meshes with native sculpt and weight-paint brushes and fixed topology
- Reversible Base Color material preparation for Substance Painter
- Vertex-group renaming with rule-based and sequential modes
- Spatial vertex-group name mapping from a source mesh to a target mesh
- Shape-key transfer between overlapping meshes with different vertex counts
- Shape-key-safe deletion for selected mesh geometry
- Batch modifier apply on selected meshes, preserving existing shape keys, including data-transfer and vertex-group edits
- One-click seam vertex-group renaming from the active mesh to the selected meshes
- Selected-loop seam weight unification with interpolation for unequal vertex counts
- One-click material creation from object names
- Split selected meshes by material and rename the pieces automatically
- Armature FK cleanup that preserves the skeleton hierarchy while stripping pose constraints and locks

## Usage

1. Install this folder as a Blender add-on, or link it into `scripts/addons/vertextool`
2. Enable `Mod Tool` in `Edit > Preferences > Add-ons`
3. Open `View3D > Sidebar > Mod Tool`

## Panels

- `Mesh`
  In Edit Mode, join the two prepared mesh parts into one object. Select exactly two closed boundary loops, make an edge or vertex on the target loop active, then run `Bridge and Weld Loops`. The target loop is retained, the source loop is welded onto it, and no bridge faces are generated.
- `Vertex Groups`
  Rename the active mesh object's vertex groups, clean empty groups, map group names, and rename selected seam groups to the active mesh.
- `Shape Keys`
  Transfer shape keys between overlapping meshes, delete selected geometry, and apply enabled modifiers while preserving existing shape keys. `Apply Enabled Modifiers` processes every selected mesh, including meshes without shape keys. Disabled modifiers remain; objects without enabled modifiers are skipped.
- `Materials`
  Create object materials, split selected objects by material, or temporarily merge the active mesh's Base Color materials into one fixed-grid SP atlas. `Restore Base Color Materials` splits the painted atlas and restores the recorded UV layout and material slots.
- `Armature`
  Convert a constrained pose rig to direct FK controls by clearing pose constraints and locks while preserving every bone and parent relationship.

### Weight Mirror

`Vertex Groups > L -> R / R -> L` copies weights to the destination X half
without changing the source half, centerline, group names, or interaction mode.
Name matching includes `Bip001_LUpArmTwist`, `Bip001_LThighTwist`, separated
`-L-` / `_L_` tokens, and `.l` / `.r` suffixes. The spatial left side is inferred
from the paired groups' weight distributions. Matching uses the current shape-key
mix without pose or modifier deformation: coincident mirrored vertices are copied
directly; differing topology uses barycentric interpolation on the source side only.
Zero weights clear previous destination assignments. Center deform groups accompany
the paired groups; masks and unpaired sided groups remain unchanged. Mirroring does
not normalize weights, so exact matches retain the source numerical values.
Locked groups that would change stop the operation before any weight is written.
Unmatched destination vertices are reported and retained. Spatial data and NumPy
weight arrays are rebuilt per operation; repainting invalidates a persistent cache.

### Joint Edit

Select at least two meshes, choose `Sculpt` or `Weight Paint`, then click
`Mesh > Joint Edit > Start Joint Edit`. During the session, use the two mode
buttons to switch between sculpting and weight painting. In Weight Paint,
choose the active vertex group from the panel's list; groups with the same
name are shared across all selected meshes, regardless of their group order.
A temporary mesh combines their current shape using the active object's origin
and orientation, with unit scale. Original objects are hidden while native Blender brushes edit the union.
`Apply` writes vertex displacements and painted weights back to the originals;
`Cancel` discards both. New groups are created only on originals that receive
nonzero weights. Explicitly added empty groups are kept on all originals.
Renaming, deleting, adding, merging and locking/unlocking groups on the proxy
are supported. Apply copies the final named weights and removes deleted groups
from the originals, including empty groups. Erased weights are removed, and no
extra normalization runs on Apply. Shared groups are initially locked if any
source locks them; unchanged source locks are retained individually, while
explicit proxy lock changes propagate. Changes to hidden source weights during
the session are rejected before writes.
Original objects stay separate. UVs, material slots, shape-key values and
modifiers are retained. Shape-key coordinates all receive the same local vertex
displacement, preserving differences between keys without baking their current
values into Basis. Armature modifiers can stay enabled in the current pose.
`Apply` measures the actual bone deformation on temporary copies, converts the
sculpt displacement back to the original mesh, and checks that the evaluated
result matches the sculpt. Bone rotation, scale, blended weights, modifier
influence masks and Preserve Volume are included.
Painted weights are applied after that check, so their new bone deformation is
not baked into mesh or shape-key coordinates. The painting surface stays in
the starting pose; Apply restores the original Armature-driven preview.

This version requires fixed topology and single-user editable mesh data. Other
enabled modifiers must be disabled first. Dynamic topology, remeshing, changed
vertex correspondence, proxy transforms, or changes to original geometry,
modifier settings or visible bone deformation during the session are rejected.
Zero-scale or otherwise non-invertible bone deformation cannot be applied.
Coordinate-dependent deformation such as B-Bones and envelopes is validated
against the actual result; an inaccurate inverse leaves the originals unchanged
and retains the sculpt session. Topology and Face Set auto-masking are
temporarily disabled on the starting brush and scene; use the same settings if
switching brushes. Sculpt masks are retained. Separate seam vertices are not
welded or synchronized by this tool.

Session references and initial positions are stored in Blender data so normal
undo and saved sessions do not rely on a Python global cache. A native operation
may replace the proxy mesh with a copy; Apply accepts it only when topology,
vertex IDs and stored starting positions remain intact. Apply and Cancel clean
up both unused proxy meshes. No mesh scans or
Python handlers run during brush strokes; sparse weight collection and NumPy bulk transforms run at start and
apply. Inverse skinning uses three batched coordinate probes only when applying
edited vertices; no persistent deformation cache is used. Materials using explicit
UV maps retain those maps; implicit UV sampling
uses a combined render UV on the proxy. Object-dependent procedural coordinates
can differ on the temporary union.

### Output Color Bake

In `Materials > Output Color Bake`, connect the desired **color output** directly
to each material's `Material Output > Surface`. Click `Read Materials`, then choose
the destination UV separately for each used material. The active UV captured when
reading materials supplies image textures with implicit UV coordinates; existing
explicit UV Map and mapping-node connections keep their original meaning.

Choose a new PNG filepath, resolution, and pixel margin, then `Bake Output Colors`.
The connected color and mixing operations are baked with emission in a temporary
Cycles scene. Original shaders, material slots, shape keys, modifiers, geometry,
UV maps, selection, and interaction mode are retained. A new `BakedUV` layer
combines the chosen destination layouts for use with the resulting single image.
The tool does not reconnect the original materials or merge their slots.

Destination UVs must lie in 0–1. Different materials must not overlap at bake
pixel centers. Zero-area UV triangles are reported, as they cannot receive image
pixels. Shader closures and EEVEE-only `Shader to RGB` in the selected color chain
are rejected; the tool does not guess which part of a shader to bake.

### Unify Seam Weights

In Mesh Edit Mode, select exactly two closed edge loops and make a vertex or edge
on the reference loop active. Click `Unify Seam Weights` in `Vertex Groups`.
Both loops may be in one mesh, or in two meshes being edited together. In the
latter case the active object's loop supplies the reference weights.

Coincident vertices copy the reference weights directly. Other vertices use the
closest point on a reference edge and linearly interpolate its endpoint weights.
The operation replaces only the non-active loop's vertex weights, matching groups
by name across objects. It does not normalize the result, alter geometry, or leave
Edit Mode. The redo panel exposes the coincidence tolerance in world units.

### Offline Group Mapping

Select one or more meshes, choose a target armature in `Vertex Groups`, choose a
mapping, then click `Rename Selected`. The selected armature's actual bone names
are used as the rename targets.

- `auto rig -> NPC`: the existing NPC naming preset.
- `autorig->zmd`: the EF_Size_A / rig_E naming family to the selected ZMD
  armature's body-bone names. Keeps separate limb twist
  destinations, merges each side's breast groups, and merges unsupported butt,
  abdominal, and pelvic deform groups into `Bip001_Pelvis`. Unmapped helper and
  mask groups are removed. Mapped weights are summed and normalized per vertex.
- `vrc->zmd`: the inspected VRC (Gemma / `pear`) group family to ZMD body names.
  This preset works with selected meshes alone; `Target Rig` is optional validation.
  It renames known body and finger groups in place. Unmapped groups, including
  physics chains, retain their names, indices, locks, and exact weights. No groups
  are deleted, no weights are merged, and no normalization or vertex scan runs.
  Conflicting destination names are rejected before changing the mesh.
  Limb controls use the game's main bones; the preset does not synthesize split
  twist weights. Finger segments retain their separate names and weights.

The preset resolves compatible breast aliases from the selected armature. For
example, `breast_base_L_a_01_jnt` falls back to `L_breast_01_jnt` when the
selected rig does not contain the former, or to `xiong_L_0_skin_jnt` on
ZhuangFangyi's skeleton. The right side resolves the corresponding aliases.

For ZMD hands, `hand.l/r` and the four `c_*1_base.l/r` metacarpal groups
merge into the corresponding `Bip001_L/R_Hand`. Each non-thumb finger's
`*1`, `c_*2`, and `c_*3` groups map to the three separate game phalanges.
Thumb assignments are unchanged. Already converted meshes with the old merged
finger weights require the original weights to recover the separate phalanges;
running the preset again cannot split that lost information.

The operation changes only vertex groups. Mesh topology, UVs, shape-key data,
parents, and modifiers are not changed. It does not bind the mesh to a different
armature. The ZMD preset preserves its output names when run again, including
already present supported limb corrective groups; it does not generate new
corrective weights or reproduce the original rig's deformation behavior.

### Animation / 动画

Enable [MMD Tools](https://github.com/MMD-Blender/blender_mmd_tools), then in
`Mod Tool > Animation / 动画` choose a target FK/game armature, then click
`Import MMD Motion / Pose`. The button opens the file browser every time;
select a `.vmd` or `.vpd` file and confirm to import it. Import creates a new
action; it retains the previous action as a
separate datablock. Existing clips are not recalculated when the addon reloads;
import the file again to use the new retargeting algorithm.

The importer adapts the DirectVmd approach from [Sasye/EIEM](https://github.com/Sasye/EIEM):

1. Sample MMD controls using their per-axis Bezier curves and quaternion slerp.
2. Evaluate the virtual Root/Center/Groove hierarchy and semi-standard rotation
   folds. D tracks compose with base tracks; Waist and ShoulderP cancellation
   are explicit. Twist controls are deform branches, not extra elbow parents.
3. Calibrate the target's bind stance against EIEM's Tda Miku reference. Limb
   axes come from child joint heads, not exported bone tails. Palm landmarks
   provide wrist roll; without them, wrist calibration retains the forearm roll.
4. Apply rotation-only FK, preserving target bone offsets and segment lengths.
5. Solve analytic two-segment leg IK with confidence-weighted knee direction,
   then minimally aim the ankle at the toe IK target. Foot IK rotates the toe
   target; its quaternion is not copied directly onto the ankle.
6. Project source twist onto its source reference axis and distribute the
   signed angle across the target's deform twist chain. Bake local transforms.

`Translation Multiplier` defaults to 1. The base conversion is automatic:
mean target leg length divided by the Tda reference leg length (10.62420198).
The multiplier changes animated translations, not the reference pivot offsets
or the model size. `Start Frame` chooses the clip's first frame. VPD gives one
pose; VMD is sampled at the scene frame rate from MMD's 30 fps. Leg/toe IK is
on by default, including controls with no keys; VMD IK toggles can disable it.

**A zero-rotation MMD pose now puts the target into the reference MMD stance.**
This changes its pose, not its rest bones or mesh data. Meshes, vertex weights,
UVs and bone collections are not modified. Supported target names include
Japanese MMD names/metadata, Bip001 game names and common humanoid names.
Known two-stage Bip001 arm/wrist twist branches are supported. Active target
constraints, pose drivers, nonstandard transform inheritance and NLA tracks are
rejected before changing the action.

VMD/VPD do not contain source rest matrices or a complete PMX constraint graph.
This version uses the explicit Tda reference; it does not import EIEM's optional
PMX dependency evaluator or Unity/FinalIK runtime. Motions made for different
source bind stances, special constraints or corrective bones can still require
manual adjustment. Expressions and unknown controls are reported as unmapped;
unsupported body-control translations are also listed in the action's
`mdt_mmd_report`. Its reference name and effective unit scale are stored in
`mdt_mmd_reference` and `mdt_mmd_translation_scale`.

Retargeting uses in-memory NumPy batches and bulk F-Curve writes. It does not
scan meshes or install frame-change handlers. Persistent `.npy` caching is
omitted because motion files, rest bones and settings can change between imports.
Source attribution and AGPL terms are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Proxy Shrinkwrap / 代理缩裹

`Mod Tool > Proxy Shrinkwrap / 代理缩裹` fits thick or layered clothing through
editable low-resolution surfaces and a single native Lattice modifier.

1. Choose `Source Clothing` and `Target Body`, or select two meshes with the
   target active. Align them in modeling/rest space first. `Create Proxies`
   creates independent source and target control surfaces; the originals stay intact.
2. Use `Source Proxy` / `Target Proxy` to select and edit either helper.
   Before binding, source proxy topology can also be edited. Create or paint an
   optional group on **Source Proxy**, then select it in `Proxy Vertex Group`.
   To restrict the actual clothing deformation, select an existing original-mesh
   group in `Clothing Mask`. Its zero-weight vertices stay fixed, fractional
   weights blend the movement once, and inversion is supported. Apply preserves
   the same masked result. The proxy control mask and clothing mask are independent.
3. Choose the native Shrinkwrap method, snap mode, offset and projection axes.
   Set `Lattice Resolution`, then click `Bind & Preview`.
   `Allowed Movement` separately constrains the final clothing displacement:
   enable X/Y and disable Z to retain every vertex's Z coordinate. Choose
   `World` or `Source Local` for rotated objects. These controls affect the volume
   field; native projection axes choose which rays search for the target.
   The constraint preserves coordinates along the locked axis, not the arc length
   of a curved sleeve. Like the other fitting settings, changes take effect on
   `Update Preview` and remain in effect when applying.
4. Edit the proxies, masks or settings and click `Update Preview` to update the
   clothing. Increasing `Volume Smoothness` spreads movement more coherently;
   increasing lattice resolution allows finer local changes. To change the source
   proxy topology or lattice resolution after binding, `Unbind`, edit, then bind again.
5. `Apply` writes the current per-vertex movement into the original mesh and all
   its shape keys, retaining their relative differences, values and animation.
   UVs, topology, materials, weights and existing modifiers stay intact. Shared
   mesh data is made independent on Apply. `Cancel` removes this session's helpers
   and preview without changing original coordinates. Both actions restore the
   initial object selection. A bound preview also survives saving/reloading the file.

Proxy reduction uses native edge collapse rather than spatial vertex clustering,
so nearby disconnected clothing layers are not welded into a single control
surface. Reduction runs only when creating proxies, not during UI drawing or fitting.
Existing simplified proxies are not automatically rebuilt; recreate them to use
the updated reduction, preserving any manual helper edits first.

Proxies use the current shape-key mix **before all modifiers**, including Armature.
The preview lattice is first in the clothing's modifier stack, so an animated
pose is not baked into the clothing. For modeling, inspect the aligned rest pose;
posed source/target silhouettes are deliberately not used to generate proxies.
Apply preserves original shape-key deltas, rather than independently deforming
each key through the nonlinear lattice. Other key values therefore retain their
original offsets from the fitted shape.

Each source proxy point supplies a displacement from its stored bind position.
A regularized, matrix-free NumPy solve fits these displacements to an anisotropic
trilinear lattice. Nearby clothing layers share this smooth volume field instead
of being individually snapped onto the body. Proxy masks control the Shrinkwrap
constraints; the volume field feathers movement into neighboring regions.
The report includes fitting error and effective strength; strength is reduced
when sampled lattice Jacobians approach folding or extreme compression.

This is a fitting tool, not a cloth collision solver. Strong compression can still
change thickness; it does not guarantee exact layer spacing or collision-free
results. Opposite limbs or clothing layers closer than the proxy/grid resolution
can share controls. Inspect the simplified proxies and increase resolution or
restrict their masks when needed. Target Normal Project depends on consistent
target proxy normals. Original modifiers are intentionally excluded, so apply any
modeling modifier whose geometry you want represented before creating proxies.

All heavy reads/writes use bulk arrays. Explicit updates avoid installing a
Python handler during every drag or paint stroke; previews then use Blender's
native lattice evaluation. No persistent numeric cache is used because proxy
geometry, masks and settings remain editable. The bounded proxy budget and
matrix-free solver avoid a dense high-resolution binding matrix.

Workflow reference: [EZ Lattice](https://github.com/kursad-k/ezlattice), for the
bounding lattice / apply / clear interaction. No implementation code is copied;
proxy reduction and the displacement-field fitter are implemented here.

## Code Layout

- `core/`
  Shared helpers and feature logic for rename, seam, and material workflows.
- `ops/`
  Blender operators, split by feature area.
- `ui/`
  Sidebar panels.
- `properties.py`
  Scene properties and property groups.
