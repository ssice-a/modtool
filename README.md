# Mod Tool

This Blender add-on focuses on a few practical mesh workflows that are common when cleaning up game or mod assets.

## Features

- Vertex-group renaming with rule-based and sequential modes
- Shape-key transfer between overlapping meshes with different vertex counts
- Shape-key-safe deletion for selected mesh geometry
- Shape-key-safe modifier apply for enabled modifiers, including data-transfer and vertex-group edits
- One-click seam vertex-group renaming from the active mesh to the selected meshes
- One-click material creation from object names
- Split selected meshes by material and rename the pieces automatically
- Armature FK cleanup that keeps weighted bones, removes control bones, strips pose constraints, and unlocks pose transforms

## Usage

1. Install this folder as a Blender add-on, or link it into `scripts/addons/vertextool`
2. Enable `Mod Tool` in `Edit > Preferences > Add-ons`
3. Open `View3D > Sidebar > Mod Tool`

## Panels

- `Vertex Groups`
  Rename the active mesh object's vertex groups, clean empty groups, map group names, and rename selected seam groups to the active mesh.
- `Shape Keys`
  Transfer shape keys between overlapping meshes, delete selected geometry, and apply enabled modifiers while preserving existing shape keys.
- `Materials`
  Create one material per selected object or split selected objects by their material slots.
- `Armature`
  Convert a rig to a direct FK skeleton by keeping bones with weighted matching vertex groups, removing unweighted bones, clearing pose constraints, and unlocking pose transforms.

## Code Layout

- `core/`
  Shared helpers and feature logic for rename, seam, and material workflows.
- `ops/`
  Blender operators, split by feature area.
- `ui/`
  Sidebar panels and UI lists.
- `properties.py`
  Scene properties and property groups.
