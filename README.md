# Mod Tool

This Blender add-on focuses on a few practical mesh workflows that are common when cleaning up game or mod assets.

## Features

- Vertex-group renaming with rule-based and sequential modes
- Shape-key transfer between overlapping meshes with different vertex counts
- Seam mapping analysis and seam-weight transfer between meshes
- One-click material creation from object names
- Split selected meshes by material and rename the pieces automatically

## Usage

1. Install this folder as a Blender add-on, or link it into `scripts/addons/vertextool`
2. Enable `Mod Tool` in `Edit > Preferences > Add-ons`
3. Open `View3D > Sidebar > Mod Tool`

## Panels

- `Vertex Groups`
  Rename the active mesh object's vertex groups and preview the result before applying it.
- `Seam Transfer`
  Analyze seam matches between two meshes, review the stored mapping list, and apply seam-weight transfer.
- `Shape Keys`
  Transfer all shape keys from one overlapping mesh surface to another mesh, even when the vertex counts differ.
- `Materials`
  Create one material per selected object or split selected objects by their material slots.

## Code Layout

- `core/`
  Shared helpers and feature logic for rename, seam, and material workflows.
- `ops/`
  Blender operators, split by feature area.
- `ui/`
  Sidebar panels and UI lists.
- `properties.py`
  Scene properties and property groups.
