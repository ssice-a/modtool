# Mod Tool Development Rules

This file is binding for future work in this repository. Read it before changing code.

## Core Priorities

1. Performance comes first. Any change that can slow the addon, the Blender UI, or large mesh workflows must be treated as risky until measured or reasoned through.
2. Reuse comes second. Prefer existing helpers and local patterns. If two paths do the same scan, conversion, matching, or validation, move the shared part into a reusable helper instead of creating another private copy.
3. Cleanliness comes third. Do not keep old compatibility paths when the new behavior replaces them. Remove obsolete code, unused helpers, and duplicate logic instead of letting the addon grow sideways.

## NumPy / npy Requirement

- When addon code can use NumPy (`import numpy as np`) or `.npy`-style array storage to clearly reduce repeated scans, Python loops, coordinate math, weight math, point-cloud scoring, shape-key delta calculation, or large mesh data transforms, use it.
- Prefer this data path for heavy numeric work: Blender `foreach_get` / `foreach_set` at the RNA boundary, NumPy arrays for bulk computation, then `foreach_set` or Blender APIs to write results back.
- Do not use NumPy as decoration. Use it when it reduces asymptotic cost, removes repeated scans, batches math, or avoids per-element RNA access.
- If NumPy is unavailable in the target Blender Python, isolate the import and fallback in one helper. Do not duplicate full algorithms just to support both paths.
- Use persistent `.npy` cache files only when cache invalidation is exact. The cache key must account for mesh identity, vertex count, topology, relevant modifiers, shape keys, and settings. If validity is uncertain, use in-memory arrays instead.

## Blender Performance Rules

- UI `draw()` and operator `poll()` must stay light. Never scan vertices, polygons, selected scene objects, modifiers, shape keys, or vertex weights from UI draw unless the result is already cached and cheap to read.
- Avoid repeated full-mesh scans. If an operation needs count plus mutation, do it in one pass and return counts from the mutating helper.
- For coordinates, polygon indices, material indices, selection-like dense attributes, and shape-key point coordinates, prefer `foreach_get` / `foreach_set` over per-element access.
- For sparse vertex-group weights, remember that Blender exposes weights through `vertex.groups`. NumPy only helps after the sparse data has been collected, so first remove duplicate scans and cache per-operation group index/name maps.
- For spatial work, use `KDTree`, `BVHTree`, grid hashing, or point-cloud compression. Avoid O(n*m) matching on real meshes.
- Minimize `context.view_layer.update()` calls. They are expensive; batch property changes before updating.

## Feature-Specific Guidance

- Vertex-group mapping should work from compact weighted point clouds. Compress or sample large point clouds before KDTree voting when the result remains stable.
- Shape-key and modifier operations should capture coordinates in flat arrays, compute deltas in batches when possible, and write back with `foreach_set`.
- Material operations should batch polygon material index reads and writes.
- Armature cleanup should avoid scanning bones, constraints, locks, and mesh weights more than once per operation stage.
- Seam tools should cache group-name lookups and avoid re-reading the same vertex weights inside matching loops.

## Verification

- Run focused syntax checks without modifying tracked `__pycache__` files. Prefer AST parsing or `compile()`-based checks that do not write bytecode; avoid whole-repo `compileall` and plain `py_compile` in this repository.
- Run `git diff --check` before finishing.
- Summaries must mention any optimization that was deliberately not implemented and why.
