# fbx-to-kn5-tool

Compile Assetto Corsa `.kn5` model files from an FBX, without ksEditor.

ksEditor only runs on Windows. This tool does its "Save kn5" step in plain Python, so you can build car models on macOS or Linux, or in CI. It reproduces ksEditor's output: on the cars it was tested with, every mesh comes out byte-identical to the kn5 ksEditor writes.

## Requirements

Python 3.8 or newer. Only the standard library is used, so there is nothing to install.

## Usage

```sh
python3 fbx2kn5.py mycar/model.fbx -o mycar/model_lod_a.kn5
```

You can list several outputs to write the same model to every LOD file in one run:

```sh
python3 fbx2kn5.py mycar/model.fbx -o mycar/model_lod_{a,b,c,d}.kn5
```

| Option | Default | Meaning |
|---|---|---|
| `fbx` | | The FBX to compile. Binary FBX only (Blender's exporter writes binary). |
| `-o`, `--output` | | One or more kn5 files to write. |
| `--ini` | `<fbx>.ini` | The ksEditor `.fbx.ini` file with materials and per-object settings. |
| `--textures` | `<fbx folder>/texture` | The folder holding the textures the materials use. |

## Inputs

It uses the same inputs as ksEditor:

- **FBX**, exported Y-up. From Blender, export with scale `0.01` so the car is the right size in game. The tool warns if the file isn't Y-up at unit scale.
- **`.fbx.ini`**, the file ksEditor saves next to the FBX. It supplies each material's shader, variables and texture slots, and each object's flags (visible, transparent, casts shadows, LOD in/out, renderable). Objects missing from the ini get ksEditor's defaults. Materials missing from it get a plain `ksPerPixel` shader and a warning, so add new materials to the ini (or create them once in ksEditor) before building.
- **Textures**, read from the texture folder and embedded in the kn5, in ksEditor's order. Missing textures are skipped with a warning.

## What it reproduces

To produce the same bytes as ksEditor, the tool follows the same steps, with the same floating-point arithmetic and in the same order:

- **Node tree:** every FBX object becomes a kn5 dummy with its local transform, plus a mesh child. A mesh with several materials is split into `NAME_SUB0`, `NAME_SUB1`, … in order of first use.
- **Triangulation:** the FBX SDK 2014.1 algorithm ksEditor uses. Polygons are projected onto their dominant axis plane and triangulated by constrained Delaunay with the SDK's edge insertion, producing the same triangle order, the same first corner per triangle, and the same handling of degenerate polygons. This includes the SDK's rule for copying per-corner normals and UVs onto the new triangles.
- **Vertex building**, in float32 as ksEditor does:
  - positions and normals go through the geometric transform, and normals are normalized;
  - V is flipped;
  - vertices are merged by exact float comparison, keeping the first match;
  - tangents use ksEditor's last-triangle-wins rule;
  - bounding spheres use the vertex average and the farthest vertex.
- **Skinned meshes:** bones with inverse bind matrices, and up to four weights per vertex in cluster order.

### Known differences

- Node transform matrices can differ from ksEditor's in float rounding noise (around 1e-16 or smaller, or a `-0.0` instead of `0.0`). This comes from the FBX SDK's transform evaluation, which depends on the Windows C runtime's `sin`, `cos` and `atan2`. It isn't visible in game, but it means whole-file checksums usually won't match ksEditor's.
- Geometric transforms (pivot offsets baked into the geometry) are supported, but their rounding hasn't been verified against ksEditor.
- A single mesh can have at most 65,535 vertices (16-bit indices), the same limit as ksEditor. The tool stops with an error instead of writing a broken file.

## Verifying a build

Load the result in the Content Manager showroom before shipping it, especially after a big change to the model. `kn5.py` can also read kn5 files, so you can compare a build with a ksEditor-made file:

```python
import kn5

a = kn5.read("ksEditor_build.kn5")
b = kn5.read("tool_build.kn5")
print(len(a.materials), len(b.materials))
```

## Files

| File | Purpose |
|---|---|
| `fbx2kn5.py` | Command line tool: scene, materials, `.fbx.ini`, kn5 assembly. |
| `fbx_triangulate.py` | The FBX SDK 2014.1 polygon triangulation. |
| `ksmesh.py` | ksEditor's float32 mesh building: vertex merging, tangents, bounding spheres. |
| `fbx.py` | Minimal binary FBX (7.x) reader. |
| `kn5.py` | kn5 (version 6) reader and writer. It writes back a file it has read byte for byte. |

## Credits

Written to build an ND2 Miata RF car mod on macOS. The ksEditor behaviour was worked out by comparing its output with this tool's and by analysing the editor's FBX import code.
