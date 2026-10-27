# fbx-to-kn5-tool

Build Assetto Corsa cars without the Windows-only tools:

- **`fbx2kn5.py`** compiles `.kn5` model files from an FBX, without ksEditor.
- **`render_previews.py`** renders each skin's `preview.jpg`, without Content Manager's showroom. See [Skin previews](#skin-previews).

ksEditor only runs on Windows. `fbx2kn5.py` does its "Save kn5" step in plain Python, so you can build car models on macOS or Linux, or in CI. It reproduces ksEditor's output: on the cars it was tested with, every mesh comes out byte-identical to the kn5 ksEditor writes.

## Requirements

`fbx2kn5.py`: Python 3.8 or newer. Only the standard library is used, so there is nothing to install.

`render_previews.py`: [uv](https://docs.astral.sh/uv/) and OpenGL 4.1. See [Skin previews](#skin-previews).

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

## Skin previews

`render_previews.py` renders the `skins/<skin>/preview.jpg` images Content Manager shows when you pick a car's colour. It imitates Content Manager's default dark-studio preview: the same camera and 1022x575 size, the same material shading and filmic tone mapping. A procedural studio stands in for the showroom. The look was fitted against previews rendered by Content Manager, and comes out within about 5/255 per pixel on average. It doesn't use any Content Manager or Kunos files.

### Setup

Install [uv](https://docs.astral.sh/uv/). It installs the Python dependencies (moderngl, numpy, Pillow) from `pyproject.toml` and `uv.lock` the first time you run the script.

The renderer needs OpenGL 4.1:

- **macOS**: built in.
- **Windows**: should work with any current GPU driver (not tested).
- **Linux without a display (CI, servers)**: Mesa's software renderer over EGL. On Ubuntu or Debian:

  ```sh
  sudo apt-get install libegl1 libgl1 libgl1-mesa-dri
  ```

  The script tries the normal context first and falls back to EGL by itself.

### Usage

Render every skin of one or more cars, writing each skin's `preview.jpg` in place:

```sh
uv run render_previews.py path/to/mycar path/to/myothercar
```

Render some skins only, or write a single skin somewhere else:

```sh
uv run render_previews.py path/to/mycar --skin red --skin blue
uv run render_previews.py path/to/mycar --skin red --out red.jpg
```

| Option | Meaning |
|---|---|
| `cars` | One or more car folders (the folder holding the kn5 files, `data/` and `skins/`). |
| `--skin` | Render only this skin. Repeat it for several. Default: every folder in `skins/`. |
| `--out` | Write the image here instead of `skins/<skin>/preview.jpg`. Use it with a single `--skin`. |

A preview takes about a second on a Mac, and around 6 seconds with Mesa's software renderer.

### What it reads from the car

- **The model**: the `LOD_0` file from `data/lods.ini`, or the first kn5 in the folder if there is no `lods.ini`.
- **Skins**: textures in `skins/<skin>/` replace the kn5's textures of the same name.
- **Car data** from the unpacked `data/` folder, as Content Manager uses it:
  - `blurred_objects.ini` picks the rims shown at standstill;
  - `ambient_shadows.ini` places `body_shadow.png` and `tyre_N_shadow.png` under the car;
  - the car is set on the ground by its wheels.

  `data.acd` is not read. If your car only has `data.acd`, unpack it (for example with Content Manager) so a `data/` folder sits next to it. Without one, the preview still renders, but without ambient shadows and with the kn5's default rims.
- **Custom Shaders Patch config**: `extension/ext_config.ini` and `skins/<skin>/ext_config.ini`. Meshes that CSP hides for a skin are hidden in its preview too:
  - `[MODEL_REPLACEMENT_...]` sections with `HIDE`, filtered by `FILE`, `SKINS` and `ACTIVE`;
  - `[MESH_ADJUSTMENT_...]` sections with `IS_RENDERABLE = 0`.

  A "roof off" skin, for example, gets a roof-off preview. Content Manager's showroom ignores this config.

Like Content Manager, it hides the cockpit LOD, the steering wheel LOD, fastened seatbelts and broken glass.

### Using it in GitHub Actions

```yaml
- uses: astral-sh/setup-uv@v6
- name: Install OpenGL
  run: sudo apt-get update && sudo apt-get install -y libegl1 libgl1 libgl1-mesa-dri
- name: Render skin previews
  run: uv run render_previews.py mycar
```

### Known differences from Content Manager

- Reflections come from a simple studio light setup, not the showroom's reflection map, so highlights on paint and glass differ slightly.
- Side windows look darker than in Content Manager, which lights the cabin behind the glass more.
- Only the dark-studio default is imitated. Content Manager's other preview presets and showrooms are not.

## Files

| File | Purpose |
|---|---|
| `fbx2kn5.py` | Command line tool: scene, materials, `.fbx.ini`, kn5 assembly. |
| `fbx_triangulate.py` | The FBX SDK 2014.1 polygon triangulation. |
| `ksmesh.py` | ksEditor's float32 mesh building: vertex merging, tangents, bounding spheres. |
| `fbx.py` | Minimal binary FBX (7.x) reader. |
| `kn5.py` | kn5 (version 6) reader and writer. It writes back a file it has read byte for byte. |
| `render_previews.py` | Command line tool: renders skin preview images. |
| `preview_scene.py` | Preview scene: loads the car, applies skin and CSP visibility, grounds it, sets up the camera. |
| `preview_gl.py` | OpenGL renderer: shading, shadows, studio, bloom and tone mapping. |
| `textures.py` | Decodes DDS (compressed and uncompressed), PNG and JPG textures. |
| `pyproject.toml`, `uv.lock` | Python dependencies for the preview renderer (managed with uv). |

## Credits

Written to build Assetto Corsa car mods on non-PC platforms that can't run ksEditor. The ksEditor behaviour was worked out by comparing its output with this tool's and by analysing the editor's FBX import code. The preview renderer follows the shading, shadows and tone mapping of Content Manager's open-source renderer ([AcTools.Render](https://github.com/gro-ove/actools)).
