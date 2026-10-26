#!/usr/bin/env python3
"""Build an Assetto Corsa .kn5 from an FBX, its .fbx.ini and a texture folder.

Replaces ksEditor's "Save kn5" step so cars can be compiled on macOS/Linux.

    python3 fbx2kn5.py mycar/model.fbx -o mycar/model_lod_a.kn5

The .fbx.ini next to the FBX supplies materials (shader, variables, texture
slots) and per-node flags; textures are read from <fbx dir>/texture unless
--textures is given. Materials missing from the ini get a plain ksPerPixel
material, and nodes missing from it get the ksEditor defaults.
"""
import argparse
import configparser
import math
import os
import struct
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbx  # noqa: E402
import kn5  # noqa: E402
import ksmesh  # noqa: E402

IDENTITY = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)


# ---------------------------------------------------------------- matrices
# Matrices are 4x4 row-major lists in the row-vector convention kn5 uses
# (translation in the last row): p' = p * M.

def mat_mul(a, b):
    return [sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4)) for r in range(4) for c in range(4)]


def mat_translate(x, y, z):
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]


def mat_scale(x, y, z):
    return [x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1]


def _rot_axis(axis, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    if axis == 0:
        return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]
    if axis == 1:
        return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]
    return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


# FBX EOrder -> axis application order (first applied first).
_EULER_ORDERS = {0: (0, 1, 2), 1: (0, 2, 1), 2: (1, 2, 0), 3: (1, 0, 2), 4: (2, 0, 1), 5: (2, 1, 0)}


def mat_euler(rx, ry, rz, order=0):
    m = list(IDENTITY)
    angles = (rx, ry, rz)
    for axis in _EULER_ORDERS.get(order, (0, 1, 2)):
        m = mat_mul(m, _rot_axis(axis, angles[axis]))
    return m


def mat_inverse(m):
    # General 4x4 inverse (Gauss-Jordan).
    a = [list(m[r * 4:r * 4 + 4]) + [1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    for col in range(4):
        piv = max(range(col, 4), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        d = a[col][col]
        if abs(d) < 1e-20:
            raise ValueError("singular matrix")
        a[col] = [v / d for v in a[col]]
        for r in range(4):
            if r != col and a[r][col]:
                f = a[r][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col])]
    return [a[r][4 + c] for r in range(4) for c in range(4)]


# ------------------------------------------------------------ fbx scene

def _name(obj):
    return obj.props[1].split("\0")[0]


def _props70(obj):
    p70 = obj.find("Properties70")
    return {p.props[0]: p.props[4:] for p in p70.children} if p70 else {}


def model_local_matrix(model):
    """FBX local transform, composed per the FBX SDK definition."""
    p = _props70(model)

    def v3(key, default=(0.0, 0.0, 0.0)):
        return tuple(p[key][:3]) if key in p else default

    order = int(p["RotationOrder"][0]) if "RotationOrder" in p else 0
    t = v3("Lcl Translation")
    r = v3("Lcl Rotation")
    s = v3("Lcl Scaling", (1.0, 1.0, 1.0))
    roff, rpiv = v3("RotationOffset"), v3("RotationPivot")
    soff, spiv = v3("ScalingOffset"), v3("ScalingPivot")
    pre = v3("PreRotation")
    post = v3("PostRotation")
    # Column-vector form: T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1.
    # In row-vector form the product is reversed.
    chain = [
        mat_translate(-spiv[0], -spiv[1], -spiv[2]),
        mat_scale(*s),
        mat_translate(*spiv),
        mat_translate(*soff),
        mat_translate(-rpiv[0], -rpiv[1], -rpiv[2]),
        mat_inverse(mat_euler(*post)),
        mat_euler(*r, order),
        mat_euler(*pre),
        mat_translate(*rpiv),
        mat_translate(*roff),
        mat_translate(*t),
    ]
    m = list(IDENTITY)
    for c in chain:
        m = mat_mul(m, c)
    return m


def geometric_matrix(model):
    p = _props70(model)
    t = tuple(p.get("GeometricTranslation", (0, 0, 0))[:3])
    r = tuple(p.get("GeometricRotation", (0, 0, 0))[:3])
    s = tuple(p.get("GeometricScaling", (1, 1, 1))[:3])
    if t == (0, 0, 0) and r == (0, 0, 0) and s == (1, 1, 1):
        return None
    return mat_mul(mat_mul(mat_scale(*s), mat_euler(*r)), mat_translate(*t))


class Scene:
    def __init__(self, path):
        self.version, root = fbx.read(path)
        self.objects = {o.props[0]: o for o in root.find("Objects").children}
        self.children = defaultdict(list)
        self.parents = defaultdict(list)
        for c in root.find("Connections").children:
            child, parent = c.props[1], c.props[2]
            self.children[parent].append(child)
            self.parents[child].append(parent)
        gs = _props70(root.find("GlobalSettings"))
        self.unit_scale = gs.get("UnitScaleFactor", [1.0])[0]
        self.up_axis = gs.get("UpAxis", [1])[0]

    def kids(self, oid, kind):
        return [self.objects[c] for c in self.children[oid] if c in self.objects and self.objects[c].name == kind]


# ---------------------------------------------------------------- geometry

def _layer(geom, name, index=0):
    for le in geom.find_all(name):
        if le.props and le.props[0] == index:
            return le
    return None


def _layer_values(le, data_name, index_name, width, polygon_of, cps, n_pv):
    """Return a per-polygon-vertex list of `width`-tuples for a layer element."""
    data = le.find(data_name).props[0]
    mapping = le.find("MappingInformationType").props[0]
    ref = le.find("ReferenceInformationType").props[0]
    idx = le.find(index_name).props[0] if ref in ("IndexToDirect", "Index") and le.find(index_name) else None
    vals = [tuple(data[i * width:(i + 1) * width]) for i in range(len(data) // width)]
    if idx is not None:
        vals = [vals[i] if i >= 0 else vals[0] for i in idx]
    if mapping == "ByPolygonVertex":
        return vals
    if mapping in ("ByVertice", "ByVertex", "ByControlPoint"):
        return [vals[cps[i]] for i in range(n_pv)]
    if mapping == "ByPolygon":
        return [vals[polygon_of[i]] for i in range(n_pv)]
    if mapping == "AllSame":
        return [vals[0]] * n_pv
    raise ValueError(f"unsupported mapping {mapping}")


def _material_per_polygon(geom, n_poly):
    le = _layer(geom, "LayerElementMaterial")
    if le is None:
        return [0] * n_poly
    mats = le.find("Materials").props[0]
    if le.find("MappingInformationType").props[0] == "AllSame":
        return [mats[0]] * n_poly
    return list(mats)


def read_geometry(geom):
    """Control points, polygons and per polygon-vertex normals/UVs of a Geometry node."""
    verts = geom.find("Vertices").props[0]
    cps = [tuple(verts[i:i + 3]) for i in range(0, len(verts), 3)]
    pvi = geom.find("PolygonVertexIndex").props[0]
    pv_cp, polygon_of, polys = [], [], []
    cur = []
    for i, v in enumerate(pvi):
        last = v < 0
        pv_cp.append(~v if last else v)
        polygon_of.append(len(polys))
        cur.append(i)
        if last:
            polys.append(cur)
            cur = []
    n_pv = len(pv_cp)
    le_n = _layer(geom, "LayerElementNormal")
    if le_n is not None:
        normals = _layer_values(le_n, "Normals", "NormalsIndex", 3, polygon_of, pv_cp, n_pv)
    else:
        # The FBX SDK leaves missing normals at zero.
        normals = [(0.0, 0.0, 0.0)] * n_pv
    le_uv = _layer(geom, "LayerElementUV")
    uvs = _layer_values(le_uv, "UV", "UVIndex", 2, polygon_of, pv_cp, n_pv) if le_uv is not None else None
    per_corner = (_by_polygon_vertex(le_n), _by_polygon_vertex(le_uv))
    return cps, pv_cp, polys, normals, uvs, _material_per_polygon(geom, len(polys)), per_corner


def _by_polygon_vertex(le):
    return le is not None and le.find("MappingInformationType").props[0] == "ByPolygonVertex"


# ---------------------------------------------------------------- ini

class FbxIni:
    def __init__(self, path):
        self.cp = configparser.ConfigParser(interpolation=None, strict=False)
        self.cp.optionxform = str
        if path and os.path.exists(path):
            with open(path, encoding="utf-8-sig", errors="replace") as f:
                self.cp.read_string(f.read())
        self.materials = {}
        for sec in self.cp.sections():
            if sec.startswith("MATERIAL_") and sec != "MATERIAL_LIST":
                m = self._material(self.cp[sec])
                self.materials[m.name] = m

    @staticmethod
    def _floats(s, n):
        vals = [float(x) for x in s.split(",")] if s else []
        return tuple((vals + [0.0] * n)[:n])

    def _material(self, s):
        m = kn5.Material(s.get("NAME"), s.get("SHADER", "ksPerPixel"),
                         int(s.get("ALPHABLEND", 0)), int(s.get("ALPHATEST", 0)), int(s.get("DEPTHMODE", 0)))
        for i in range(int(s.get("VARCOUNT", 0))):
            pre = f"VAR_{i}_"
            m.props.append(kn5.MatProp(
                s.get(pre + "NAME"),
                float(s.get(pre + "FLOAT1", 0)),
                self._floats(s.get(pre + "FLOAT2"), 2),
                self._floats(s.get(pre + "FLOAT3"), 3),
                self._floats(s.get(pre + "FLOAT4"), 4)))
        for i in range(int(s.get("RESCOUNT", 0))):
            pre = f"RES_{i}_"
            m.mappings.append((s.get(pre + "NAME"), int(s.get(pre + "SLOT", 0)), s.get(pre + "TEXTURE")))
        return m

    def node(self, key):
        return self.cp[key] if self.cp.has_section(key) else {}


def default_material(name):
    m = kn5.Material(name, "ksPerPixel")
    for pname, val in (("ksAmbient", 0.6), ("ksDiffuse", 0.6), ("ksSpecular", 0.0),
                       ("ksSpecularEXP", 1.0), ("ksEmissive", 0.0), ("ksAlphaRef", 0.0)):
        m.props.append(kn5.MatProp(pname, val))
    return m


# ---------------------------------------------------------------- builder

class Builder:
    def __init__(self, scene, ini, texture_dir, log):
        self.scene, self.ini, self.texture_dir, self.log = scene, ini, texture_dir, log
        self.materials, self.material_index = [], {}

    def material_id(self, name):
        if name not in self.material_index:
            m = self.ini.materials.get(name)
            if m is None:
                self.log(f"warning: material {name!r} not in fbx.ini, using default ksPerPixel")
                m = default_material(name)
            self.material_index[name] = len(self.materials)
            self.materials.append(m)
        return self.material_index[name]

    def _flags(self, node, key, mesh):
        s = self.ini.node(key)
        node.active = int(s.get("ACTIVE", 1))
        if mesh:
            node.visible = int(s.get("VISIBLE", 1))
            node.transparent = int(s.get("TRANSPARENT", 0))
            node.cast_shadows = int(s.get("CAST_SHADOWS", 1))
            node.lod_in = float(s.get("LOD_IN", 0))
            node.lod_out = float(s.get("LOD_OUT", 0))
            node.renderable = int(s.get("RENDERABLE", 1))

    def build(self, fbx_name):
        root = kn5.Node(1, f"FBX: {fbx_name}")
        key = f"model_FBX: {fbx_name}"
        self._flags(root, key, False)
        root.children = [self.model_node(m, key) for m in self.scene.kids(0, "Model")]
        return root

    def model_node(self, model, parent_key):
        name = _name(model)
        key = f"{parent_key}_{name}"
        node = kn5.Node(1, name, matrix=tuple(model_local_matrix(model)))
        self._flags(node, key, False)
        geoms = self.scene.kids(model.props[0], "Geometry")
        if geoms:
            node.children.extend(self.mesh_nodes(model, geoms[0], key))
        for child in self.scene.kids(model.props[0], "Model"):
            node.children.append(self.model_node(child, key))
        return node

    def mesh_nodes(self, model, geom, key):
        name = _name(model)
        mats = [_name(m) for m in self.scene.kids(model.props[0], "Material")] or ["DEFAULT"]
        skin = self.skin_data(geom)
        geo = geometric_matrix(model) or IDENTITY
        rows = [ksmesh.r32(geo[r * 4:r * 4 + 4]) for r in range(4)]
        batches = ksmesh.build_batches(*read_geometry(geom), rows)
        out = []
        for sub, batch in enumerate(batches):
            mname = mats[batch.material] if batch.material < len(mats) else mats[0]
            # A multi-material mesh is split into NAME_SUB0, NAME_SUB1, ... like ksEditor does.
            mesh_name = f"{name}_SUB{sub}" if len(batches) > 1 else name
            out.append(self.make_mesh(mesh_name, batch, mname, skin, f"{key}_{mesh_name}"))
        return out

    def skin_data(self, geom):
        skins = [d for d in self.scene.kids(geom.props[0], "Deformer") if d.props[2] == "Skin"]
        if not skins:
            return None
        # Up to four (weight, bone) slots per control point, filled in cluster order.
        bones, weights = [], defaultdict(list)
        for cluster in self.scene.kids(skins[0].props[0], "Deformer"):
            links = self.scene.kids(cluster.props[0], "Model")
            if not links:
                continue
            bi = len(bones)
            link = cluster.find("TransformLink")
            bind = link.props[0] if link else IDENTITY
            bones.append((_name(links[0]), tuple(mat_inverse(bind))))
            idx = cluster.find("Indexes")
            w = cluster.find("Weights")
            if idx and w:
                for cp, wt in zip(idx.props[0], w.props[0]):
                    if len(weights[cp]) < 4:
                        weights[cp].append((ksmesh.f32(wt), bi))
                    else:
                        self.log(f"warning: control point {cp} has more than 4 bone weights; extra ignored")
        return bones, weights

    def make_mesh(self, name, batch, mname, skin, key):
        if len(batch.pos) > 65535:
            raise SystemExit(f"error: mesh {name!r} has {len(batch.pos)} vertices; kn5 meshes are limited to 65535")
        tan = ksmesh.tangents(batch)
        node = kn5.Node(3 if skin else 2, name)
        self._flags(node, key, True)
        node.material = self.material_id(mname)
        node.indices = batch.indices
        buf = bytearray()
        if skin:
            bones, weights = skin
            node.bones = list(bones)
            for p, n, uv, t, cp in zip(batch.pos, batch.nrm, batch.uv, tan, batch.cp):
                ws = weights.get(cp, [])
                wv = [w for w, _ in ws] + [0.0] * (4 - len(ws))
                iv = [float(b) for _, b in ws] + [-1.0] * (4 - len(ws))
                if iv[0] == -1.0:
                    iv[0] = 0.0
                buf += struct.pack("<19f", *p, *n, *uv, *t, *wv, *iv)
        else:
            for p, n, uv, t in zip(batch.pos, batch.nrm, batch.uv, tan):
                buf += struct.pack("<11f", *p, *n, *uv, *t)
            node.sphere = ksmesh.bounding_sphere(batch.pos)
        node.vertices = bytes(buf)
        return node

    def textures(self):
        out, seen = [], set()
        for m in self.materials:
            for _, _, tex in m.mappings:
                if tex in seen:
                    continue
                seen.add(tex)
                path = os.path.join(self.texture_dir, tex)
                if not os.path.exists(path):
                    self.log(f"warning: texture {tex!r} not found in {self.texture_dir}, skipped")
                    continue
                with open(path, "rb") as f:
                    out.append(kn5.Texture(tex, f.read()))
        return out


def convert(fbx_path, ini_path=None, texture_dir=None, log=print):
    ini_path = ini_path or fbx_path + ".ini"
    texture_dir = texture_dir or os.path.join(os.path.dirname(os.path.abspath(fbx_path)), "texture")
    if not os.path.exists(ini_path):
        log(f"warning: {ini_path} not found, using default materials and flags")
    scene = Scene(fbx_path)
    if scene.unit_scale != 1.0 or scene.up_axis != 1:
        log(f"warning: FBX has UnitScaleFactor={scene.unit_scale}, UpAxis={scene.up_axis}; "
            "expected 1.0 and Y-up (export from Blender with scale 0.01, Y up), output is not rescaled")
    b = Builder(scene, FbxIni(ini_path), texture_dir, log)
    root = b.build(os.path.basename(fbx_path))
    return kn5.Kn5(textures=b.textures(), materials=b.materials, root=root)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fbx")
    ap.add_argument("-o", "--output", required=True, nargs="+", help="kn5 file(s) to write (e.g. every LOD)")
    ap.add_argument("--ini", help="fbx.ini path (default: <fbx>.ini)")
    ap.add_argument("--textures", help="texture folder (default: <fbx dir>/texture)")
    args = ap.parse_args()
    model = convert(args.fbx, args.ini, args.textures, log=lambda m: print(m, file=sys.stderr))
    for out in args.output:
        kn5.write(model, out)
        print(f"wrote {out}: {len(model.textures)} textures, {len(model.materials)} materials")


if __name__ == "__main__":
    main()
